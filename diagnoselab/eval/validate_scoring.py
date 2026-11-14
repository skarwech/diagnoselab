"""Measure the F15-F18 valid-when conditions (plan section 8) -> results/step4_validation.json.

Run: uv run python -m diagnoselab.eval.validate_scoring   (a few minutes)
"""

from __future__ import annotations

import itertools
import json
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from diagnoselab.baselines.policies import BASELINES
from diagnoselab.scoring.reward import (diagnosis_score, efficiency, magnitude_credit, restoration_score,
                                        score_episode, total_reward)
from diagnoselab.server.episode import Episode
from diagnoselab.server.tools import ToolRejected
from diagnoselab.sim.faults import PHYSICAL_TYPES, sample_faults
from diagnoselab.sim.generator import FAULTS, generate_plant, stream
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Fault, Instance

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("easy", "medium", "hard")
REPAIR = {"fouling": "clean_pipe", "stuck_valve": "free_valve", "pump_wear": "replace_pump", "leak": "fix_leak",
          "sensor_bias": "recalibrate_sensor", "sensor_stuck": "recalibrate_sensor"}


def entry(f: Fault, magnitude: float | None = None) -> dict:
    out = {"component": f.component, "type": f.type}
    if f.type != "sensor_stuck":
        out["magnitude"] = f.magnitude if magnitude is None else magnitude
    return out


# --- F15 ----------------------------------------------------------------------------------------
def f15(n_cases: int = 100_000) -> dict:
    t = [Fault("P1", "fouling", 1.0), Fault("V1", "stuck_valve", 0.5), Fault("PT-J1", "sensor_stuck", None)]
    units = {
        "perfect": (diagnosis_score(t, [entry(f) for f in t])["D"], 1.0),
        "empty": (diagnosis_score(t, [])["D"], 0.0),
        "all_wrong": (diagnosis_score(t, [{"component": "P9", "type": "fouling", "magnitude": 1.0}] * 3)["D"], 0.0),
        "boundary_0.30": (magnitude_credit(1.3, 1.0), 1.0),
        "boundary_0.60": (magnitude_credit(0.16, 0.1), 0.5),
        "beyond_0.30": (magnitude_credit(1.300002, 1.0), 0.5),
        "beyond_0.60": (magnitude_credit(1.600002, 1.0), 0.0),
        "stuck_no_magnitude": (diagnosis_score(t[2:], [entry(t[2])])["D"], 1.0),
        "wild_magnitude_is_match_not_fp": (diagnosis_score(t[:1], [entry(t[0], 99.0)])["false_positives"], 0.0),
        "duplicate_is_fp": (diagnosis_score(t[:1], [entry(t[0]), entry(t[0])])["false_positives"], 1.0),
    }
    rng = random.Random(15)
    pool = [(f"P{i}", "fouling") for i in range(8)] + [(f"J{i}", "leak") for i in range(6)] + [
        (f"PT-J{i}", "sensor_bias") for i in range(4)] + [(f"FT-P{i}", "sensor_stuck") for i in range(3)]
    viol = {"range": 0, "permutation": 0, "add_correct": 0, "add_correct_type_any_magnitude": 0, "add_wrong": 0}
    for _ in range(n_cases):
        truth = [Fault(c, ty, None if ty == "sensor_stuck" else round(rng.uniform(0.05, 5.0), 4))
                 for c, ty in rng.sample(pool, rng.randint(1, 5))]
        diag = []
        for _ in range(rng.randint(0, 8)):
            if rng.random() < 0.5:
                f = rng.choice(truth)
                diag.append(entry(f, None if f.type == "sensor_stuck" else f.magnitude * rng.uniform(0.2, 2.5)))
            else:
                c, ty = rng.choice(pool)
                diag.append({"component": c, "type": ty, **({} if ty == "sensor_stuck" else {"magnitude": 1.0})})
        d = diagnosis_score(truth, diag)["D"]
        viol["range"] += not 0.0 <= d <= 1.0
        shuffled = diag[:]
        rng.shuffle(shuffled)
        viol["permutation"] += abs(diagnosis_score(truth, shuffled)["D"] - d) > 1e-12
        fresh = [f for f in truth if (f.component, f.type) not in {(e["component"], e["type"]) for e in diag}]
        if fresh:  # a true fault not yet submitted (a resubmitted one is a duplicate, i.e. a false positive)
            f = rng.choice(fresh)
            viol["add_correct"] += diagnosis_score(truth, diag + [entry(f)])["D"] < d - 1e-12
            any_mag = entry(f, None if f.type == "sensor_stuck" else rng.uniform(0, 50))
            viol["add_correct_type_any_magnitude"] += diagnosis_score(truth, diag + [any_mag])["D"] < d - 1e-12
        wrong = {"component": "NOPE", "type": "fouling", "magnitude": 1.0}
        viol["add_wrong"] += diagnosis_score(truth, diag + [wrong])["D"] > d + 1e-12
    return {"units": {k: {"got": v[0], "want": v[1], "ok": abs(v[0] - v[1]) < 1e-12} for k, v in units.items()},
            "property_cases": n_cases, "property_violations": viol}


# --- F16 ----------------------------------------------------------------------------------------
def repairs_for(faults) -> list[dict]:
    return [{"action": REPAIR[f.type], "target": f.component} for f in faults]


def f16(n_instances: int = 10_000) -> dict:
    out = {"instances": n_instances, "exact_not_1": 0, "none_not_0": 0, "partial_violations": 0, "worst_drop": 0.0,
           "min_gap": float("inf"), "healthy_repair_wrong": 0, "duplicate_wrong": 0, "order_dependent": 0}
    for k in range(n_instances):
        tier = TIERS[k % 3]
        spec = generate_plant(k // 3 % 400, tier)
        faults = sample_faults(spec, tier, np.random.default_rng([k, 16]))  # uncertified draws: fast, wider
        inst = Instance(spec, faults, None, 0)
        phys = [f for f in faults if f.type in PHYSICAL_TYPES]
        exact = restoration_score(inst, repairs_for(faults))  # sensor recalibrations of faulty sensors included
        out["exact_not_1"] += abs(exact["S"] - 1.0) > 1e-9
        out["none_not_0"] += restoration_score(inst, [])["S"] != 0.0
        out["min_gap"] = min(out["min_gap"], exact["J_healthy"] - exact["J_before"])
        for r in range(1, len(phys)):
            for subset in itertools.combinations(phys, r):
                s = restoration_score(inst, repairs_for(subset))
                drop = s["J_before"] - s["J_after"]
                out["partial_violations"] += drop > 1e-9
                out["worst_drop"] = max(out["worst_drop"], drop)
        if k % 10 == 0:
            healthy = next(ln.id for ln in spec.links if ln.kind == "pipe" and ln.id not in {f.component for f in faults})
            plus = restoration_score(inst, repairs_for(faults) + [{"action": "clean_pipe", "target": healthy}])
            out["healthy_repair_wrong"] += abs(plus["S"] - 0.85) > 1e-9 or abs(plus["J_after"] - exact["J_after"]) > 1e-12
            dup = restoration_score(inst, repairs_for(faults) + repairs_for(phys[:1]))
            out["duplicate_wrong"] += abs(dup["S"] - 0.85) > 1e-9
            rev = restoration_score(inst, list(reversed(repairs_for(faults))))
            out["order_dependent"] += abs(rev["S"] - exact["S"]) > 1e-12
    return out


def blanket_attack(n_seeds: int = 60) -> dict:
    """Empty diagnosis + k random distinct repairs, k = 1..10, on certified instances (S is the only term)."""
    rng = random.Random(18)
    by_k = {k: [] for k in range(1, 11)}
    for seed in range(n_seeds):
        for tier in TIERS:
            inst = make_instance(seed, tier)
            comps = [(ln.id, ln.kind) for ln in inst.spec.links] + [(j.id, "junction") for j in inst.spec.junctions]
            comps += [(s.id, "sensor") for s in inst.spec.sensors]
            action = {"pipe": "clean_pipe", "valve": "free_valve", "pump": "replace_pump", "junction": "fix_leak",
                      "sensor": "recalibrate_sensor"}
            for k in by_k:
                pick = rng.sample(comps, k)
                s = restoration_score(inst, [{"action": action[kd], "target": c} for c, kd in pick])["S"]
                by_k[k].append(total_reward(0.0, s, 0.0))
    return {str(k): {"mean_R": statistics.mean(v), "max_R": max(v)} for k, v in by_k.items()}


# --- F17 ----------------------------------------------------------------------------------------
def f17() -> dict:
    grid = [round(x, 2) for x in np.linspace(0, 1, 101)]
    monotone = all(total_reward(d, s, sp) > total_reward(d, s, sp + 1) for d in grid if d >= 0.7
                   for s in (0.0, 0.5, 1.0) for sp in range(40))
    rng = random.Random(17)
    in_range = all(0.0 <= total_reward(rng.random(), rng.random(), rng.uniform(0, 40)) <= 1.0 for _ in range(100_000))
    return {"stop_early_efficiency": efficiency(0.0, 0.0), "efficiency_below_0.4": efficiency(0.39, 0.0),
            "lower_spend_strictly_higher": monotone, "R_in_range": in_range}


# --- F18 ----------------------------------------------------------------------------------------
def run_inprocess(policy, inst: Instance, seed: int) -> dict:
    ep = Episode(inst)
    try:
        policy(ep.call, random.Random(seed))
    except ToolRejected:
        pass  # a policy that trips a rejection simply stops; it is scored on what it submitted
    return score_episode(inst, ep._state)


def f18(n_seeds: int = 50) -> dict:
    table = {}
    for name, policy in BASELINES.items():
        table[name] = {}
        for tier in TIERS:
            scores = [run_inprocess(policy, make_instance(s, tier), s) for s in range(n_seeds)]
            r = [sc["reward"] for sc in scores]
            table[name][tier] = {"mean": statistics.mean(r), "sd": statistics.pstdev(r), "max": max(r),
                                 "D": statistics.mean(sc["D"] for sc in scores),
                                 "S": statistics.mean(sc["S"] for sc in scores),
                                 "spent": statistics.mean(sc["spent"] for sc in scores)}
    return {"seeds_per_tier": n_seeds, "table": table}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mcp_crosscheck(seeds: tuple[int, ...] = (0, 1), policies: tuple[str, ...] = ("lazy_heuristic", "random_guess",
                                                                                 "everything_faulty")) -> dict:
    """The same policy over the real MCP transport scores exactly as in-process (server finalize vs in-process)."""
    import anyio
    from mcp import Client

    rows = []
    for tier in TIERS:
        for seed in seeds:
            for name in policies:
                port, state = _free_port(), Path(tempfile.mkdtemp())
                proc = subprocess.Popen([sys.executable, "-m", "diagnoselab.server.app", "--host", "127.0.0.1",
                                         "--port", str(port), "--seed", str(seed), "--tier", tier, "--state",
                                         str(state)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    for _ in range(300):
                        try:
                            socket.create_connection(("127.0.0.1", port), 0.2).close()
                            break
                        except OSError:
                            time.sleep(0.1)
                    with anyio.from_thread.start_blocking_portal() as portal:
                        with portal.wrap_async_context_manager(Client(f"http://127.0.0.1:{port}/mcp")) as client:
                            def call(tool: str, args: dict) -> dict:
                                r = portal.call(client.call_tool, tool, args)
                                if r.is_error:
                                    code, _, msg = r.content[0].text.partition(": ")
                                    raise ToolRejected(code, msg)
                                return r.structured_content
                            try:
                                BASELINES[name](call, random.Random(seed))
                            except ToolRejected:
                                pass
                    fin = subprocess.run([sys.executable, "-m", "diagnoselab.server.finalize", "--state", str(state),
                                          "--out", str(state / "score.json")], cwd=ROOT, capture_output=True,
                                         text=True, check=True)
                    via_mcp = json.loads((state / "score.json").read_text())
                finally:
                    proc.kill()
                inproc = run_inprocess(BASELINES[name], make_instance(seed, tier), seed)
                rows.append({"tier": tier, "seed": seed, "policy": name, "mcp": via_mcp["reward"],
                             "inprocess": inproc["reward"], "identical": via_mcp == inproc})
    return {"rows": rows, "all_identical": all(r["identical"] for r in rows)}


def static_import_check() -> dict:
    src = (ROOT / "diagnoselab/baselines/policies.py").read_text()
    bad = [line for line in src.splitlines() if line.startswith(("import", "from")) and "diagnoselab" in line
           and "diagnoselab.server.tools import ToolRejected" not in line]
    return {"forbidden_imports": bad}


def main() -> None:
    out = {}
    for name, fn in (("F15", f15), ("F16", f16), ("blanket_attack", blanket_attack), ("F17", f17), ("F18", f18),
                     ("mcp_crosscheck", mcp_crosscheck), ("static_imports", static_import_check)):
        t0 = time.perf_counter()
        out[name] = fn()
        print(f"{name} done in {time.perf_counter() - t0:.1f}s", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/step4_validation.json").write_text(json.dumps(out, default=float, indent=1))
    print("wrote results/step4_validation.json")


if __name__ == "__main__":
    main()
