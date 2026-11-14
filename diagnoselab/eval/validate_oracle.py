"""Step 5 validation: F19 (oracle), plus the rows that needed it: F5, F17 (normalised), F18 (normalised).

Evaluation side, so the truth is allowed here; the oracle itself is checked for imports separately.
Writes results/step5_validation.json. Spec: docs/oracle_spec.md (sections 3-4), docs/scoring_spec.md section 4.
"""

from __future__ import annotations

import ast
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

from diagnoselab.baselines.policies import BASELINES
from diagnoselab.eval.run_oracle import TIERS, run_many
from diagnoselab.eval.validate_scoring import run_inprocess
from diagnoselab.scoring.reward import NORM_FLOOR, oracle_scores
from diagnoselab.sim.certifier import FD_STEP
from diagnoselab.sim.experiments import COSTS, experiment_sensors, experiment_setpoint
from diagnoselab.sim.faults import PHYSICAL_TYPES, apply_faults
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Fault
from diagnoselab.sim.sensors import physical_readings
from diagnoselab.sim.solver import solve

ROOT = Path(__file__).resolve().parents[2]
ORACLE_FILE = ROOT / "diagnoselab" / "oracle" / "oracle.py"
F19_SEEDS = range(0, 30)  # plan: 30 dev seeds per tier
F5_SEEDS = range(0, 200)  # plan: 200 accepted dev instances
HELDOUT = range(50, 100)  # reported separately: never used for tuning
SPEND_CAP, RUNTIME_CAP = 28, 120.0
N_RANDOM = 5
EXPERIMENTS = ("speed_step", "valve_step", "isolate_branch", "pump_off_static")


# --- probes as (kind, args) ---------------------------------------------------------------------
def parse_probe(label: str) -> tuple[str, dict]:
    kind, *rest = label.split(":")
    if kind == "read_sensor":
        return "read", {"sensor": rest[0]}
    if kind == "speed_step":
        return kind, {"s": float(rest[0]), **({"pump": rest[1]} if len(rest) > 1 else {})}
    if kind == "valve_step":
        return kind, {"valve": rest[0], "theta": float(rest[1])}
    if kind == "isolate_branch":
        return kind, {"branch": rest[0]}
    return "pump_off_static", {}


def probe_menu(spec) -> list[tuple[str, dict]]:
    """The oracle's probe kinds (docs/oracle_spec.md section 2) for the random comparator."""
    pumps = [ln.id for ln in spec.links if ln.kind == "pump"]
    menu = [("read", {"sensor": s.id}) for s in spec.sensors]
    menu += [("speed_step", {"s": 1.2}), ("pump_off_static", {})]
    menu += [("valve_step", {"valve": ln.id, "theta": 0.3}) for ln in spec.links if ln.kind == "valve"]
    menu += [("isolate_branch", {"branch": b.id}) for b in spec.branches]
    if len(pumps) == 1:
        menu.append(("speed_step", {"s": 0.75}))
    else:
        menu += [("speed_step", {"s": s, "pump": p}) for p in pumps for s in (0.75, 1.2)]
    return menu


# --- information efficiency (oracle_spec section 3) -----------------------------------------------
class TrueInfo:
    """Whitened finite-difference rows at the true plant, columns = true support (stuck sensors excluded)."""

    def __init__(self, inst):
        self.inst, self.spec = inst, inst.spec
        self.sigma = {s.id: s.sigma for s in self.spec.sensors}
        self.stuck = {f.component for f in inst.faults if f.type == "sensor_stuck"}
        self.phys = tuple(f for f in inst.faults if f.type in PHYSICAL_TYPES)
        self.bias = [f for f in inst.faults if f.type == "sensor_bias"]
        self.mags = np.array([f.magnitude for f in self.phys] + [f.magnitude for f in self.bias])
        self.prior = np.diag(1.0 / (10.0 * self.mags) ** 2)
        self._rows: dict = {}

    def rows(self, kind: str, args: dict) -> np.ndarray:
        key = (kind, tuple(sorted(args.items())))
        if key not in self._rows:
            if kind == "read":
                sp, sensors = experiment_setpoint(self.spec, "nominal"), [args["sensor"]]
            else:
                sp, sensors = experiment_setpoint(self.spec, kind, args), list(experiment_sensors(self.spec, kind, args))
            sensors = [s for s in sensors if s not in self.stuck]
            base = physical_readings(self.spec, solve(self.spec, apply_faults(self.spec, self.phys), sp))
            cols = []
            for f in self.phys:
                h = FD_STEP[f.type]
                moved = tuple(Fault(g.component, g.type, g.magnitude + (h if g is f else 0.0)) for g in self.phys)
                y = physical_readings(self.spec, solve(self.spec, apply_faults(self.spec, moved), sp))
                cols.append([(y[s] - base[s]) / h / self.sigma[s] for s in sensors])
            cols += [[1.0 if s == f.component else 0.0 for s in sensors] for f in self.bias]
            self._rows[key] = np.array(cols).T.reshape(len(sensors), len(self.mags))
        return self._rows[key]

    def info(self, probes: list[tuple[str, dict]]) -> float:
        m = self.prior.copy()
        for kind, args in probes:
            j = self.rows(kind, args)
            m += j.T @ j
        return float(np.linalg.slogdet(m)[1] - np.linalg.slogdet(self.prior)[1])


def cost(kind: str) -> int:
    return COSTS["read" if kind == "read" else kind]


def random_probes(spec, target: int, rng: random.Random) -> list[tuple[str, dict]]:
    menu, out, spent = probe_menu(spec), [], 0
    while True:
        fits = [p for p in menu if spent + cost(p[0]) <= target]
        if not fits:
            return out
        p = rng.choice(fits)
        out.append(p)
        spent += cost(p[0])


def info_efficiency(r: dict) -> dict:
    inst = make_instance(r["seed"], r["tier"])
    ti = TrueInfo(inst)
    probes = [parse_probe(t["probe"]) for t in r["trace"]]
    spend = int(sum(cost(k) for k, _ in probes))
    curve, acc = [], []
    for p in probes:
        acc.append(p)
        curve.append((int(sum(cost(k) for k, _ in acc)), ti.info(acc)))
    oracle_info = ti.info(probes)
    rand = [ti.info(random_probes(inst.spec, spend, random.Random(f"{r['tier']}:{r['seed']}:{k}")))
            for k in range(N_RANDOM)]
    return {"seed": r["seed"], "tier": r["tier"], "spend": spend, "oracle": oracle_info, "random": rand,
            "random_median": statistics.median(rand), "above": oracle_info > statistics.median(rand),
            "curve": curve}


# --- checks -------------------------------------------------------------------------------------
def static_imports() -> dict:
    tree = ast.parse(ORACLE_FILE.read_text())
    mods = sorted({(n.module or "") if isinstance(n, ast.ImportFrom) else a.name
                   for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in (n.names if isinstance(n, ast.Import) else [n])})
    top = sorted({m.split(".")[0] for m in mods if m})
    non_std = [m for m in top if m not in sys.stdlib_module_names]
    return {"modules": mods, "non_stdlib": non_std, "pass": not non_std}


def per_tier(results: list[dict], fn) -> dict:
    return {t: fn([r for r in results if r["tier"] == t]) for t in TIERS}


def summary(rs: list[dict]) -> dict:
    n = len(rs)
    usage = {k: sum(1 for r in rs if r["usage"].get(k)) / n for k in (*EXPERIMENTS, "read_sensor")}
    return {"n": n, "D": statistics.mean(r["score"]["D"] for r in rs), "R": statistics.mean(r["score"]["reward"] for r in rs),
            "S": statistics.mean(r["score"]["S"] for r in rs), "spent_mean": statistics.mean(r["score"]["spent"] for r in rs),
            "spent_max": max(r["score"]["spent"] for r in rs), "runtime_max": max(r["runtime_s"] for r in rs),
            "simulate_max": max(r["simulate_used"] for r in rs), "notes": sum(1 for r in rs if r["note"]),
            "errors": sum(1 for r in rs if r["error"]), "usage": usage,
            "D_ge_085": sum(1 for r in rs if r["score"]["D"] >= 0.85) / n,
            "R_min": min(r["score"]["reward"] for r in rs), "R_lt_05": sorted(r["seed"] for r in rs if r["score"]["reward"] < 0.5)}


def f19(main: list[dict], heldout: list[dict]) -> dict:
    lazy = json.loads((ROOT / "results" / "step4_validation.json").read_text())["F18"]["table"]["lazy_heuristic"]
    f19_runs = [r for r in main if r["seed"] in F19_SEEDS]
    repeats = [run_many(list(F19_SEEDS), list(TIERS), 8) for _ in range(2)]
    first = {(r["tier"], r["seed"]): (r["calls_hash"], r["score"]["reward"]) for r in f19_runs}
    same = all({(r["tier"], r["seed"]): (r["calls_hash"], r["score"]["reward"]) for r in rep} == first for rep in repeats)
    eff = [info_efficiency(r) for r in f19_runs]
    target = {"easy": 0.9, "medium": 0.9, "hard": 0.8}
    tiers = {}
    for t in TIERS:
        s = summary([r for r in f19_runs if r["tier"] == t])
        e = [x for x in eff if x["tier"] == t]
        s.update(D_target=target[t], D_pass=s["D"] >= target[t], spend_pass=s["spent_max"] <= SPEND_CAP,
                 runtime_pass=s["runtime_max"] < RUNTIME_CAP, info_above=sum(x["above"] for x in e) / len(e),
                 R_lazy=lazy[t]["mean"], gap=s["R"] - lazy[t]["mean"], gap_pass=s["R"] >= lazy[t]["mean"] + 0.3,
                 usage_pass=all(s["usage"][k] >= 0.10 for k in EXPERIMENTS) and s["usage"]["read_sensor"] >= 0.5)
        tiers[t] = s
    linux = json.loads((ROOT / "results" / "oracle_cache_linux.json").read_text())["records"]
    cross = [k for k, v in linux.items() if (k.split(":")[0], int(k.split(":")[1])) in first
             and first[(k.split(":")[0], int(k.split(":")[1]))][0] != v["calls_hash"]]
    return {"seeds": f"{F19_SEEDS.start}-{F19_SEEDS.stop - 1}", "tiers": tiers, "determinism_repeats": 3,
            "deterministic": same, "cross_platform_mismatch": cross,
            "info_efficiency": [{k: v for k, v in x.items()} for x in eff],
            "heldout": per_tier(heldout, summary), "static_imports": static_imports()}


def f5_f17(main: list[dict]) -> dict:
    cache = oracle_scores()
    recompute = {f"{r['tier']}:{r['seed']}": r["score"]["reward"] for r in main}
    mismatch = [k for k, v in recompute.items() if abs(cache.get(k, float("nan")) - v) > 1e-12]
    return {"F5": per_tier(main, lambda rs: {"n": len(rs), "D_ge_085": sum(r["score"]["D"] >= 0.85 for r in rs) / len(rs),
                                             "pass": sum(r["score"]["D"] >= 0.85 for r in rs) / len(rs) >= 0.9}),
            "F17": {"cache_entries": len(cache), "cache_equals_recompute": not mismatch, "mismatch": mismatch,
                    "per_tier": per_tier(main, lambda rs: {"R_min": min(r["score"]["reward"] for r in rs),
                                                           "below_floor": sorted(r["seed"] for r in rs if r["score"]["reward"] < NORM_FLOOR)}),
                    "floor": NORM_FLOOR}}


def f18_normalized(n_seeds: int = 50) -> dict:
    cache, table = oracle_scores(), {}
    for name, policy in BASELINES.items():
        table[name] = {}
        for t in TIERS:
            vals = []
            for s in range(n_seeds):
                r_o = cache.get(f"{t}:{s}")
                if r_o is None or r_o < NORM_FLOOR:
                    continue
                vals.append(min(run_inprocess(policy, make_instance(s, t), s)["reward"] / r_o, 1.0))
            table[name][t] = {"mean": statistics.mean(vals), "max": max(vals), "n": len(vals)}
    return {"seeds": f"0-{n_seeds - 1}", "table": table}


def main() -> None:
    main_runs = run_many(list(F5_SEEDS), list(TIERS), 8)
    heldout = [r for r in main_runs if r["seed"] in HELDOUT]
    out = {"F19": f19(main_runs, heldout), **f5_f17(main_runs), "F18_normalized": f18_normalized(),
           "all_200": per_tier(main_runs, summary)}
    path = ROOT / "results" / "step5_validation.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    for t in TIERS:
        s = out["F19"]["tiers"][t]
        print(t, {k: s[k] for k in ("D", "R", "spent_max", "info_above", "gap", "usage")})
    print("deterministic", out["F19"]["deterministic"], "cross-platform mismatches", out["F19"]["cross_platform_mismatch"])
    print("F17", out["F17"]["cache_equals_recompute"], "F5", {t: v["D_ge_085"] for t, v in out["F5"].items()})


if __name__ == "__main__":
    main()
