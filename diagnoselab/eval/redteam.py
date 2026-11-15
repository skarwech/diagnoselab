"""F23 red-team suite: every attack from plan section 5.3 as a measured run -> results/step8_redteam.json.

    uv run python -m diagnoselab.eval.redteam

Scoring attacks report the normalised score (oracle cache, frozen configuration) on dev seeds; the bar is 0.10 and
the lazy heuristic is a baseline, not an attack. Information attacks compare two instances with the SAME plant and
DIFFERENT hidden faults: anything an agent can observe must be identical between them.
"""

from __future__ import annotations

import json
import random
import statistics
import threading
from pathlib import Path

import numpy as np

from diagnoselab.baselines.policies import BASELINES, MID, REPAIR, _components
from diagnoselab.scoring.reward import normalization, score_episode
from diagnoselab.server.episode import Episode, response_hash
from diagnoselab.server.tools import ToolRejected, public_topology
from diagnoselab.sim.faults import sample_faults
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Instance

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("easy", "medium", "hard")
SEEDS = range(50)
BAR = 0.10


def normalised(inst: Instance, ep: Episode, seed: int, tier: str) -> float:
    s = score_episode(inst, ep._state)
    n = normalization(s["reward"], seed, tier)
    return n["score_norm"] if n["normalized"] else float("nan")


def play(policy, seed: int, tier: str) -> float:
    inst = make_instance(seed, tier)
    ep = Episode(inst)
    try:
        policy(ep.call, random.Random(seed))
    except ToolRejected:
        pass
    return normalised(inst, ep, seed, tier)


def summary(values: list[float]) -> dict:
    v = [x for x in values if x == x]
    return {"mean": statistics.fmean(v), "max": max(v), "n": len(v)}


# --- scoring attacks -----------------------------------------------------------------------------
def blanket(k: int):
    def policy(call, rng):
        comps = [c for c in _components(call("get_topology", {})) if c[1] != "sensor_bias"]
        call("submit", {"diagnosis": [], "repairs": [{"action": REPAIR[t], "target": c} for c, t in rng.sample(comps, k)]})
    return policy


def static_values(topo: dict) -> dict[str, float]:
    heads = {s["id"]: s["head"] for s in topo["sources"]}
    feed = {ln["to"]: heads[ln["frm"]] for ln in topo["links"] if ln["frm"] in heads}
    return {j["id"]: (feed.get(j["id"], topo["tank_head"]) if j["side"] == "suction" else topo["tank_head"]) - j["z"]
            for j in topo["junctions"]}


def static_sweep(blind_repairs: int):
    """One static test; flag every pressure/flow sensor more than 5 sigma off its zero-flow value as biased,
    plus `blind_repairs` random physical repairs (the Step 4 combined attack)."""
    def policy(call, rng):
        topo = call("get_topology", {})
        static = static_values(topo)
        sig = {s["id"]: (s["sigma"], s["kind"], s["target"]) for s in topo["sensors"]}
        r = call("run_experiment", {"type": "pump_off_static", "args": {}})["readings"]
        diag = []
        for sid, v in r.items():
            sigma, kind, target = sig[sid]
            dev = (v - static[target]) / sigma if kind == "pressure" else v / sigma
            if abs(dev) > 5:
                diag.append({"component": sid, "type": "sensor_bias", "magnitude": round(abs(dev), 6)})
        phys = [c for c in _components(topo) if c[1] != "sensor_bias"]
        repairs = [{"action": REPAIR[t], "target": c} for c, t in rng.sample(phys, blind_repairs)]
        call("submit", {"diagnosis": diag[:10], "repairs": repairs})
    return policy


def guess_all_types(n: int):
    """Topology-only: name n random components with their kind's fault at mid magnitude and repair them."""
    def policy(call, rng):
        comps = [c for c in _components(call("get_topology", {})) if c[1] != "sensor_bias"]
        pick = rng.sample(comps, n)
        call("submit", {"diagnosis": [{"component": c, "type": t, "magnitude": MID[t]} for c, t in pick],
                        "repairs": [{"action": REPAIR[t], "target": c} for c, t in pick]})
    return policy


def scoring_attacks() -> dict:
    attacks = {f"baseline:{n}": p for n, p in BASELINES.items() if n != "lazy_heuristic"}
    attacks |= {f"blanket_repair_k{k}": blanket(k) for k in (1, 2, 3, 5, 10)}
    attacks |= {f"guess_{n}_components": guess_all_types(n) for n in (1, 2, 3)}
    attacks |= {f"static_sweep_plus_{b}_blind_repairs": static_sweep(b) for b in (0, 1, 2)}
    out = {}
    for name, policy in attacks.items():
        out[name] = {t: summary([play(policy, s, t) for s in SEEDS]) for t in TIERS}
        worst = max(v["mean"] for v in out[name].values())
        out[name]["worst_tier_mean"] = worst
        out[name]["pass"] = worst <= BAR
        print(f"{name}: worst tier mean {worst:.3f}", flush=True)
    out["lazy_heuristic (baseline, not an attack)"] = {t: summary([play(BASELINES["lazy_heuristic"], s, t) for s in SEEDS])
                                                       for t in TIERS}
    return out


# --- information attacks: same plant, different hidden faults -------------------------------------
def twin_pair(seed: int, tier: str) -> tuple[Instance, Instance]:
    a = make_instance(seed, tier)
    b = Instance(a.spec, sample_faults(a.spec, tier, np.random.default_rng(10_000 + seed)), None, a.noise_seed)
    return a, b


def probe_calls(topo: dict) -> list[tuple[str, dict]]:
    links = topo["links"]
    junction = topo["junctions"][0]["id"]
    valve = next((ln["id"] for ln in links if ln["kind"] == "valve"), None)
    sensor = topo["sensors"][0]["id"]
    calls = [("simulate", {"overrides": []}),
             ("simulate", {"overrides": [{"component": junction, "type": "leak", "magnitude": 0.05}],
                           "setpoint": {"experiment": "pump_off_static"}}),
             ("simulate", {"overrides": [{"component": sensor, "type": "sensor_stuck"}]}),
             ("simulate", {"overrides": [{"component": links[0]["id"], "type": "fouling", "magnitude": 1.0}],
                           "setpoint": {"experiment": "speed_step", "s": 0.8}})]
    if valve:
        calls.append(("simulate", {"overrides": [], "setpoint": {"experiment": "valve_step", "valve": valve, "theta": 0.3}}))
    calls += [("read_manual", {"topic": "faults"}), ("get_topology", {}), ("budget_status", {})]
    bad = [("read_sensor", {"sensor_id": "NOPE"}), ("read_sensor", {"sensor_id": sensor, "x": 1}),
           ("run_experiment", {"type": "speed_step", "args": {"s": 1.0}}),
           ("run_experiment", {"type": "valve_step", "args": {"valve": "V999", "theta": 0.3}}),
           ("simulate", {"overrides": [{"component": junction, "type": "fouling", "magnitude": 1}]}),
           ("simulate", {"overrides": [{"component": junction, "type": "leak", "magnitude": float("nan")}]}),
           ("simulate", {"overrides": [{"component": junction, "type": "leak", "magnitude": 10 ** 400}]}),
           ("submit", {"diagnosis": [{"component": "ZZZ", "type": "leak", "magnitude": 1}], "repairs": []}),
           ("nonexistent_tool", {}), ("read_manual", {"topic": "secrets"}), ("get_topology", {"seed": 1}),
           ("simulate", {"overrides": [{"component": junction, "type": "leak", "magnitude": 0.1}] * 2}),
           ("simulate", {"overrides": [], "sensors": ["x" * 20000]})]
    return calls + bad


def observe(inst: Instance, calls: list[tuple[str, dict]]) -> list[str]:
    ep = Episode(inst)
    out = []
    for tool, args in calls:
        try:
            out.append("ok:" + response_hash(ep.call(tool, args)))
        except ToolRejected as e:
            out.append(f"err:{e.code}:{e.message}")
    return out


def information_attacks(n: int = 30) -> dict:
    diffs, internal, total = [], 0, 0
    for seed in range(n):
        for tier in TIERS:
            a, b = twin_pair(seed, tier)
            calls = probe_calls(public_topology(a.spec))
            oa, ob = observe(a, calls), observe(b, calls)
            total += len(calls)
            internal += sum("E_INTERNAL" in x for x in oa + ob)
            diffs += [f"{tier}:{seed}:{calls[i][0]}" for i, (x, y) in enumerate(zip(oa, ob)) if x != y]
    return {"pairs": n * 3, "calls_per_pair": total // (n * 3), "observable_differences": diffs,
            "internal_errors": internal, "pass": not diffs and internal == 0}


def submit_probing(n: int = 30) -> dict:
    """Resubmission and acknowledgement probing: the ack never depends on correctness, a second submit is
    rejected, and only the first counts."""
    acks, second_ok, scored_first = set(), 0, 0
    for seed in range(n):
        for tier in TIERS:
            inst = make_instance(seed, tier)
            right = [{"component": f.component, "type": f.type, **({"magnitude": f.magnitude} if f.magnitude else {})}
                     for f in inst.faults]
            for first, second in ((right, []), ([], right)):
                ep = Episode(inst)
                acks.add(response_hash(ep.call("submit", {"diagnosis": first, "repairs": []})))
                try:
                    ep.call("submit", {"diagnosis": second, "repairs": []})
                    second_ok += 1
                except ToolRejected as e:
                    pass
                stored = ep._state["submission"]["diagnosis"]
                scored_first += len(stored) == len(first)
    return {"distinct_acks": len(acks), "second_submit_accepted": second_ok, "first_counted": scored_first,
            "episodes": n * 3 * 2, "pass": len(acks) == 1 and second_ok == 0 and scored_first == n * 3 * 2}


def budget_race(trials: int = 20, threads: int = 8, per_thread: int = 10) -> dict:
    """Concurrent costed calls must never overspend: exactly B reads succeed, the rest are E_BUDGET."""
    worst = 0
    bad = 0
    for t in range(trials):
        inst = make_instance(t, "medium")
        ep = Episode(inst)
        sid = inst.spec.sensors[0].id
        ok = []

        def worker():
            for _ in range(per_thread):
                try:
                    ep.call("read_sensor", {"sensor_id": sid})
                    ok.append(1)
                except ToolRejected:
                    pass

        ts = [threading.Thread(target=worker) for _ in range(threads)]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        worst = max(worst, ep._state["spent"])
        bad += ep._state["spent"] > 40 or len(ok) != min(40, threads * per_thread)
    return {"trials": trials, "max_spent": worst, "violations": bad, "pass": bad == 0}


def cross_talk(n: int = 20) -> dict:
    """Two interleaved episodes give exactly the responses each gives alone (no shared state or noise)."""
    mismatch = 0
    for seed in range(n):
        a, b = make_instance(seed, "medium"), make_instance(seed, "hard")
        def script(inst):
            sids = [s.id for s in inst.spec.sensors][:5]
            return [("read_sensor", {"sensor_id": s}) for s in sids] + [("run_experiment", {"type": "pump_off_static", "args": {}})]
        alone_a, alone_b = observe(a, script(a)), observe(b, script(b))
        ea, eb, inter_a, inter_b = Episode(a), Episode(b), [], []
        for (ta, xa), (tb, xb) in zip(script(a), script(b)):
            inter_a.append("ok:" + response_hash(ea.call(ta, xa)))
            inter_b.append("ok:" + response_hash(eb.call(tb, xb)))
        mismatch += (inter_a != alone_a) + (inter_b != alone_b)
    return {"pairs": n, "mismatches": mismatch, "pass": mismatch == 0}


def timing() -> dict:
    from diagnoselab.eval.validate_server import f6_timing
    t = f6_timing()
    return {**t, "pass": t["p_padded"] > 0.01}


def main() -> None:
    out = {"bar": BAR, "seeds": f"dev {SEEDS.start}-{SEEDS.stop - 1}", "scoring": scoring_attacks(),
           "information": information_attacks(), "submit_probing": submit_probing(), "budget_race": budget_race(),
           "cross_talk": cross_talk(), "timing": timing()}
    iso = ROOT / "results" / "isolation.json"
    out["isolation"] = json.loads(iso.read_text()) if iso.exists() else None
    (ROOT / "results" / "step8_redteam.json").write_text(json.dumps(out, indent=1, default=float))
    for k in ("information", "submit_probing", "budget_race", "cross_talk", "timing"):
        print(k, {x: y for x, y in out[k].items() if x not in ("observable_differences", "median_ms")})


if __name__ == "__main__":
    main()
