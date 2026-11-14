"""F18 baselines (docs/scoring_spec.md section 7). Each policy sees only call(tool, args), exactly like an agent:
the same code runs in-process (Episode.call) and over MCP. Nothing here may import the simulator or server
internals (checked by tests/test_f18_baselines.py)."""

from __future__ import annotations

import random
from typing import Any, Callable

from diagnoselab.server.tools import ToolRejected

Call = Callable[[str, dict], dict]
KIND_TYPE = {"pipe": "fouling", "valve": "stuck_valve", "pump": "pump_wear", "junction": "leak", "sensor": "sensor_bias"}
REPAIR = {"fouling": "clean_pipe", "stuck_valve": "free_valve", "pump_wear": "replace_pump", "leak": "fix_leak",
          "sensor_bias": "recalibrate_sensor", "sensor_stuck": "recalibrate_sensor"}
RANGE = {"fouling": (0.3, 2.0), "stuck_valve": (0.2, 0.8), "pump_wear": (0.10, 0.35), "leak": (0.02, 0.08),
         "sensor_bias": (4.0, 10.0)}  # typical ranges from the manual
MID = {t: (lo + hi) / 2 for t, (lo, hi) in RANGE.items()}


def _components(topo: dict) -> list[tuple[str, str]]:
    """(component id, its kind's fault type) for every diagnosable component."""
    out = [(ln["id"], KIND_TYPE[ln["kind"]]) for ln in topo["links"]]
    out += [(j["id"], "leak") for j in topo["junctions"]]
    return out + [(s["id"], "sensor_bias") for s in topo["sensors"]]


def _submit(call: Call, picks: list[tuple[str, str, float]], repair: bool = True) -> None:
    diagnosis = [{"component": c, "type": t, "magnitude": round(m, 6)} for c, t, m in picks]
    repairs = [{"action": REPAIR[t], "target": c} for c, t, _ in picks] if repair else []
    call("submit", {"diagnosis": diagnosis, "repairs": repairs})


def noop(call: Call, rng: random.Random) -> None:
    """Never calls anything and never submits."""


def stop_early(call: Call, rng: random.Random) -> None:
    call("submit", {"diagnosis": [], "repairs": []})


def everything_faulty(call: Call, rng: random.Random) -> None:
    comps = rng.sample(_components(call("get_topology", {})), 10)
    _submit(call, [(c, t, MID[t]) for c, t in comps])


def blanket_repair(call: Call, rng: random.Random) -> None:
    comps = rng.sample(_components(call("get_topology", {})), 10)
    call("submit", {"diagnosis": [], "repairs": [{"action": REPAIR[t], "target": c} for c, t in comps]})


def random_guess(call: Call, rng: random.Random) -> None:
    topo = call("get_topology", {})
    sensors = [s["id"] for s in topo["sensors"]]
    for _ in range(20):
        call("read_sensor", {"sensor_id": rng.choice(sensors)})
    comps = rng.sample(_components(topo), rng.randint(1, 3))
    _submit(call, [(c, t, rng.uniform(*RANGE[t])) for c, t in comps])


def prior_guess(call: Call, rng: random.Random) -> None:
    """Most common fault type (the kind with the most components) on two random components, no probes."""
    topo = call("get_topology", {})
    by_type: dict[str, list[str]] = {}
    for c, t in _components(topo):
        if t != "sensor_bias":
            by_type.setdefault(t, []).append(c)
    t = max(by_type, key=lambda k: len(by_type[k]))
    _submit(call, [(c, t, MID[t]) for c in rng.sample(by_type[t], min(2, len(by_type[t])))])


def lazy_heuristic(call: Call, rng: random.Random, min_drop: float = 25.0, max_faults: int = 4) -> None:
    """Read every sensor once at nominal, then greedily add the single fault whose linearised simulate response
    best explains the whitened residual, while the squared residual drops by >= min_drop (5 sigma)."""
    topo = call("get_topology", {})
    sigma = {s["id"]: s["sigma"] for s in topo["sensors"]}
    readings: dict[str, float] = {}
    for sid in sigma:
        try:
            readings[sid] = call("read_sensor", {"sensor_id": sid})["value"]
        except ToolRejected:
            break  # budget exhausted
    ids = list(readings)
    base = call("simulate", {"overrides": [], "sensors": ids})["readings"]
    residual = [(readings[s] - base[s]) / sigma[s] for s in ids]
    columns: dict[tuple[str, str], list[float]] = {}
    for c, t in _components(topo):
        if t == "sensor_bias":
            if c in readings:
                columns[(c, t)] = [1.0 if s == c else 0.0 for s in ids]  # bias in sigma units moves one reading
            continue
        sim = call("simulate", {"overrides": [{"component": c, "type": t, "magnitude": MID[t]}], "sensors": ids})
        columns[(c, t)] = [(sim["readings"][s] - base[s]) / sigma[s] / MID[t] for s in ids]
    picks: list[tuple[str, str, float]] = []
    for _ in range(max_faults):
        best = None
        for key, col in columns.items():
            if key[0] in {p[0] for p in picks}:
                continue
            norm = sum(v * v for v in col)
            if norm == 0.0:
                continue
            m = sum(r * v for r, v in zip(residual, col)) / norm
            if key[1] != "sensor_bias" and m <= 0:
                continue  # physical magnitudes are non-negative
            drop = m * m * norm
            if best is None or drop > best[0]:
                best = (drop, key, m, col)
        if best is None or best[0] < min_drop:
            break
        _, (c, t), m, col = best
        residual = [r - m * v for r, v in zip(residual, col)]
        picks.append((c, t, abs(m)))
    _submit(call, picks)


BASELINES: dict[str, Callable[[Call, random.Random], None]] = {
    "noop": noop, "stop_early": stop_early, "everything_faulty": everything_faulty,
    "blanket_repair": blanket_repair, "random_guess": random_guess, "prior_guess": prior_guess,
    "lazy_heuristic": lazy_heuristic,
}
