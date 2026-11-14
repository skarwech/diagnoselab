"""F8 simulate (digital twin): fidelity, hidden-state isolation, rate limit, no effect on budget or plant.

Contract: docs/server_spec.md sections 3, 5, 6, 10; plan section 8, F8.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np
import pytest

from diagnoselab.server.episode import SIMULATE_CAP, Episode, ToolRejected
from diagnoselab.sim.experiments import experiment_setpoint
from diagnoselab.sim.faults import apply_faults, sample_faults
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Fault, Instance, PlantSpec
from diagnoselab.sim.sensors import physical_readings
from diagnoselab.sim.solver import solve

CASES = [(1, "easy"), (2, "medium"), (4, "medium"), (3, "hard")]


@functools.lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


@functools.lru_cache(maxsize=None)
def twin(seed: int, tier: str) -> Instance:
    """Same nominal plant and noise seed, different hidden faults."""
    base = inst(seed, tier)
    rng = np.random.default_rng(20_000 + seed)
    for _ in range(50):
        other = sample_faults(base.spec, tier, rng)
        if {f.component for f in other} != {f.component for f in base.faults}:
            return Instance(base.spec, other, None, base.noise_seed)
    raise AssertionError("could not draw a different fault set")


def ids(spec: PlantSpec, kind: str) -> list[str]:
    return [ln.id for ln in spec.links if ln.kind == kind]


def running(instance: Instance) -> Episode:
    ep = Episode(instance, clock=lambda: 0.0)
    ep.call("budget_status", {})  # INIT -> RUNNING before any hash comparison
    return ep


def override(f: Fault) -> dict:
    """A true fault as a simulate override (section 5): sign for bias, value for stuck, no magnitude for stuck."""
    d: dict[str, Any] = {"component": f.component, "type": f.type}
    if f.type != "sensor_stuck":
        d["magnitude"] = f.magnitude
    if f.type == "sensor_bias":
        d["sign"] = f.sign
    if f.type == "sensor_stuck":
        d["value"] = f.value
    return d


def setpoints(spec: PlantSpec) -> list[tuple[str, dict]]:
    pumps = ids(spec, "pump")
    out = [("nominal", {}), ("speed_step", {"s": 0.85}), ("speed_step", {"s": 1.1, "pump": pumps[-1]}),
           ("pump_off_static", {})]
    out += [("valve_step", {"valve": v, "theta": 0.4}) for v in ids(spec, "valve")]
    return out + [("isolate_branch", {"branch": b.id}) for b in spec.branches]


def reference(instance: Instance, kind: str, args: dict, faults: tuple[Fault, ...] | None = None) -> dict[str, float]:
    """Sim-side noise-free readings: physics of the faulted plant, then sensor bias / stuck (sim_spec 4-5)."""
    spec, faults = instance.spec, instance.faults if faults is None else faults
    phys = physical_readings(spec, solve(spec, apply_faults(spec, faults), experiment_setpoint(spec, kind, args)))
    sigma = {s.id: s.sigma for s in spec.sensors}
    by_sensor = {f.component: f for f in faults if f.type in ("sensor_bias", "sensor_stuck")}

    def sensed(sid: str, v: float) -> float:
        f = by_sensor.get(sid)
        if f is None:
            return v
        return float(f.value) if f.type == "sensor_stuck" else v + f.sign * f.magnitude * sigma[sid]

    return {sid: sensed(sid, v) for sid, v in phys.items()}


def same(a: float, b: float, tol: float = 1e-9) -> bool:
    """Equal to tol; if one side is already rounded to 6 decimals (spec section 3), compare it with the other
    side rounded the same way."""
    return (abs(a - b) <= tol or (a == round(a, 6) and abs(a - round(b, 6)) <= tol)
            or (b == round(b, 6) and abs(b - round(a, 6)) <= tol))


@pytest.mark.parametrize("case", CASES, ids=[f"{s}-{t}" for s, t in CASES])
def test_true_overrides_match_true_readings(case: tuple[int, str]) -> None:
    """F8 valid-when 'fidelity below 1e-9': simulate with the true faults as overrides equals
    ep.true_readings(setpoint) for nominal and every experiment type; both match the sim-side reference."""
    instance = inst(*case)
    ep = running(instance)
    sensors = [s.id for s in instance.spec.sensors]
    overrides = [override(f) for f in instance.faults]
    for kind, args in setpoints(instance.spec):
        sp = {"experiment": kind, **args}
        r = ep.call("simulate", {"overrides": overrides, "setpoint": sp, "sensors": sensors})
        truth, ref = ep.true_readings(sp), reference(instance, kind, args)
        assert set(r["readings"]) == set(sensors), kind
        for sid in sensors:
            assert same(r["readings"][sid], truth[sid]), (kind, sid, r["readings"][sid], truth[sid])
            assert abs(truth[sid] - ref[sid]) <= 1e-6 + 1e-9 * abs(ref[sid]), (kind, sid)
    default = ep.call("simulate", {"overrides": overrides, "sensors": sensors})["readings"]
    nominal = ep.true_readings({"experiment": "nominal"})
    assert all(same(default[sid], nominal[sid]) for sid in sensors), "missing setpoint must mean nominal"


@pytest.mark.parametrize("case", CASES, ids=[f"{s}-{t}" for s, t in CASES])
def test_empty_overrides_identical_across_hidden_faults(case: tuple[int, str]) -> None:
    """F8 valid-when 'isolation differential identical': empty (and fixed non-empty) overrides give identical
    simulate output for two instances with the same nominal plant and different faults; empty overrides equal the
    healthy plant."""
    a, b = running(inst(*case)), running(twin(*case))
    spec = inst(*case).spec
    for kind, args in setpoints(spec):
        empty = {"overrides": [], "setpoint": {"experiment": kind, **args}}
        fixed = {**empty, "overrides": [{"component": ids(spec, "pipe")[-1], "type": "fouling", "magnitude": 2.5}]}
        ra, rb = a.call("simulate", empty), b.call("simulate", empty)
        assert ra == rb, kind
        healthy = reference(inst(*case), kind, args, faults=())
        assert ra["readings"] and all(abs(v - healthy[sid]) <= 1e-6 + 1e-9 * abs(v) for sid, v in ra["readings"].items())
        assert a.call("simulate", fixed) == b.call("simulate", fixed), kind


def test_rate_limit_301st_call_rejected() -> None:
    """F8 valid-when 'the 301st call is rejected': calls 1..300 succeed with simulate_left counting down, call 301
    gives E_RATE_LIMIT, leaves the state unchanged, and the episode keeps running."""
    assert SIMULATE_CAP == 300
    instance = inst(1, "easy")
    ep = running(instance)
    sid = instance.spec.sensors[0].id
    for k in range(1, SIMULATE_CAP + 1):
        assert ep.call("simulate", {"overrides": [], "sensors": [sid]})["simulate_left"] == SIMULATE_CAP - k
    before = ep.state_hash()
    with pytest.raises(ToolRejected) as e:
        ep.call("simulate", {"overrides": [], "sensors": [sid]})
    assert e.value.code == "E_RATE_LIMIT" and ep.state_hash() == before and ep.status == "RUNNING"
    assert ep.call("read_sensor", {"sensor_id": sid})["cost"] == 1


def test_simulate_never_changes_budget_or_plant() -> None:
    """F8 valid-when 'budget unchanged' / 'no effect on the plant state': simulate costs 0, leaves true_readings
    unchanged and takes no noise index (a later read equals the same read in an episode without simulates)."""
    instance = inst(4, "medium")
    spec = instance.spec
    sid = spec.sensors[0].id
    wild = {"overrides": [{"component": ids(spec, "pipe")[0], "type": "fouling", "magnitude": 10.0},
                          {"component": ids(spec, "pump")[0], "type": "pump_wear", "magnitude": 0.9},
                          {"component": spec.junctions[-1].id, "type": "leak", "magnitude": 0.3}],
            "setpoint": {"experiment": "pump_off_static"}}
    a, b = running(instance), running(instance)
    truth = a.true_readings({"experiment": "nominal"})
    first = a.call("read_sensor", {"sensor_id": sid})
    for _ in range(5):
        a.call("simulate", wild)
        assert a.trace[-1]["spent"] == 1 and a.trace[-1]["cost"] == 0
    second = a.call("read_sensor", {"sensor_id": sid})
    assert a.true_readings({"experiment": "nominal"}) == truth
    assert (first, second) == (b.call("read_sensor", {"sensor_id": sid}), b.call("read_sensor", {"sensor_id": sid}))
    assert a.call("budget_status", {})["spent"] == 2


def ov(component: str, typ: str, magnitude: float | None = None, **extra: Any) -> dict:
    d: dict[str, Any] = {"component": component, "type": typ, **extra}
    return d if magnitude is None else {**d, "magnitude": magnitude}


def components(spec: PlantSpec) -> list[dict]:
    """One magnitude-0 override per component, all kinds (healthy, but each a valid entry)."""
    kind_type = {"pipe": "fouling", "valve": "stuck_valve", "pump": "pump_wear"}
    out = [ov(ln.id, kind_type[ln.kind], 0.0) for ln in spec.links]
    out += [ov(j.id, "leak", 0.0) for j in spec.junctions]
    return out + [ov(s.id, "sensor_bias", 0.0) for s in spec.sensors]


BAD_SIMULATE = [
    ("fouling>10", lambda s: {"overrides": [ov(ids(s, "pipe")[0], "fouling", 10.000001)]}),
    ("stuck>0.9", lambda s: {"overrides": [ov(ids(s, "valve")[0], "stuck_valve", 0.91)]}),
    ("wear>0.9", lambda s: {"overrides": [ov(ids(s, "pump")[0], "pump_wear", 0.95)]}),
    ("leak>0.3", lambda s: {"overrides": [ov(s.junctions[-1].id, "leak", 0.31)]}),
    ("bias>50", lambda s: {"overrides": [ov(s.sensors[0].id, "sensor_bias", 50.5)]}),
    ("negative", lambda s: {"overrides": [ov(ids(s, "pipe")[0], "fouling", -0.1)]}),
    ("no-magnitude", lambda s: {"overrides": [ov(ids(s, "pipe")[0], "fouling")]}),
    ("kind-mismatch", lambda s: {"overrides": [ov(ids(s, "pipe")[0], "leak", 0.1)]}),
    ("unknown-component", lambda s: {"overrides": [ov("P99", "fouling", 1.0)]}),
    ("duplicate", lambda s: {"overrides": [ov(ids(s, "pipe")[0], "fouling", 1.0), ov(ids(s, "pipe")[0], "fouling", 2.0)]}),
    ("confidence", lambda s: {"overrides": [ov(ids(s, "pipe")[0], "fouling", 1.0, confidence=0.5)]}),
    ("sign-0", lambda s: {"overrides": [ov(s.sensors[0].id, "sensor_bias", 5.0, sign=0)]}),
    ("21-overrides", lambda s: {"overrides": components(s)[:21]}),
    ("not-a-list", lambda s: {"overrides": ids(s, "pipe")[0]}),
    ("unknown-experiment", lambda s: {"overrides": [], "setpoint": {"experiment": "warp"}}),
    ("theta<0.1", lambda s: {"overrides": [], "setpoint": {"experiment": "valve_step", "valve": ids(s, "valve")[0],
                                                           "theta": 0.05}}),
    ("s>1.2", lambda s: {"overrides": [], "setpoint": {"experiment": "speed_step", "s": 1.3}}),
    ("unknown-sensor", lambda s: {"overrides": [], "sensors": ["XX-J0"]}),
    ("sensors-not-list", lambda s: {"overrides": [], "sensors": s.sensors[0].id}),
    ("extra-field", lambda s: {"overrides": [], "zz_unknown": 1}),
]


@pytest.mark.parametrize("build", [b for _, b in BAD_SIMULATE], ids=[n for n, _ in BAD_SIMULATE])
def test_invalid_overrides_rejected(build: Callable[[PlantSpec], dict]) -> None:
    """F8 overrides contract (section 5: simulate ranges, one entry per component, at most 20, override-only
    fields): violations give E_INVALID_ARGUMENT and leave the state unchanged."""
    instance = inst(3, "hard")
    ep = running(instance)
    before = ep.state_hash()
    with pytest.raises(ToolRejected) as e:
        ep.call("simulate", build(instance.spec))
    assert e.value.code == "E_INVALID_ARGUMENT" and ep.state_hash() == before


def test_override_range_boundaries_accepted() -> None:
    """F8 overrides contract: the closed range ends of section 5 and exactly 20 overrides are accepted."""
    instance = inst(3, "hard")
    spec = instance.spec
    ep = running(instance)
    for o in [ov(ids(spec, "pipe")[0], "fouling", 10.0), ov(ids(spec, "valve")[0], "stuck_valve", 0.9),
              ov(ids(spec, "pump")[0], "pump_wear", 0.9), ov(spec.junctions[-1].id, "leak", 0.3),
              ov(spec.sensors[0].id, "sensor_bias", 50.0, sign=-1), ov(spec.sensors[1].id, "sensor_stuck", value=3.0)]:
        assert ep.call("simulate", {"overrides": [o]})["readings"], o
    assert ep.call("simulate", {"overrides": components(spec)[:20]})["readings"]
