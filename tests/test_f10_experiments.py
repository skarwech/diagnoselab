"""F10 run_experiment: exact costs, documented sensor subsets, noisy values, revert to nominal, argument ranges.

Contract: docs/server_spec.md sections 2, 3, 6, 10; docs/sim_spec.md section 6; plan section 8, F10.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence

import pytest

from diagnoselab.server.episode import BUDGET, Episode, ToolRejected
from diagnoselab.sim.experiments import experiment_sensors, experiment_setpoint
from diagnoselab.sim.faults import apply_faults
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Instance, PlantSpec
from diagnoselab.sim.sensors import observe, physical_readings
from diagnoselab.sim.solver import solve

COSTS = {"speed_step": 4, "valve_step": 4, "isolate_branch": 6, "pump_off_static": 8}
EXP_KEYS = {"type", "readings", "cost", "budget_left", "events"}
TOL = 1e-6
NAN, INF = float("nan"), float("inf")
CASES = [(1, "easy"), (2, "medium"), (3, "hard")]


@functools.lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


def ids(spec: PlantSpec, kind: str) -> list[str]:
    return [ln.id for ln in spec.links if ln.kind == kind]


def running(instance: Instance) -> Episode:
    ep = Episode(instance, clock=lambda: 0.0)
    ep.call("get_topology", {})  # INIT -> RUNNING before any hash comparison
    return ep


def expected(instance: Instance, sensor_ids: Sequence[str], k: int, kind: str = "nominal",
             args: dict | None = None) -> dict[str, float]:
    """Noisy readings for the k-th accepted costed call at the experiment setpoint of the true plant."""
    spec = instance.spec
    state = solve(spec, apply_faults(spec, instance.faults), experiment_setpoint(spec, kind, args))
    return observe(instance, physical_readings(spec, state), sensor_ids, k)


def experiments(spec: PlantSpec) -> list[tuple[str, dict]]:
    out = [("speed_step", {"s": 0.8}), ("speed_step", {"s": 1.15, "pump": ids(spec, "pump")[-1]})]
    out += [("valve_step", {"valve": v, "theta": 0.3}) for v in ids(spec, "valve")]
    out += [("isolate_branch", {"branch": b.id}) for b in spec.branches]
    return out + [("pump_off_static", {})]


@pytest.mark.parametrize("case", CASES, ids=[f"{s}-{t}" for s, t in CASES])
def test_experiment_cost_sensors_values_and_revert(case: tuple[int, str]) -> None:
    """F10 valid-when 'costs exact' and 'revert hashes equal': each experiment charges exactly 4/4/6/8, returns
    exactly experiment_sensors(...) with the F4 noisy values, events []; afterwards true_readings(nominal) is
    unchanged and a read of every sensor gives the nominal (reverted) values."""
    instance = inst(*case)
    spec = instance.spec
    sensors = [s.id for s in spec.sensors]
    for kind, args in experiments(spec):
        ep = running(instance)
        ep.call("read_sensor", {"sensor_id": sensors[0]})
        truth = ep.true_readings({"experiment": "nominal"})
        r = ep.call("run_experiment", {"type": kind, "args": args})
        cost = COSTS[kind]
        assert set(r) == EXP_KEYS and r["type"] == kind and r["events"] == [], kind
        assert r["cost"] == cost and r["budget_left"] == BUDGET - 1 - cost and ep.trace[-1]["spent"] == 1 + cost
        want = experiment_sensors(spec, kind, args)
        assert set(r["readings"]) == set(want) and len(r["readings"]) == len(want), kind
        exp = expected(instance, want, 1, kind, args)
        assert all(abs(r["readings"][sid] - exp[sid]) <= TOL for sid in want), kind
        assert ep.true_readings({"experiment": "nominal"}) == truth, kind
        for k, sid in enumerate(sensors, start=2):
            value = ep.call("read_sensor", {"sensor_id": sid})["value"]
            assert abs(value - expected(instance, [sid], k)[sid]) <= TOL, (kind, sid, "plant did not revert")


def exp(kind: str, **args: Any) -> dict:
    return {"type": kind, "args": args}


BAD_EXPERIMENTS = [
    ("s=0.69", lambda s: exp("speed_step", s=0.69)),
    ("s=1.21", lambda s: exp("speed_step", s=1.21)),
    ("s=0.7-eps", lambda s: exp("speed_step", s=0.7 - 1e-9)),
    ("s=1.2+eps", lambda s: exp("speed_step", s=1.2 + 1e-9)),
    ("s=-1", lambda s: exp("speed_step", s=-1.0)),
    ("s=nan", lambda s: exp("speed_step", s=NAN)),
    ("s=inf", lambda s: exp("speed_step", s=INF)),
    ("s=str", lambda s: exp("speed_step", s="1.0")),
    ("s=bool", lambda s: exp("speed_step", s=True)),
    ("s=null", lambda s: exp("speed_step", s=None)),
    ("s-missing", lambda s: exp("speed_step")),
    ("pump-unknown", lambda s: exp("speed_step", s=1.0, pump="PU9")),
    ("pump-is-valve", lambda s: exp("speed_step", s=1.0, pump=ids(s, "valve")[0])),
    ("speed-extra", lambda s: exp("speed_step", s=1.0, zz=1)),
    ("theta=0.09", lambda s: exp("valve_step", valve=ids(s, "valve")[0], theta=0.09)),
    ("theta=1.01", lambda s: exp("valve_step", valve=ids(s, "valve")[0], theta=1.01)),
    ("theta=0", lambda s: exp("valve_step", valve=ids(s, "valve")[0], theta=0.0)),
    ("theta=-0.5", lambda s: exp("valve_step", valve=ids(s, "valve")[0], theta=-0.5)),
    ("theta=nan", lambda s: exp("valve_step", valve=ids(s, "valve")[0], theta=NAN)),
    ("valve-unknown", lambda s: exp("valve_step", valve="V99", theta=0.5)),
    ("valve-is-pipe", lambda s: exp("valve_step", valve=ids(s, "pipe")[0], theta=0.5)),
    ("valve-is-pump", lambda s: exp("valve_step", valve=ids(s, "pump")[0], theta=0.5)),
    ("valve-missing", lambda s: exp("valve_step", theta=0.5)),
    ("theta-missing", lambda s: exp("valve_step", valve=ids(s, "valve")[0])),
    ("branch-unknown", lambda s: exp("isolate_branch", branch="B99")),
    ("branch-B0", lambda s: exp("isolate_branch", branch="B0")),
    ("branch-int", lambda s: exp("isolate_branch", branch=1)),
    ("branch-is-link", lambda s: exp("isolate_branch", branch=s.branches[0].entry)),
    ("branch-missing", lambda s: exp("isolate_branch")),
    ("static-with-args", lambda s: exp("pump_off_static", s=1.0)),
    ("type-nominal", lambda s: exp("nominal")),
    ("type-unknown", lambda s: exp("speed-step", s=1.0)),
    ("type-empty", lambda s: exp("")),
    ("args-list", lambda s: {"type": "speed_step", "args": [0.9]}),
    ("args-str", lambda s: {"type": "speed_step", "args": "s=0.9"}),
    ("top-extra", lambda s: {**exp("speed_step", s=1.0), "zz": 1}),
]


@pytest.mark.parametrize("build", [b for _, b in BAD_EXPERIMENTS], ids=[n for n, _ in BAD_EXPERIMENTS])
def test_out_of_range_arguments_rejected(build: Callable[[PlantSpec], dict]) -> None:
    """F10 'out-of-range arguments are rejected': s outside [0.7, 1.2], theta outside [0.1, 1], unknown
    valve/pump/branch, wrong types and unknown fields give E_INVALID_ARGUMENT with no charge and no state change."""
    instance = inst(3, "hard")
    ep = running(instance)
    before = ep.state_hash()
    with pytest.raises(ToolRejected) as e:
        ep.call("run_experiment", build(instance.spec))
    assert e.value.code == "E_INVALID_ARGUMENT" and ep.state_hash() == before and ep.trace[-1]["spent"] == 0


def test_range_boundaries_accepted() -> None:
    """F10 argument ranges are closed: s = 0.7, 0.95, 1.05, 1.2 and theta = 0.1, 0.95 are accepted at cost 4 each.
    Steps that reproduce the nominal setpoint (|s - 1| < 0.05, theta > 0.95) are rejected (attack review: a no-op
    step would be a cheap bulk nominal read)."""
    instance = inst(3, "hard")
    valve = ids(instance.spec, "valve")[0]
    ep = running(instance)
    calls = [exp("speed_step", s=0.7), exp("speed_step", s=1.2), exp("speed_step", s=0.95), exp("speed_step", s=1.05),
             exp("valve_step", valve=valve, theta=0.1), exp("valve_step", valve=valve, theta=0.95)]
    assert [ep.call("run_experiment", c)["cost"] for c in calls] == [4, 4, 4, 4, 4, 4]
    assert ep.trace[-1]["spent"] == 24
    for noop in (exp("speed_step", s=1.0), exp("speed_step", s=0.97), exp("valve_step", valve=valve, theta=1.0)):
        with pytest.raises(ToolRejected) as e:
            ep.call("run_experiment", noop)
        assert e.value.code == "E_INVALID_ARGUMENT"
