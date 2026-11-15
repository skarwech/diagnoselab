"""F9 read_sensor: exact cost, response shape, F3/F4 values, refusal without partial effect, ledger property.

Contract: docs/server_spec.md sections 2, 3, 10; docs/sim_spec.md section 5; plan section 8, F9.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import numpy as np
import pytest

from diagnoselab.server.episode import BUDGET, Episode, ToolRejected
from diagnoselab.sim.experiments import experiment_setpoint
from diagnoselab.sim.faults import apply_faults
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Instance
from diagnoselab.sim.sensors import observe, physical_readings
from diagnoselab.sim.solver import solve

UNITS = {"pressure": "m", "flow": "L/s", "power": "kW", "vibration": "mm/s"}
READ_KEYS = {"sensor_id", "value", "unit", "cost", "budget_left", "events"}
TOL = 1e-6  # tool values are rounded to 6 decimals
BAD_IDS = ["PT-J999", "", "FT-PU1x", "J1", 7, None, ["PT-J1"]]


@functools.lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


def running(instance: Instance) -> Episode:
    ep = Episode(instance, clock=lambda: 0.0)
    ep.call("get_topology", {})  # INIT -> RUNNING before any hash comparison
    return ep


def expected(instance: Instance, sensor_ids: Sequence[str], k: int, kind: str = "nominal",
             args: dict | None = None) -> dict[str, float]:
    """Noisy reading the server must return for the k-th accepted costed call (spec section 2 noise key)."""
    spec = instance.spec
    state = solve(spec, apply_faults(spec, instance.faults), experiment_setpoint(spec, kind, args))
    return observe(instance, physical_readings(spec, state), sensor_ids, k)


def pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    return seq[int(rng.integers(len(seq)))]


@pytest.mark.parametrize("case", [(4, "medium"), (2, "medium"), (3, "hard")], ids=["bias", "stuck", "hard"])
def test_read_response_cost_and_value(case: tuple[int, str]) -> None:
    """F9 'costs 1 and returns {value, unit, cost, budget_left, events}', 'exact cost accounting' and 'value
    distribution per F4': every sensor read once gives the documented fields, cost 1, events [], and the F4 value
    keyed by (noise_seed, accepted costed calls before, sensor index)."""
    instance = inst(*case)
    ep = running(instance)
    for k, sensor in enumerate(instance.spec.sensors):
        r = ep.call("read_sensor", {"sensor_id": sensor.id})
        assert set(r) == READ_KEYS and r["sensor_id"] == sensor.id and r["unit"] == UNITS[sensor.kind]
        assert r["cost"] == 1 and r["budget_left"] == BUDGET - (k + 1) and r["events"] == []
        assert abs(r["value"] - expected(instance, [sensor.id], k)[sensor.id]) <= TOL, sensor.id
        assert ep.trace[-1]["spent"] == k + 1


def test_stuck_and_biased_sensors_per_f3() -> None:
    """F9 valid-when 'stuck and biased sensors behave per F3': a stuck sensor returns its constant on every read;
    a biased sensor returns physical + sign * m * sigma + sigma * N(0, 1)."""
    stuck_inst, bias_inst = inst(2, "medium"), inst(4, "medium")
    stuck = next(f for f in stuck_inst.faults if f.type == "sensor_stuck")
    bias = next(f for f in bias_inst.faults if f.type == "sensor_bias")
    ep = running(stuck_inst)
    values = [ep.call("read_sensor", {"sensor_id": stuck.component})["value"] for _ in range(4)]
    assert len(set(values)) == 1 and abs(values[0] - stuck.value) <= TOL

    spec = bias_inst.spec
    sensor = next(s for s in spec.sensors if s.id == bias.component)
    index = [s.id for s in spec.sensors].index(sensor.id)
    physical = physical_readings(spec, solve(spec, apply_faults(spec, bias_inst.faults)))[sensor.id]
    ep = running(bias_inst)
    for k in range(5):
        noise = np.random.default_rng([bias_inst.noise_seed, k, index]).standard_normal()
        value = ep.call("read_sensor", {"sensor_id": sensor.id})["value"]
        assert abs(value - (physical + bias.sign * bias.magnitude * sensor.sigma + sensor.sigma * noise)) <= TOL


def test_budget_refusal_has_no_partial_effect() -> None:
    """F9 'refusal when the budget is insufficient, with no partial effect': an unaffordable call is E_BUDGET,
    leaves state_hash and the ledger unchanged and takes no noise index; reads past B are E_BUDGET too."""
    instance = inst(1, "easy")
    sid = instance.spec.sensors[0].id
    ep = running(instance)
    n_static = (BUDGET - 4) // 8  # then a speed step leaves fewer than 8 units
    for _ in range(n_static):
        ep.call("run_experiment", {"type": "pump_off_static", "args": {}})
    ep.call("run_experiment", {"type": "speed_step", "args": {"s": 0.9}})
    spent = 8 * n_static + 4
    before = ep.state_hash()
    with pytest.raises(ToolRejected) as e:
        ep.call("run_experiment", {"type": "pump_off_static", "args": {}})
    assert e.value.code == "E_BUDGET" and ep.state_hash() == before
    line = ep.trace[-1]
    assert (line["ok"], line["code"], line["cost"], line["spent"], line["budget_left"]) == (
        False, "E_BUDGET", 0, spent, BUDGET - spent)
    r = ep.call("read_sensor", {"sensor_id": sid})
    assert abs(r["value"] - expected(instance, [sid], n_static + 1)[sid]) <= TOL, "refused call consumed a noise index"
    left = BUDGET - spent - 1
    assert [ep.call("read_sensor", {"sensor_id": sid})["budget_left"] for _ in range(left)] == list(range(left - 1, -1, -1))
    before = ep.state_hash()
    with pytest.raises(ToolRejected) as e:
        ep.call("read_sensor", {"sensor_id": sid})
    assert e.value.code == "E_BUDGET" and ep.state_hash() == before and ep.trace[-1]["spent"] == BUDGET


def ledger_run(instance: Instance, rng: np.random.Generator) -> int:
    """Random valid/invalid reads (plus free calls); after each call spent == accepted reads."""
    ep = running(instance)
    sensors = [s.id for s in instance.spec.sensors]
    accepted = 0
    for _ in range(int(rng.integers(1, 61))):
        if rng.random() < 0.1:
            ep.call("budget_status", {})
            assert ep.trace[-1]["spent"] == accepted
            continue
        valid = rng.random() < 0.8
        sid = pick(rng, sensors) if valid else pick(rng, BAD_IDS)
        try:
            r = ep.call("read_sensor", {"sensor_id": sid})
            accepted += 1
            assert valid and r["cost"] == 1 and r["budget_left"] == BUDGET - accepted
        except ToolRejected as e:
            exhausted = accepted == BUDGET
            assert e.code in ({"E_BUDGET"} if valid else {"E_INVALID_ARGUMENT"} | ({"E_BUDGET"} if exhausted else set()))
            assert exhausted or not valid
        assert ep.trace[-1]["spent"] == accepted and ep.trace[-1]["budget_left"] == BUDGET - accepted
    return accepted


def test_ledger_equals_accepted_reads_small() -> None:
    """F9 valid-when 'the ledger equals calls x cost' (small count): 40 random sequences."""
    rng = np.random.default_rng(3)
    for k in range(40):
        ledger_run(inst(*[(1, "easy"), (4, "medium")][k % 2]), rng)


@pytest.mark.slow
def test_ledger_equals_accepted_reads_1000() -> None:
    """F9 valid-when 'the ledger equals calls x cost over 1,000 random sequences'."""
    rng = np.random.default_rng(4)
    for k in range(1000):
        ledger_run(inst(*[(1, "easy"), (4, "medium"), (3, "hard")][k % 3]), rng)
