"""F11 budget manager: ledger property over random sequences, free tools, call cap, time cap.

Contract: docs/server_spec.md section 2 and 10; plan section 8, F11.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from diagnoselab.server.episode import BUDGET, CALL_CAP, SIMULATE_CAP, TIME_CAP_S, Episode, ToolRejected
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Instance, PlantSpec

COSTS = {"speed_step": 4, "valve_step": 4, "isolate_branch": 6, "pump_off_static": 8}
TOPICS = ("overview", "tools", "physics", "faults", "experiments", "sensors", "submission", "scoring")
STATUS_KEYS = {"status", "budget", "spent", "remaining", "calls_used", "call_cap", "simulate_used", "simulate_cap",
               "time_used_s", "time_cap_s"}
WEIGHTS = {"read_sensor": 0.3, "run_experiment": 0.25, "simulate": 0.1, "budget_status": 0.1, "read_manual": 0.08,
           "get_topology": 0.07, "submit": 0.05, "nope": 0.05}
NAMES, PROBS = list(WEIGHTS), np.array(list(WEIGHTS.values())) / sum(WEIGHTS.values())
CASES = [(1, "easy"), (4, "medium"), (3, "hard")]


@functools.lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


def ids(spec: PlantSpec, kind: str) -> list[str]:
    return [ln.id for ln in spec.links if ln.kind == kind]


def pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    return seq[int(rng.integers(len(seq)))]


def running(instance: Instance, clock: Callable[[], float] = lambda: 0.0) -> Episode:
    ep = Episode(instance, clock=clock)
    ep.call("get_topology", {})  # INIT -> RUNNING before any hash comparison
    return ep


def random_call(spec: PlantSpec, rng: np.random.Generator) -> tuple[str, Any, bool]:
    """(tool, args, valid): a valid call, or one that is invalid by construction."""
    tool = NAMES[int(rng.choice(len(NAMES), p=PROBS))]
    valid = bool(rng.random() < 0.75) and tool != "nope"
    sensors, pipe = [s.id for s in spec.sensors], ids(spec, "pipe")[0]
    if tool == "read_sensor":
        return tool, {"sensor_id": pick(rng, sensors) if valid else "XX-J0"}, valid
    if tool == "run_experiment":
        kind, args = pick(rng, [("speed_step", {"s": 0.9}), ("valve_step", {"valve": ids(spec, "valve")[0], "theta": 0.5}),
                                ("isolate_branch", {"branch": spec.branches[0].id}), ("pump_off_static", {})])
        return tool, {"type": kind, "args": args if valid else {**args, "zz": 1}}, valid
    if tool == "simulate":
        bad = [{"component": pipe, "type": "fouling", "magnitude": 99.0}]
        return tool, {"overrides": [] if valid else bad, "sensors": [pick(rng, sensors)]}, valid
    if tool == "read_manual":
        return tool, {"topic": pick(rng, TOPICS) if valid else "nope"}, valid
    if tool == "submit":
        typ = "fouling" if valid else "leak"
        return tool, {"diagnosis": [{"component": pipe, "type": typ, "magnitude": 1.0}], "repairs": []}, valid
    return tool, ({} if valid else {"zz": 1}), valid


def cost_of(tool: str, args: Any) -> int:
    return 1 if tool == "read_sensor" else COSTS[args["type"]] if tool == "run_experiment" else 0


def check_sequence(instance: Instance, rng: np.random.Generator, n_calls: int) -> None:
    """Ledger property: spent <= B after every call, spent == sum of accepted costs, unaffordable calls refused,
    free tools cost 0, rejected calls leave state_hash unchanged (except a strike from an invalid submit)."""
    ep, spent = running(instance), 0
    for _ in range(n_calls):
        tool, args, valid = random_call(instance.spec, rng)
        status, before = ep.status, ep.state_hash()
        cost = cost_of(tool, args) if valid else 0
        try:
            r, ok = ep.call(tool, args), True
        except ToolRejected as e:
            ok, code = False, e.code
        should_pass = valid and status == "RUNNING" and cost <= BUDGET - spent
        assert ok == should_pass, (tool, args, status, spent)
        if ok:
            spent += cost
            if cost:
                assert r["cost"] == cost and r["budget_left"] == BUDGET - spent
        else:
            if valid and status == "RUNNING":
                assert code == "E_BUDGET", (tool, code)
            if status != "RUNNING":
                assert code == "E_EPISODE_OVER", (tool, code)
            if tool != "submit" or status != "RUNNING":
                assert ep.state_hash() == before, (tool, args)
        line = ep.trace[-1]
        assert line["spent"] == spent <= BUDGET and line["budget_left"] == BUDGET - spent
        assert line["cost"] == (cost if ok else 0) and line["ok"] is ok


def test_constants() -> None:
    """F11 / spec section 2: B = 40, 150-call cap, 300-simulate cap, 30-minute time cap."""
    assert (BUDGET, CALL_CAP, SIMULATE_CAP, TIME_CAP_S) == (40, 150, 300, 1800)


def test_free_tools_cost_zero() -> None:
    """F11 'free tools cost 0': read_manual (every topic and none), get_topology, simulate and budget_status
    never charge; budget_status reports the documented fields."""
    ep = running(inst(1, "easy"))
    calls = [("read_manual", {})] + [("read_manual", {"topic": t}) for t in TOPICS]
    calls += [("get_topology", {}), ("simulate", {"overrides": []}), ("budget_status", {})]
    for tool, args in calls:
        r = ep.call(tool, args)
        assert ep.trace[-1]["cost"] == 0 and ep.trace[-1]["spent"] == 0, tool
        if tool == "read_manual":
            assert set(r) == {"topic", "text"} and isinstance(r["text"], str) and r["text"], args
    s = ep.call("budget_status", {})
    assert set(s) == STATUS_KEYS and s["status"] == "RUNNING" and s["simulate_used"] == 1
    assert (s["budget"], s["spent"], s["remaining"], s["call_cap"], s["simulate_cap"], s["time_cap_s"]) == (
        40, 0, 40, 150, 300, 1800)


def test_random_sequences_small() -> None:
    """F11 property test (small count): 300 random mixed valid/invalid sequences keep spent <= B with exact
    accounting and unchanged state on rejected calls."""
    rng = np.random.default_rng(11)
    for k in range(300):
        check_sequence(inst(*CASES[k % 3]), rng, int(rng.integers(1, 30)))


@pytest.mark.slow
def test_random_sequences_100k(record_property: Callable[[str, Any], None]) -> None:
    """F11 valid-when 'the property test passes on 100,000 sequences' (stops after ~60 s; count recorded)."""
    rng = np.random.default_rng(12)
    deadline, count = time.monotonic() + 60, 0
    while count < 100_000 and time.monotonic() < deadline:
        check_sequence(inst(*CASES[count % 3]), rng, int(rng.integers(1, 16)))
        count += 1
    record_property("f11_sequences", count)
    print(f"F11 property test: {count} sequences")
    assert count >= 1000


def test_call_cap_151_finalizes() -> None:
    """F11 'at the call cap the episode finalizes': 150 calls (rejected ones counted, simulate not) run normally;
    call 151 finalizes and is rejected with E_EPISODE_OVER, and later calls (simulate too) are rejected."""
    instance = inst(1, "easy")
    sid = instance.spec.sensors[0].id
    ep = Episode(instance, clock=lambda: 0.0)
    for i in range(CALL_CAP):
        if i % 10 == 0:
            ep.call("simulate", {"overrides": [], "sensors": [sid]})
        if i % 3 == 0:
            with pytest.raises(ToolRejected) as e:
                ep.call("read_sensor", {"sensor_id": "XX-J0"})
            assert e.value.code == "E_INVALID_ARGUMENT", i
        else:
            s = ep.call("budget_status", {})
    assert ep.status == "RUNNING" and s["simulate_used"] == 15 and s["calls_used"] in (CALL_CAP - 1, CALL_CAP)
    with pytest.raises(ToolRejected) as e:
        ep.call("get_topology", {})
    assert e.value.code == "E_EPISODE_OVER" and ep.status == "FINALIZED"
    for tool, args in [("simulate", {"overrides": []}), ("read_sensor", {"sensor_id": sid})]:
        with pytest.raises(ToolRejected) as e:
            ep.call(tool, args)
        assert e.value.code == "E_EPISODE_OVER" and ep.trace[-1]["spent"] == 0


def test_time_cap_with_injected_clock() -> None:
    """F11 'at the time cap the episode finalizes': the clock starts at the first call; before the deadline calls
    work and time_used_s follows the clock; the first call after it finalizes with E_EPISODE_OVER."""
    instance = inst(1, "easy")
    sid = instance.spec.sensors[0].id
    now = [1000.0]
    ep = Episode(instance, clock=lambda: now[0])
    assert ep.status == "INIT"
    assert ep.call("budget_status", {})["time_used_s"] == pytest.approx(0.0, abs=1e-6)
    assert ep.status == "RUNNING"
    now[0] = 1100.0
    assert ep.call("budget_status", {})["time_used_s"] == pytest.approx(100.0, abs=1e-3)
    now[0] = 1000.0 + TIME_CAP_S - 1
    assert ep.call("read_sensor", {"sensor_id": sid})["cost"] == 1
    now[0] = 1000.0 + TIME_CAP_S + 1
    with pytest.raises(ToolRejected) as e:
        ep.call("budget_status", {})
    assert e.value.code == "E_EPISODE_OVER" and ep.status == "FINALIZED" and ep.trace[-1]["spent"] == 1
