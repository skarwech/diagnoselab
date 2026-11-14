"""F13 trace logger and replay: documented trace fields, atomic episode.json, perfect replay, tamper detection.

Contract: docs/server_spec.md sections 7 and 10; plan section 8, F13.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from diagnoselab.server.episode import BUDGET, Episode, ToolRejected
from diagnoselab.server.replay import replay
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Instance, PlantSpec

FIELDS = {"i", "tool", "args", "ok", "code", "cost", "spent", "budget_left", "response_hash"}
TIERS = ("easy", "medium", "hard")


@functools.lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


def ids(spec: PlantSpec, kind: str) -> list[str]:
    return [ln.id for ln in spec.links if ln.kind == kind]


def pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    return seq[int(rng.integers(len(seq)))]


def random_call(spec: PlantSpec, rng: np.random.Generator, submit: bool = True) -> tuple[str, Any]:
    """A JSON-safe call, valid or invalid, over every tool (submit optional, so a long trace reaches the cap)."""
    sensors, pipe, valve = [s.id for s in spec.sensors], ids(spec, "pipe")[0], ids(spec, "valve")[0]
    menu = [
        ("read_sensor", {"sensor_id": pick(rng, sensors)}),
        ("read_sensor", {"sensor_id": "XX-J0"}),
        ("run_experiment", {"type": "valve_step", "args": {"valve": valve, "theta": float(rng.uniform(0.1, 1.0))}}),
        ("run_experiment", {"type": "speed_step", "args": {"s": float(rng.uniform(0.5, 1.4))}}),
        ("run_experiment", {"type": "isolate_branch", "args": {"branch": pick(rng, spec.branches).id}}),
        ("run_experiment", {"type": "pump_off_static", "args": {}}),
        ("simulate", {"overrides": [{"component": pipe, "type": "fouling", "magnitude": float(rng.uniform(0, 12))}],
                      "sensors": [pick(rng, sensors)]}),
        ("simulate", {"overrides": [], "setpoint": {"experiment": "valve_step", "valve": valve, "theta": 0.5}}),
        ("read_manual", {"topic": pick(rng, ["physics", "faults", "nope"])}),
        ("get_topology", {}),
        ("budget_status", {}),
        ("budget_status", {"zz": 1}),
        ("submit", {"diagnosis": [{"component": pipe, "type": pick(rng, ["fouling", "leak"]), "magnitude": 1.0}],
                    "repairs": []}),
        ("nope", {}),
        ("read_sensor", None),
    ]
    return pick(rng, [c for c in menu if submit or c[0] != "submit"])


def call(ep: Episode, tool: str, args: Any) -> tuple[bool, Any]:
    try:
        return True, ep.call(tool, args)
    except ToolRejected as e:
        return False, {"error": f"{e.code}: {e.message}"}


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_traced(seed: int, tier: str, state: Path, rng: np.random.Generator, n: int,
               submit: bool = True) -> tuple[Episode, list]:
    """Random episode with state_dir; after every call the trace has one more line and episode.json parses."""
    ep, calls = Episode(inst(seed, tier), state_dir=state), []
    for k in range(n):
        tool, args = random_call(inst(seed, tier).spec, rng, submit=submit)
        calls.append((tool, args, *call(ep, tool, args)))
        assert len(lines(state / "trace.jsonl")) == k + 1, "trace must be flushed once per call"
        doc = json.loads((state / "episode.json").read_text())
        assert doc["seed"] == seed and doc["tier"] == tier
    return ep, calls


def traced(tmp_path: Path, k: int, n: int, rng: np.random.Generator) -> tuple[Path, int, str]:
    seed, tier = k % 5, TIERS[k % 3]
    state = tmp_path / f"t{k}"
    state.mkdir()
    run_traced(seed, tier, state, rng, n, submit=n <= 150)
    return state / "trace.jsonl", seed, tier


def test_replay_reproduces_every_response(tmp_path: Path) -> None:
    """F13 valid-when 'replay equality is 100%' (small count): 5 random valid+invalid traces, one past the
    150-call cap, replay with no differing index."""
    rng = np.random.default_rng(13)
    for k, n in enumerate((12, 40, 25, 210, 60)):
        trace, seed, tier = traced(tmp_path, k, n, rng)
        assert replay(trace, seed, tier) == [], (seed, tier, n)
        if n > 150:  # no submit in this trace: the 151st non-simulate call hit the call cap
            counted = [r for r in lines(trace) if r["tool"] != "simulate"]
            assert len(counted) > 150 and all(r["code"] == "E_EPISODE_OVER" for r in counted[150:])


@pytest.mark.slow
def test_replay_50_traces(tmp_path: Path) -> None:
    """F13 valid-when 'replay equality is 100% on 50 traces'."""
    rng = np.random.default_rng(14)
    for k in range(50):
        trace, seed, tier = traced(tmp_path, k, int(rng.integers(5, 171)), rng)
        assert replay(trace, seed, tier) == [], (k, seed, tier)


def test_replay_detects_tampering(tmp_path: Path) -> None:
    """F13 negative control: a trace whose response_hash at index 7 was altered replays as exactly [7]."""
    trace, seed, tier = traced(tmp_path, 1, 20, np.random.default_rng(15))
    tampered = tmp_path / "tampered.jsonl"
    rows = lines(trace)
    rows[7] = {**rows[7], "response_hash": "0" * 64}
    tampered.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert replay(trace, seed, tier) == [] and replay(tampered, seed, tier) == [7]


def test_replay_edge_arguments(tmp_path: Path) -> None:
    """F13 replay of the hard inputs: NaN/Inf arguments, non-object arguments and arguments over 16 KB (logged as
    {"__too_large__": n_bytes}) still replay perfectly."""
    state = tmp_path / "edge"
    state.mkdir()
    instance = inst(2, "medium")
    ep = Episode(instance, state_dir=state)
    sid = instance.spec.sensors[0].id
    calls = [("read_sensor", {"sensor_id": sid}), ("run_experiment", {"type": "speed_step", "args": {"s": float("nan")}}),
             ("simulate", {"overrides": [], "setpoint": {"experiment": "speed_step", "s": float("inf")}}),
             ("read_sensor", {"sensor_id": "x" * 20_000}), ("read_sensor", [sid]), ("budget_status", None),
             ("read_sensor", {"sensor_id": sid})]
    for tool, args in calls:
        call(ep, tool, args)
    rows = lines(state / "trace.jsonl")
    assert len(rows) == len(calls)
    big = rows[3]["args"]
    assert set(big) == {"__too_large__"} and big["__too_large__"] > 16_000 and rows[3]["code"] == "E_TOO_LARGE"
    assert replay(state / "trace.jsonl", 2, "medium") == []


def hashes(obj: object) -> set[str]:
    """sha256 of the canonical JSON under the usual canonical forms (spec section 7 does not fix separators)."""
    return {hashlib.sha256(json.dumps(obj, sort_keys=True, separators=sep, ensure_ascii=asc).encode()).hexdigest()
            for sep in ((",", ":"), (", ", ": ")) for asc in (True, False)}


def test_trace_lines_have_documented_fields(tmp_path: Path) -> None:
    """F13 'writes an append-only JSONL: call index, tool, arguments, response hash, cost, budget left' and 'the
    log is complete': one line per call with every documented field, consistent ledger, response_hash of the
    response (time_used_s excluded) or of {"error": "<code>: <message>"}, identical to ep.trace."""
    state = tmp_path / "fields"
    state.mkdir()
    ep, calls = run_traced(3, "hard", state, np.random.default_rng(16), 60, submit=False)
    rows, spent = lines(state / "trace.jsonl"), 0
    assert len(rows) == len(calls) and rows == ep.trace and sum(r["ok"] for r in rows) >= 15
    for i, (row, (tool, args, ok, res)) in enumerate(zip(rows, calls)):
        assert FIELDS <= set(row), i
        assert (row["i"], row["tool"], row["args"], row["ok"]) == (i, tool, args, ok), i
        assert (row["code"] is None) == ok and (ok or res["error"].startswith(row["code"] + ": ")), i
        assert isinstance(row["cost"], int) and row["cost"] >= 0 and (ok or row["cost"] == 0), i
        spent += row["cost"]
        assert row["spent"] == spent <= BUDGET and row["budget_left"] == BUDGET - spent, i
        hashed = {k: v for k, v in res.items() if k != "time_used_s"} if tool == "budget_status" and ok else res
        assert row["response_hash"] in hashes(hashed), (i, tool)


def test_no_disk_writes_without_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F13 / spec section 10: with state_dir=None nothing is written to disk; the trace is kept in memory."""
    monkeypatch.chdir(tmp_path)
    instance = inst(1, "easy")
    ep = Episode(instance)
    rng = np.random.default_rng(17)
    for _ in range(10):
        call(ep, *random_call(instance.spec, rng))
    assert list(tmp_path.iterdir()) == [] and len(ep.trace) == 10
