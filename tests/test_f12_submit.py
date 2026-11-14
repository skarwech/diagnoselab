"""F12 submit and lifecycle: one-shot terminal submit, field-level rejections, strikes, no score anywhere.

Contract: docs/server_spec.md sections 2, 3, 5, 7, 10; plan section 8, F12.
"""

from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from diagnoselab.server.episode import MAX_STRIKES, Episode, ToolRejected
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Instance, PlantSpec

ROOT = Path(__file__).resolve().parents[1]
REPAIR = {"fouling": "clean_pipe", "stuck_valve": "free_valve", "pump_wear": "replace_pump", "leak": "fix_leak",
          "sensor_bias": "recalibrate_sensor", "sensor_stuck": "recalibrate_sensor"}
CASES = [(1, "easy"), (2, "medium"), (3, "hard")]
NAN = float("nan")
TOPICS = ("overview", "tools", "physics", "faults", "experiments", "sensors", "submission", "scoring")


@functools.lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


def ids(spec: PlantSpec, kind: str) -> list[str]:
    return [ln.id for ln in spec.links if ln.kind == kind]


def pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    return seq[int(rng.integers(len(seq)))]


def running(instance: Instance, **kw: Any) -> Episode:
    ep = Episode(instance, clock=lambda: 0.0, **kw)
    ep.call("get_topology", {})  # INIT -> RUNNING before any hash comparison
    return ep


def true_submission(instance: Instance) -> dict:
    """The true faults as a submission (magnitude omitted for sensor_stuck, as section 5 allows)."""
    diagnosis = [{"component": f.component, "type": f.type, "confidence": 0.7,
                  **({} if f.type == "sensor_stuck" else {"magnitude": f.magnitude})} for f in instance.faults]
    return {"diagnosis": diagnosis, "repairs": [{"action": REPAIR[f.type], "target": f.component} for f in instance.faults]}


def sub(diagnosis: list | None = None, repairs: list | None = None) -> dict:
    return {"diagnosis": diagnosis or [], "repairs": repairs or []}


def foul(s: PlantSpec, **kw: Any) -> dict:
    return {"component": ids(s, "pipe")[0], "type": "fouling", "magnitude": 1.0, **kw}


def without(d: dict, key: str) -> dict:
    return {k: v for k, v in d.items() if k != key}


# Content errors that are unambiguously submission-validation failures (section 2: E_INVALID_SUBMISSION).
STRICT = [
    ("missing-magnitude", lambda s: sub([without(foul(s), "magnitude")]), ("magnitude",)),
    ("bias-missing-magnitude", lambda s: sub([{"component": s.sensors[0].id, "type": "sensor_bias"}]), ("magnitude",)),
    ("negative-magnitude", lambda s: sub([foul(s, magnitude=-0.5)]), ("magnitude",)),
    ("kind-mismatch", lambda s: sub([{**foul(s), "type": "leak"}]), ("type", "component")),
    ("confidence>1", lambda s: sub([foul(s, confidence=1.5)]), ("confidence",)),
    ("confidence<0", lambda s: sub([foul(s, confidence=-0.1)]), ("confidence",)),
    ("11-diagnoses", lambda s: sub([foul(s)] * 11), ("diagnosis",)),
    ("11-repairs", lambda s: sub([], [{"action": "clean_pipe", "target": ids(s, "pipe")[0]}] * 11), ("repairs",)),
    ("action-kind-mismatch", lambda s: sub([], [{"action": "free_valve", "target": ids(s, "pipe")[0]}]),
     ("action", "target")),
]
# Sections 2 and 3 conflict for these (unknown ids, wrong types, NaN, unknown fields): either code is accepted.
LENIENT = [
    ("unknown-type", lambda s: sub([{**foul(s), "type": "corrosion"}]), ("type",)),
    ("unknown-component", lambda s: sub([{**foul(s), "component": "Q99"}]), ("component",)),
    ("unknown-action", lambda s: sub([], [{"action": "polish", "target": ids(s, "pipe")[0]}]), ("action",)),
    ("unknown-target", lambda s: sub([], [{"action": "clean_pipe", "target": "Q99"}]), ("target",)),
    ("string-magnitude", lambda s: sub([foul(s, magnitude="1.0")]), ("magnitude",)),
    ("nan-magnitude", lambda s: sub([foul(s, magnitude=NAN)]), ("magnitude",)),
    ("sign-in-submit", lambda s: sub([{"component": s.sensors[0].id, "type": "sensor_bias", "magnitude": 5.0,
                                       "sign": -1}]), ("sign",)),
    ("diagnosis-not-list", lambda s: {"diagnosis": ids(s, "pipe")[0], "repairs": []}, ("diagnosis",)),
    ("entry-not-object", lambda s: {"diagnosis": [5], "repairs": []}, ("diagnosis",)),
]
INVALID_CODES = {"E_INVALID_SUBMISSION", "E_INVALID_ARGUMENT"}


def test_max_strikes_constant() -> None:
    """F12 'an invalid format is rejected up to 3 times'."""
    assert MAX_STRIKES == 3


@pytest.mark.parametrize("case", CASES, ids=[f"{s}-{t}" for s, t in CASES])
def test_valid_submit_locks_episode(case: tuple[int, str]) -> None:
    """F12 'a valid submit locks the episode and returns {"status": "submitted"}, never a score'; 'a second submit
    is rejected'; 'post-submit calls are rejected with the budget unchanged'."""
    instance = inst(*case)
    spec = instance.spec
    ep = running(instance)
    ep.call("read_sensor", {"sensor_id": spec.sensors[0].id})
    assert ep.call("submit", true_submission(instance)) == {"status": "submitted"}
    assert ep.status == "SUBMITTED"
    locked, spent = ep.state_hash(), ep.trace[-1]["spent"]
    after = [("submit", true_submission(instance)), ("submit", sub()), ("read_sensor", {"sensor_id": spec.sensors[0].id}),
             ("run_experiment", {"type": "pump_off_static", "args": {}}), ("simulate", {"overrides": []}),
             ("get_topology", {}), ("read_manual", {}), ("budget_status", {})]
    for tool, args in after:
        with pytest.raises(ToolRejected) as e:
            ep.call(tool, args)
        assert e.value.code == "E_EPISODE_OVER", tool
        assert ep.state_hash() == locked and ep.trace[-1]["spent"] == spent == 1 and ep.status == "SUBMITTED"


def test_duplicates_and_empty_lists_allowed() -> None:
    """F12 / section 5: duplicate diagnosis entries are allowed (scored as false positives), as are empty lists."""
    instance = inst(1, "easy")
    assert running(instance).call("submit", sub([foul(instance.spec)] * 2)) == {"status": "submitted"}
    assert running(instance).call("submit", sub()) == {"status": "submitted"}


@pytest.mark.parametrize("build,fields", [(b, f) for _, b, f in STRICT], ids=[n for n, _, _ in STRICT])
def test_invalid_submission_field_level(build: Callable[[PlantSpec], dict], fields: tuple[str, ...]) -> None:
    """F12 'rejected ... with field-level errors only': E_INVALID_SUBMISSION naming the field; the episode keeps
    running and nothing is charged."""
    instance = inst(2, "medium")
    ep = running(instance)
    with pytest.raises(ToolRejected) as e:
        ep.call("submit", build(instance.spec))
    assert e.value.code == "E_INVALID_SUBMISSION" and any(f in e.value.message for f in fields), e.value.message
    assert ep.status == "RUNNING" and ep.trace[-1]["spent"] == 0


@pytest.mark.parametrize("build,fields", [(b, f) for _, b, f in LENIENT], ids=[n for n, _, _ in LENIENT])
def test_invalid_submission_ambiguous_code(build: Callable[[PlantSpec], dict], fields: tuple[str, ...]) -> None:
    """F12 field-level rejection where spec sections 2 and 3 disagree on the code: rejected, field named, the
    episode keeps running."""
    instance = inst(2, "medium")
    ep = running(instance)
    with pytest.raises(ToolRejected) as e:
        ep.call("submit", build(instance.spec))
    assert e.value.code in INVALID_CODES and any(f in e.value.message for f in fields), e.value.message
    assert ep.status == "RUNNING" and ep.trace[-1]["spent"] == 0


def strike_cases(spec: PlantSpec) -> list[dict]:
    return [build(spec) for name, build, _ in STRICT if name in ("missing-magnitude", "kind-mismatch",
                                                                   "confidence>1", "11-diagnoses")]


def test_three_strikes_then_valid_submit_accepted() -> None:
    """F12 'rejected up to 3 times': after three invalid submits a valid submit is still accepted."""
    instance = inst(2, "medium")
    ep = running(instance)
    for bad in strike_cases(instance.spec)[:MAX_STRIKES]:
        with pytest.raises(ToolRejected) as e:
            ep.call("submit", bad)
        assert e.value.code == "E_INVALID_SUBMISSION" and ep.status == "RUNNING"
    assert ep.call("submit", true_submission(instance)) == {"status": "submitted"}


def test_fourth_invalid_submit_finalizes() -> None:
    """F12 / spec section 2: after 3 strikes the next invalid submit finalizes the episode (empty submission);
    every later call is E_EPISODE_OVER."""
    instance = inst(2, "medium")
    ep = running(instance)
    bad = strike_cases(instance.spec)
    for b in bad[:MAX_STRIKES]:
        with pytest.raises(ToolRejected):
            ep.call("submit", b)
    with pytest.raises(ToolRejected) as e:
        ep.call("submit", bad[MAX_STRIKES])
    assert e.value.code in {"E_INVALID_SUBMISSION", "E_EPISODE_OVER"} and ep.status == "FINALIZED"
    final = ep.state_hash()
    for tool, args in [("submit", true_submission(instance)), ("budget_status", {})]:
        with pytest.raises(ToolRejected) as e:
            ep.call(tool, args)
        assert e.value.code == "E_EPISODE_OVER" and ep.state_hash() == final


def test_finalize_idempotent() -> None:
    """F12 'a crash or timeout finalizes': ep.finalize() returns None, is idempotent, and locks the episode."""
    instance = inst(1, "easy")
    ep = running(instance)
    assert ep.finalize() is None and ep.status == "FINALIZED"
    final = ep.state_hash()
    assert ep.finalize() is None and ep.state_hash() == final
    with pytest.raises(ToolRejected) as e:
        ep.call("submit", true_submission(instance))
    assert e.value.code == "E_EPISODE_OVER"


def forbidden_keys(obj: object) -> list[str]:
    out, stack = [], [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            out += [k for k in x if isinstance(k, str) and k.lower() in ("score", "reward")]
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return out


def random_episode(instance: Instance, rng: np.random.Generator) -> list[object]:
    """Every response of one random episode: all info tools, random probes, then a valid or invalid submit."""
    spec = instance.spec
    ep, responses = Episode(instance, clock=lambda: 0.0), []
    sensors = [s.id for s in spec.sensors]
    menu = [lambda: ("read_sensor", {"sensor_id": pick(rng, sensors)}),
            lambda: ("run_experiment", {"type": "isolate_branch", "args": {"branch": pick(rng, spec.branches).id}}),
            lambda: ("simulate", {"overrides": [], "sensors": sensors[:2]}),
            lambda: ("budget_status", {}),
            lambda: ("submit", pick(rng, strike_cases(spec)))]
    calls = [("get_topology", {}), ("read_manual", {})] + [("read_manual", {"topic": t}) for t in TOPICS]
    calls += [pick(rng, menu)() for _ in range(int(rng.integers(1, 12)))]
    calls += [("submit", true_submission(instance) if rng.random() < 0.5 else sub([foul(spec, magnitude=-1.0)]))]
    calls += [("budget_status", {})]
    for tool, args in calls:
        try:
            responses.append(ep.call(tool, args))
        except ToolRejected as e:
            responses.append({"code": e.code, "message": e.message})
    return responses


def test_no_score_or_reward_key_small() -> None:
    """F12 'the score never appears in any response' (small count): no response of 30 random episodes contains a
    key named score or reward."""
    rng = np.random.default_rng(5)
    for k in range(30):
        assert not forbidden_keys(random_episode(inst(*CASES[k % 3]), rng))


@pytest.mark.slow
def test_no_score_or_reward_key_10000() -> None:
    """F12 'grep across 10,000 episodes for score fields'."""
    rng = np.random.default_rng(6)
    for k in range(10_000):
        assert not forbidden_keys(random_episode(inst(*CASES[k % 3]), rng)), k


@pytest.mark.parametrize("ending", ["crash", "four-strikes"])
def test_crash_still_yields_verifier_result(ending: str, tmp_path: Path) -> None:
    """F12 valid-when 'an agent crash still yields a verifier result': with no submit (crash) or after the fourth
    invalid submit, finalize (spec section 7) rebuilds from episode.json and writes a reward of 0 (empty
    submission)."""
    instance = inst(2, "medium")
    state = tmp_path / "state"
    state.mkdir()
    ep = running(instance, state_dir=state)
    ep.call("read_sensor", {"sensor_id": instance.spec.sensors[0].id})
    if ending == "four-strikes":
        for b in strike_cases(instance.spec):
            with pytest.raises(ToolRejected):
                ep.call("submit", b)
        assert ep.status == "FINALIZED"
    out = tmp_path / "reward.json"
    proc = subprocess.run([sys.executable, "-m", "diagnoselab.server.finalize", "--state", str(state), "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    reward = json.loads(out.read_text())["reward"]
    assert isinstance(reward, (int, float)) and not isinstance(reward, bool) and reward == 0.0
