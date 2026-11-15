"""F19 oracle and Step 5 score normalisation, written from the specs.

Contract: docs/oracle_spec.md (sections 1, 2 and 4) and docs/scoring_spec.md sections 4 and 5.
The oracle is run in-process against a real `Episode` on a few dev seeds per tier; every call it makes is
recorded through a wrapper, so the checks below only use what crossed the tool boundary.
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from diagnoselab.config_hash import config_hash, config_id

from diagnoselab.oracle.oracle import find_stuck, run_oracle
from diagnoselab.scoring import reward as rw
from diagnoselab.scoring.reward import NORM_FLOOR, normalization, oracle_scores, score_episode
from diagnoselab.server.episode import SIMULATE_CAP, Episode
from diagnoselab.server.tools import REPAIR_KIND, SIM_RANGES, ToolRejected, component_kinds
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Fault, Instance

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "diagnoselab" / "oracle" / "oracle.py"
HARBOR_ORACLE = ROOT / "harbor_task" / "solution" / "oracle.py"
TIERS = ("easy", "medium", "hard")
SEEDS = (0, 1, 2)
CASES = [(seed, tier) for tier in TIERS for seed in SEEDS]
SPEND_CAP = 28  # oracle_spec section 2, step 5: 70% of B
MAX_ENTRIES = 10
PHYSICAL = {"fouling", "stuck_valve", "pump_wear", "leak"}
ORACLE_TOOLS = {"get_topology", "simulate", "read_sensor", "run_experiment", "submit"}
BASE_KEYS = {"reward", "D", "S", "efficiency", "spent", "submitted", "n_true", "matched_credit", "false_positives",
             "unneeded_repairs", "restoration"}
NORM_KEYS = {"r_oracle", "ratio_uncapped", "score_norm"}
ACTION = {kind: action for action, kind in REPAIR_KIND.items()}


# --- helpers -------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


def play(seed: int, tier: str, state_dir: Path | None = None) -> dict:
    """One fresh oracle episode; returns the recorded calls, rejections, oracle result and server state."""
    i = inst(seed, tier)
    ep = Episode(i, state_dir=state_dir)
    calls: list[tuple[str, dict]] = []
    rejected: list[tuple[str, str]] = []
    after_submit: list[str] = []
    submitted = False

    def call(tool: str, args: dict) -> dict:
        nonlocal submitted
        if submitted:
            after_submit.append(tool)
        calls.append((tool, json.loads(json.dumps(args))))
        try:
            res = ep.call(tool, args)
        except ToolRejected as e:
            rejected.append((tool, e.code))
            raise
        if tool == "submit":
            submitted = True
        return res

    result = run_oracle(call)
    if state_dir is not None:
        ep.finalize()
    state = dict(ep._state)
    return {"calls": calls, "rejected": rejected, "after_submit": after_submit, "result": result,
            "state": state, "reward": score_episode(i, state)["reward"], "instance": i}


@lru_cache(maxsize=None)
def run(seed: int, tier: str) -> dict:
    return play(seed, tier)


def canon(calls: list) -> str:
    return json.dumps([[t, a] for t, a in calls], sort_keys=True)


def entry(f: Fault) -> dict:
    out = {"component": f.component, "type": f.type}
    return out if f.type == "sensor_stuck" else {**out, "magnitude": f.magnitude}


def perfect(i: Instance) -> dict:
    kinds = component_kinds(i.spec)
    return {"diagnosis": [entry(f) for f in i.faults],
            "repairs": [{"action": ACTION[kinds[f.component]], "target": f.component} for f in i.faults]}


def finalize(state_dir: Path, tmp_path: Path) -> dict:
    out = tmp_path / "score.json"
    proc = subprocess.run([sys.executable, "-m", "diagnoselab.server.finalize", "--state", str(state_dir),
                           "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(out.read_text())


def check_flat_finite(result: dict) -> None:
    for key, value in result.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (key, value)
        assert math.isfinite(value), (key, value)
    json.dumps(result, allow_nan=False)


def check_finalize_object(result: dict) -> None:
    """scoring_spec section 5 and section 4's finalize rules."""
    check_flat_finite(result)
    if result["normalized"] == 1:
        assert set(result) == BASE_KEYS | {"reward_raw", "normalized", "config_id"} | NORM_KEYS
        assert result["r_oracle"] >= NORM_FLOOR
        assert result["ratio_uncapped"] == pytest.approx(result["reward_raw"] / result["r_oracle"], abs=1e-12)
        assert result["score_norm"] == min(result["ratio_uncapped"], 1.0)
        assert result["reward"] == result["score_norm"]
    else:
        assert result["normalized"] == 0
        assert set(result) == BASE_KEYS | {"reward_raw", "normalized", "config_id"}
        assert result["reward"] == result["reward_raw"]


# --- 1. static ---------------------------------------------------------------------------------
def test_oracle_imports_stdlib_only() -> None:
    """oracle_spec 1: no diagnoselab import, no numpy; standard library only (checked statically)."""
    tree = ast.parse(ORACLE.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            mods.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("__import__", "exec", "eval"), node.func.id
    assert mods, "no imports found"
    assert mods <= set(sys.stdlib_module_names), mods - set(sys.stdlib_module_names)
    assert "diagnoselab" not in mods and "numpy" not in mods and "importlib" not in mods


def test_harbor_copy_byte_identical() -> None:
    assert HARBOR_ORACLE.read_bytes() == ORACLE.read_bytes()


# --- 2. episode contract -----------------------------------------------------------------------
@pytest.mark.parametrize("seed,tier", CASES)
def test_episode_contract(seed: int, tier: str) -> None:
    """Exactly one accepted submit, nothing rejected, no call after submit, spend <= 28, simulate <= cap."""
    r = run(seed, tier)
    tools = [t for t, _ in r["calls"]]
    assert r["rejected"] == []
    assert tools.count("submit") == 1 and tools[-1] == "submit"
    assert r["after_submit"] == []
    assert set(tools) <= ORACLE_TOOLS
    assert r["state"]["status"] == "SUBMITTED" and r["state"]["submission"] is not None
    assert r["state"]["spent"] <= SPEND_CAP
    assert r["state"]["simulate"] <= SIMULATE_CAP
    assert tools.count("simulate") == r["state"]["simulate"]
    assert r["result"]["spent"] == r["state"]["spent"]


# --- 3. determinism ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed,tier", [(1, "easy"), (2, "medium"), (0, "hard")])
def test_repeat_runs_identical(seed: int, tier: str) -> None:
    """oracle_spec 4: 3 repeats give the same call sequence and score."""
    first = run(seed, tier)
    for _ in range(2):
        again = play(seed, tier)
        assert canon(again["calls"]) == canon(first["calls"])
        assert again["reward"] == first["reward"]


SUBPROCESS_SCRIPT = """
import json, sys
from diagnoselab.oracle.oracle import run_oracle
from diagnoselab.scoring.reward import score_episode
from diagnoselab.server.episode import Episode
from diagnoselab.sim.instance import make_instance
seed, tier = int(sys.argv[1]), sys.argv[2]
i = make_instance(seed, tier)
ep = Episode(i)
calls = []
def call(tool, args):
    calls.append([tool, json.loads(json.dumps(args))])
    return ep.call(tool, args)
run_oracle(call)
print(json.dumps({"calls": calls, "reward": score_episode(i, ep._state)["reward"]}))
"""


@pytest.mark.parametrize("seed,tier,hashseed", [(1, "easy", "1"), (0, "hard", "12345")])
def test_subprocess_hashseed_identical(seed: int, tier: str, hashseed: str) -> None:
    """The same call sequence and reward under a different PYTHONHASHSEED (no set/dict-order dependence)."""
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    proc = subprocess.run([sys.executable, "-c", SUBPROCESS_SCRIPT, str(seed), tier], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    r = run(seed, tier)
    assert json.dumps(out["calls"], sort_keys=True) == canon(r["calls"])
    assert out["reward"] == r["reward"]


# --- 4. repairs --------------------------------------------------------------------------------
@pytest.mark.parametrize("seed,tier", CASES)
def test_repairs_only_diagnosed_physical(seed: int, tier: str) -> None:
    """oracle_spec 1 and 2.7: repairs only diagnosed physical components; no sensor repairs; each diagnosed
    physical component repaired exactly once with its kind's action."""
    r = run(seed, tier)
    sub = r["state"]["submission"]
    kinds = component_kinds(r["instance"].spec)
    diagnosed_phys = [d["component"] for d in sub["diagnosis"] if d["type"] in PHYSICAL]
    targets = [rep["target"] for rep in sub["repairs"]]
    assert all(kinds[t] != "sensor" for t in targets)
    assert all(rep["action"] != "recalibrate_sensor" for rep in sub["repairs"])
    assert set(targets) <= set(diagnosed_phys)
    assert len(targets) == len(set(targets))
    if len(diagnosed_phys) <= MAX_ENTRIES:
        assert sorted(targets) == sorted(diagnosed_phys)
    for rep in sub["repairs"]:
        assert REPAIR_KIND[rep["action"]] == kinds[rep["target"]]


# --- 5. stuck-sensor rule ----------------------------------------------------------------------
NOM, STATIC, SPEED = ("nominal",), ("static",), ("speed", 1.2, "")


def test_find_stuck_exact_repeat_across_setpoints() -> None:
    """Manual fact (g): a stuck sensor returns the same value exactly at every operating point."""
    obs = [("P1", NOM, 30.123456), ("P2", NOM, 12.5), ("P1", SPEED, 30.123456), ("P2", SPEED, 14.25)]
    assert find_stuck(obs) == ["P1"]


def test_find_stuck_noisy_sensor_not_stuck() -> None:
    obs = [("P1", NOM, 30.123456), ("P1", NOM, 30.123457), ("P1", STATIC, 30.123455), ("F1", NOM, 0.0),
           ("F1", STATIC, 0.000001)]
    assert find_stuck(obs) == []


def test_find_stuck_per_sensor_and_order() -> None:
    """Equal values on different sensors never count; detection order is first-repeat order; no duplicates."""
    obs = [("A", NOM, 1.0), ("B", NOM, 1.0), ("B", STATIC, 2.0), ("B", SPEED, 1.0), ("A", STATIC, 1.0),
           ("A", SPEED, 1.0), ("B", NOM, 1.0)]
    assert find_stuck(obs) == ["B", "A"]
    assert find_stuck([]) == []
    assert find_stuck([("A", NOM, 5.0)]) == []


def test_find_stuck_repeat_after_other_values() -> None:
    """A repeat of any earlier value (not only the previous one) is exact, so it counts."""
    assert find_stuck([("A", NOM, 1.0), ("A", STATIC, 2.0), ("A", SPEED, 1.0)]) == ["A"]


def test_find_stuck_does_not_mutate_input() -> None:
    obs = [("A", NOM, 1.0), ("A", STATIC, 1.0)]
    before = list(obs)
    find_stuck(obs)
    assert obs == before


# --- 6. cache ----------------------------------------------------------------------------------
def test_cache_keys_and_values() -> None:
    """scoring_spec 4: {"tier:seed": R} for dev seeds 0-199 and test seeds 1000-1039 of every tier, finite R in [0, 1]."""
    cache = json.loads((ROOT / "diagnoselab" / "scoring" / "oracle_scores.json").read_text())
    assert set(cache) == {f"{t}:{s}" for t in TIERS for s in (*range(200), *range(1000, 1040))}
    for key, value in cache.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), key
        assert math.isfinite(value) and 0.0 <= value <= 1.0, (key, value)
    assert oracle_scores() == cache


def test_cache_below_floor_count() -> None:
    """scoring_spec 4: 10 of 600 dev seeds fall below the floor."""
    dev = {k: v for k, v in oracle_scores().items() if int(k.split(":")[1]) < 1000}
    assert sum(v < NORM_FLOOR for v in dev.values()) == 10


@pytest.mark.parametrize("seed,tier", CASES + [(94, "medium"), (143, "hard")])
def test_cache_reproduced_in_process(seed: int, tier: str) -> None:
    """The Linux-generated cache reproduces on this platform to 1e-12 (incl. a below-floor seed and a seed that
    uses the whole simulate cap)."""
    assert run(seed, tier)["reward"] == pytest.approx(oracle_scores()[f"{tier}:{seed}"], abs=1e-12)


# --- 7. normalisation --------------------------------------------------------------------------
@pytest.mark.parametrize("r_oracle", [None, 0.0, 0.3, NORM_FLOOR - 1e-9, math.nan, math.inf])
def test_normalization_not_normalised(monkeypatch: pytest.MonkeyPatch, r_oracle: float | None) -> None:
    cache = {} if r_oracle is None else {"easy:7": r_oracle}
    monkeypatch.setattr(rw, "oracle_scores", lambda: cache)
    out = normalization(0.42, 7, "easy")
    assert out == {"reward_raw": 0.42, "normalized": 0.0, "config_id": float(config_id())}
    check_flat_finite(out)


@pytest.mark.parametrize("reward,r_oracle", [(0.0, 0.8), (0.4, 0.8), (0.8, 0.8), (0.95, 0.8), (0.3, NORM_FLOOR),
                                             (1.0, NORM_FLOOR)])
def test_normalization_normalised(monkeypatch: pytest.MonkeyPatch, reward: float, r_oracle: float) -> None:
    monkeypatch.setattr(rw, "oracle_scores", lambda: {"hard:3": r_oracle})
    out = normalization(reward, 3, "hard")
    check_flat_finite(out)
    assert out["normalized"] == 1.0 and out["reward_raw"] == reward and out["r_oracle"] == r_oracle
    assert out["ratio_uncapped"] == reward / r_oracle
    assert out["score_norm"] == min(reward / r_oracle, 1.0)
    assert 0.0 <= out["score_norm"] <= 1.0


def test_normalization_key_is_tier_and_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rw, "oracle_scores", lambda: {"medium:5": 0.9})
    assert normalization(0.5, 5, "medium")["normalized"] == 1.0
    assert normalization(0.5, 5, "easy")["normalized"] == 0.0
    assert normalization(0.5, 6, "medium")["normalized"] == 0.0


def test_finalize_oracle_episode_normalised(tmp_path: Path) -> None:
    """The oracle's own episode on a cached seed normalises to exactly 1 (R == R_oracle)."""
    seed, tier = 0, "easy"
    assert oracle_scores()[f"{tier}:{seed}"] >= NORM_FLOOR
    state = tmp_path / "state"
    play(seed, tier, state_dir=state)
    out = finalize(state, tmp_path)
    check_finalize_object(out)
    assert out["normalized"] == 1 and out["submitted"] == 1
    assert out["ratio_uncapped"] == pytest.approx(1.0, abs=1e-12) and out["reward"] == pytest.approx(1.0, abs=1e-12)


def test_finalize_below_floor_not_normalised(tmp_path: Path) -> None:
    seed, tier = 94, "medium"
    assert oracle_scores()[f"{tier}:{seed}"] < NORM_FLOOR
    state = tmp_path / "state"
    play(seed, tier, state_dir=state)
    out = finalize(state, tmp_path)
    check_finalize_object(out)
    assert out["normalized"] == 0 and out["reward"] == out["reward_raw"]


def test_finalize_perfect_agent_capped(tmp_path: Path) -> None:
    """An agent beating the oracle gets score_norm = 1 with ratio_uncapped > 1 kept in the log."""
    seed, tier = next((s, t) for t in ("hard", "medium") for s in range(200)
                      if NORM_FLOOR <= oracle_scores()[f"{t}:{s}"] < 0.95)
    state = tmp_path / "state"
    ep = Episode(inst(seed, tier), state_dir=state)
    ep.call("submit", perfect(inst(seed, tier)))
    ep.finalize()
    out = finalize(state, tmp_path)
    check_finalize_object(out)
    assert out["normalized"] == 1 and out["ratio_uncapped"] > 1 and out["reward"] == 1.0


def test_finalize_uncached_seed_raw(tmp_path: Path) -> None:
    """Seeds outside the cache (here 1100) are not normalised: reward stays raw R."""
    seed, tier = 1100, "easy"
    assert f"{tier}:{seed}" not in oracle_scores()
    i = make_instance(seed, tier)
    state = tmp_path / "state"
    ep = Episode(i, state_dir=state)
    ep.call("read_sensor", {"sensor_id": i.spec.sensors[0].id})
    ep.call("submit", perfect(i))
    ep.finalize()
    out = finalize(state, tmp_path)
    check_finalize_object(out)
    assert out["normalized"] == 0 and out["reward"] == out["reward_raw"] > 0


def test_finalize_no_submission(tmp_path: Path) -> None:
    seed, tier = 0, "medium"
    state = tmp_path / "state"
    ep = Episode(inst(seed, tier), state_dir=state)
    ep.call("read_sensor", {"sensor_id": inst(seed, tier).spec.sensors[0].id})
    ep.finalize()
    out = finalize(state, tmp_path)
    check_finalize_object(out)
    assert out["reward_raw"] == 0.0 and out["reward"] == 0.0


# --- 8. further spec-vs-behaviour checks -------------------------------------------------------
@pytest.mark.parametrize("seed,tier", CASES)
def test_submission_shape(seed: int, tier: str) -> None:
    """<= 10 entries each; one entry per (component, type); a sensor is never both biased and stuck; physical
    magnitudes are >= 0 within the simulate ranges (oracle_spec 2.2)."""
    sub = run(seed, tier)["state"]["submission"]
    diag = sub["diagnosis"]
    assert len(diag) <= MAX_ENTRIES and len(sub["repairs"]) <= MAX_ENTRIES
    pairs = [(d["component"], d["type"]) for d in diag]
    assert len(pairs) == len(set(pairs))
    sensor_comps = [d["component"] for d in diag if d["type"].startswith("sensor_")]
    assert len(sensor_comps) == len(set(sensor_comps)), "sensor both biased and stuck"
    for d in diag:
        if d["type"] in PHYSICAL:
            lo, hi = SIM_RANGES[d["type"]]
            assert lo <= d["magnitude"] <= hi, d
        elif d["type"] == "sensor_bias":
            assert d["magnitude"] >= 0 and math.isfinite(d["magnitude"])


@pytest.mark.parametrize("seed,tier", CASES)
def test_stuck_claims_backed_by_exact_repeat(seed: int, tier: str) -> None:
    """Every sensor_stuck claim is backed by an exact repeated reading the oracle actually received."""
    r = run(seed, tier)
    stuck = {d["component"] for d in r["state"]["submission"]["diagnosis"] if d["type"] == "sensor_stuck"}
    values: dict[str, list[float]] = {}
    # rebuild the readings the oracle received: replay its calls (minus submit) on a fresh episode
    ep = Episode(r["instance"])
    for tool, args in r["calls"][:-1]:
        res = ep.call(tool, args)
        if tool == "read_sensor":
            values.setdefault(args["sensor_id"], []).append(res["value"])
        elif tool == "run_experiment":
            for s, v in res["readings"].items():
                values.setdefault(s, []).append(v)
    for s in stuck:
        vals = values.get(s, [])
        assert len(vals) != len(set(vals)), s


def _experiments(seed: int, tier: str) -> list[dict]:
    return [a for t, a in run(seed, tier)["calls"] if t == "run_experiment"]


@pytest.mark.parametrize("seed,tier", CASES)
def test_probes_in_spec_catalogue(seed: int, tier: str) -> None:
    """oracle_spec 2: all-pump speed steps at s = 1.2, valve steps at theta = 0.3, branch isolations, static test."""
    for a in _experiments(seed, tier):
        if a["type"] == "speed_step" and "pump" not in a["args"]:
            assert a["args"]["s"] in (1.2, 0.75), a  # 0.75 all-pump is the one-pump-plant local step
        elif a["type"] == "valve_step":
            assert a["args"]["theta"] == 0.3, a
        else:
            assert a["type"] in ("speed_step", "isolate_branch", "pump_off_static"), a


def test_pump_local_speed_steps_use_spec_speeds() -> None:
    """oracle_spec 2: on two-pump plants each pump is stepped at s = 0.75 or s = 1.2."""
    local = next((loc for s in range(30) if (loc := [a for a in _experiments(s, "hard")
                                                     if a["type"] == "speed_step" and "pump" in a["args"]])), [])
    if not local:
        pytest.skip("no hard seed in 0-29 runs a pump-local speed step at this budget")
    assert all(a["args"]["s"] in (0.75, 1.2) for a in local), local


def test_normalization_refuses_cache_from_other_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """scoring_spec 4: a cache built under another configuration hash is never used to normalise."""
    monkeypatch.setattr(rw, "oracle_scores", lambda: {"hard:3": 0.8})
    monkeypatch.setattr(rw, "cache_config_hash", lambda: "0" * 64)
    out = normalization(0.6, 3, "hard")
    assert out["normalized"] == 0.0 and "score_norm" not in out and out["config_id"] == float(config_id())


def test_shipped_cache_matches_running_code() -> None:
    """The installed oracle cache was generated under the code in this checkout."""
    assert rw.cache_config_hash() == config_hash()
