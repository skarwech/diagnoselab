"""F17 efficiency, total reward and the finalize object.

Contract: docs/scoring_spec.md sections 4, 5 and 6; plan section 5.1 and section 8, F17.
Episode states are built the way finalize sees them: a real `Episode` persists `episode.json`, and its
`state` goes to `score_episode`. The oracle-dependent conditions wait for F19 (Step 5).
"""

from __future__ import annotations

import itertools
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import pytest

from diagnoselab.scoring.reward import diagnosis_score, efficiency, restoration_score, score_episode, total_reward
from diagnoselab.server.episode import Episode
from diagnoselab.server.tools import FAULT_KIND, REPAIR_KIND, component_kinds
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Fault, Instance

TIERS = ("easy", "medium", "hard")
CASES = [(seed, tier) for tier in TIERS for seed in range(4)]  # dev seeds
BUDGET = 40
TOL = 1e-6  # plan decision 8: scoring rounds to 1e-6
KEYS = {"reward", "D", "S", "efficiency", "spent", "submitted", "n_true", "matched_credit", "false_positives",
        "unneeded_repairs", "restoration"}
ACTION = {kind: action for action, kind in REPAIR_KIND.items()}
TYPES_OF = {kind: [t for t, k in FAULT_KIND.items() if k == kind] for kind in set(FAULT_KIND.values())}


@lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


# --- helpers -------------------------------------------------------------------------------------
def entry(f: Fault) -> dict:
    out = {"component": f.component, "type": f.type}
    return out if f.type == "sensor_stuck" else {**out, "magnitude": f.magnitude}


def perfect(i: Instance) -> dict:
    """Every true fault at its true magnitude, and a repair of every faulty component."""
    kinds = component_kinds(i.spec)
    return {"diagnosis": [entry(f) for f in i.faults],
            "repairs": [{"action": ACTION[kinds[f.component]], "target": f.component} for f in i.faults]}


def random_submission(rng: random.Random, i: Instance) -> dict:
    """A valid submission (stored shape) mixing true faults, wrong faults, needed and unneeded repairs."""
    kinds = component_kinds(i.spec)
    comps = sorted(kinds)
    diagnosis = []
    for _ in range(rng.randint(0, 10)):
        if rng.random() < 0.5:
            f = rng.choice(i.faults)
            e = entry(f)
            if "magnitude" in e:
                e["magnitude"] = f.magnitude * rng.uniform(0.0, 2.0)
        else:
            comp = rng.choice(comps)
            e = {"component": comp, "type": rng.choice(TYPES_OF[kinds[comp]])}
            if e["type"] != "sensor_stuck":
                e["magnitude"] = rng.uniform(0.0, 5.0)
        diagnosis.append(e)
    pool = [f.component for f in i.faults] + comps
    repairs = [{"action": ACTION[kinds[c]], "target": c} for c in (rng.choice(pool) for _ in range(rng.randint(0, 10)))]
    return {"diagnosis": diagnosis, "repairs": repairs}


def state(submission: dict | None, spent: int) -> dict:
    return {"status": "SUBMITTED" if submission is not None else "FINALIZED", "spent": spent, "costed": 0,
            "simulate": 0, "strikes": 0, "submission": submission}


def episode_state(i: Instance, tmp_path: Path, reads: int, submission: dict | None) -> dict:
    """Run a real episode (reads at cost 1 each, then an optional submit) and return the persisted state."""
    ep = Episode(i, state_dir=tmp_path)
    for sid in itertools.islice(itertools.cycle([s.id for s in i.spec.sensors]), reads):
        ep.call("read_sensor", {"sensor_id": sid})
    if submission is not None:
        ep.call("submit", submission)
    ep.finalize()
    return json.loads((tmp_path / "episode.json").read_text())["state"]


def check_finalize_object(result: dict) -> None:
    """scoring_spec section 5: a flat object with exactly the listed keys, finite numbers only."""
    assert set(result) == KEYS, f"extra {set(result) - KEYS}, missing {KEYS - set(result)}"
    for key, value in result.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (key, value)
        assert math.isfinite(value), (key, value)
    json.dumps(result, allow_nan=False)


# --- efficiency and total reward -----------------------------------------------------------------
@pytest.mark.parametrize("d", [0.0, 0.1, 0.2, 0.3, 0.39, 0.399999, 0.4])
def test_efficiency_zero_when_d_below_gate(d: float) -> None:
    """F17: efficiency with D below 0.4 is 0 (g(D) = clip((D - 0.4)/0.3, 0, 1)), whatever the spend."""
    for spent in range(BUDGET + 1):
        assert efficiency(d, spent) == pytest.approx(0.0, abs=1e-12), spent


@pytest.mark.parametrize("d, spent, expected", [
    (0.55, 0, 0.5), (0.7, 0, 1.0), (1.0, 0, 1.0), (1.0, 20, 0.5), (0.85, 10, 0.75), (1.0, 40, 0.0),
    (0.7, 30, 0.25), (0.475, 8, 0.2),
])
def test_efficiency_formula(d: float, spent: int, expected: float) -> None:
    """F17 / scoring_spec section 4: efficiency = g(D) * (1 - spent/40)."""
    assert efficiency(d, spent) == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("d, s, spent, expected", [
    (0.0, 0.0, 0, 0.0), (1.0, 1.0, 0, 1.0), (1.0, 1.0, 40, 0.8), (0.0, 1.0, 0, 0.35), (0.5, 0.0, 0, 0.225 + 0.2 / 3),
    (0.3, 0.6, 10, 0.135 + 0.21), (0.85, 0.5, 20, 0.3825 + 0.175 + 0.1),
])
def test_total_reward_formula(d: float, s: float, spent: int, expected: float) -> None:
    """F17 / scoring_spec section 4: R = 0.45 D + 0.35 S + 0.20 efficiency."""
    assert total_reward(d, s, spent) == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("d", [0.7, 0.75, 0.85, 1.0])
@pytest.mark.parametrize("s", [0.0, 0.3, 1.0])
def test_lower_spend_strictly_higher_reward(d: float, s: float) -> None:
    """F17: at fixed D >= 0.7, lower spend gives strictly higher R (integer spends 0-40)."""
    rewards = [total_reward(d, s, spent) for spent in range(BUDGET + 1)]
    for lo, hi in itertools.combinations(range(BUDGET + 1), 2):
        assert rewards[lo] > rewards[hi], (d, s, lo, hi, rewards[lo], rewards[hi])


def test_reward_in_unit_interval() -> None:
    """F17: R is in [0, 1] for D, S in [0, 1] and spent in [0, 40]."""
    rng = random.Random(1700)
    grid = [0.0, 0.4, 0.7, 1.0]
    points = [(d, s, spent) for d in grid for s in grid for spent in (0, 40)]
    points += [(rng.random(), rng.random(), rng.choice((rng.randint(0, BUDGET), rng.uniform(0, BUDGET))))
               for _ in range(5000)]
    for d, s, spent in points:
        r = total_reward(d, s, spent)
        assert -1e-12 <= r <= 1.0 + 1e-9, (d, s, spent, r)


# --- score_episode ---------------------------------------------------------------------------------
@pytest.mark.parametrize("seed, tier", CASES)
def test_stop_early_gives_zero(seed: int, tier: str, tmp_path: Path) -> None:
    """F17: stop-early (empty diagnosis, no repairs, spent 0) gives efficiency 0 and R = 0."""
    i = inst(seed, tier)
    st = episode_state(i, tmp_path, reads=0, submission={"diagnosis": [], "repairs": []})
    assert st["spent"] == 0 and st["submission"] is not None
    for result in (score_episode(i, st), score_episode(i, state({"diagnosis": [], "repairs": []}, 0))):
        check_finalize_object(result)
        assert result["reward"] == pytest.approx(0.0, abs=1e-12)
        assert result["efficiency"] == pytest.approx(0.0, abs=1e-12)
        assert result["D"] == pytest.approx(0.0, abs=1e-12) and result["S"] == pytest.approx(0.0, abs=1e-12)
        assert result["submitted"] == 1


@pytest.mark.parametrize("seed, tier", CASES)
@pytest.mark.parametrize("reads", [0, 7])
def test_no_submission_scores_zero(seed: int, tier: str, reads: int, tmp_path: Path) -> None:
    """F17 / scoring_spec sections 1 and 5 (plan 5.1): with submission None, R = 0 (D = 0, S = 0, efficiency
    gated to 0), submitted = 0, and the object still has exactly the section 5 keys."""
    i = inst(seed, tier)
    st = episode_state(i, tmp_path, reads=reads, submission=None)
    assert st["submission"] is None and st["spent"] == reads
    result = score_episode(i, st)
    check_finalize_object(result)
    assert result["reward"] == 0
    assert result["D"] == 0 and result["S"] == 0 and result["efficiency"] == 0
    assert result["submitted"] == 0
    assert result["spent"] == reads
    assert result["n_true"] == len(i.faults)


@pytest.mark.parametrize("seed, tier", CASES)
def test_perfect_submission_through_episode(seed: int, tier: str, tmp_path: Path) -> None:
    """F17: a perfect submission after k reads scores D = 1, S = 1, efficiency = 1 - k/40,
    R = 0.45 + 0.35 + 0.20 (1 - k/40)."""
    i, reads = inst(seed, tier), 9
    result = score_episode(i, episode_state(i, tmp_path, reads=reads, submission=perfect(i)))
    check_finalize_object(result)
    assert result["D"] == pytest.approx(1.0, abs=TOL) and result["S"] == pytest.approx(1.0, abs=TOL)
    assert result["efficiency"] == pytest.approx(1.0 - reads / BUDGET, abs=TOL)
    assert result["reward"] == pytest.approx(0.8 + 0.2 * (1.0 - reads / BUDGET), abs=TOL)
    assert result["spent"] == reads and result["submitted"] == 1 and result["n_true"] == len(i.faults)
    assert result["matched_credit"] == pytest.approx(len(i.faults), abs=TOL)
    assert result["false_positives"] == 0 and result["unneeded_repairs"] == 0
    assert result["restoration"] == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("seed, tier", CASES)
def test_score_episode_lower_spend_strictly_higher(seed: int, tier: str) -> None:
    """F17: at fixed D >= 0.7 (here D = 1), lower spend gives strictly higher R through score_episode."""
    i = inst(seed, tier)
    rewards = [score_episode(i, state(perfect(i), spent))["reward"] for spent in range(BUDGET + 1)]
    assert all(a > b for a, b in zip(rewards, rewards[1:])), rewards


@pytest.mark.parametrize("seed, tier", CASES)
def test_score_episode_consistent_and_in_range(seed: int, tier: str) -> None:
    """F17 / scoring_spec sections 5-6: on random valid submissions the finalize object has exactly the
    section 5 keys, finite numbers, R in [0, 1], and agrees with the component functions."""
    i, rng = inst(seed, tier), random.Random(1701 + seed)
    for _ in range(25):
        sub, spent = random_submission(rng, i), rng.randint(0, BUDGET)
        result = score_episode(i, state(sub, spent))
        check_finalize_object(result)
        d, r = diagnosis_score(i.faults, sub["diagnosis"]), restoration_score(i, sub["repairs"])
        assert result["D"] == pytest.approx(d["D"], abs=TOL)
        assert result["matched_credit"] == pytest.approx(d["matched_credit"], abs=TOL)
        assert result["false_positives"] == d["false_positives"]
        assert result["S"] == pytest.approx(r["S"], abs=TOL)
        assert result["restoration"] == pytest.approx(r["restoration"], abs=TOL)
        assert result["unneeded_repairs"] == r["unneeded_repairs"]
        assert result["efficiency"] == pytest.approx(efficiency(d["D"], spent), abs=TOL)
        assert result["reward"] == pytest.approx(total_reward(d["D"], r["S"], spent), abs=TOL)
        assert -1e-12 <= result["reward"] <= 1.0 + 1e-9
        assert result["spent"] == spent and result["submitted"] == 1 and result["n_true"] == len(i.faults)


# --- oracle-dependent (Step 5) -----------------------------------------------------------------
def test_oracle_reward_at_least_half_per_seed() -> None:
    """F17 valid-when: R_oracle per seed is at least 0.5 (no tiny denominators)."""
    pytest.skip("needs the F19 oracle (Step 5)")


def test_cached_oracle_equals_recomputation() -> None:
    """F17 valid-when: cached oracle scores (results/oracle_scores.json) equal recomputation."""
    pytest.skip("needs the F19 oracle (Step 5)")
