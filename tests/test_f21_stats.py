"""F21: the statistics code matches known toy values, and the aggregation is deterministic."""

from __future__ import annotations

import statistics

import pytest

from diagnoselab.eval.validate_experiments import bootstrap_ci, f21


def test_bootstrap_constant_values() -> None:
    assert bootstrap_ci([0.4] * 12) == (pytest.approx(0.4), pytest.approx(0.4))


def test_bootstrap_brackets_the_mean_and_is_seeded() -> None:
    values = [0.0, 1.0] * 20
    lo, hi = bootstrap_ci(values)
    assert lo < statistics.fmean(values) < hi
    assert (lo, hi) == bootstrap_ci(values)  # reruns reproduce exactly
    assert hi - lo == pytest.approx(2 * 1.96 * 0.5 / 40 ** 0.5, abs=0.04)  # normal approximation of the percentile CI


def _trials(scores: dict[tuple[str, int], list[float]]) -> list[dict]:
    return [{"tier": t, "seed": s, "reward": r, "started": f"{i}", "trial": f"{t}-{s}-{i}"}
            for (t, s), rs in scores.items() for i, r in enumerate(rs)]


def test_f21_toy_gap_and_repeats() -> None:
    scores = {("easy", 1000): [0.5, 0.6, 0.7], ("medium", 1000): [0.4, 0.4, 0.4], ("hard", 1000): [0.2, 0.3, 0.1]}
    out = f21(_trials(scores))
    o = out["overall"]
    assert out["repeats"] == 3 and out["seeds"] == 3
    assert o["agent_mean"] == pytest.approx((0.6 + 0.4 + 0.2) / 3)
    assert o["gap"] == pytest.approx(1 - 0.4)
    assert o["per_repeat_gap"] == [pytest.approx(1 - (0.5 + 0.4 + 0.2) / 3), pytest.approx(1 - (0.6 + 0.4 + 0.3) / 3),
                                   pytest.approx(1 - (0.7 + 0.4 + 0.1) / 3)]
    assert out["valid"]["gap_ge_025"] and out["valid"]["each_repeat_ge_015"]


def test_f21_near_oracle_agent_fails() -> None:
    scores = {("easy", 1000 + i): [0.95, 0.96, 0.94] for i in range(5)}
    out = f21(_trials(scores))
    assert not out["valid"]["gap_ge_025"] and not out["pass"]
