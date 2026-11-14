"""F18 baselines (plan section 8, F18; docs/scoring_spec.md section 7)."""

import random
import statistics
from pathlib import Path

import pytest

from diagnoselab.baselines.policies import BASELINES
from diagnoselab.eval.validate_scoring import mcp_crosscheck, run_inprocess
from diagnoselab.sim.instance import make_instance

ROOT = Path(__file__).resolve().parents[1]


def test_baselines_see_only_the_tool_api() -> None:
    """F18: baselines run through the same API as an agent; they import no simulator or server internals."""
    src = (ROOT / "diagnoselab/baselines/policies.py").read_text()
    imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from ")) and "diagnoselab" in ln]
    assert imports == ["from diagnoselab.server.tools import ToolRejected"], imports


@pytest.mark.parametrize("name", ["noop", "stop_early"])
def test_noop_and_stop_early_score_exactly_zero(name: str) -> None:
    """F18 valid-when: no-op exactly 0; stop-early at most 0.02 (it is exactly 0 by construction)."""
    for tier in ("easy", "medium", "hard"):
        for seed in range(5):
            assert run_inprocess(BASELINES[name], make_instance(seed, tier), seed)["reward"] == 0.0


@pytest.mark.parametrize("name", ["everything_faulty", "blanket_repair"])
def test_junk_submissions_score_near_zero(name: str) -> None:
    """F18 valid-when: everything-faulty <= 0.02 and blanket-repair <= 0.05 (raw R here; normalized needs Step 5)."""
    for tier in ("easy", "medium", "hard"):
        rewards = [run_inprocess(BASELINES[name], make_instance(s, tier), s)["reward"] for s in range(10)]
        assert statistics.mean(rewards) <= 0.05, (tier, rewards)


@pytest.mark.parametrize("name", list(BASELINES))
def test_every_baseline_runs_and_is_deterministic(name: str) -> None:
    """F18: every baseline completes through the tool API, and the same seed gives the same score."""
    inst = make_instance(2, "medium")
    a, b = run_inprocess(BASELINES[name], inst, 2), run_inprocess(BASELINES[name], inst, 2)
    assert a == b and 0.0 <= a["reward"] <= 1.0


@pytest.mark.slow
def test_same_score_over_real_mcp() -> None:
    """F18: 'all baselines run through the same MCP API' - the real transport gives the identical score."""
    assert mcp_crosscheck(seeds=(0,), policies=("lazy_heuristic", "random_guess"))["all_identical"]
