"""F7: the agent-facing manual (docs/manual.md) is complete, split-safe and discloses nothing hidden."""

import re
from pathlib import Path

import pytest

MANUAL = Path(__file__).resolve().parents[1] / "docs" / "manual.md"
TOPICS = ["overview", "tools", "physics", "faults", "experiments", "sensors", "submission", "scoring"]

# Plan section 2.1: (candidate-cause keywords, disambiguating experiment), checked on one line (one table row).
CONFOUNDS = [
    (("fouling", "stuck valve", "sensor bias", "leak"), "pump_off_static"),
    (("fouling", "stuck valve"), "valve_step"),
    (("leak", "flow-meter bias"), "speed_step"),
    (("pump wear", "fouling"), "speed_step"),
]


def _text() -> str:
    return MANUAL.read_text(encoding="utf-8")


def _sections() -> dict[str, str]:
    parts = re.split(r"^## ", _text(), flags=re.MULTILINE)
    return {p.split("\n", 1)[0].strip(): p for p in parts[1:]}


def test_exactly_the_eight_topic_headings_in_order() -> None:
    text = _text()
    assert re.findall(r"^## (.*)$", text, flags=re.MULTILINE) == TOPICS
    assert text.count("## ") == len(TOPICS)  # no ### subheadings or stray splits
    assert not re.search(r"^#{3,}", text, flags=re.MULTILINE)
    assert text.split("## ", 1)[0].strip(), "table of contents before the first heading"


@pytest.mark.parametrize(("causes", "experiment"), CONFOUNDS, ids=lambda x: x if isinstance(x, str) else "+".join(x))
def test_every_confound_names_its_disambiguating_experiment(causes: tuple[str, ...], experiment: str) -> None:
    lines = _sections()["experiments"].lower().splitlines()
    assert any(experiment in ln and all(c in ln for c in causes) for ln in lines)


def test_no_seed_tier_or_fault_count_disclosure() -> None:
    hits = re.findall(r"\b(seeds?|tiers?|easy|medium|hard)\b", _text(), flags=re.IGNORECASE)
    assert hits == []
    assert "DL_" not in _text()


def test_tools_section_covers_every_tool_and_error_code() -> None:
    tools = _sections()["tools"]
    for name in ("read_manual", "get_topology", "simulate", "read_sensor", "run_experiment", "budget_status",
                 "submit", "E_INVALID_ARGUMENT", "E_BUDGET", "E_EPISODE_OVER", "E_RATE_LIMIT", "E_TOO_LARGE",
                 "E_INVALID_SUBMISSION", "E_UNKNOWN_TOOL"):
        assert f"`{name}`" in tools, name


def test_scoring_states_reward_and_unscored_confidence() -> None:
    scoring = _sections()["scoring"]
    assert "R    = 0.45 * D + 0.35 * S + 0.20 * g(D) * (1 - spent / B)" in scoring
    assert "`confidence` is not scored" in scoring
