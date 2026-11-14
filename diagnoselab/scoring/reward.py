"""Reward (F15-F17): diagnosis D, restoration S, efficiency and total R. docs/scoring_spec.md is the contract.

Runs server-side in the sim container (numpy only). Truth comes from the certified instance; the agent's
claims are never trusted beyond the validated submission.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from diagnoselab.sim.faults import PHYSICAL_TYPES, apply_faults
from diagnoselab.sim.model import Fault, Instance
from diagnoselab.sim.solver import kpi, solve

BUDGET = 40
FP_PENALTY = 0.5
UNNEEDED_PENALTY = 0.15
BANDS = ((0.30, 1.0), (0.60, 0.5))  # relative magnitude error -> credit


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def magnitude_credit(m_sub: float | None, m_true: float) -> float:
    """Credit for a component+type match; the error is rounded to 1e-6 (plan decision 8), so 0.30 is in band."""
    if m_sub is None:
        return 0.0
    error = round(abs(float(m_sub) - m_true) / m_true, 6)
    return next((credit for limit, credit in BANDS if error <= limit), 0.0)


def diagnosis_score(true_faults: Sequence[Fault], diagnosis: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """D = clip(sum(credit) - 0.5 * FP, 0, n_true) / n_true. A component+type match with a wildly wrong magnitude
    is a match with credit 0, not a false positive (plan 5.2; the Appendix A sketch would break F15's monotonicity)."""
    used: set[int] = set()
    credit = 0.0
    for f in true_faults:
        best, best_i = -1.0, None
        for i, entry in enumerate(diagnosis):
            if i in used or (entry["component"], entry["type"]) != (f.component, f.type):
                continue
            c = 1.0 if f.type == "sensor_stuck" else magnitude_credit(entry.get("magnitude"), f.magnitude)
            if c > best:
                best, best_i = c, i
        if best_i is not None:
            used.add(best_i)
            credit += best
    fp = len(diagnosis) - len(used)
    n = len(true_faults)
    return {"D": _clip(credit - FP_PENALTY * fp, 0.0, n) / n, "matched_credit": credit, "false_positives": float(fp)}


def _kpi_with(instance: Instance, physical: tuple[Fault, ...]) -> float:
    return kpi(instance.spec, solve(instance.spec, apply_faults(instance.spec, physical)))


def restoration_score(instance: Instance, repairs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """S = clip(restoration - 0.15 * unneeded, 0, 1) on the true plant at nominal. A repair is unneeded if its
    target is not faulty or repeats an earlier repair of the same target."""
    faulty = {f.component for f in instance.faults}
    physical = tuple(f for f in instance.faults if f.type in PHYSICAL_TYPES)
    seen: set[str] = set()
    unneeded = 0
    for r in repairs:
        unneeded += r["target"] not in faulty or r["target"] in seen
        seen.add(r["target"])
    j_before, j_healthy = _kpi_with(instance, physical), _kpi_with(instance, ())
    remaining = tuple(f for f in physical if f.component not in seen)
    j_after = j_before if remaining == physical else _kpi_with(instance, remaining)
    restoration = (j_after - j_before) / (j_healthy - j_before)
    return {"S": _clip(restoration - UNNEEDED_PENALTY * unneeded, 0.0, 1.0), "restoration": restoration,
            "unneeded_repairs": float(unneeded), "J_before": j_before, "J_after": j_after, "J_healthy": j_healthy}


def efficiency(D: float, spent: float) -> float:
    """Counts only once the diagnosis is decent: g(D) = clip((D - 0.4) / 0.3, 0, 1)."""
    return _clip((D - 0.4) / 0.3, 0.0, 1.0) * (1.0 - spent / BUDGET)


def total_reward(D: float, S: float, spent: float) -> float:
    return 0.45 * D + 0.35 * S + 0.20 * efficiency(D, spent)


def score_episode(instance: Instance, state: Mapping[str, Any]) -> dict[str, float]:
    """The flat finalize object (scoring_spec section 5): finite floats only."""
    sub, spent = state.get("submission"), float(state.get("spent", 0))
    zero = {"reward": 0.0, "D": 0.0, "S": 0.0, "efficiency": 0.0, "spent": spent, "submitted": 0.0,
            "n_true": float(len(instance.faults)), "matched_credit": 0.0, "false_positives": 0.0,
            "unneeded_repairs": 0.0, "restoration": 0.0}
    if not sub:
        return zero
    d = diagnosis_score(instance.faults, sub["diagnosis"])
    s = restoration_score(instance, sub["repairs"])
    return {**zero, "reward": total_reward(d["D"], s["S"], spent), "D": d["D"], "S": s["S"],
            "efficiency": efficiency(d["D"], spent), "submitted": 1.0, "matched_credit": d["matched_credit"],
            "false_positives": d["false_positives"], "unneeded_repairs": s["unneeded_repairs"],
            "restoration": s["restoration"]}
