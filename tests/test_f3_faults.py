"""F3 fault library: signature tests, uniformity, KPI rule, drift trigger (plan section 8, F3)."""

import numpy as np
import pytest

from diagnoselab.eval.validate_sim import f3
from diagnoselab.sim.faults import apply_drift, drift_fires, sample_drift, sample_faults
from diagnoselab.sim.generator import generate_plant, stream


@pytest.fixture(scope="module")
def m() -> dict:
    return f3()


def test_signatures(m: dict) -> None:
    """F3 valid-when: noise-free signature tests (fouling, stuck valve, wear, leak, bias, stuck sensor)."""
    assert m["fouling_r_rel_err"] < 1e-6
    assert m["stuck_valve_max_head_change"] == 0.0
    assert m["wear_head_rel_err"] < 1e-9 and m["wear_exact_law_rel_err"] < 1e-9
    assert m["eta_decreasing_in_w"] and m["vib_monotone_in_w"]
    assert m["leak_imbalance_abs_err"] < 1e-9
    assert m["bias_offset_max_err"] < 1e-12 and m["stuck_distinct_values"] == 1


def test_uniform_pairs_chi_square(m: dict) -> None:
    """F3 valid-when: chi-square on 10,000 draws shows uniform (component, type), p > 0.01."""
    assert m["chi_square_p"] > 0.01


def test_kpi_rule_and_one_fault_per_component(m: dict) -> None:
    """F3 valid-when: every physical fault degrades KPI >= 5%; at most one fault per component."""
    assert m["kpi_min_individual_impact"] >= 0.05
    assert m["multi_fault_components"] == 0 and m["magnitudes_in_range"]
    assert m["tier_counts_ok_of_100"] == {"easy": 100, "medium": 100, "hard": 100}


def test_drift_fires_exactly_once(m: dict) -> None:
    """F3 valid-when: drift fires once, at the costed call that crosses 0.4 B."""
    assert {str(k): v for k, v in m["drift_fires_counts"].items()} == {"1": 1000}
    assert drift_fires(15, 1) and not drift_fires(16, 1) and not drift_fires(10, 4) and drift_fires(12, 4)


def test_drift_content_grows_or_adds() -> None:
    """F3: drift either grows an existing physical fault (capped) or adds one on a healthy component."""
    for seed in range(20):
        spec = generate_plant(seed, "hard")
        faults = sample_faults(spec, "hard", stream(seed, "hard", 1))
        drift = sample_drift(spec, faults, stream(seed, "hard", 3))
        after = apply_drift(spec, faults, drift)
        if drift.kind == "new":
            assert len(after) == len(faults) + 1 and after[-1].onset == "drift"
            assert after[-1].component not in {f.component for f in faults}
        else:
            grown = after[drift.index]
            assert grown.magnitude >= faults[drift.index].magnitude and grown.onset == "drift"


def test_sampling_is_deterministic() -> None:
    """F1/F3: same (seed, tier) gives the same fault set."""
    spec = generate_plant(5, "medium")
    a = sample_faults(spec, "medium", stream(5, "medium", 1))
    b = sample_faults(spec, "medium", stream(5, "medium", 1))
    assert a == b and np.all([f.magnitude is None or f.magnitude >= 0 for f in a])
