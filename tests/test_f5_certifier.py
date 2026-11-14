"""F5 instance certifier (plan section 8, F5). Coherence rule per the option-E decision: a true-fault column
with |cos| > 0.95 against a competitor passes only if it is still >= 5 sigma apart at its true magnitude."""

import pytest

from diagnoselab.eval.validate_sim import f5
from diagnoselab.sim.certifier import COHERENCE_MAX, CRB_MAX, SEPARATION_MIN, certify
from diagnoselab.sim.instance import make_instance


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_make_instance_deterministic_and_certified(tier: str) -> None:
    """F5 valid-when: decisions identical on repeated runs; accepted instances meet every rule; no drift."""
    a, b = make_instance(3, tier), make_instance(3, tier)
    assert a.faults == b.faults and a.cert == b.cert and a.drift is None
    assert a.cert.accepted and max(a.cert.crb_rel.values()) <= CRB_MAX
    assert all(c <= COHERENCE_MAX or a.cert.separation[k] >= SEPARATION_MIN for k, c in a.cert.coherence.items())
    assert certify(a.spec, a.faults) == a.cert


@pytest.mark.slow
def test_discard_rate_below_35pct() -> None:
    """F5 valid-when: discard rate below 35% (first-draw rejection, 100 dev seeds per tier)."""
    m = f5(100)
    assert all(v < 0.35 for v in m["discard"].values()), m["discard"]


@pytest.mark.skip(reason="needs the F19 oracle (plan Step 5)")
def test_oracle_succeeds_on_accepted_instances() -> None:
    """F5 valid-when: on 200 accepted dev instances, oracle D >= 0.85 in >= 90% of them."""
