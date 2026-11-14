"""F15 diagnosis score D: handcrafted unit cases and property tests.

Contract: docs/scoring_spec.md sections 2 and 6; plan section 5.2 and section 8, F15.
Entries are built in the shape the episode stores (server_spec section 5): `magnitude` only for types
other than `sensor_stuck`, never negative, optional `confidence`, no `sign`.
"""

from __future__ import annotations

import random
from typing import Sequence

import pytest

from diagnoselab.scoring.reward import diagnosis_score
from diagnoselab.sim.model import Fault, rnd

TOL = 1e-6  # plan decision 8: scoring rounds to 1e-6
EXACT = 1e-12
N_FAST, N_FULL = 2_000, 100_000
CASES = [N_FAST, pytest.param(N_FULL, marks=pytest.mark.slow)]
MAX_ENTRIES = 10  # server_spec section 5: at most 10 diagnosis entries

SENSOR_TYPES = ("sensor_bias", "sensor_stuck")
PHYS_TYPE = {"pipe": "fouling", "valve": "stuck_valve", "pump": "pump_wear", "junction": "leak"}
RANGES = {"fouling": (0.3, 2.0), "stuck_valve": (0.2, 0.8), "pump_wear": (0.10, 0.35), "leak": (0.02, 0.08),
          "sensor_bias": (4.0, 10.0)}
POOL = ([(f"P{i}", "pipe") for i in range(1, 9)] + [(f"V{i}", "valve") for i in range(1, 4)]
        + [(f"PU{i}", "pump") for i in range(1, 3)] + [(f"J{i}", "junction") for i in range(1, 9)]
        + [(f"S{i}", "sensor") for i in range(1, 9)])

PIPE = Fault("P1", "fouling", 0.8)
VALVE = Fault("V1", "stuck_valve", 0.5)
PUMP = Fault("PU1", "pump_wear", 0.2)
LEAK = Fault("J3", "leak", 0.05)
BIAS = Fault("S2", "sensor_bias", 6.0, sign=-1)
STUCK = Fault("S4", "sensor_stuck", None, value=12.3)
TRUTH = (PIPE, VALVE, PUMP, LEAK, BIAS, STUCK)


# --- helpers ---------------------------------------------------------------------------------
def entry(component: str, typ: str, magnitude: float | None = None, confidence: float | None = None) -> dict:
    out: dict = {"component": component, "type": typ}
    if magnitude is not None:
        out["magnitude"] = magnitude
    if confidence is not None:
        out["confidence"] = confidence
    return out


def correct(f: Fault, magnitude: float | None = None) -> dict:
    """Entry for true fault f: its component and type, the true magnitude unless one is given."""
    if f.type == "sensor_stuck":
        return entry(f.component, f.type)
    return entry(f.component, f.type, f.magnitude if magnitude is None else magnitude)


def score(truth: Sequence[Fault], diagnosis: list[dict]) -> dict:
    return diagnosis_score(tuple(truth), [dict(e) for e in diagnosis])


def section2(truth: Sequence[Fault], diagnosis: list[dict]) -> tuple[float, float, int]:
    """Transcription of scoring_spec section 2: (D, matched_credit, false_positives)."""
    by_key = {(f.component, f.type): f for f in truth}
    best: dict[tuple[str, str], float] = {}
    fp = 0
    for e in diagnosis:
        key = (e["component"], e["type"])
        f = by_key.get(key)
        if f is None:
            fp += 1
            continue
        if f.type == "sensor_stuck":
            credit = 1.0
        else:
            err = round(abs(e["magnitude"] - f.magnitude) / f.magnitude, 6)
            credit = 1.0 if err <= 0.30 else 0.5 if err <= 0.60 else 0.0
        if key in best:
            fp += 1
        best[key] = max(best.get(key, 0.0), credit)
    total = sum(best.values())
    return min(max(total - 0.5 * fp, 0.0), len(truth)) / len(truth), total, fp


# --- random cases ----------------------------------------------------------------------------
def random_fault(rng: random.Random, comp: str, kind: str) -> Fault:
    if kind == "sensor":
        if rng.random() < 0.5:
            return Fault(comp, "sensor_stuck", None, value=rnd(rng.uniform(-5.0, 50.0)))
        return Fault(comp, "sensor_bias", rnd(rng.uniform(*RANGES["sensor_bias"])), sign=rng.choice((-1, 1)))
    typ = PHYS_TYPE[kind]
    return Fault(comp, typ, rnd(rng.uniform(*RANGES[typ])))


def random_truth(rng: random.Random) -> tuple[Fault, ...]:
    return tuple(random_fault(rng, c, k) for c, k in rng.sample(POOL, rng.randint(1, 5)))


def guess(rng: random.Random, m_true: float) -> float:
    """A non-negative submitted magnitude: exact, at a band edge (float noise), anywhere, or wild."""
    u = rng.random()
    if u < 0.2:
        return m_true
    if u < 0.4:
        return m_true * (1.0 + rng.choice((-1.0, 1.0)) * rng.choice((0.3, 0.6)))
    if u < 0.9:
        return m_true * rng.uniform(0.0, 2.5)
    return m_true * rng.uniform(5.0, 50.0)


def with_confidence(rng: random.Random, e: dict) -> dict:
    return {**e, "confidence": rng.random()} if rng.random() < 0.3 else e


def guess_entry(rng: random.Random, f: Fault) -> dict:
    """Correct component and type for true fault f, any magnitude."""
    return with_confidence(rng, correct(f, None if f.type == "sensor_stuck" else guess(rng, f.magnitude)))


def wrong_entry(rng: random.Random, truth: Sequence[Fault]) -> dict:
    """An entry whose (component, type) is not in the truth: a healthy component, or a faulty sensor
    with the other sensor fault type."""
    faulty_sensors = [f for f in truth if f.type in SENSOR_TYPES]
    if faulty_sensors and rng.random() < 0.3:
        f = rng.choice(faulty_sensors)
        typ, comp = SENSOR_TYPES[1 - SENSOR_TYPES.index(f.type)], f.component
    else:
        used = {f.component for f in truth}
        comp, kind = rng.choice([p for p in POOL if p[0] not in used])
        typ = PHYS_TYPE.get(kind) or rng.choice(SENSOR_TYPES)
    magnitude = None if typ == "sensor_stuck" else rnd(rng.uniform(0.0, 12.0))
    return with_confidence(rng, entry(comp, typ, magnitude))


def random_diagnosis(rng: random.Random, truth: Sequence[Fault], exclude: Fault | None = None,
                     max_len: int = MAX_ENTRIES) -> list[dict]:
    """Random entries (duplicates possible); none for `exclude`'s (component, type)."""
    targets = [f for f in truth if f is not exclude]
    out = []
    for _ in range(rng.randint(0, max_len)):
        out.append(guess_entry(rng, rng.choice(targets)) if targets and rng.random() < 0.6
                   else wrong_entry(rng, truth))
    return out


def insert(rng: random.Random, diagnosis: list[dict], e: dict) -> list[dict]:
    i = rng.randint(0, len(diagnosis))
    return diagnosis[:i] + [e] + diagnosis[i:]


# --- handcrafted unit cases --------------------------------------------------------------------
@pytest.mark.parametrize("truth", [(PIPE,), (PUMP, BIAS), (PIPE, VALVE, STUCK), TRUTH])
def test_perfect_diagnosis_scores_one(truth: tuple[Fault, ...]) -> None:
    """F15 unit case: perfect = 1 (every true fault, true magnitude, no extras)."""
    r = score(truth, [correct(f) for f in truth])
    assert r["D"] == pytest.approx(1.0, abs=TOL)
    assert r["matched_credit"] == pytest.approx(len(truth), abs=TOL)
    assert r["false_positives"] == 0


@pytest.mark.parametrize("truth", [(PIPE,), (PUMP, BIAS), TRUTH])
def test_empty_diagnosis_scores_zero(truth: tuple[Fault, ...]) -> None:
    """F15 unit case: empty = 0."""
    r = score(truth, [])
    assert r["D"] == pytest.approx(0.0, abs=TOL)
    assert r["matched_credit"] == pytest.approx(0.0, abs=TOL)
    assert r["false_positives"] == 0


def test_all_wrong_scores_zero() -> None:
    """F15 unit case: all wrong = 0 (healthy components, and faulty sensors with the other sensor type)."""
    wrong = [entry("P2", "fouling", 0.8), entry("V2", "stuck_valve", 0.5), entry("J1", "leak", 0.05),
             entry("S2", "sensor_stuck"), entry("S4", "sensor_bias", 6.0), entry("S7", "sensor_bias", 5.0)]
    r = score(TRUTH, wrong)
    assert r["D"] == pytest.approx(0.0, abs=TOL)
    assert r["matched_credit"] == pytest.approx(0.0, abs=TOL)
    assert r["false_positives"] == len(wrong)


@pytest.mark.parametrize("m_true, m_sub, credit", [
    # exactly 0.30 relative error -> 1.0, several of them float-unfriendly (raw ratio just above 0.30)
    (1.0, 1.3, 1.0), (1.0, 0.7, 1.0), (0.3, 0.39, 1.0), (0.7, 0.91, 1.0), (0.35, 0.455, 1.0),
    (0.6, 0.78, 1.0), (0.15, 0.195, 1.0), (0.1, 0.13, 1.0),
    # exactly 0.60 relative error -> 0.5
    (0.1, 0.16, 0.5), (1.0, 1.6, 0.5), (1.0, 0.4, 0.5), (0.7, 1.12, 0.5), (0.35, 0.56, 0.5), (0.1, 0.04, 0.5),
    # just beyond a band edge -> the lower band
    (1.0, 1.3001, 0.5), (1.0, 0.6999, 0.5), (0.1, 0.13001, 0.5),
    (1.0, 1.6001, 0.0), (1.0, 0.3999, 0.0), (0.1, 0.16001, 0.0),
    # rounding to 1e-6: 0.3000004 rounds onto the edge, 0.3000006 does not
    (1.0, 1.3000004, 1.0), (1.0, 1.3000006, 0.5),
    # interior points
    (1.0, 1.0, 1.0), (1.0, 1.15, 1.0), (1.0, 1.45, 0.5), (1.0, 3.0, 0.0), (1.0, 0.0, 0.0),
])
def test_magnitude_band_boundaries(m_true: float, m_sub: float, credit: float) -> None:
    """F15 unit case: boundary relative errors at exactly 0.30 -> 1.0 and 0.60 -> 0.5 (e rounded to 1e-6),
    just beyond -> lower band."""
    r = score((Fault("P1", "fouling", m_true),), [entry("P1", "fouling", m_sub)])
    assert r["matched_credit"] == pytest.approx(credit, abs=TOL)
    assert r["D"] == pytest.approx(credit, abs=TOL)
    assert r["false_positives"] == 0


@pytest.mark.parametrize("f", [VALVE, PUMP, LEAK, BIAS, Fault("S5", "sensor_bias", 6.0, sign=1)])
@pytest.mark.parametrize("rel, credit", [(0.0, 1.0), (0.3, 1.0), (-0.3, 1.0), (0.45, 0.5), (0.6, 0.5),
                                         (-0.6, 0.5), (0.61, 0.0), (4.0, 0.0)])
def test_bands_apply_to_every_magnitude_type(f: Fault, rel: float, credit: float) -> None:
    """F15 unit case: the 0.30 / 0.60 bands use the excess quantity of every magnitude-bearing type."""
    r = score((f,), [correct(f, f.magnitude * (1.0 + rel))])
    assert r["matched_credit"] == pytest.approx(credit, abs=TOL)
    assert r["false_positives"] == 0


def test_sensor_stuck_without_magnitude_earns_full_credit() -> None:
    """F15 unit case: `sensor_stuck` without magnitude = 1.0 on a component-and-type match."""
    r = score((STUCK,), [entry(STUCK.component, "sensor_stuck")])
    assert r["matched_credit"] == pytest.approx(1.0, abs=TOL)
    assert r["D"] == pytest.approx(1.0, abs=TOL)
    r = score((STUCK, PIPE), [entry(STUCK.component, "sensor_stuck")])
    assert r["D"] == pytest.approx(0.5, abs=TOL) and r["false_positives"] == 0


@pytest.mark.parametrize("m_sub, credit", [(6.0, 1.0), (7.8, 1.0), (4.2, 1.0), (9.0, 0.5), (9.6, 0.5), (12.0, 0.0)])
def test_sensor_bias_sign_ignored(m_sub: float, credit: float) -> None:
    """F15 unit case: `sensor_bias` magnitudes are unsigned |offset|/sigma; the true fault's sign is ignored."""
    for sign in (-1, 1):
        r = score((Fault("S2", "sensor_bias", 6.0, sign=sign),), [entry("S2", "sensor_bias", m_sub)])
        assert r["matched_credit"] == pytest.approx(credit, abs=TOL), f"sign={sign}"
        assert r["false_positives"] == 0


def test_wild_magnitude_is_a_zero_credit_match_not_a_false_positive() -> None:
    """F15 unit case (scoring_spec section 2 choice, plan 5.2): a correct component and type with a wildly
    wrong magnitude is a match with credit 0, NOT a false positive."""
    r = score((PIPE, PUMP), [correct(PIPE), correct(PUMP, 5.0)])
    assert r["D"] == pytest.approx(0.5, abs=TOL)  # 0.25 if it were a false positive
    assert r["matched_credit"] == pytest.approx(1.0, abs=TOL)
    assert r["false_positives"] == 0
    r = score((PUMP,), [correct(PUMP, 50.0)])
    assert r["D"] == pytest.approx(0.0, abs=TOL) and r["false_positives"] == 0


def test_duplicates_count_as_false_positives() -> None:
    """F15 unit case: duplicates count as false positives; the best-credit entry is the match, in any order."""
    r = score((PIPE, PUMP), [correct(PIPE), correct(PIPE)])
    assert (r["D"], r["matched_credit"], r["false_positives"]) == (pytest.approx(0.25, abs=TOL),
                                                                   pytest.approx(1.0, abs=TOL), 1)
    r = score((PIPE, PUMP), [correct(PIPE)] * 3)
    assert r["D"] == pytest.approx(0.0, abs=TOL) and r["false_positives"] == 2
    for diag in ([correct(PIPE, 5.0), correct(PIPE)], [correct(PIPE), correct(PIPE, 5.0)],
                 [correct(PIPE, 1.2), correct(PIPE), correct(PIPE, 5.0)]):
        r = score((PIPE, PUMP), diag)
        assert r["matched_credit"] == pytest.approx(1.0, abs=TOL)
        assert r["false_positives"] == len(diag) - 1
    r = score((STUCK,), [entry("S4", "sensor_stuck"), entry("S4", "sensor_stuck")])
    assert r["D"] == pytest.approx(0.5, abs=TOL) and r["false_positives"] == 1


@pytest.mark.parametrize("truth, diag, expected", [
    ((PIPE, VALVE, PUMP, LEAK), [correct(PIPE)], 0.25),
    ((PIPE, VALVE, PUMP, LEAK), [correct(PIPE), correct(VALVE, 0.75)], 0.375),
    ((PIPE, VALVE, PUMP), [correct(PIPE)], 1 / 3),
    ((PIPE, VALVE, PUMP), [correct(PIPE), correct(VALVE), entry("J1", "leak", 0.05)], 0.5),
    (TRUTH, [correct(f) for f in TRUTH[:3]], 0.5),
])
def test_normalized_by_n_true(truth: tuple[Fault, ...], diag: list[dict], expected: float) -> None:
    """F15 unit case: D = (sum credit - 0.5 FP) / n_true."""
    assert score(truth, diag)["D"] == pytest.approx(expected, abs=TOL)


def test_clipped_at_zero() -> None:
    """F15 unit case: D is clipped at 0 (false positives cannot drive it negative); counts stay raw."""
    r = score((PIPE,), [entry(f"J{i}", "leak", 0.05) for i in range(1, 6)])
    assert r["D"] == pytest.approx(0.0, abs=TOL) and r["false_positives"] == 5
    r = score((PIPE, PUMP), [correct(PIPE)] + [entry(f"J{i}", "leak", 0.05) for i in range(1, 6)])
    assert r["D"] == pytest.approx(0.0, abs=TOL)
    assert r["matched_credit"] == pytest.approx(1.0, abs=TOL) and r["false_positives"] == 5


def test_confidence_never_changes_the_score() -> None:
    """F15 / plan decision 10: the optional confidence is logged, never rewarded."""
    diag = [correct(PIPE), correct(PUMP, 0.27), entry("J1", "leak", 0.05), entry("S4", "sensor_stuck")]
    base = score(TRUTH, diag)
    for conf in (0.0, 0.37, 1.0):
        r = score(TRUTH, [{**e, "confidence": conf} for e in diag])
        assert (r["D"], r["matched_credit"], r["false_positives"]) == (base["D"], base["matched_credit"],
                                                                       base["false_positives"])


def test_returns_the_section6_fields() -> None:
    """F15 / scoring_spec section 6: diagnosis_score returns D, matched_credit and false_positives."""
    r = score(TRUTH, [correct(PIPE)])
    assert {"D", "matched_credit", "false_positives"} <= set(r)


def test_handcrafted_cases_agree_with_section2_transcription() -> None:
    """F15: the in-test transcription of scoring_spec section 2 reproduces the handcrafted cases."""
    assert section2((PIPE, PUMP), [correct(PIPE), correct(PUMP, 5.0)]) == (0.5, 1.0, 0)
    assert section2((PIPE, PUMP), [correct(PIPE), correct(PIPE)]) == (0.25, 1.0, 1)
    assert section2((Fault("P1", "fouling", 1.0),), [entry("P1", "fouling", 1.3)]) == (1.0, 1.0, 0)
    assert section2((Fault("P1", "fouling", 0.1),), [entry("P1", "fouling", 0.16)]) == (0.5, 0.5, 0)


# --- property tests ------------------------------------------------------------------------------
@pytest.mark.parametrize("n", CASES)
def test_matches_section2_formula(n: int) -> None:
    """F15 property: on random cases D, matched_credit and FP equal the scoring_spec section 2 rule."""
    rng = random.Random(1500)
    bad = []
    for k in range(n):
        truth = random_truth(rng)
        diag = random_diagnosis(rng, truth)
        r, (d, credit, fp) = score(truth, diag), section2(truth, diag)
        if abs(r["D"] - d) > TOL or abs(r["matched_credit"] - credit) > TOL or r["false_positives"] != fp:
            bad.append((k, truth, diag, r, (d, credit, fp)))
    assert not bad, f"{len(bad)} of {n} cases differ from section 2; first: {bad[0]}"


@pytest.mark.parametrize("n", CASES)
def test_d_in_unit_interval(n: int) -> None:
    """F15 property: D is in [0, 1]."""
    rng = random.Random(1501)
    for k in range(n):
        truth = random_truth(rng)
        diag = random_diagnosis(rng, truth)
        d = score(truth, diag)["D"]
        assert -EXACT <= d <= 1.0 + EXACT, (k, truth, diag, d)


@pytest.mark.parametrize("n", CASES)
def test_permutation_invariant(n: int) -> None:
    """F15 property: D (and the counts) do not depend on the order of the diagnosis entries."""
    rng = random.Random(1502)
    for k in range(n):
        truth = random_truth(rng)
        diag = random_diagnosis(rng, truth)
        base = score(truth, diag)
        r = score(truth, rng.sample(diag, len(diag)))
        assert r["D"] == pytest.approx(base["D"], abs=EXACT), (k, truth, diag)
        assert r["matched_credit"] == pytest.approx(base["matched_credit"], abs=EXACT), (k, truth, diag)
        assert r["false_positives"] == base["false_positives"], (k, truth, diag)


@pytest.mark.parametrize("n", CASES)
def test_adding_a_fully_correct_fault_never_lowers_d(n: int) -> None:
    """F15 property: adding a fully correct fault (true component, type and magnitude) never lowers D.
    Precondition: no existing entry has that (component, type); otherwise the new entry is a duplicate,
    which section 2 counts as a false positive."""
    rng = random.Random(1503)
    for k in range(n):
        truth = random_truth(rng)
        target = rng.choice(truth)
        diag = random_diagnosis(rng, truth, exclude=target, max_len=MAX_ENTRIES - 1)
        before, after = score(truth, diag), score(truth, insert(rng, diag, correct(target)))
        assert after["D"] >= before["D"] - EXACT, (k, truth, diag, target)
        assert after["false_positives"] == before["false_positives"], (k, truth, diag, target)


@pytest.mark.parametrize("n", CASES)
def test_adding_correct_component_and_type_never_lowers_d(n: int) -> None:
    """F15 property: adding a correct component+type with any magnitude never lowers D and is never a false
    positive. Precondition: no existing entry has that (component, type) (duplicates are false positives)."""
    rng = random.Random(1504)
    for k in range(n):
        truth = random_truth(rng)
        target = rng.choice(truth)
        diag = random_diagnosis(rng, truth, exclude=target, max_len=MAX_ENTRIES - 1)
        before, after = score(truth, diag), score(truth, insert(rng, diag, guess_entry(rng, target)))
        assert after["D"] >= before["D"] - EXACT, (k, truth, diag, target)
        assert after["false_positives"] == before["false_positives"], (k, truth, diag, target)
        assert after["matched_credit"] >= before["matched_credit"] - EXACT, (k, truth, diag, target)


@pytest.mark.parametrize("n", CASES)
def test_adding_a_wrong_fault_never_raises_d(n: int) -> None:
    """F15 property: adding a wrong fault (a (component, type) not in the truth) never raises D; it adds
    exactly one false positive and no credit."""
    rng = random.Random(1505)
    for k in range(n):
        truth = random_truth(rng)
        diag = random_diagnosis(rng, truth, max_len=MAX_ENTRIES - 1)
        before, after = score(truth, diag), score(truth, insert(rng, diag, wrong_entry(rng, truth)))
        assert after["D"] <= before["D"] + EXACT, (k, truth, diag)
        assert after["false_positives"] == before["false_positives"] + 1, (k, truth, diag)
        assert after["matched_credit"] == pytest.approx(before["matched_credit"], abs=EXACT), (k, truth, diag)
