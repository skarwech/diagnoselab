"""F2 hydraulic solver (plan section 8 F2; sim_spec.md sections 2, 4, 8, 9).

Analytic cases use hand-built specs (PlantSpec.from_dict) whose link drops are all about
1 m or more, so the EPS/DELTA smoothing error (EPS^2/(4 D^2) relative) stays near 1e-8,
far below the 1e-6 tolerance; the closed forms ignore smoothing. Random cases build Fault
objects directly (not via sample_faults) so F2 does not depend on F3.

Agreement metric (differential test): |x - x_ref| <= 1e-6 * max(1, |x_ref|) for heads (m),
flows and leaks (L/s); near-zero flows are compared absolutely.
"""

from __future__ import annotations

import math
import time
from typing import Any, NamedTuple

import numpy as np
import pytest

from diagnoselab.sim.faults import apply_faults
from diagnoselab.sim.generator import generate_plant
from diagnoselab.sim.model import Fault, PlantSpec, Setpoint
from diagnoselab.sim.solver import solve
from tests.reference_solver import ref_solve

REL = 1e-6  # analytic tolerance
AGREE = 1e-6  # differential tolerance
RESIDUAL_MAX = 1e-8
MAX_ITER = 30
DELTA = 1e-6  # sim_spec section 2
TIERS = ("easy", "medium", "hard")
FAULT_TYPE = {"pipe": "fouling", "valve": "stuck_valve", "pump": "pump_wear", "junction": "leak"}
PHYSICAL_MAX = {"fouling": 4.0, "stuck_valve": 0.9, "pump_wear": 0.6, "leak": 0.15}  # section 4 table
EXTREMES = {"fouling": (2.0,), "stuck_valve": (0.9,), "pump_wear": (0.6,), "leak": (0.08, 0.15)}
N_PLANTS_FAST, N_PLANTS_FULL = 50, 500
CASES_PER_PLANT = 20  # 1000 random cases fast, 10,000 slow
N_EXTREME_PLANTS = 9
N_DIFF_FAST, N_DIFF_FULL = 100, 1000
N_MONO_FAST, N_MONO_FULL = 6, 60
MEDIAN_MAX, P99_MAX = 0.005, 0.020  # s


# --- hand-built specs ---------------------------------------------------------------------

def link(lid: str, kind: str, frm: str, to: str, r: float = 0.0, a: float = 0.0, b: float = 0.0) -> dict:
    pump = kind == "pump"
    return {"id": lid, "kind": kind, "frm": frm, "to": to, "r": r, "a": a, "b": b,
            "q_bep": 30.0 if pump else 0.0, "eta_max": 0.8 if pump else 0.0, "v0": 2.0 if pump else 0.0}


def hand_spec(junctions: list[tuple[str, float, str]], source_head: float, links: list[dict],
              outlets: tuple[str, ...] = ("T1",)) -> PlantSpec:
    return PlantSpec.from_dict({
        "seed": 0, "tier": "easy", "tank_head": 50.0,
        "junctions": [{"id": j, "z": z, "side": side} for j, z, side in junctions],
        "sources": [{"id": "S1", "head": source_head}],
        "outlets": [{"id": o, "q_req": 1.0} for o in outlets],
        "links": links, "sensors": [], "branches": [], "design": {}})


def pump_line(dz: float, a: float, b: float, r1: float, r2: float) -> PlantSpec:
    """S1 (50 - dz) -> P1 -> J1 (suction) -> PU1 -> J2 (discharge) -> P2 -> T1 (50)."""
    return hand_spec([("J1", 50.0 - dz - 4.0, "suction"), ("J2", 40.0, "discharge")], 50.0 - dz,
                     [link("P1", "pipe", "S1", "J1", r1), link("PU1", "pump", "J1", "J2", a=a, b=b),
                      link("P2", "pipe", "J2", "T1", r2)])


@pytest.mark.parametrize("reverse", [False, True])
def test_series_resistances_add(reverse: bool) -> None:
    """F2 valid-when: analytic error < 1e-6 relative (series resistances add)."""
    r1, r2 = 0.002, 0.003  # dH = 10 m: drops 4 m and 6 m
    p2 = link("P2", "pipe", "T1", "J1", r2) if reverse else link("P2", "pipe", "J1", "T1", r2)
    st = solve(hand_spec([("J1", 40.0, "discharge")], 60.0, [link("P1", "pipe", "S1", "J1", r1), p2]))
    q = math.sqrt(10.0 / (r1 + r2))
    assert st.q["P1"] == pytest.approx(q, rel=REL)
    assert st.q["P2"] == pytest.approx(-q if reverse else q, rel=REL)
    assert st.h["J1"] == pytest.approx(60.0 - r1 * q * q, rel=REL)


def test_parallel_split_between_junctions() -> None:
    """F2 valid-when: analytic error < 1e-6 relative (parallel split q_i ~ 1/sqrt(r_i))."""
    r0, r1, r2, r3 = 0.002, 0.004, 0.009, 0.002  # drops 7.4, 5.3, 7.4 m
    links = [link("P0", "pipe", "S1", "J1", r0), link("P1", "pipe", "J1", "J2", r1),
             link("P2", "pipe", "J1", "J2", r2), link("P3", "pipe", "J2", "T1", r3)]
    st = solve(hand_spec([("J1", 40.0, "discharge"), ("J2", 40.0, "discharge")], 70.0, links))
    g1, g2 = 1 / math.sqrt(r1), 1 / math.sqrt(r2)
    q = math.sqrt(20.0 / (r0 + 1 / (g1 + g2) ** 2 + r3))
    assert st.q["P0"] == pytest.approx(q, rel=REL)
    assert st.q["P1"] == pytest.approx(q * g1 / (g1 + g2), rel=REL)
    assert st.q["P2"] == pytest.approx(q * g2 / (g1 + g2), rel=REL)


def test_parallel_split_into_two_outlets() -> None:
    """F2 valid-when: analytic error < 1e-6 relative (parallel outlets at the common tank head)."""
    r0, r1, r2 = 0.002, 0.004, 0.009  # drops 5.8 m and 4.2 m
    links = [link("P0", "pipe", "S1", "J1", r0), link("P1", "pipe", "J1", "T1", r1), link("P2", "pipe", "J1", "T2", r2)]
    st = solve(hand_spec([("J1", 40.0, "discharge")], 60.0, links, outlets=("T1", "T2")))
    g1, g2 = 1 / math.sqrt(r1), 1 / math.sqrt(r2)
    q = math.sqrt(10.0 / (r0 + 1 / (g1 + g2) ** 2))
    assert st.q["P1"] == pytest.approx(q * g1 / (g1 + g2), rel=REL)
    assert st.q["P1"] / st.q["P2"] == pytest.approx(math.sqrt(r2 / r1), rel=REL)


@pytest.mark.parametrize("dz", [8.0, 0.0])
@pytest.mark.parametrize("s", [1.0, 0.8])
def test_pump_system_intersection(dz: float, s: float) -> None:
    """F2 valid-when: analytic error < 1e-6 relative (q = sqrt((a s^2 - dz)/(b + r)))."""
    a, b, r1, r2 = 20.0, 0.002, 0.001, 0.0025  # smallest drop (dz=8, s=0.8) is ~0.9 m
    st = solve(pump_line(dz, a, b, r1, r2), setpoint=Setpoint(speed={"PU1": s}))
    q = math.sqrt((a * s * s - dz) / (b + r1 + r2))
    assert st.q["PU1"] == pytest.approx(q, rel=REL)
    assert st.h["J1"] == pytest.approx(50.0 - dz - r1 * q * q, rel=REL)
    assert st.h["J2"] == pytest.approx(50.0 + r2 * q * q, rel=REL)


def test_affinity_law_zero_static_lift() -> None:
    """F2 valid-when: analytic error < 1e-6 relative (zero static lift: flow linear in speed)."""
    spec = pump_line(0.0, 20.0, 0.002, 0.001, 0.0025)
    q1 = solve(spec).q["PU1"]
    for s in (0.5, 0.8, 1.2):
        assert solve(spec, setpoint=Setpoint(speed={"PU1": s})).q["PU1"] == pytest.approx(s * q1, rel=REL)


def test_pump_off_static_zero_flow_uniform_heads() -> None:
    """F2 verifies: pump off gives zero flow and uniform static heads (hand-built).

    The smoothed check valve passes ~DELTA/(2 sqrt(b dz)); b*dz = 5 keeps it below 1e-6 L/s.
    """
    st = solve(pump_line(10.0, 40.0, 0.5, 0.001, 0.0025), setpoint=Setpoint(speed={"PU1": 0.0}))
    assert all(abs(q) < 1e-6 for q in st.q.values())
    assert st.h["J2"] == pytest.approx(50.0, abs=1e-6)
    assert st.h["J1"] == pytest.approx(40.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_pump_off_static_on_generated_plants(seed: int) -> None:
    """F2 verifies / sim_spec section 6: pumps off -> discharge heads = tank_head, suction heads =
    source head, flows effectively 0 (bounded by the smoothed check-valve leakage DELTA/sqrt(b dz))."""
    spec = generate_plant(seed, TIERS[seed % 3])
    d = spec.to_dict()
    pumps = [lk for lk in d["links"] if lk["kind"] == "pump"]
    st = solve(spec, setpoint=Setpoint(speed={pu["id"]: 0.0 for pu in pumps}))
    src = {s["id"]: s["head"] for s in d["sources"]}
    feed = {lk["to"]: src[lk["frm"]] for lk in d["links"] if lk["frm"] in src}
    feed |= {lk["frm"]: src[lk["to"]] for lk in d["links"] if lk["to"] in src}
    bound = sum(DELTA / math.sqrt(pu["b"] * max(d["tank_head"] - feed[pu["frm"]], 1e-9)) for pu in pumps)
    for j in d["junctions"]:
        want = d["tank_head"] if j["side"] == "discharge" else feed[j["id"]]
        assert st.h[j["id"]] == pytest.approx(want, abs=1e-6), j["id"]
    assert max(abs(q) for q in st.q.values()) <= bound + 1e-12


# --- random plants, faults and setpoints ---------------------------------------------------

class Case(NamedTuple):
    spec: PlantSpec
    d: dict
    faults: tuple[Fault, ...]
    speed: dict[str, float]
    valve: dict[str, float]
    closed: frozenset[str]


class Solved(NamedTuple):
    case: Case
    state: Any
    error: str | None
    seconds: float


def fault_pairs(d: dict) -> list[tuple[str, str]]:
    return [(lk["id"], FAULT_TYPE[lk["kind"]]) for lk in d["links"]] + [(j["id"], "leak") for j in d["junctions"]]


def ids_of(d: dict, kind: str) -> list[str]:
    return [lk["id"] for lk in d["links"] if lk["kind"] == kind]


def random_case(spec: PlantSpec, d: dict, rng: np.random.Generator) -> Case:
    pairs = fault_pairs(d)
    picks = rng.choice(len(pairs), size=int(rng.integers(0, 4)), replace=False)
    faults = tuple(Fault(pairs[i][0], pairs[i][1], float(rng.uniform(0.0, PHYSICAL_MAX[pairs[i][1]])))
                   for i in picks)
    speed = {p: float(rng.uniform(0.5, 1.2)) for p in ids_of(d, "pump") if rng.random() < 0.5}
    if rng.random() < 0.1:
        speed = {p: 0.0 for p in ids_of(d, "pump")}
    valve = {v: float(rng.uniform(0.1, 1.0)) for v in ids_of(d, "valve") if rng.random() < 0.5}
    entries = [b["entry"] for b in d["branches"]]
    closed = frozenset({entries[int(rng.integers(len(entries)))]}) if entries and rng.random() < 0.25 else frozenset()
    return Case(spec, d, faults, speed, valve, closed)


def extreme_cases(spec: PlantSpec, d: dict) -> list[Case]:
    """Each extreme alone (fouling x3, theta 0.1, wear 0.6, leak 8% and 15%), then one of each together
    at nominal, at all valves commanded to 0.1 and at s = 0.5 and 1.2."""
    singles = [(Fault(c, t, m),) for c, t in fault_pairs(d) for m in EXTREMES[t]]
    first = {t: next(c for c, tt in fault_pairs(d) if tt == t) for t in FAULT_TYPE.values()}
    combo = tuple(Fault(c, t, EXTREMES[t][-1]) for t, c in first.items())
    pumps, valves = ids_of(d, "pump"), ids_of(d, "valve")
    cases = [Case(spec, d, f, {}, {}, frozenset()) for f in singles]
    cases += [Case(spec, d, combo, {}, {}, frozenset()),
              Case(spec, d, combo, {}, {v: 0.1 for v in valves}, frozenset()),
              Case(spec, d, combo, {p: 0.5 for p in pumps}, {}, frozenset()),
              Case(spec, d, combo, {p: 1.2 for p in pumps}, {}, frozenset())]
    return cases


def plants(n: int) -> list[tuple[PlantSpec, dict]]:
    out = []
    for seed in range(n):
        spec = generate_plant(seed, TIERS[seed % 3])
        out.append((spec, spec.to_dict()))
    return out


def random_cases(n_plants: int) -> list[Case]:
    rng = np.random.default_rng(2026)
    return [random_case(spec, d, rng) for spec, d in plants(n_plants) for _ in range(CASES_PER_PLANT)]


def solve_case(case: Case) -> Solved:
    """Solve one case; an exception is recorded (and asserted on later) so it cannot hide the rest."""
    try:
        params = apply_faults(case.spec, case.faults)
        setpoint = Setpoint(speed=case.speed, valve=case.valve, closed=case.closed)
        t0 = time.perf_counter()
        state = solve(case.spec, params, setpoint)
        return Solved(case, state, None, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - reported by the convergence test
        return Solved(case, None, f"{type(exc).__name__}: {exc}", math.nan)


def solve_all(cases: list[Case]) -> list[Solved]:
    if cases:
        solve_case(cases[0])  # warm-up before timing
    return [solve_case(c) for c in cases]


@pytest.fixture(scope="module")
def solved() -> dict[str, list[Solved]]:
    extremes = [c for spec, d in plants(N_EXTREME_PLANTS) for c in extreme_cases(spec, d)]
    return {"random": solve_all(random_cases(N_PLANTS_FAST)), "extreme": solve_all(extremes)}


def label(case: Case) -> str:
    return (f"{case.d['tier']}/{case.d['seed']} faults={[(f.component, f.type, f.magnitude) for f in case.faults]} "
            f"speed={case.speed} valve={case.valve} closed={set(case.closed)}")


def junction_imbalance(d: dict, state: Any) -> float:
    """Max |sum(q in) - sum(q out) - q_leak| over junctions, recomputed from state.q and state.leak."""
    bal = {j["id"]: -state.leak[j["id"]] for j in d["junctions"]}
    for lk in d["links"]:
        q = state.q.get(lk["id"], 0.0)
        if lk["to"] in bal:
            bal[lk["to"]] += q
        if lk["frm"] in bal:
            bal[lk["frm"]] -= q
    return max(abs(v) for v in bal.values())


def convergence_errors(results: list[Solved]) -> list[str]:
    errors = []
    for r in results:
        if r.error:
            errors.append(f"{label(r.case)}: {r.error}")
        elif r.state.iterations > MAX_ITER or not r.state.residual < RESIDUAL_MAX:
            errors.append(f"{label(r.case)}: {r.state.iterations} it, residual {r.state.residual:.2e}")
    return errors


def mass_errors(results: list[Solved]) -> list[str]:
    errors = []
    for r in results:
        if r.state is None:
            continue  # reported by the convergence tests
        imbalance = junction_imbalance(r.case.d, r.state)
        if not imbalance < RESIDUAL_MAX:
            errors.append(f"{label(r.case)}: imbalance {imbalance:.2e}")
    return errors


def disagreements(r: Solved) -> list[str]:
    if r.state is None:
        return [f"solve failed: {r.error}"]
    c = r.case
    try:
        ref = ref_solve(c.d, faults=[{"component": f.component, "type": f.type, "magnitude": f.magnitude}
                                     for f in c.faults],
                        speed=c.speed, valve=c.valve, closed=set(c.closed))
    except Exception as exc:  # noqa: BLE001 - a reference failure is reported, not hidden
        return [f"ref_solve failed: {type(exc).__name__}: {exc}"]
    out = []
    for field in ("h", "q", "leak"):
        mine = getattr(r.state, field)
        for key, want in ref[field].items():
            got = mine.get(key)
            if got is None or abs(got - want) > AGREE * max(1.0, abs(want)):
                out.append(f"{field}[{key}] = {got} vs ref {want}")
    return out


def differential_errors(results: list[Solved]) -> list[str]:
    return [f"{label(r.case)}: {e}" for r in results for e in disagreements(r)[:3]]


def report(errors: list[str]) -> None:
    assert not errors, f"{len(errors)} failure(s); first: {errors[:5]}"


def test_random_plants_converge_within_30_iterations(solved: dict[str, list[Solved]]) -> None:
    """F2 valid-when: random plants with random faults/setpoints converge in <= 30 iterations,
    zero failures (1000 cases here; 10,000 in the slow test)."""
    report(convergence_errors(solved["random"]))


def test_extreme_faults_converge(solved: dict[str, list[Solved]]) -> None:
    """F2 verifies: extreme cases converge (fouling x3 = magnitude 2.0, stuck at theta 0.1 = 0.9,
    leak 0.08 and 0.15, wear 0.6) in <= 30 iterations."""
    report(convergence_errors(solved["extreme"]))


def test_mass_conservation_residual_below_1e8(solved: dict[str, list[Solved]]) -> None:
    """F2 valid-when: junction residuals < 1e-8 (recomputed from state.q and state.leak)."""
    report(mass_errors(solved["random"] + solved["extreme"]))


def test_agrees_with_reference_solver(solved: dict[str, list[Solved]]) -> None:
    """F2 valid-when: agreement with the second (reference) solver below 1e-6."""
    report(differential_errors(solved["random"][:N_DIFF_FAST] + solved["extreme"][::10]))


def test_solve_time_median_5ms_p99_20ms(solved: dict[str, list[Solved]]) -> None:
    """F2 expect: median solve under 5 ms, p99 under 20 ms."""
    t = np.array([r.seconds for r in solved["random"] + solved["extreme"] if r.state is not None])
    assert np.median(t) < MEDIAN_MAX, f"median {np.median(t) * 1e3:.2f} ms"
    assert np.percentile(t, 99) < P99_MAX, f"p99 {np.percentile(t, 99) * 1e3:.2f} ms"


# --- monotonicity ---------------------------------------------------------------------------

def strictly_rising(values: list[float]) -> bool:
    """Each step rises; steps that start from a near-zero flow may only stay level (1e-9)."""
    return all(b > a if a > 1e-6 else b >= a - 1e-9 for a, b in zip(values, values[1:]))


def monotonicity_errors(n_plants: int) -> list[str]:
    errors = []
    for spec, d in plants(n_plants):
        for p in ids_of(d, "pipe"):
            flows = [abs(solve(spec, apply_faults(spec, (Fault(p, "fouling", m),) if m else ())).q[p])
                     for m in (0.0, 0.5, 1.5, 3.0)]
            if not strictly_rising(flows[::-1]):
                errors.append(f"{d['tier']}/{d['seed']} pipe {p}: |q| {flows} as r rises")
        for v in ids_of(d, "valve"):
            flows = [abs(solve(spec, setpoint=Setpoint(valve={v: th})).q[v]) for th in (0.2, 0.5, 1.0)]
            if not strictly_rising(flows):
                errors.append(f"{d['tier']}/{d['seed']} valve {v}: |q| {flows} as it opens")
    return errors


def test_monotonicity_resistance_and_valve_opening() -> None:
    """F2 valid-when: zero monotonicity violations (raising a pipe's r lowers its |flow|;
    opening a valve raises its flow)."""
    report(monotonicity_errors(N_MONO_FAST))


# --- slow: plan counts ------------------------------------------------------------------------

@pytest.mark.slow
def test_full_plan_counts() -> None:
    """F2 valid-when (plan counts): 10,000 random plants+faults converge in <= 30 iterations with
    zero failures, residuals < 1e-8, 1000 differential cases agree < 1e-6, zero monotonicity
    violations on 60 plants."""
    results = solve_all(random_cases(N_PLANTS_FULL))
    report(convergence_errors(results))
    report(mass_errors(results))
    report(differential_errors(results[:: len(results) // N_DIFF_FULL]))
    report(monotonicity_errors(N_MONO_FULL))
