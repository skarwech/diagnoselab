"""Closed-form checks of the independent reference solver (plan F2).

Tolerance policy: relative 1e-6. The smoothed pipe law q = D/sqrt(r*sqrt(D^2+EPS^2))
deviates from the unsmoothed sqrt(D/r) by a relative EPS^2/(4 D^2), and the pump
law by about DELTA^2/(8 x^2). Every case below keeps each link head drop at about
1 m or more (and pump x >> DELTA), so the smoothing error is at most ~1e-8,
far below the 1e-6 tolerance; the closed forms ignore smoothing.
"""

import math

import numpy as np
import pytest

from tests.reference_solver import DELTA, pipe_drop, pipe_flow, ref_solve

REL = 1e-6
HEAD_ABS = 1e-8  # m; heads that must equal a fixed head


def pipe(lid: str, frm: str, to: str, r: float, kind: str = "pipe") -> dict:
    return {"id": lid, "kind": kind, "frm": frm, "to": to, "r": r,
            "a": 0.0, "b": 0.0, "q_bep": 0.0, "eta_max": 0.0, "v0": 0.0}


def pump(lid: str, frm: str, to: str, a: float, b: float) -> dict:
    return {"id": lid, "kind": "pump", "frm": frm, "to": to, "r": 0.0,
            "a": a, "b": b, "q_bep": 1.0, "eta_max": 0.8, "v0": 2.0}


def make_spec(junctions: list[tuple[str, float, str]], source_head: float, links: list[dict],
              tank_head: float = 50.0, design: dict | None = None) -> dict:
    return {"seed": 0, "tier": "easy", "tank_head": tank_head,
            "junctions": [{"id": j, "z": z, "side": side} for j, z, side in junctions],
            "sources": [{"id": "S1", "head": source_head}],
            "outlets": [{"id": "T1", "q_req": 1.0}],
            "links": links, "sensors": [], "branches": [], "design": design or {}}


def pump_line(source_head: float, a: float, b: float, r1: float, r2: float) -> dict:
    """S1 -> P1 -> J1 (suction) -> PU1 -> J2 (discharge) -> P2 -> T1 (tank_head 50)."""
    return make_spec([("J1", source_head - 4.0, "suction"), ("J2", 40.0, "discharge")], source_head,
                     [pipe("P1", "S1", "J1", r1), pump("PU1", "J1", "J2", a, b), pipe("P2", "J2", "T1", r2)])


# --- (a) series -----------------------------------------------------------------

@pytest.mark.parametrize("reverse_p2", [False, True])
def test_series_resistances_add(reverse_p2: bool) -> None:
    r1, r2, h_src = 0.002, 0.003, 60.0  # dH = 10 m, drops 4 m and 6 m
    p2 = pipe("P2", "T1", "J1", r2) if reverse_p2 else pipe("P2", "J1", "T1", r2)
    out = ref_solve(make_spec([("J1", 40.0, "discharge")], h_src, [pipe("P1", "S1", "J1", r1), p2]))
    q = math.sqrt((h_src - 50.0) / (r1 + r2))
    assert out["q"]["P1"] == pytest.approx(q, rel=REL)
    assert out["q"]["P2"] == pytest.approx(-q if reverse_p2 else q, rel=REL)
    assert out["h"]["J1"] == pytest.approx(h_src - r1 * q * q, rel=REL)


# --- (b) parallel ---------------------------------------------------------------

def test_parallel_between_fixed_heads() -> None:
    r1, r2 = 0.002, 0.008  # dH = 10 m across each pipe
    out = ref_solve(make_spec([], 60.0, [pipe("P1", "S1", "T1", r1), pipe("P2", "S1", "T1", r2)]))
    assert out["q"]["P1"] == pytest.approx(math.sqrt(10.0 / r1), rel=REL)
    assert out["q"]["P1"] / out["q"]["P2"] == pytest.approx(math.sqrt(r2 / r1), rel=REL)


def test_parallel_split_between_junctions() -> None:
    r0, r1, r2, r3, h_src = 0.002, 0.004, 0.009, 0.002, 70.0  # drops 7.4, 5.3, 7.4 m
    links = [pipe("P0", "S1", "J1", r0), pipe("P1", "J1", "J2", r1),
             pipe("P2", "J1", "J2", r2), pipe("P3", "J2", "T1", r3)]
    out = ref_solve(make_spec([("J1", 40.0, "discharge"), ("J2", 40.0, "discharge")], h_src, links))
    g1, g2 = 1 / math.sqrt(r1), 1 / math.sqrt(r2)
    q_tot = math.sqrt((h_src - 50.0) / (r0 + 1 / (g1 + g2) ** 2 + r3))
    assert out["q"]["P0"] == pytest.approx(q_tot, rel=REL)
    assert out["q"]["P1"] == pytest.approx(q_tot * g1 / (g1 + g2), rel=REL)
    assert out["q"]["P2"] == pytest.approx(q_tot * g2 / (g1 + g2), rel=REL)


def test_closed_link_is_removed() -> None:
    links = [pipe("P1", "S1", "T1", 0.002), pipe("P2", "S1", "T1", 0.008)]
    out = ref_solve(make_spec([], 60.0, links), closed={"P2"})
    assert out["q"]["P2"] == 0.0
    assert out["q"]["P1"] == pytest.approx(math.sqrt(10.0 / 0.002), rel=REL)


# --- (c) pump-system intersection and (d) affinity --------------------------------

@pytest.mark.parametrize("dz", [8.0, 0.0])
@pytest.mark.parametrize("s", [1.0, 0.8])
def test_pump_system_intersection(dz: float, s: float) -> None:
    a, b, r1, r2 = 20.0, 0.002, 0.001, 0.0025  # smallest drop (r1, dz=8, s=0.8) is 1.1 m
    out = ref_solve(pump_line(50.0 - dz, a, b, r1, r2), speed={"PU1": s})
    q = math.sqrt((a * s * s - dz) / (b + r1 + r2))
    assert out["q"]["PU1"] == pytest.approx(q, rel=REL)
    assert out["q"]["P1"] == pytest.approx(q, rel=REL)
    assert out["h"]["J1"] == pytest.approx(50.0 - dz - r1 * q * q, rel=REL)
    assert out["h"]["J2"] == pytest.approx(50.0 + r2 * q * q, rel=REL)


def test_affinity_law_zero_lift() -> None:
    spec = pump_line(50.0, 20.0, 0.002, 0.001, 0.0025)  # dz = 0; r1 drop at s=0.5 is 0.91 m
    q1 = ref_solve(spec)["q"]["PU1"]
    for s in (0.5, 0.8, 1.2):
        assert ref_solve(spec, speed={"PU1": s})["q"]["PU1"] == pytest.approx(s * q1, rel=REL)


# --- (e) pump off ------------------------------------------------------------------

def test_pump_off_static_heads() -> None:
    # Smoothed check valve: q = sqrt(sp(-dz)/b) ~ DELTA/(2*sqrt(b*dz)), so |q| < 1e-6
    # needs b*dz > 0.25. Here b*dz = 5 -> q ~ 2.2e-7 L/s.
    out = ref_solve(pump_line(40.0, 40.0, 0.5, 0.001, 0.0025), speed={"PU1": 0.0})
    assert all(abs(q) < 1e-6 for q in out["q"].values())
    assert out["h"]["J2"] == pytest.approx(50.0, abs=HEAD_ABS)
    assert out["h"]["J1"] == pytest.approx(40.0, abs=HEAD_ABS)


def test_pump_off_check_valve_leakage_at_generator_scale() -> None:
    # Generator-scale b ~ 1.6e-3 and dz = 3 m: the smoothed law gives ~7e-6 L/s,
    # i.e. "effectively 0" but NOT below 1e-6 (see report on sim_spec section 2).
    b, dz = 1.6e-3, 3.0
    out = ref_solve(pump_line(50.0 - dz, 15.0, b, 0.001, 0.0025), speed={"PU1": 0.0})
    assert out["q"]["PU1"] == pytest.approx(DELTA / (2 * math.sqrt(b * dz)), rel=REL)
    assert out["h"]["J2"] == pytest.approx(50.0, abs=HEAD_ABS)


# --- fault application (section 4) --------------------------------------------------

def test_fouling_scales_resistance() -> None:
    spec = make_spec([("J1", 40.0, "discharge")], 60.0, [pipe("P1", "S1", "J1", 0.002), pipe("P2", "J1", "T1", 0.003)])
    faults = [{"component": "P1", "type": "fouling", "magnitude": 1.5},
              {"component": "PT-J1", "type": "sensor_stuck", "magnitude": None}]  # sensor faults ignored
    out = ref_solve(spec, faults=faults)
    assert out["q"]["P1"] == pytest.approx(math.sqrt(10.0 / (0.002 * 2.5 + 0.003)), rel=REL)


@pytest.mark.parametrize("stuck", [None, 0.5])
def test_valve_command_and_stuck_valve(stuck: float | None) -> None:
    r1, rv = 0.002, 0.001
    spec = make_spec([("J1", 40.0, "discharge")], 60.0, [pipe("P1", "S1", "J1", r1), pipe("V1", "J1", "T1", rv, "valve")])
    faults = [] if stuck is None else [{"component": "V1", "type": "stuck_valve", "magnitude": stuck}]
    out = ref_solve(spec, faults=faults, valve={"V1": 0.3})
    theta = 0.3 if stuck is None else 1.0 - stuck  # a stuck valve ignores the command
    assert out["q"]["V1"] == pytest.approx(math.sqrt(10.0 / (r1 + rv / theta**2)), rel=REL)


def test_pump_wear_scales_shutoff_head() -> None:
    a, b, r1, r2, w, dz = 20.0, 0.002, 0.001, 0.0025, 0.3, 4.0
    out = ref_solve(pump_line(50.0 - dz, a, b, r1, r2), faults=[{"component": "PU1", "type": "pump_wear", "magnitude": w}])
    assert out["q"]["PU1"] == pytest.approx(math.sqrt((a * (1 - w) - dz) / (b + r1 + r2)), rel=REL)


def test_dead_end_leak() -> None:
    # S1 -> P1 -> J1 (leak only). sqrt((H - h)/r) = C*sqrt(h - z) => h = (H + r C^2 z)/(1 + r C^2).
    h_src, r, z, f, q_design, h_design = 60.0, 0.2, 30.0, 0.05, 50.0, 55.0  # P1 drop 1.4 m
    spec = make_spec([("J1", z, "discharge")], h_src, [pipe("P1", "S1", "J1", r)],
                     design={"h": {"J1": h_design}, "Q_pump": q_design})
    out = ref_solve(spec, faults=[{"component": "J1", "type": "leak", "magnitude": f}])
    c = f * q_design / math.sqrt(h_design - z)
    h = (h_src + r * c * c * z) / (1 + r * c * c)
    assert out["h"]["J1"] == pytest.approx(h, rel=REL)
    assert out["leak"]["J1"] == pytest.approx(c * math.sqrt(h - z), rel=REL)
    assert out["q"]["P1"] == pytest.approx(out["leak"]["J1"], rel=REL)


# --- (f) inverse design on a looped two-pump network ------------------------------------

LOOP_H = {"J1": 41.2, "J2": 64.0, "J3": 43.0, "J4": 63.5, "J5": 62.0, "J6": 59.0, "J7": 58.0, "J8": 56.0,
          "S1": 42.0, "S2": 44.0, "T1": 50.0, "T2": 50.0, "T3": 50.0, "T4": 50.0}
LOOP_Q = {"P1": 25.2, "PU1": 25.2, "P2": 25.2, "P3": 30.8, "PU2": 30.8, "P4": 30.8, "P5": 33.0,
          "P6": 23.0, "V1": 12.0, "P7": 21.0, "V2": 15.0, "V3": 9.0, "P8": 20.0, "P9": 3.0}  # P9 = chord
LOOP_TOPO = [("P1", "S1", "J1"), ("PU1", "J1", "J2"), ("P2", "J2", "J5"), ("P3", "S2", "J3"),
             ("PU2", "J3", "J4"), ("P4", "J4", "J5"), ("P5", "J5", "J6"), ("P6", "J5", "J7"),
             ("V1", "J6", "T1"), ("P7", "J6", "J8"), ("V2", "J8", "T2"), ("V3", "J8", "T4"),
             ("P8", "J7", "T3"), ("P9", "J7", "J8")]


def loop_spec(with_design_heads: bool) -> dict:
    """Parameters derived from a chosen design point as in sim_spec section 7 (r = D/q^2, a = 1.4 rise)."""
    links = []
    for lid, frm, to in LOOP_TOPO:
        drop, q = LOOP_H[frm] - LOOP_H[to], LOOP_Q[lid]
        if lid.startswith("PU"):
            links.append(pump(lid, frm, to, 1.4 * -drop, 0.4 * -drop / q**2))
        else:
            links.append(pipe(lid, frm, to, drop / q**2, "valve" if lid[0] == "V" else "pipe"))
    junctions = [(j, 38.0 if j in ("J1", "J3") else 40.0, "suction" if j in ("J1", "J3") else "discharge")
                 for j in LOOP_H if j[0] == "J"]
    spec = make_spec(junctions, 42.0, links, design={"Q_pump": 56.0, "h": {j: LOOP_H[j] for j, _, _ in junctions}
                                                     if with_design_heads else {}})
    spec["sources"].append({"id": "S2", "head": 44.0})
    spec["outlets"] = [{"id": t, "q_req": 1.0} for t in ("T1", "T2", "T3", "T4")]
    return spec


@pytest.mark.parametrize("with_design_heads", [True, False])  # False exercises the linear-network guess
def test_looped_network_recovers_design_point(with_design_heads: bool) -> None:
    # Every design drop is >= 1 m, so smoothing moves the solution by < 1e-8 relative.
    out = ref_solve(loop_spec(with_design_heads))
    for lid, q in LOOP_Q.items():
        assert out["q"][lid] == pytest.approx(q, rel=REL)
    for node, h in LOOP_H.items():
        assert out["h"][node] == pytest.approx(h, rel=REL)


def test_looped_network_extreme_faults_conserve_mass() -> None:
    faults = [{"component": "P7", "type": "fouling", "magnitude": 4.0},
              {"component": "V2", "type": "stuck_valve", "magnitude": 0.9},
              {"component": "PU1", "type": "pump_wear", "magnitude": 0.6},
              {"component": "J6", "type": "leak", "magnitude": 0.15}]
    out = ref_solve(loop_spec(True), faults=faults, valve={"V1": 0.1})
    for j in (n for n in LOOP_H if n[0] == "J"):
        net_in = sum(out["q"][l] * ((to == j) - (frm == j)) for l, frm, to in LOOP_TOPO)
        assert net_in - out["leak"][j] == pytest.approx(0.0, abs=1e-8)
    assert out["leak"]["J6"] > 0.0


# --- internals and input validation ---------------------------------------------------

def test_pipe_drop_inverts_pipe_flow() -> None:
    q = np.array([-80.0, -1.0, -1e-3, -1e-7, 0.0, 1e-7, 1e-3, 1.0, 80.0])
    r = np.full_like(q, 0.003)
    np.testing.assert_allclose(pipe_flow(pipe_drop(q, r), r), q, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("bad", [
    {"faults": [{"component": "P1", "type": "corrosion", "magnitude": 0.1}]},
    {"faults": [{"component": "P1", "type": "pump_wear", "magnitude": 0.1}]},
    {"faults": [{"component": "J1", "type": "leak", "magnitude": 0.05}]},  # no design point
    {"speed": {"P1": 0.5}},
    {"closed": {"P9"}},
])
def test_rejects_invalid_input(bad: dict) -> None:
    spec = make_spec([("J1", 40.0, "discharge")], 60.0, [pipe("P1", "S1", "J1", 0.002), pipe("P2", "J1", "T1", 0.003)])
    with pytest.raises(ValueError):
        ref_solve(spec, **bad)


def test_rejects_disconnected_junction() -> None:
    spec = make_spec([("J1", 40.0, "discharge")], 60.0, [pipe("P1", "S1", "J1", 0.002), pipe("P2", "J1", "T1", 0.003)])
    with pytest.raises(ValueError, match="not connected"):
        ref_solve(spec, closed={"P1", "P2"})
