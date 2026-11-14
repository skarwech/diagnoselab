"""Fault library (F3): application, KPI-eligible sampling, drift. docs/sim_spec.md section 4."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from diagnoselab.sim.model import Drift, Fault, Params, PlantSpec, rnd
from diagnoselab.sim.sensors import SENSOR_TYPES, physical_readings
from diagnoselab.sim.solver import kpi, solve

MAGNITUDE_RANGES = {"fouling": (0.3, 2.0), "stuck_valve": (0.2, 0.8), "pump_wear": (0.10, 0.35),
                    "leak": (0.02, 0.08), "sensor_bias": (4.0, 10.0)}
PHYSICAL_MAX = {"fouling": 4.0, "stuck_valve": 0.9, "pump_wear": 0.6, "leak": 0.15, "sensor_bias": 14.0}
PHYSICAL_TYPES = ("fouling", "stuck_valve", "pump_wear", "leak")
TYPE_OF_KIND = {"pipe": "fouling", "valve": "stuck_valve", "pump": "pump_wear"}
TIER_COUNTS = {"easy": (1, 0), "medium": (2, 1), "hard": (3, 2)}
MIN_IMPACT = 0.05
DRIFT_FRACTION = 0.4

__all__ = ["apply_faults", "sample_faults", "eligible_pairs", "physical_pairs", "sample_drift", "apply_drift",
           "drift_fires", "impact", "MAGNITUDE_RANGES", "PHYSICAL_MAX", "PHYSICAL_TYPES", "SENSOR_TYPES"]


def apply_faults(spec: PlantSpec, faults: tuple[Fault, ...] = ()) -> Params:
    fields: dict[str, dict[str, float]] = {"fouling": {}, "stuck": {}, "wear": {}, "leak": {}}
    target = {"fouling": "fouling", "stuck_valve": "stuck", "pump_wear": "wear", "leak": "leak"}
    for f in faults:
        if f.type in SENSOR_TYPES:
            continue
        if f.type not in target:
            raise ValueError(f"unknown fault type {f.type!r}")
        value = 1.0 - f.magnitude if f.type == "stuck_valve" else f.magnitude
        fields[target[f.type]][f.component] = float(value)
    return Params(**fields)


def physical_pairs(spec: PlantSpec) -> tuple[tuple[str, str], ...]:
    pairs = [(ln.id, TYPE_OF_KIND[ln.kind]) for ln in spec.links]
    return tuple(pairs + [(j.id, "leak") for j in spec.junctions])


@lru_cache(maxsize=512)
def _healthy_kpi(spec: PlantSpec) -> float:
    return kpi(spec, solve(spec))


def impact(spec: PlantSpec, faults: tuple[Fault, ...]) -> float:
    """Relative KPI degradation (J_healthy - J_faulty) / |J_healthy|."""
    j_h = _healthy_kpi(spec)
    return (j_h - kpi(spec, solve(spec, apply_faults(spec, faults)))) / abs(j_h)


@lru_cache(maxsize=8192)
def _impact_at(spec: PlantSpec, comp: str, typ: str, m: float) -> float:
    return impact(spec, (Fault(comp, typ, m),))


@lru_cache(maxsize=512)
def eligible_pairs(spec: PlantSpec) -> tuple[tuple[str, str], ...]:
    """Physical pairs that degrade the KPI by >= 5% alone at their maximum range magnitude."""
    return tuple(p for p in physical_pairs(spec) if _impact_at(spec, *p, MAGNITUDE_RANGES[p[1]][1]) >= MIN_IMPACT)


@lru_cache(maxsize=8192)
def min_magnitude(spec: PlantSpec, comp: str, typ: str) -> float:
    """Smallest in-range magnitude reaching MIN_IMPACT (bisection; KPI impact is monotone in magnitude)."""
    lo, hi = MAGNITUDE_RANGES[typ]
    if _impact_at(spec, comp, typ, lo) >= MIN_IMPACT:
        return lo
    for _ in range(40):
        if hi - lo < 1e-6:
            break
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if _impact_at(spec, comp, typ, mid) >= MIN_IMPACT else (mid, hi)
    return hi


def _draw_physical(spec: PlantSpec, rng: np.random.Generator, used: set[str]) -> Fault:
    avail = [p for p in eligible_pairs(spec) if p[0] not in used]
    if not avail:
        raise ValueError("not enough eligible (component, type) pairs")
    comp, typ = avail[int(rng.integers(len(avail)))]
    return Fault(comp, typ, rnd(rng.uniform(min_magnitude(spec, comp, typ), MAGNITUDE_RANGES[typ][1])))


def sample_faults(spec: PlantSpec, tier: str, rng: np.random.Generator) -> tuple[Fault, ...]:
    n_phys, n_sens = TIER_COUNTS[tier]
    used: set[str] = set()
    physical = []
    for _ in range(n_phys):
        physical.append(_draw_physical(spec, rng, used))
        used.add(physical[-1].component)
    pairs = [(s.id, t) for s in spec.sensors for t in SENSOR_TYPES]
    picked = []
    for _ in range(n_sens):
        avail = [p for p in pairs if p[0] not in used]
        sid, typ = avail[int(rng.integers(len(avail)))]
        used.add(sid)
        if typ == "sensor_bias":
            lo, hi = MAGNITUDE_RANGES[typ]
            picked.append(Fault(sid, typ, rnd(rng.uniform(lo, hi)), sign=1 if rng.random() < 0.5 else -1))
        else:
            picked.append(Fault(sid, typ, None))
    if any(f.type == "sensor_stuck" for f in picked):
        nominal = physical_readings(spec, solve(spec, apply_faults(spec, tuple(physical))))
        picked = [Fault(f.component, f.type, None, value=rnd(nominal[f.component])) if f.type == "sensor_stuck" else f
                  for f in picked]
    return tuple(physical) + tuple(picked)


def sample_drift(spec: PlantSpec, faults: tuple[Fault, ...], rng: np.random.Generator) -> Drift:
    grow = rng.random() < 0.5
    used = {f.component for f in faults}
    if not grow and any(p[0] not in used for p in eligible_pairs(spec)):
        f = _draw_physical(spec, rng, used)
        return Drift("new", fault=Fault(f.component, f.type, f.magnitude, onset="drift"))
    idx = [i for i, f in enumerate(faults) if f.type in PHYSICAL_TYPES]
    return Drift("grow", index=idx[int(rng.integers(len(idx)))], factor=rnd(1.0 + rng.uniform(0.4, 0.8)))


def apply_drift(spec: PlantSpec, faults: tuple[Fault, ...], drift: Drift) -> tuple[Fault, ...]:
    if drift.kind == "new":
        return faults + (drift.fault,)
    f = faults[drift.index]
    grown = Fault(f.component, f.type, rnd(min(f.magnitude * drift.factor, PHYSICAL_MAX[f.type])), f.sign, f.value,
                  onset="drift")
    return faults[: drift.index] + (grown,) + faults[drift.index + 1:]


def drift_fires(spent_before: float, cost: float, budget: float = 40) -> bool:
    """The costed call that crosses DRIFT_FRACTION * budget fires the drift (exactly once per episode)."""
    return spent_before < DRIFT_FRACTION * budget <= spent_before + cost
