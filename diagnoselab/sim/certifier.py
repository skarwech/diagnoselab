"""Instance certifier (F5): accept only instances identifiable within the budget.

docs/sim_spec.md section 10. Columns are whitened finite differences d(reading)/d(magnitude)/sigma
over a fixed probe menu. CRB rule: greedy allocation of B (repeats allowed) on the true support.
Coherence rule: full menu once; a true-fault column whose |cosine| with another candidate exceeds 0.95
is accepted only if it is still >= 5 sigma from the best-fitting competitor at its true magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diagnoselab.sim.experiments import BUDGET, COSTS, experiment_sensors, experiment_setpoint
from diagnoselab.sim.faults import PHYSICAL_TYPES, apply_faults, physical_pairs
from diagnoselab.sim.model import CertResult, Fault, PlantSpec, Setpoint
from diagnoselab.sim.sensors import physical_readings
from diagnoselab.sim.solver import solve

CRB_MAX = 0.10
COHERENCE_MAX = 0.95
SEPARATION_MIN = 5.0  # option E (user decision): a cos > 0.95 pair passes if still >= 5 sigma apart
STUCK_SIGMA = 5.0
SPEEDS = (0.7, 0.85, 1.1)
THETAS = (0.3, 0.6)
FD_STEP = {"fouling": 1e-4, "stuck_valve": 1e-4, "pump_wear": 1e-4, "leak": 1e-5}


@dataclass(frozen=True)
class Probe:
    label: str
    point: str  # operating-point key
    cost: int
    sensors: tuple[str, ...]


def probe_menu(spec: PlantSpec) -> tuple[dict[str, Setpoint], tuple[Probe, ...]]:
    points: dict[str, Setpoint] = {"nominal": Setpoint()}
    probes = [Probe(f"read:{s.id}", "nominal", COSTS["read"], (s.id,)) for s in spec.sensors]
    exps: list[tuple[str, dict]] = [("pump_off_static", {})]
    exps += [("speed_step", {"s": s, "pump": ln.id}) for ln in spec.links if ln.kind == "pump" for s in SPEEDS]
    exps += [("valve_step", {"valve": ln.id, "theta": t}) for ln in spec.links if ln.kind == "valve" for t in THETAS]
    exps += [("isolate_branch", {"branch": b.id}) for b in spec.branches]
    for kind, args in exps:
        key = ":".join([kind, *map(str, args.values())])
        points[key] = experiment_setpoint(spec, kind, args)
        probes.append(Probe(key, key, COSTS[kind], experiment_sensors(spec, kind, args)))
    return points, tuple(probes)


class _Design:
    """Readings of every probe row under a given physical fault set, warm-started from the base plant."""

    def __init__(self, spec: PlantSpec, base_faults: tuple[Fault, ...]):
        self.spec = spec
        self.points, self.probes = probe_menu(spec)
        self.rows = [(p.point, sid) for p in self.probes for sid in p.sensors]
        self.row_probe = np.array([i for i, p in enumerate(self.probes) for _ in p.sensors])
        sig = {s.id: s.sigma for s in spec.sensors}
        self.sigma = np.array([sig[sid] for _, sid in self.rows])
        self.base_states = {k: solve(spec, apply_faults(spec, base_faults), sp) for k, sp in self.points.items()}
        self.base = self._gather({k: physical_readings(spec, st) for k, st in self.base_states.items()})

    def _gather(self, readings: dict[str, dict[str, float]]) -> np.ndarray:
        return np.array([readings[pt][sid] for pt, sid in self.rows])

    def readings(self, faults: tuple[Fault, ...]) -> np.ndarray:
        params = apply_faults(self.spec, faults)
        return self._gather({k: physical_readings(self.spec, solve(self.spec, params, sp, h0=self.base_states[k].h))
                             for k, sp in self.points.items()})


def _columns(spec: PlantSpec, faults: tuple[Fault, ...], design: _Design):
    """Whitened candidate columns; returns (keys, matrix rows x candidates, true-fault keys, magnitudes)."""
    phys = tuple(f for f in faults if f.type in PHYSICAL_TYPES)
    truth = {f.component: f for f in faults}
    stuck = {f.component for f in faults if f.type == "sensor_stuck"}
    live = np.array([sid not in stuck for _, sid in design.rows], dtype=float)  # stuck rows carry no information
    keys, cols = [], []
    for comp, typ in physical_pairs(spec):
        rest = tuple(f for f in phys if f.component != comp)
        m0 = truth[comp].magnitude if comp in truth else 0.0
        y0 = design.base if comp in truth or typ != "stuck_valve" else design.readings(rest + (Fault(comp, typ, 0.0),))
        y1 = design.readings(rest + (Fault(comp, typ, m0 + FD_STEP[typ]),))
        keys.append(f"{comp}:{typ}")
        cols.append((y1 - y0) / FD_STEP[typ] / design.sigma * live)
    for s in spec.sensors:
        if s.id in stuck:
            continue
        keys.append(f"{s.id}:sensor_bias")
        sign = truth[s.id].sign if s.id in truth else 1
        cols.append(np.array([sign * (sid == s.id) for _, sid in design.rows], dtype=float))
    true_keys = [f"{f.component}:{f.type}" for f in faults if f.type != "sensor_stuck"]
    mags = np.array([f.magnitude for f in faults if f.type != "sensor_stuck"], dtype=float)
    return keys, np.column_stack(cols), true_keys, mags


def _greedy_crb(design: _Design, sub: np.ndarray, mags: np.ndarray, budget: int) -> tuple[dict[str, int], np.ndarray]:
    """Spend the budget greedily (repeats allowed) to minimise sum_i CRB_ii / m_i^2 on the true support."""
    info_p = [sub[design.row_probe == i].T @ sub[design.row_probe == i] for i in range(len(design.probes))]
    info = np.diag(1.0 / (10.0 * mags) ** 2)  # weak prior: std = 10x magnitude
    spend: dict[str, int] = {}
    left = budget

    def crb(m: np.ndarray) -> np.ndarray:
        return np.sqrt(np.diag(np.linalg.inv(m))) / mags

    while left > 0 and np.max(crb(info)) > CRB_MAX:
        obj = np.sum(crb(info) ** 2)
        best, best_gain = None, 0.0
        for i, p in enumerate(design.probes):
            if p.cost <= left:
                gain = (obj - np.sum(crb(info + info_p[i]) ** 2)) / p.cost
                if gain > best_gain:
                    best, best_gain = i, gain
        if best is None:
            break
        info = info + info_p[best]
        left -= design.probes[best].cost
        spend[design.probes[best].label] = spend.get(design.probes[best].label, 0) + 1
    return spend, crb(info)


def _stuck_sensor_reasons(spec, faults, design) -> list[str]:
    reasons = []
    for f in faults:
        if f.type == "sensor_stuck":
            rows = np.array([sid == f.component for _, sid in design.rows])
            sep = np.max(np.abs(design.base[rows] - f.value) / design.sigma[rows])
            if sep < STUCK_SIGMA:
                reasons.append(f"stuck sensor {f.component} never moves {STUCK_SIGMA} sigma ({sep:.2f})")
    return reasons


def _stuck_valve_reasons(spec, faults, design) -> list[str]:
    reasons = []
    stuck_sensors = {f.component for f in faults if f.type == "sensor_stuck"}
    phys = tuple(f for f in faults if f.type in PHYSICAL_TYPES)
    for f in phys:
        if f.type != "stuck_valve":
            continue
        keys = {f"valve_step:{f.component}:{t}" for t in THETAS}  # stuck at 0.3 looks healthy under a 0.3 command
        healthy = design.readings(tuple(g for g in phys if g is not f))
        rows = np.array([pt in keys and sid not in stuck_sensors for pt, sid in design.rows])
        sep = np.max(np.abs(design.base[rows] - healthy[rows]) / design.sigma[rows], initial=0.0)
        if sep < STUCK_SIGMA:
            reasons.append(f"stuck valve {f.component} indistinguishable from healthy under valve_step ({sep:.2f})")
    return reasons


def _max_cos(target: np.ndarray, cols: np.ndarray, skip: int | None) -> float:
    norms = np.linalg.norm(cols, axis=0)
    t = np.linalg.norm(target)
    if t == 0:
        return 1.0
    cos = np.abs(cols.T @ target) / (np.where(norms > 0, norms, 1.0) * t)
    cos[norms == 0] = 0.0
    if skip is not None:
        cos[skip] = 0.0
    return float(np.max(cos))


def _stuck_valve_coherence(spec: PlantSpec, faults: tuple[Fault, ...], f: Fault, design: _Design) -> float:
    """A stuck valve is a discrete change (commands ignored), so a column taken in the stuck world
    hides its signature. Compare its secant signature against alternatives in the world without it."""
    rest = tuple(g for g in faults if g is not f)
    world = _Design(spec, tuple(g for g in rest if g.type in PHYSICAL_TYPES))
    stuck = {g.component for g in faults if g.type == "sensor_stuck"}
    live = np.array([sid not in stuck for _, sid in design.rows], dtype=float)
    signature = (design.base - world.base) / f.magnitude / design.sigma * live
    keys, cols, _, _ = _columns(spec, rest, world)
    own = f"{f.component}:stuck_valve"
    return _max_cos(signature, cols, keys.index(own) if own in keys else None), float(np.linalg.norm(signature))


def certify(spec: PlantSpec, faults: tuple[Fault, ...], budget: int = BUDGET) -> CertResult:
    design = _Design(spec, tuple(f for f in faults if f.type in PHYSICAL_TYPES))
    keys, cols, true_keys, mags = _columns(spec, faults, design)
    idx = [keys.index(k) for k in true_keys]
    spend, crb_rel = _greedy_crb(design, cols[:, idx], mags, budget)

    by_key = {f"{f.component}:{f.type}": f for f in faults}
    coherence, separation = {}, {}
    for k, i, m in zip(true_keys, idx, mags):
        if k.endswith(":stuck_valve"):
            cos, norm = _stuck_valve_coherence(spec, faults, by_key[k], design)
        else:
            cos, norm = _max_cos(cols[:, i], cols, i), float(np.linalg.norm(cols[:, i]))
        coherence[k] = cos
        # whitened distance from the best-fitting competitor at the true magnitude (sigma units)
        separation[k] = float(m * norm * np.sqrt(max(0.0, 1.0 - cos * cos)))

    reasons = [f"CRB {k} = {v:.3f} > {CRB_MAX}" for k, v in zip(true_keys, crb_rel) if v > CRB_MAX]
    reasons += [f"coherence {k} = {coherence[k]:.3f} > {COHERENCE_MAX} and separation {separation[k]:.1f} sigma "
                f"< {SEPARATION_MIN}" for k in true_keys
                if coherence[k] > COHERENCE_MAX and separation[k] < SEPARATION_MIN]
    reasons += _stuck_sensor_reasons(spec, faults, design) + _stuck_valve_reasons(spec, faults, design)
    return CertResult(accepted=not reasons, crb_rel=dict(zip(true_keys, map(float, crb_rel))),
                      coherence=coherence, reasons=tuple(reasons), spend=spend, separation=separation)
