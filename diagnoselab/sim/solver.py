"""Steady-state hydraulic solver (F2): damped Newton on junction continuity.

Laws, smoothing constants and outputs follow docs/sim_spec.md section 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np

from diagnoselab.sim.model import Params, PlantSpec, Setpoint

EPS = 1e-4  # m, pipe/valve smoothing
DELTA = 1e-6  # smooth max(x, 0) for pumps and leaks
TOL = 1e-10  # L/s, max junction residual
MAX_ITER = 50
MAX_STEP = 20.0  # m, per-iteration head change cap
HEALTHY = Params()
NOMINAL = Setpoint()


class SolverError(RuntimeError):
    pass


@dataclass(frozen=True, eq=False)
class Net:
    jid: tuple[str, ...]
    jindex: Mapping[str, int]
    z: np.ndarray
    fixed: Mapping[str, float]
    lid: tuple[str, ...]
    kind: tuple[str, ...]
    frm_j: np.ndarray  # junction index or -1
    to_j: np.ndarray
    frm_h: np.ndarray  # fixed head or nan
    to_h: np.ndarray
    r: np.ndarray
    a: np.ndarray
    b: np.ndarray
    is_pump: np.ndarray
    is_valve: np.ndarray
    h0: np.ndarray  # initial guess
    h_static: np.ndarray  # exact leak-free solution with every pump stopped
    q0: np.ndarray  # initial flows (design, or 1.0)
    inc: np.ndarray  # links x junctions: +1 at frm, -1 at to  (D = inc @ h + dfix)
    dfix: np.ndarray  # fixed-head contribution to D


@dataclass(frozen=True)
class State:
    h: Mapping[str, float]
    q: Mapping[str, float]
    leak: Mapping[str, float]
    power: Mapping[str, float]
    vib: Mapping[str, float]
    iterations: int
    residual: float


@lru_cache(maxsize=512)
def compile_net(spec: PlantSpec) -> Net:
    jid = tuple(j.id for j in spec.junctions)
    jindex = {j: i for i, j in enumerate(jid)}
    fixed = {s.id: s.head for s in spec.sources} | {o.id: spec.tank_head for o in spec.outlets}

    def end(node: str) -> tuple[int, float]:
        if node in jindex:
            return jindex[node], np.nan
        if node in fixed:
            return -1, fixed[node]
        raise ValueError(f"link endpoint {node!r} is neither a junction nor a fixed-head node")

    ends = [(end(ln.frm), end(ln.to)) for ln in spec.links]
    design_h = spec.design.get("h", {}) if spec.design else {}
    design_q = spec.design.get("q", {}) if spec.design else {}
    h0 = np.array([design_h.get(j, np.mean(list(fixed.values()))) for j in jid], dtype=float)
    inc = np.zeros((len(spec.links), len(jid)))
    for k, ((fi, _), (ti, _)) in enumerate(ends):
        if fi >= 0:
            inc[k, fi] += 1.0
        if ti >= 0:
            inc[k, ti] -= 1.0
    dfix = np.array([(fh if fi < 0 else 0.0) - (th if ti < 0 else 0.0) for (fi, fh), (ti, th) in ends])
    feed = {ln.to: fixed[ln.frm] for ln in spec.links if ln.frm in fixed and ln.to in jindex}
    h_static = np.array([feed.get(j.id, spec.tank_head) if j.side == "suction" else spec.tank_head
                         for j in spec.junctions], dtype=float)
    return Net(
        h_static=h_static,
        q0=np.array([design_q.get(ln.id, 1.0) for ln in spec.links], dtype=float), inc=inc, dfix=dfix,
        jid=jid, jindex=jindex, z=np.array([j.z for j in spec.junctions], dtype=float), fixed=fixed,
        lid=tuple(ln.id for ln in spec.links), kind=tuple(ln.kind for ln in spec.links),
        frm_j=np.array([e[0][0] for e in ends], dtype=int), to_j=np.array([e[1][0] for e in ends], dtype=int),
        frm_h=np.array([e[0][1] for e in ends], dtype=float), to_h=np.array([e[1][1] for e in ends], dtype=float),
        r=np.array([ln.r for ln in spec.links], dtype=float), a=np.array([ln.a for ln in spec.links], dtype=float),
        b=np.array([ln.b for ln in spec.links], dtype=float),
        is_pump=np.array([ln.kind == "pump" for ln in spec.links]),
        is_valve=np.array([ln.kind == "valve" for ln in spec.links]), h0=h0,
    )


def _smax(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable smooth max(x, 0) = (x + sqrt(x^2 + DELTA^2)) / 2; returns (value, root)."""
    root = np.sqrt(x * x + DELTA * DELTA)
    neg = DELTA * DELTA / (2.0 * (root - np.minimum(x, 0.0)) + 1e-300)
    return np.where(x >= 0, 0.5 * (x + root), neg), root


def leak_coefficients(spec: PlantSpec, net: Net, leak: Mapping[str, float]) -> np.ndarray:
    c = np.zeros(len(net.jid))
    if not leak:
        return c
    q_design = spec.design["Q_pump"]
    for j, f in leak.items():
        i = net.jindex[j]
        c[i] = f * q_design / np.sqrt(spec.design["h"][j] - net.z[i])
    return c


def _link_arrays(spec: PlantSpec, net: Net, params: Params, sp: Setpoint) -> tuple[np.ndarray, ...]:
    r_eff = net.r.copy()
    a_eff = np.zeros_like(net.r)
    active = np.ones(len(net.lid), dtype=bool)
    for k, link in enumerate(net.lid):
        if link in sp.closed:
            active[k] = False
        if net.kind[k] == "pipe":
            r_eff[k] = net.r[k] * (1.0 + params.fouling.get(link, 0.0))
        elif net.kind[k] == "valve":
            theta = params.stuck.get(link, sp.valve.get(link, 1.0))
            r_eff[k] = net.r[k] / theta**2
        else:
            s = sp.speed.get(link, 1.0)
            a_eff[k] = net.a[k] * (1.0 - params.wear.get(link, 0.0)) * s * s
    return r_eff, a_eff, active


def _flows(net: Net, hj: np.ndarray, r_eff: np.ndarray, a_eff: np.ndarray, active: np.ndarray):
    hf = np.where(net.frm_j >= 0, hj[np.maximum(net.frm_j, 0)], net.frm_h)
    ht = np.where(net.to_j >= 0, hj[np.maximum(net.to_j, 0)], net.to_h)
    d = hf - ht
    s = np.sqrt(d * d + EPS * EPS)
    rs = np.where(net.is_pump, 1.0, r_eff) * s
    q_pipe = d / np.sqrt(rs)
    g_pipe = (1.0 - d * d / (2.0 * s * s)) / np.sqrt(rs)
    spx, root = _smax(a_eff + d)  # x = a_eff - (h_to - h_frm)
    b = np.where(net.is_pump, net.b, 1.0)
    q_pump = np.sqrt(spx / b)
    g_pump = q_pump / (2.0 * root)
    q = np.where(net.is_pump, q_pump, q_pipe) * active
    g = np.where(net.is_pump, g_pump, g_pipe) * active
    return q, g, hf, ht


def _residual(net: Net, hj, r_eff, a_eff, active, c):
    q, g, _, _ = _flows(net, hj, r_eff, a_eff, active)
    n = len(net.jid)
    into = net.to_j >= 0
    out = net.frm_j >= 0
    spp, root = _smax(hj - net.z)
    ql = c * np.sqrt(spp)
    f = (np.bincount(net.to_j[into], q[into], n) - np.bincount(net.frm_j[out], q[out], n) - ql)
    return f, q, g, ql, root


def _inverse(net: Net, q: np.ndarray, r_eff: np.ndarray, a_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form inverse link laws D = R(q) and dR/dq (pumps need q > 0)."""
    rr = np.where(net.is_pump, 1.0, r_eff)
    k = (rr * q * q) ** 2
    d_pipe = np.sign(q) * np.sqrt(0.5 * (k + np.sqrt(k * k + 4.0 * k * EPS * EPS)))
    s = np.sqrt(d_pipe * d_pipe + EPS * EPS)
    g_pipe = (1.0 - d_pipe * d_pipe / (2.0 * s * s)) / np.sqrt(rr * s)
    qp = np.where(net.is_pump, np.maximum(q, 1e-300), 1.0)
    y = np.where(net.is_pump, net.b, 1.0) * qp * qp
    x = y - DELTA * DELTA / (4.0 * y)
    g_pump = qp / (2.0 * np.sqrt(x * x + DELTA * DELTA))
    d = np.where(net.is_pump, x - a_eff, d_pipe)
    return d, 1.0 / np.where(net.is_pump, g_pump, g_pipe)


def _gga(net: Net, hj, q, r_eff, a_eff, active, c, max_iter: int = 40) -> tuple[np.ndarray, int]:
    """Global phase: Newton on (flows, heads) with the inverse laws (Todini-Pilati GGA)."""
    act = np.nonzero(active)[0]
    inc = net.inc[act]
    pump = net.is_pump[act]
    q = q[act].copy()

    def residuals(h, qa):
        d_q, rd = _inverse(net, _scatter(qa), r_eff, a_eff)
        r1 = (inc @ h + net.dfix[act]) - d_q[act]
        spp, root = _smax(h - net.z)
        ql = c * np.sqrt(spp)
        r2 = -inc.T @ qa - ql
        return r1, r2, rd[act], ql / (2.0 * root)

    def _scatter(qa):
        full = np.zeros(len(net.lid))
        full[act] = qa
        return full

    r1, r2, rd, ld = residuals(hj, q)
    for it in range(max_iter):
        if np.max(np.abs(r2), initial=0.0) < 1e-7 and np.max(np.abs(r1), initial=0.0) < 1e-7:
            return hj, it
        m = inc.T @ (inc / rd[:, None]) + np.diag(ld)
        dh = np.linalg.solve(m, r2 - inc.T @ (r1 / rd))
        dq = (r1 + inc @ dh) / rd
        shrink = pump & (dq < 0)  # keep pump flows positive (check valve)
        alpha = min(1.0, float(np.min(0.9 * q[shrink] / -dq[shrink], initial=1.0)))
        merit = np.sum(r1 * r1) + np.sum(r2 * r2)
        for _ in range(30):
            h_t, q_t = hj + alpha * dh, q + alpha * dq
            res = residuals(h_t, q_t)
            if np.sum(res[0] ** 2) + np.sum(res[1] ** 2) < merit:
                break
            alpha *= 0.5
        hj, q, (r1, r2, rd, ld) = h_t, q_t, res
    return hj, max_iter


def _content(y: np.ndarray) -> np.ndarray:
    """Antiderivative of sqrt(smax(x)) w.r.t. x, written in y = smax(x) (exact, y > 0)."""
    return (2.0 / 3.0) * y**1.5 - 0.5 * DELTA * DELTA / np.sqrt(y)


def _energy(net: Net, hj, r_eff, a_eff, active, c) -> float:
    """Convex network content E(h) with F(h) = -grad E(h); a line search on E makes Newton global."""
    hf = np.where(net.frm_j >= 0, hj[np.maximum(net.frm_j, 0)], net.frm_h)
    ht = np.where(net.to_j >= 0, hj[np.maximum(net.to_j, 0)], net.to_h)
    d = hf - ht
    phi_pipe = (2.0 / 3.0) * (d * d + EPS * EPS) ** 0.75 / np.sqrt(np.where(net.is_pump, 1.0, r_eff))
    y_pump, _ = _smax(a_eff + d)
    phi_pump = _content(y_pump) / np.sqrt(np.where(net.is_pump, net.b, 1.0))
    phi = np.where(net.is_pump, phi_pump, phi_pipe) * active
    y_leak, _ = _smax(hj - net.z)
    return float(phi.sum() + (c * _content(y_leak)).sum())


def _jacobian(net: Net, g: np.ndarray, ql: np.ndarray, root: np.ndarray) -> np.ndarray:
    n = len(net.jid)
    jac = np.zeros((n, n))
    fj, tj = net.frm_j, net.to_j
    for k in np.nonzero(g)[0]:
        i, o = fj[k], tj[k]
        if o >= 0:
            if i >= 0:
                jac[o, i] += g[k]
            jac[o, o] -= g[k]
        if i >= 0:
            jac[i, i] -= g[k]
            if o >= 0:
                jac[i, o] += g[k]
    jac[np.diag_indices(n)] -= ql / (2.0 * root)
    return jac


def solve(spec: PlantSpec, params: Params | None = None, setpoint: Setpoint | None = None,
          h0: Mapping[str, float] | None = None) -> State:
    params, sp = params or HEALTHY, setpoint or NOMINAL
    net = compile_net(spec)
    r_eff, a_eff, active = _link_arrays(spec, net, params, sp)
    c = leak_coefficients(spec, net, params.leak)
    all_off = bool(net.is_pump.any()) and all(sp.speed.get(k, 1.0) == 0.0 for k, p in zip(net.lid, net.is_pump) if p)
    hj = np.array([h0[j] for j in net.jid]) if h0 else (net.h_static.copy() if all_off else net.h0.copy())
    f, q, g, ql, root = _residual(net, hj, r_eff, a_eff, active, c)
    it = 0
    if np.max(np.abs(f), initial=0.0) > 1e-6:  # far from the solution: global (q, h) phase first
        speed = np.array([sp.speed.get(k, 1.0) if p else 1.0 for k, p in zip(net.lid, net.is_pump)])
        q_init = np.where(net.is_pump, np.maximum(net.q0 * speed, 1e-6), net.q0)
        hj, it = _gga(net, hj, q_init, r_eff, a_eff, active, c)
        f, q, g, ql, root = _residual(net, hj, r_eff, a_eff, active, c)
    energy = _energy(net, hj, r_eff, a_eff, active, c)
    while np.max(np.abs(f), initial=0.0) > TOL:
        if it >= MAX_ITER:
            raise SolverError(f"no convergence after {MAX_ITER} iterations (residual {np.max(np.abs(f)):.3g})")
        try:
            step = np.linalg.solve(_jacobian(net, g, ql, root), -f)
        except np.linalg.LinAlgError as e:
            raise SolverError("singular Jacobian: a junction is disconnected from every fixed head") from e
        step *= min(1.0, MAX_STEP / max(np.max(np.abs(step)), 1e-300))
        slope, alpha = float(-f @ step), 1.0  # directional derivative of E (< 0: descent)
        while True:  # Armijo backtracking on the convex content E
            trial = hj + alpha * step
            e_trial = _energy(net, trial, r_eff, a_eff, active, c)
            if e_trial <= energy + 1e-4 * alpha * slope + 1e-13 * abs(energy) or alpha < 1e-8:
                break
            alpha *= 0.5
        hj, energy = trial, e_trial
        f, q, g, ql, root = _residual(net, hj, r_eff, a_eff, active, c)
        it += 1
    return _state(spec, net, params, sp, hj, q, ql, it, float(np.max(np.abs(f), initial=0.0)))


def _state(spec, net, params, sp, hj, q, ql, it, resid) -> State:
    h = dict(zip(net.jid, hj.tolist())) | dict(net.fixed)
    qd = dict(zip(net.lid, q.tolist()))
    power, vib = {}, {}
    for ln in spec.links:
        if ln.kind != "pump":
            continue
        s, w, qp = sp.speed.get(ln.id, 1.0), params.wear.get(ln.id, 0.0), qd[ln.id]
        if s <= 0.0 or ln.id in sp.closed:
            power[ln.id], vib[ln.id] = 0.0, 0.0
            continue
        u = qp / (s * ln.q_bep)
        eta = float(np.clip(ln.eta_max * (1.0 - 0.5 * w) * (1.0 - (u - 1.0) ** 2), 0.05, ln.eta_max))
        power[ln.id] = 9.81 * qp * (h[ln.to] - h[ln.frm]) / (1000.0 * eta)
        vib[ln.id] = ln.v0 * s * s * (1.0 + 2.0 * w) * (1.0 + 3.0 * (u - 1.0) ** 2)
    return State(h=h, q=qd, leak=dict(zip(net.jid, ql.tolist())), power=power, vib=vib,
                 iterations=it, residual=resid)


def outlet_flows(spec: PlantSpec, state: State) -> dict[str, float]:
    flows = {o.id: 0.0 for o in spec.outlets}
    for ln in spec.links:
        if ln.to in flows:
            flows[ln.to] += state.q[ln.id]
        if ln.frm in flows:
            flows[ln.frm] -= state.q[ln.id]
    return flows


def kpi(spec: PlantSpec, state: State) -> float:
    """J = -(sum unmet fraction + SE / SE_design), docs/sim_spec.md section 3."""
    delivered = outlet_flows(spec, state)
    unmet = sum(max(0.0, o.q_req - delivered[o.id]) / o.q_req for o in spec.outlets)
    # floor the delivered flow so a deadheaded pump or tank backfeed keeps J finite and monotone
    total = max(sum(delivered.values()), 0.1 * spec.design["Q_pump"])
    se = sum(state.power.values()) / (3.6 * total)
    return -(unmet + se / spec.design["SE"])
