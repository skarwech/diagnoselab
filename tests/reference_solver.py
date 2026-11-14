"""Independent reference hydraulic solver for differential testing (plan F2).

Written from docs/sim_spec.md sections 2 and 4 only; it shares no code with
diagnoselab/sim/. The formulation is deliberately different from nodal Newton on
junction heads: an augmented system whose unknowns are the flow of every open
link AND the head of every junction, one equation per link and per junction.

* pipe / valve : (h_frm - h_to) - drop(q) = 0, where drop() is the exact
                 algebraic inverse of the spec law q = D / sqrt(r*sqrt(D^2+EPS^2))
* pump         : q - sqrt(sp(a*(1-w)*s^2 - (h_to - h_frm)) / b) = 0
* junction     : sum(q in) - sum(q out) - C*sqrt(sp(h - z)) = 0

It is solved with scipy.optimize.root(method="hybr"). The answer is accepted only
after every law is re-checked in the spec's own forward form (q = f(D)).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

EPS = 1e-4
DELTA = 1e-6
XTOL = 1e-14
ACCEPT_TOL = 1e-9  # max forward-form residual (L/s) for a solution to be accepted
THETA_SLACK = 1e-12  # roundoff allowed on the valve opening range [0.1, 1]
FAULT_KIND = {"fouling": "pipe", "stuck_valve": "valve", "pump_wear": "pump", "leak": "junction"}
SENSOR_FAULTS = {"sensor_bias", "sensor_stuck"}


def sp(x: np.ndarray) -> np.ndarray:
    """Smooth max(x, 0) = (x + sqrt(x^2 + DELTA^2)) / 2, evaluated without cancellation.

    For x < 0 the literal formula subtracts two nearly equal numbers (0.08% error at
    x = -3, 1.9% at x = -12), so the algebraically identical DELTA^2 / (2*(sqrt - x))
    is used there.
    """
    x = np.asarray(x, dtype=float)
    root_ = np.sqrt(x * x + DELTA * DELTA)
    return np.where(x >= 0, 0.5 * (x + root_), DELTA * DELTA / (2.0 * (root_ - np.minimum(x, 0.0))))


def pipe_flow(d: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Spec section 2 pipe law (forward form): q = D / sqrt(r * sqrt(D^2 + EPS^2))."""
    return d / np.sqrt(r * np.sqrt(d * d + EPS * EPS))


def pipe_drop(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Exact inverse of pipe_flow.

    Squaring the law gives D^4 = r^2 q^4 (D^2 + EPS^2). With k = r q^2 its positive
    root is D^2 = k (k + sqrt(k^2 + 4 EPS^2)) / 2, and D has the sign of q, so
    D = q * sqrt(r * (k + sqrt(k^2 + 4 EPS^2)) / 2).
    """
    k = r * q * q
    return q * np.sqrt(r * (k + np.sqrt(k * k + 4.0 * EPS * EPS)) / 2.0)


def pump_flow(lift: np.ndarray, a_eff: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spec section 2 pump law; lift = h_to - h_frm, a_eff = a*(1-w)*s^2."""
    return np.sqrt(sp(a_eff - lift) / b)


def _fault_effects(spec: dict, faults: list[dict] | tuple) -> dict[str, dict[str, float]]:
    """Map each physical fault type to {component: magnitude} (section 4)."""
    kinds = {l["id"]: l["kind"] for l in spec["links"]} | {j["id"]: "junction" for j in spec["junctions"]}
    effects: dict[str, dict[str, float]] = {t: {} for t in FAULT_KIND}
    seen: set[str] = set()
    for f in faults:
        ftype, comp, mag = f["type"], f["component"], f.get("magnitude")
        if ftype in SENSOR_FAULTS:
            continue
        if ftype not in FAULT_KIND:
            raise ValueError(f"unknown fault type {ftype!r}")
        if kinds.get(comp) != FAULT_KIND[ftype]:
            raise ValueError(f"{ftype} needs a {FAULT_KIND[ftype]}, got {comp!r}")
        if mag is None or mag < 0:
            raise ValueError(f"fault magnitude must be >= 0, got {mag!r} on {comp}")
        if comp in seen:
            raise ValueError(f"more than one physical fault on {comp}")
        seen.add(comp)
        effects[ftype][comp] = float(mag)
    return effects


def _check_setpoint(links: dict[str, dict], speed: dict, valve: dict, closed: frozenset) -> None:
    for name, ids, kind in (("speed", speed, "pump"), ("valve", valve, "valve"), ("closed", closed, None)):
        bad = [i for i in ids if i not in links or (kind and links[i]["kind"] != kind)]
        if bad:
            raise ValueError(f"{name}: unknown or wrong-kind link ids {bad}")
    if any(s < 0 for s in speed.values()):
        raise ValueError(f"negative pump speed in {speed}")


def _leak_coefficients(spec: dict, leaks: dict[str, float]) -> dict[str, float]:
    """C_j = f_j * Q_design / sqrt(h_design_j - z_j) (section 2)."""
    if not leaks:
        return {}
    design = spec.get("design") or {}
    z = {j["id"]: j["z"] for j in spec["junctions"]}
    coeffs = {}
    for j, f in leaks.items():
        if "Q_pump" not in design or j not in design.get("h", {}):
            raise ValueError(f"leak on {j} needs design.Q_pump and design.h[{j!r}]")
        p_design = design["h"][j] - z[j]
        if p_design <= 0:
            raise ValueError(f"leak on {j}: design pressure head {p_design} is not positive")
        coeffs[j] = f * design["Q_pump"] / np.sqrt(p_design)
    return coeffs


class _Network:
    """The open network with every fault and setpoint already folded into its parameters."""

    def __init__(self, spec: dict, effects: dict, speed: dict, valve: dict, closed: frozenset) -> None:
        self.junctions = [j["id"] for j in spec["junctions"]]
        self.z = np.array([j["z"] for j in spec["junctions"]], dtype=float)
        fixed = {s["id"]: float(s["head"]) for s in spec["sources"]}
        fixed |= {o["id"]: float(spec["tank_head"]) for o in spec["outlets"]}
        self.fixed_ids, self.fixed_h = list(fixed), np.array(list(fixed.values()))
        node = {n: i for i, n in enumerate(self.junctions + self.fixed_ids)}
        self.all_links = [l["id"] for l in spec["links"]]
        open_links = [l for l in spec["links"] if l["id"] not in closed]
        self.link_ids = [l["id"] for l in open_links]
        self.frm = np.array([node[l["frm"]] for l in open_links], dtype=int)
        self.to = np.array([node[l["to"]] for l in open_links], dtype=int)
        self.is_pump = np.array([l["kind"] == "pump" for l in open_links], dtype=bool)
        self.r = np.array([self._r_eff(l, effects, valve) for l in open_links])
        self.a_eff = np.array([self._a_eff(l, effects, speed) for l in open_links])
        self.b = np.array([l["b"] if l["kind"] == "pump" else 1.0 for l in open_links], dtype=float)
        leak_c = _leak_coefficients(spec, effects["leak"])
        self.c = np.array([leak_c.get(j, 0.0) for j in self.junctions])
        nj, nl = len(self.junctions), len(open_links)
        self.inc = np.zeros((nj, nl))  # +1 where the link enters the junction, -1 where it leaves
        for k in range(nl):
            if self.to[k] < nj:
                self.inc[self.to[k], k] += 1.0
            if self.frm[k] < nj:
                self.inc[self.frm[k], k] -= 1.0
        self._check_connected()

    @staticmethod
    def _r_eff(link: dict, effects: dict, valve: dict) -> float:
        if link["kind"] == "pipe":
            return link["r"] * (1.0 + effects["fouling"].get(link["id"], 0.0))
        if link["kind"] == "valve":
            stuck = effects["stuck_valve"].get(link["id"])
            theta = 1.0 - stuck if stuck is not None else valve.get(link["id"], 1.0)
            # Roundoff slack: the stuck-valve physical max 0.9 gives 1 - 0.9 = 0.0999...98.
            if not 0.1 - THETA_SLACK <= theta <= 1.0 + THETA_SLACK:
                raise ValueError(f"valve {link['id']}: opening {theta} outside [0.1, 1]")
            return link["r"] / theta**2
        if link["kind"] == "pump":
            return 1.0  # unused placeholder
        raise ValueError(f"unknown link kind {link['kind']!r}")

    @staticmethod
    def _a_eff(link: dict, effects: dict, speed: dict) -> float:
        if link["kind"] != "pump":
            return 0.0
        s = speed.get(link["id"], 1.0)
        return link["a"] * (1.0 - effects["pump_wear"].get(link["id"], 0.0)) * s * s

    def _check_connected(self) -> None:
        n = len(self.junctions) + len(self.fixed_ids)
        graph = coo_matrix((np.ones(len(self.frm)), (self.frm, self.to)), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        anchored = set(label[len(self.junctions):])
        loose = [j for i, j in enumerate(self.junctions) if label[i] not in anchored]
        if loose:
            raise ValueError(f"junctions not connected to any fixed-head node: {loose}")

    def _drops(self, h: np.ndarray) -> np.ndarray:
        heads = np.concatenate([h, self.fixed_h])
        return heads[self.frm] - heads[self.to]

    def link_flows(self, h: np.ndarray) -> np.ndarray:
        """Forward-form flows of every open link for given junction heads."""
        d = self._drops(h)
        return np.where(self.is_pump, pump_flow(-d, self.a_eff, self.b), pipe_flow(d, self.r))

    def leak(self, h: np.ndarray) -> np.ndarray:
        return self.c * np.sqrt(sp(h - self.z))

    def residual(self, x: np.ndarray) -> np.ndarray:
        nl = len(self.link_ids)
        q, h = x[:nl], x[nl:]
        d = self._drops(h)
        link_res = np.where(self.is_pump, q - pump_flow(-d, self.a_eff, self.b), d - pipe_drop(q, self.r))
        return np.concatenate([link_res, self.inc @ q - self.leak(h)])

    def forward_residual(self, x: np.ndarray) -> float:
        """Max |residual| of the spec's laws as written (L/s), the acceptance test."""
        nl = len(self.link_ids)
        q, h = x[:nl], x[nl:]
        res = np.concatenate([q - self.link_flows(h), self.inc @ q - self.leak(h)])
        return float(np.max(np.abs(res), initial=0.0))

    def linear_heads(self) -> np.ndarray:
        """Robust head guess: a linear network with q = g*(h_frm - h_to + e)."""
        nj = len(self.junctions)
        g = np.where(self.is_pump, np.where(self.a_eff > 0, 1.0, 1e-6) / np.sqrt(self.b), 1.0 / np.sqrt(self.r))
        e = np.where(self.is_pump, self.a_eff, 0.0)
        lap = (self.inc * g) @ self.inc.T
        fixed_term = np.zeros(len(self.link_ids))  # h_frm - h_to contribution of fixed nodes
        for k in range(len(self.link_ids)):
            if self.frm[k] >= nj:
                fixed_term[k] += self.fixed_h[self.frm[k] - nj]
            if self.to[k] >= nj:
                fixed_term[k] -= self.fixed_h[self.to[k] - nj]
        return np.linalg.solve(lap, self.inc @ (g * (fixed_term + e))) if nj else np.zeros(0)

    def initial_guesses(self, spec: dict) -> list[np.ndarray]:
        design_h = (spec.get("design") or {}).get("h", {})
        heads = [self.linear_heads()]
        if self.junctions and all(j in design_h for j in self.junctions):
            heads.insert(0, np.array([design_h[j] for j in self.junctions], dtype=float))
        return [np.concatenate([self.link_flows(h), h]) for h in heads]

    def solve(self, guesses: list[np.ndarray]) -> np.ndarray:
        best = np.inf
        for x0 in guesses:
            # hybr may report "not making good progress" at a converged point when
            # xtol is this tight, so the forward residual decides, not sol.success.
            sol = root(self.residual, x0, method="hybr", options={"xtol": XTOL})
            err = self.forward_residual(sol.x)
            if err < ACCEPT_TOL:
                return sol.x
            best = min(best, err)
        raise RuntimeError(f"reference solver did not converge (best forward residual {best:.3e} L/s)")

    def result(self, x: np.ndarray) -> dict:
        nl = len(self.link_ids)
        q, h = x[:nl], x[nl:]
        flows = dict.fromkeys(self.all_links, 0.0) | {k: float(v) for k, v in zip(self.link_ids, q)}
        heads = {j: float(v) for j, v in zip(self.junctions, h)}
        heads |= {n: float(v) for n, v in zip(self.fixed_ids, self.fixed_h)}
        leaks = {j: float(v) for j, v in zip(self.junctions, self.leak(h))}
        return {"h": heads, "q": flows, "leak": leaks}


def ref_solve(
    spec: dict,
    faults: list[dict] | tuple = (),
    speed: dict[str, float] | None = None,
    valve: dict[str, float] | None = None,
    closed: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """Solve the steady state of `spec` (sim_spec.md section 8 dict) under faults and a setpoint.

    Returns {"h": heads of junctions and fixed nodes, "q": flow of every link (0 if
    closed), "leak": leak flow of every junction}.
    """
    speed, valve, closed = dict(speed or {}), dict(valve or {}), frozenset(closed)
    _check_setpoint({l["id"]: l for l in spec["links"]}, speed, valve, closed)
    net = _Network(spec, _fault_effects(spec, faults), speed, valve, closed)
    return net.result(net.solve(net.initial_guesses(spec)))
