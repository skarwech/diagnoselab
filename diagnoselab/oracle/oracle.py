"""DiagnoseLab reference oracle (docs/oracle_spec.md).

Self-contained on purpose: Python standard library only, no diagnoselab import, so this one file runs in-process
for evaluation and unchanged inside the agent container as Harbor's solve.sh. It sees exactly what an agent sees:
tool responses, the manual's constants, and simulate (the free digital twin, within its 300-call cap).

run_oracle(call) plays one episode. call(tool, args) returns the tool's result dict and raises on a rejected call.
"""

from __future__ import annotations

import math
from typing import Callable

Call = Callable[[str, dict], dict]
Param = tuple  # (component, fault type) for a physical fault; ("bias", sensor id) for a sensor bias
Key = tuple  # operating point: ("nominal",), ("speed", s, pump or ""), ("valve", v, theta), ("isolate", b), ("static",)

# --- manual constants (topics faults, tools, submission) ---------------------------------------------------------
LINK_TYPE = {"pipe": "fouling", "valve": "stuck_valve", "pump": "pump_wear"}
TYPICAL = {"fouling": (0.3, 2.0), "stuck_valve": (0.2, 0.8), "pump_wear": (0.10, 0.35), "leak": (0.02, 0.08),
           "sensor_bias": (4.0, 10.0)}
MID = {t: (lo + hi) / 2 for t, (lo, hi) in TYPICAL.items()}
SIM_MAX = {"fouling": 10.0, "stuck_valve": 0.9, "pump_wear": 0.9, "leak": 0.3}
REPAIR = {"fouling": "clean_pipe", "stuck_valve": "free_valve", "pump_wear": "replace_pump", "leak": "fix_leak"}
BUDGET = 40
NOMINAL: Key = ("nominal",)
STATIC: Key = ("static",)

# --- oracle settings (docs/oracle_spec.md section 2) -------------------------------------------------------------
BROAD_SPEED = 1.2  # the all-pump speed step screened for every candidate
LOCAL_SPEED = 0.75  # pump-local speed steps: all pumps on one-pump plants; per pump at 0.75 and 1.2 on two
SCREENED: tuple = (NOMINAL, ("speed", BROAD_SPEED, ""), STATIC)  # operating points screened for every candidate
THETA = 0.3  # valve_step command
SPEND_CAP = 28  # 70% of B
JOIN, KEEP = 25.0, 9.0  # squared whitened residual drop to add a fault (5 sigma) and to keep it (3 sigma)
SWAP_MARGIN = 25.0  # an alternative explanation closer than this (squared sigma) leaves an element unresolved
REL_SD_MAX = 0.15  # an element whose relative sd exceeds this is unresolved (the credit band is 30%)
PRIOR_INCLUSION = 0.1  # design prior variance = PRIOR_INCLUSION * MID^2 per candidate
COVER_READS = 2  # every live sensor is read twice: a stuck sensor repeats a value exactly (manual fact (g))
COVER_BONUS = 0.0  # design bonus (nats) per needed read a probe supplies
GAIN_MIN = 0.05  # stop when the best probe gains less than this per budget unit
SIM_RESERVE = 10  # simulate calls kept for the final fit, plus (support + 2) per operating point in use
FD_REL, ANCHOR_DRIFT = 0.1, 0.15  # finite-difference step and relinearisation threshold (relative)
INPLAY_MAX, LOCAL_MAX, LOCAL_PARAMS, MAX_SUPPORT, TRY_MAX = 6, 3, 4, 8, 2
LM_TOL = 0.5  # stop Levenberg-Marquardt when the linear model promises less than this (squared sigma)
LM_REJECT_MAX = 9  # rejected Levenberg-Marquardt steps tolerated per pass (each costs one simulate per point)
CONFIRM_FIRST = False  # tried and rejected (oracle_spec tuning log): confirming each fitted bias with a read at once
FIT_DURING_COVERAGE = False  # fit only once every live sensor has been read COVER_READS times
BIAS_MIN_READS = 1  # readings before a bias may be fitted; 2 was tried and rejected (coverage-first fitting covers it)
POLISH = True
PAIRS, PAIR_POOL, PAIR_GAIN, PAIR_Z = True, 6, 9.0, 1.5
EXPLORE_PAIRS = False  # one-for-two replacement search in the final fit
POLISH_MARGIN, POLISH_RIVALS = 50.0, 2  # final-fit swap check: linear margin below which rivals are refit
FIT_SIM_MAX = 35  # simulate calls one exploration fit may spend on optional additions, pruning and swaps
ROUND = 9  # decision scores are rounded to 1e-9 before any comparison, so ties break by a fixed order


class SimExhausted(Exception):
    """The simulate cap is near: stop exploring and submit the current estimate."""


# --- small dense linear algebra (matrices are at most ~60 x 60) --------------------------------------------------
def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cholesky(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix not positive definite")
                low[i][i] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return low


def chol_solve(low: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(low[i][k] * y[k] for k in range(i))) / low[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
    return x


def spd_factor(a: list[list[float]]) -> list[list[float]]:
    """Cholesky with a growing ridge when the matrix is singular to working precision."""
    ridge = 0.0
    for _ in range(12):
        try:
            return cholesky([[v + (ridge if i == j else 0.0) for j, v in enumerate(row)] for i, row in enumerate(a)])
        except ValueError:
            ridge = max(ridge * 10.0, 1e-10 * max(1.0, max(abs(a[i][i]) for i in range(len(a)))))
    raise ValueError("matrix cannot be factored")


def inverse(a: list[list[float]]) -> list[list[float]]:
    low = spd_factor(a)
    n = len(a)
    cols = [chol_solve(low, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def logdet(a: list[list[float]]) -> float:
    return 2.0 * sum(math.log(v[i]) for i, v in enumerate(spd_factor(a)))


def gram(cols: list[list[float]]) -> list[list[float]]:
    return [[dot(ci, cj) for cj in cols] for ci in cols]


def orthonormal(cols: list[list[float]]) -> list[list[float]]:
    basis: list[list[float]] = []
    for c in cols:
        v = list(c)
        for _ in range(2):  # re-orthogonalise once for stability
            for q in basis:
                d = dot(q, v)
                v = [x - d * y for x, y in zip(v, q)]
        n = math.sqrt(dot(v, v))
        if n > 1e-9:
            basis.append([x / n for x in v])
    return basis


def project_out(v: list[float], basis: list[list[float]]) -> list[float]:
    for q in basis:
        d = dot(q, v)
        v = [x - d * y for x, y in zip(v, q)]
    return v


# --- the plant as the datasheet and manual describe it -----------------------------------------------------------
def sim_setpoint(key: Key) -> dict:
    if key[0] == "nominal":
        return {"experiment": "nominal"}
    if key[0] == "speed":
        return {"experiment": "speed_step", "s": key[1], **({"pump": key[2]} if key[2] else {})}
    if key[0] == "valve":
        return {"experiment": "valve_step", "valve": key[1], "theta": key[2]}
    if key[0] == "isolate":
        return {"experiment": "isolate_branch", "branch": key[1]}
    return {"experiment": "pump_off_static"}


class Plant:
    def __init__(self, topo: dict):
        self.sensors = [s["id"] for s in topo["sensors"]]
        self.sigma = {s["id"]: float(s["sigma"]) for s in topo["sensors"]}
        self.kind = {s["id"]: s["kind"] for s in topo["sensors"]}
        self.target = {s["id"]: s["target"] for s in topo["sensors"]}
        self.links = {ln["id"]: ln for ln in topo["links"]}
        self.junctions = [j["id"] for j in topo["junctions"]]
        self.pumps = [ln["id"] for ln in topo["links"] if ln["kind"] == "pump"]
        self.valves = [ln["id"] for ln in topo["links"] if ln["kind"] == "valve"]
        self.branches = {b["id"]: b for b in topo["branches"]}
        self.costs = dict(topo["costs"])
        self.candidates: list[Param] = [(ln["id"], LINK_TYPE[ln["kind"]]) for ln in topo["links"]]
        self.candidates += [(j, "leak") for j in self.junctions]
        self.order = {p: i for i, p in enumerate(self.candidates)}
        self.order.update({("bias", s): len(self.candidates) + i for i, s in enumerate(self.sensors)})

    def cost(self, key: Key) -> int:
        kind = {"nominal": "read_sensor", "speed": "speed_step", "valve": "valve_step", "isolate": "isolate_branch",
                "static": "pump_off_static"}[key[0]]
        return int(self.costs[kind])

    def returned(self, key: Key) -> list[str]:
        """Sensors an experiment returns (manual, topic experiments), in datasheet order."""
        jset = set(self.junctions)
        kind = key[0]
        if kind == "static":
            return [s for s in self.sensors if self.kind[s] in ("pressure", "flow")]
        if kind == "speed":
            pumps = [key[2]] if key[2] else self.pumps
            nodes = {n for p in pumps for n in (self.links[p]["frm"], self.links[p]["to"])}
            return [s for s in self.sensors if self.kind[s] == "flow"
                    or (self.kind[s] == "pressure" and self.target[s] in nodes)
                    or (self.kind[s] in ("power", "vibration") and self.target[s] in pumps)]
        if kind == "valve":
            v = self.links[key[1]]
            ends = {n for n in (v["frm"], v["to"]) if n in jset}
            touching = [ln for ln in self.links.values() if ln["frm"] in ends or ln["to"] in ends]
            near = ends | {n for ln in touching for n in (ln["frm"], ln["to"]) if n in jset}
            ids = {ln["id"] for ln in touching}
            return [s for s in self.sensors if (self.kind[s] == "pressure" and self.target[s] in near)
                    or (self.kind[s] == "flow" and self.target[s] in ids)]
        if kind == "isolate":
            b = self.branches[key[1]]
            nodes = set(b["junctions"]) | {self.links[b["entry"]]["frm"]}
            return [s for s in self.sensors if self.kind[s] == "flow"
                    or (self.kind[s] == "pressure" and self.target[s] in nodes)]
        return list(self.sensors)

    def nodes_of(self, comps: list[str]) -> set[str]:
        out = set()
        for c in comps:
            if c in self.links:
                out |= {self.links[c]["frm"], self.links[c]["to"]}
            else:
                out.add(c)
        return out

    def local_keys(self, comps: list[str]) -> list[Key]:
        """Local probes that touch the given components (valve steps, branch isolations, per-pump steps)."""
        nodes, keys = self.nodes_of(comps), []
        for v in self.valves:
            if v in comps or {self.links[v]["frm"], self.links[v]["to"]} & nodes:
                keys.append(("valve", v, THETA))
        for bid, b in self.branches.items():
            if (set(b["links"]) | set(b["junctions"])) & set(comps) or self.links[b["entry"]]["frm"] in nodes:
                keys.append(("isolate", bid))
        pumps = [p for p in self.pumps if p in comps or {self.links[p]["frm"], self.links[p]["to"]} & nodes]
        if len(self.pumps) == 1 and pumps:
            keys.append(("speed", LOCAL_SPEED, ""))
        elif len(self.pumps) > 1:
            keys += [("speed", s, p) for p in pumps for s in (LOCAL_SPEED, BROAD_SPEED)]
        return keys


def hyp_of(phys: dict) -> tuple:
    return tuple(sorted((c, t, round(m, ROUND)) for (c, t), m in phys.items()))


def fd_step(typ: str, m: float) -> float:
    h = FD_REL * max(m, TYPICAL[typ][0])
    return -h if m + h > SIM_MAX[typ] else h


class Twin:
    """simulate with a cache keyed by (operating point, rounded hypothesis)."""

    def __init__(self, call: Call):
        self.call, self.left, self.used, self.cache = call, 300, 0, {}

    def predict(self, key: Key, hyp: tuple, reserve: int = 0) -> dict[str, float]:
        ck = (key, hyp)
        if ck not in self.cache:
            if self.left <= reserve:
                raise SimExhausted
            res = self.call("simulate", {"overrides": [{"component": c, "type": t, "magnitude": m} for c, t, m in hyp],
                                         "setpoint": sim_setpoint(key)})
            self.left, self.used = int(res["simulate_left"]), self.used + 1
            self.cache[ck] = res["readings"]
        return self.cache[ck]


def find_stuck(obs: list[tuple[str, Key, float]]) -> list[str]:
    """Stuck sensors: any exact repeat of a reading (manual fact (g)). Isolated here because the stuck-sensor
    behaviour is an open decision; a healthy sensor draws fresh noise on every call."""
    seen: dict[str, list[float]] = {}
    stuck: list[str] = []
    for sid, _, value in obs:
        if value in seen.get(sid, []) and sid not in stuck:
            stuck.append(sid)
        seen.setdefault(sid, []).append(value)
    return stuck


# --- the oracle --------------------------------------------------------------------------------------------------
class Oracle:
    def __init__(self, call: Call):
        self.call = call
        self.plant = Plant(call("get_topology", {}))
        self.twin = Twin(call)
        self.obs: list[tuple[str, Key, float]] = []
        self.spent = 0
        self.phys: dict[Param, float] = {}
        self.bias: dict[str, float] = {}
        self.stuck: list[str] = []
        self.pcols: dict[Param, dict] = {}  # per-parameter Jacobian anchors
        self.screen: dict[Key, dict[Param, dict[str, float]]] = {}
        self.status: dict = {"margins": {}, "competitors": {}, "ambiguous": [], "rel_sd": {}}
        self.trace: list[dict] = []
        self.reserve = 0  # simulate calls to keep in hand (raised for exploration, 0 for the final fit)

    # columns -------------------------------------------------------------------------------------------------
    def screening(self, key: Key, p: Param) -> dict[str, float]:
        """Whitened secant sensitivity per unit magnitude at the healthy plant (step = typical magnitude)."""
        per_key = self.screen.setdefault(key, {})
        if p not in per_key:
            if key[0] == "static" and p[1] != "leak":
                per_key[p] = {}  # no flow at the static test: fouling, valves and wear are invisible (topic faults)
            else:
                base = self.twin.predict(key, (), self.reserve)
                hit = self.twin.predict(key, ((p[0], p[1], MID[p[1]]),), self.reserve)
                per_key[p] = {s: (hit[s] - base[s]) / MID[p[1]] / self.plant.sigma[s] for s in self.plant.sensors}
        return per_key[p]

    def has_screen(self, key: Key, p: Param) -> bool:
        return p in self.screen.get(key, {}) or (key[0] == "static" and p[1] != "leak")

    def cand_col(self, p: Param, rows: list, compute: bool) -> list[float]:
        if p[0] == "bias":
            return [1.0 if s == p[1] else 0.0 for s, _, _ in rows]
        col = []
        for s, k, _ in rows:
            if compute or self.has_screen(k, p):
                col.append(self.screening(k, p).get(s, 0.0))
            else:
                col.append(0.0)  # local column not computed yet (filled in for the leading candidates)
        return col

    def pcol(self, p: Param, phys: dict, key: Key) -> dict[str, float]:
        """Whitened finite-difference column of a support parameter, anchored at the estimate when it was first
        needed; re-anchored once the parameter drifts more than ANCHOR_DRIFT (cross-effects are not tracked)."""
        e, m = self.pcols.get(p), phys[p]
        if e is None or abs(m - e["at"]) > ANCHOR_DRIFT * max(e["at"], TYPICAL[p[1]][0]):
            e = self.pcols[p] = {"at": m, "phys": dict(phys), "keys": {}}
        if key not in e["keys"]:
            h = fd_step(p[1], e["at"])
            base = self.twin.predict(key, hyp_of(e["phys"]), self.reserve)
            moved = self.twin.predict(key, hyp_of({**e["phys"], p: e["at"] + h}), self.reserve)
            e["keys"][key] = {s: (moved[s] - base[s]) / h / self.plant.sigma[s] for s in self.plant.sensors}
        return e["keys"][key]

    def drifted(self, phys: dict) -> bool:
        return any(p in self.pcols and abs(m - self.pcols[p]["at"]) > ANCHOR_DRIFT * max(self.pcols[p]["at"],
                                                                                      TYPICAL[p[1]][0])
                   for p, m in phys.items())

    def columns(self, phys: dict, bias: dict, rows: list, fresh: tuple = ()) -> list[list[float]]:
        """Whitened Jacobian: support columns by finite differences (a just-proposed parameter in `fresh` uses its
        free screening column), bias columns are indicators."""
        cols = []
        for p in phys:
            if p in fresh:
                cols.append(self.cand_col(p, rows, True))
            else:
                cols.append([self.pcol(p, phys, k).get(s, 0.0) for s, k, _ in rows])
        return cols + [self.cand_col(("bias", s), rows, False) for s in bias]

    def residuals(self, phys: dict, bias: dict, rows: list) -> list[float]:
        hyp, preds, out = hyp_of(phys), {}, []
        for s, k, y in rows:
            if k not in preds:
                preds[k] = self.twin.predict(k, hyp, self.reserve)
            out.append((y - preds[k][s]) / self.plant.sigma[s] - bias.get(s, 0.0))
        return out

    # fit -----------------------------------------------------------------------------------------------------
    @staticmethod
    def names(phys: dict, bias: dict) -> list[Param]:
        return list(phys) + [("bias", s) for s in bias]

    @staticmethod
    def value(phys: dict, bias: dict, n: Param) -> float:
        return bias[n[1]] if n[0] == "bias" else phys[n]

    def refit(self, phys: dict, bias: dict, rows: list, fresh: tuple = (), iters: int = 6):
        """Bounded Levenberg-Marquardt on the support with a chord Jacobian; one more pass with refreshed
        columns if the estimate drifted from their anchors."""
        phys, bias = dict(phys), dict(bias)
        r = self.residuals(phys, bias, rows)
        ssr = dot(r, r)
        names = self.names(phys, bias)
        for attempt in range(2):
            cols = self.columns(phys, bias, rows, fresh if attempt == 0 else ())
            a = gram(cols)
            lam, rejected = 1e-3, 0
            for _ in range(iters):
                g = [dot(c, r) for c in cols]
                m = [[v + (lam * row[i] + 1e-12 if i == j else 0.0) for j, v in enumerate(row)]
                     for i, row in enumerate(a)]
                delta = chol_solve(spd_factor(m), g)
                p2, b2, eff = dict(phys), dict(bias), []
                for n, d in zip(names, delta):
                    if n[0] == "bias":
                        b2[n[1]] = bias[n[1]] + d
                        eff.append(d)
                    else:
                        p2[n] = min(max(phys[n] + d, 0.0), SIM_MAX[n[1]])
                        eff.append(p2[n] - phys[n])
                if 2.0 * dot(eff, g) - dot(eff, [dot(row, eff) for row in a]) < LM_TOL:
                    break  # the linear model promises no meaningful gain
                r2 = self.residuals(p2, b2, rows)
                ssr2 = dot(r2, r2)
                if ssr2 < ssr - 1e-9:
                    phys, bias, r, ssr, lam = p2, b2, r2, ssr2, max(lam / 3.0, 1e-6)
                else:
                    lam, rejected = lam * 5.0, rejected + 1
                    if rejected >= LM_REJECT_MAX:
                        break
            if not (fresh and attempt == 0) and not self.drifted(phys):
                break
        return phys, bias, r, ssr

    def scan(self, names: list[Param], cols: list[list[float]], rows: list, r: list, exclude: Param | None = None):
        """Candidate drops on the residual projected off the support columns (minus `exclude`), with the sign
        rule for physical magnitudes. Returns (drop, param, coefficient), best first."""
        keep = [c for n, c in zip(names, cols) if n != exclude]
        basis = orthonormal(keep)
        rp = project_out(r, basis)
        used = {n[1] if n[0] == "bias" else n[0] for n in names if n != exclude} | set(self.stuck)
        cands = [p for p in self.plant.candidates if p[0] not in used]
        seen: dict[str, int] = {}
        for s, _, _ in rows:
            seen[s] = seen.get(s, 0) + 1
        # a bias on a sensor read once would just delete that row, beating every physical explanation of it
        cands += [("bias", s) for s in self.plant.sensors if s not in used and seen.get(s, 0) >= BIAS_MIN_READS]
        local = any(k not in SCREENED for _, k, _ in rows)

        def score(p: Param, compute: bool):
            cp = project_out(self.cand_col(p, rows, compute), basis)
            cc = dot(cp, cp)
            if cc < 1e-12:
                return None
            coef = dot(cp, rp) / cc
            if p[0] != "bias" and coef <= 0.0:
                return None
            return (round(coef * coef * cc, ROUND), p, coef)

        out = [x for x in (score(p, False) for p in cands) if x is not None]
        out.sort(key=lambda x: (-x[0], self.plant.order[x[1]]))
        if local:  # fill in local columns for the leading physical candidates and rescore them
            redo = {x[1]: score(x[1], True) for x in out[:5] if x[1][0] != "bias"}
            out = [redo.get(x[1], x) for x in out]
            out = [x for x in out if x is not None]
            out.sort(key=lambda x: (-x[0], self.plant.order[x[1]]))
        return out

    def start(self, p: Param, coef: float) -> float:
        return coef if p[0] == "bias" else min(max(coef, 0.5 * TYPICAL[p[1]][0]), SIM_MAX[p[1]])

    def with_param(self, phys: dict, bias: dict, p: Param, coef: float, drop: Param | None = None):
        tp = {q: m for q, m in phys.items() if q != drop}
        tb = {s: b for s, b in bias.items() if ("bias", s) != drop}
        if p[0] == "bias":
            tb[p[1]] = self.start(p, coef)
        else:
            tp[p] = self.start(p, coef)
        return tp, tb

    def fit(self) -> None:
        self.stuck = find_stuck(self.obs)
        rows = [o for o in self.obs if o[0] not in self.stuck]
        phys = dict(self.phys)
        bias = {s: b for s, b in self.bias.items() if s not in self.stuck}
        phys, bias, r, ssr = self.refit(phys, bias, rows)
        start = self.twin.used
        phys, bias, r, ssr = self.grow(phys, bias, rows, r, ssr, start, MAX_SUPPORT)
        phys, bias, r, ssr = self.prune(phys, bias, rows, r, ssr)
        if self.within(start):
            phys, bias, r, ssr = self.swaps(phys, bias, rows, r, ssr, start)
        if EXPLORE_PAIRS and self.within(start):
            dof = max(1, len(rows) - len(self.names(phys, bias)))
            if ssr > dof + PAIR_Z * math.sqrt(2.0 * dof):
                phys, bias, r, ssr = self.pair_swaps(phys, bias, rows, r, ssr)
        self.phys, self.bias = phys, bias
        self.status = self.diagnose(phys, bias, rows, r)
        self.status.update(ssr=ssr, n_rows=len(rows))

    def grow(self, phys: dict, bias: dict, rows: list, r: list, ssr: float, start: int, limit: int):
        """Greedy additions: the best of TRY_MAX leading candidates that cuts the real SSR by at least JOIN."""
        for _ in range(limit):
            if len(phys) + len(bias) >= MAX_SUPPORT or not self.within(start):
                break
            added = False
            for drop, p, coef in self.scan(self.names(phys, bias), self.columns(phys, bias, rows), rows, r)[:TRY_MAX]:
                if drop < JOIN:
                    break
                tp, tb, tr, tssr = self.refit(*self.with_param(phys, bias, p, coef), rows,
                                              fresh=(p,) if p[0] != "bias" else ())
                if ssr - tssr >= JOIN:
                    phys, bias, r, ssr, added = tp, tb, tr, tssr, True
                    break
            if not added:
                break
        return phys, bias, r, ssr

    def polish(self) -> None:
        """Final fit only. Borderline elements are checked against their leading rivals with real nonlinear swaps
        (linear margins mix linearisation points); a poor overall fit triggers remove-one-and-regrow. Every
        improvement is committed at once, so reaching the simulate cap keeps what was found."""
        rows = [o for o in self.obs if o[0] not in self.stuck]
        phys, bias = dict(self.phys), dict(self.bias)
        r = self.residuals(phys, bias, rows)
        ssr = dot(r, r)
        for n in self.names(phys, bias):
            if n not in self.names(phys, bias):
                continue
            own, alt = self.rivals(phys, bias, rows, r, self.columns(phys, bias, rows), n)
            if not alt or own - alt[0][0] >= POLISH_MARGIN:
                continue
            best = None
            for _, q, coef in alt[:POLISH_RIVALS]:
                trial = self.refit(*self.with_param(phys, bias, q, coef, drop=n), rows,
                                   fresh=(q,) if q[0] != "bias" else ())
                if trial[3] < ssr - 1.0 and (best is None or trial[3] < best[3]):
                    best = trial
            if best is not None:
                phys, bias, r, ssr = best
                self.phys, self.bias = phys, bias
        dof = max(1, len(rows) - len(self.names(phys, bias)))
        if PAIRS and (ssr > dof + PAIR_Z * math.sqrt(2.0 * dof) or self.status.get("ambiguous")):
            phys, bias, r, ssr = self.pair_swaps(phys, bias, rows, r, ssr)
        dof = max(1, len(rows) - len(self.names(phys, bias)))
        if ssr > dof + 3.0 * math.sqrt(2.0 * dof):
            for n in self.names(phys, bias):
                if n not in self.names(phys, bias):
                    continue
                tp = {p: m for p, m in phys.items() if p != n}
                tb = {q: b for q, b in bias.items() if ("bias", q) != n}
                tp, tb, tr, tssr = self.refit(tp, tb, rows)
                tp, tb, tr, tssr = self.grow(tp, tb, rows, tr, tssr, self.twin.used, 2)
                if tssr < ssr - 1.0:
                    phys, bias, r, ssr = tp, tb, tr, tssr
                    self.phys, self.bias = phys, bias
        phys, bias, r, ssr = self.prune(phys, bias, rows, r, ssr)
        self.phys, self.bias = phys, bias
        self.status = self.diagnose(phys, bias, rows, r)
        self.status.update(ssr=ssr, n_rows=len(rows))

    def pair_swaps(self, phys: dict, bias: dict, rows: list, r: list, ssr: float):
        """Replace one element by two (a one-for-two local minimum: a leak standing in for a neighbouring leak plus
        a fouled pipe). Pairs among the element's leading rivals are scored by a two-column least squares on the
        projected residual (free); only a pair that beats the element by PAIR_GAIN gets a real refit."""
        for n in self.names(phys, bias):
            if n not in self.names(phys, bias):
                continue
            names, cols = self.names(phys, bias), self.columns(phys, bias, rows)
            ce, theta = cols[names.index(n)], self.value(phys, bias, n)
            basis = orthonormal([c for m, c in zip(names, cols) if m != n])
            rp = project_out([x + theta * c for x, c in zip(r, ce)], basis)
            cp = project_out(ce, basis)
            own = dot(cp, rp) ** 2 / dot(cp, cp) if dot(cp, cp) > 1e-12 else 0.0
            lead = [x for x in self.scan(names, cols, rows, [x + theta * c for x, c in zip(r, ce)], exclude=n)
                    if x[1] != n][:PAIR_POOL]
            proj = {x[1]: project_out(self.cand_col(x[1], rows, True), basis) for x in lead}
            best = None
            for i, (_, a, _) in enumerate(lead):
                for _, b, _ in lead[i + 1:]:
                    if (a[1] if a[0] == "bias" else a[0]) == (b[1] if b[0] == "bias" else b[0]):
                        continue
                    pa, pb = proj[a], proj[b]
                    g = [[dot(pa, pa), dot(pa, pb)], [dot(pa, pb), dot(pb, pb)]]
                    det = g[0][0] * g[1][1] - g[0][1] ** 2
                    if det <= 1e-12 * max(g[0][0] * g[1][1], 1e-300):
                        continue
                    ya, yb = dot(pa, rp), dot(pb, rp)
                    ba, bb = (g[1][1] * ya - g[0][1] * yb) / det, (g[0][0] * yb - g[0][1] * ya) / det
                    if (a[0] != "bias" and ba <= 0.0) or (b[0] != "bias" and bb <= 0.0):
                        continue
                    drop = round(ba * ya + bb * yb, ROUND)
                    if drop >= own + PAIR_GAIN and (best is None or drop > best[0]):
                        best = (drop, a, ba, b, bb)
            if best is None:
                continue
            _, a, ba, b, bb = best
            tp, tb = self.with_param(phys, bias, a, ba, drop=n)
            tp, tb = self.with_param(tp, tb, b, bb)
            fresh = tuple(q for q in (a, b) if q[0] != "bias")
            tp, tb, tr, tssr = self.refit(tp, tb, rows, fresh=fresh)
            if tssr <= ssr - KEEP:
                phys, bias, r, ssr = tp, tb, tr, tssr
                self.phys, self.bias = phys, bias
        return phys, bias, r, ssr

    def prune(self, phys: dict, bias: dict, rows: list, r: list, ssr: float):
        while phys or bias:
            names = self.names(phys, bias)
            cov = inverse(gram(self.columns(phys, bias, rows)))
            dssr = {n: self.value(phys, bias, n) ** 2 / max(cov[i][i], 1e-300) for i, n in enumerate(names)}
            weakest = min(names, key=lambda n: (round(dssr[n], ROUND), self.plant.order[n]))
            if dssr[weakest] >= 2.0 * KEEP:
                break
            tp = {p: m for p, m in phys.items() if p != weakest}
            tb = {s: b for s, b in bias.items() if ("bias", s) != weakest}
            tp, tb, tr, tssr = self.refit(tp, tb, rows)
            if tssr - ssr >= KEEP:
                break  # the explicit removal costs too much: the weakest element stays, so do the others
            phys, bias, r, ssr = tp, tb, tr, tssr
        return phys, bias, r, ssr

    def rivals(self, phys: dict, bias: dict, rows: list, r: list, cols: list[list[float]], n: Param):
        """Own drop and the best alternatives for element n, on the residual with n's contribution added back."""
        names = self.names(phys, bias)
        ce, theta = cols[names.index(n)], self.value(phys, bias, n)
        alt = self.scan(names, cols, rows, [x + theta * c for x, c in zip(r, ce)], exclude=n)
        cp = project_out(ce, orthonormal([c for m, c in zip(names, cols) if m != n]))
        cc = dot(cp, cp)
        re = project_out([x + theta * c for x, c in zip(r, ce)], orthonormal([c for m, c in zip(names, cols) if m != n]))
        own = dot(cp, re) ** 2 / cc if cc > 1e-12 else 0.0
        return round(own, ROUND), [x for x in alt if x[1] != n]

    def within(self, start: int) -> bool:
        """Optional fit work stays inside FIT_SIM_MAX during exploration (no limit in the final fit)."""
        return self.reserve == 0 or self.twin.used - start < FIT_SIM_MAX

    def swaps(self, phys: dict, bias: dict, rows: list, r: list, ssr: float, start: int):
        tried = 0
        for n in self.names(phys, bias):
            if tried >= 2 or n not in self.names(phys, bias) or not self.within(start):
                continue
            own, alt = self.rivals(phys, bias, rows, r, self.columns(phys, bias, rows), n)
            if alt and alt[0][0] > own + 1.0:
                tried += 1
                q = alt[0][1]
                tp, tb, tr, tssr = self.refit(*self.with_param(phys, bias, q, alt[0][2], drop=n), rows,
                                              fresh=(q,) if q[0] != "bias" else ())
                if tssr < ssr - 1.0:
                    phys, bias, r, ssr = tp, tb, tr, tssr
        return phys, bias, r, ssr

    def diagnose(self, phys: dict, bias: dict, rows: list, r: list) -> dict:
        """Swap margins, competitors and relative sd per element, plus ambiguous candidates (stop-rule inputs)."""
        names = self.names(phys, bias)
        cols = self.columns(phys, bias, rows)
        margins, competitors, rel_sd = {}, {}, {}
        if names:
            cov = inverse(gram(cols))
            for i, n in enumerate(names):
                rel_sd[n] = math.sqrt(max(cov[i][i], 0.0)) / max(abs(self.value(phys, bias, n)), 1e-12)
                own, alt = self.rivals(phys, bias, rows, r, cols, n)
                margins[n] = own - (alt[0][0] if alt else 0.0)
                competitors[n] = [x[1] for x in alt[:2]]
        ambiguous = [x[1] for x in self.scan(names, cols, rows, r) if KEEP <= x[0] < JOIN]
        return {"margins": margins, "competitors": competitors, "rel_sd": rel_sd, "ambiguous": ambiguous}

    # design ------------------------------------------------------------------------------------------------------
    def unresolved(self) -> list[Param]:
        """Elements whose magnitude is loose or whose best alternative is close, their competitors, and
        candidates that explain between 3 and 5 sigma of the residual."""
        st, out = self.status, []
        for n in self.names(self.phys, self.bias):
            if st["rel_sd"].get(n, 1.0) > REL_SD_MAX or st["margins"].get(n, 0.0) < SWAP_MARGIN:
                out += [n, *st["competitors"].get(n, [])]
        out += st["ambiguous"]
        return list(dict.fromkeys(out))

    def design_rows(self, params: list[Param], rows: list[tuple[str, Key]]) -> list[list[float]]:
        return [[(1.0 if p[1] == s else 0.0) if p[0] == "bias"
                 else (self.screening(k, p).get(s, 0.0) if self.has_screen(k, p) else 0.0) for p in params]
                for s, k in rows]

    def covered(self) -> bool:
        """Every live sensor has COVER_READS readings (biases are testable offsets, stuck sensors are exposed)."""
        stuck = find_stuck(self.obs)
        reads = {s: 0 for s in self.plant.sensors if s not in stuck}
        for s, _, _ in self.obs:
            if s in reads:
                reads[s] += 1
        return all(n >= COVER_READS for n in reads.values())

    def choose(self):
        """Bayesian D-optimal probe per budget unit (screening columns, prior variance PRIOR_INCLUSION * MID^2),
        plus a bonus for reads still needed for stuck detection. None when nothing is left to resolve."""
        left = SPEND_CAP - self.spent
        reads = {s: 0 for s in self.plant.sensors if s not in self.stuck}
        for s, _, _ in self.obs:
            if s in reads:
                reads[s] += 1
        need = {s: max(0, COVER_READS - n) for s, n in reads.items()}
        broad = any(need.values())
        focus = [p for p in self.unresolved() if p[0] != "bias" or p[1] in reads]
        if left < 1 or (not broad and not focus):
            return None
        for s in self.bias if CONFIRM_FIRST else ():  # a fitted bias seen once gets a confirming read first
            if reads.get(s) == 1:
                return ("read", s), float("inf")
        if broad:
            params = list(self.plant.candidates) + [("bias", s) for s in reads]
        else:
            params = focus[:INPLAY_MAX]
        open_phys = [p for p in focus if p[0] != "bias"][:LOCAL_PARAMS]
        local = self.plant.local_keys([p[0] for p in open_phys])[:LOCAL_MAX]
        for k in local:  # local design columns for the unresolved physical parameters only
            for p in open_phys:
                self.screening(k, p)
        prior = [1.0 / (PRIOR_INCLUSION * MID["sensor_bias" if p[0] == "bias" else p[1]] ** 2) for p in params]
        info = [[prior[i] if i == j else 0.0 for j in range(len(params))] for i in range(len(params))]
        for v in self.design_rows(params, [(s, k) for s, k, _ in self.obs if s in reads]):
            nz = [i for i, x in enumerate(v) if x != 0.0]
            for i in nz:
                for j in nz:
                    info[i][j] += v[i] * v[j]
        cov = inverse(info)
        keys = [("speed", BROAD_SPEED, ""), STATIC] + [k for k in local if k != ("speed", BROAD_SPEED, "")]
        options = [(("read", s), [(s, NOMINAL)]) for s in reads]
        options += [(("exp", k), [(s, k) for s in self.plant.returned(k) if s in reads]) for k in keys]
        best = None
        for probe, prow in options:
            cost = self.plant.cost(NOMINAL if probe[0] == "read" else probe[1])
            if cost > left or not prow:
                continue
            vs = self.design_rows(params, prow)
            cv = [[dot(row, v) for row in cov] for v in vs]
            m = [[dot(a, b) + (1.0 if i == j else 0.0) for j, b in enumerate(vs)] for i, a in enumerate(cv)]
            supplied: dict[str, int] = {}
            for s, _ in prow:
                supplied[s] = supplied.get(s, 0) + 1
            bonus = COVER_BONUS * sum(min(need[s], n) for s, n in supplied.items())
            score = round((logdet(m) + bonus) / cost, ROUND)
            if best is None or score > best[0]:
                best = (score, probe)
        if best is None or best[0] < GAIN_MIN:
            return None
        return best[1], best[0]

    # episode -----------------------------------------------------------------------------------------------------
    def execute(self, probe: tuple) -> None:
        if probe[0] == "read":
            res = self.call("read_sensor", {"sensor_id": probe[1]})
            self.obs.append((probe[1], NOMINAL, float(res["value"])))
        else:
            sp = sim_setpoint(probe[1])
            kind = sp.pop("experiment")
            res = self.call("run_experiment", {"type": kind, "args": sp})
            self.obs += [(s, probe[1], float(res["readings"][s])) for s in self.plant.sensors if s in res["readings"]]
        self.spent = BUDGET - int(res["budget_left"])

    def submission(self) -> dict:
        diagnosis = [{"component": c, "type": t, "magnitude": round(m, 6)} for (c, t), m in self.phys.items()]
        diagnosis += [{"component": s, "type": "sensor_bias", "magnitude": round(abs(b), 6)} for s, b in self.bias.items()]
        diagnosis += [{"component": s, "type": "sensor_stuck"} for s in self.stuck]
        repairs = [{"action": REPAIR[t], "target": c} for (c, t) in self.phys]  # sensor repairs never move J
        return {"diagnosis": diagnosis[:10], "repairs": repairs[:10]}

    def run(self) -> dict:
        note = None
        try:
            self.reserve = SIM_RESERVE + 2 * len(SCREENED)
            for k in SCREENED:  # screening columns for every candidate
                for p in self.plant.candidates:
                    self.screening(k, p)
            while (pick := self.choose()) is not None:
                points = len({k for _, k, _ in self.obs}) + 1
                self.reserve = SIM_RESERVE + (len(self.phys) + 2) * points
                self.execute(pick[0])
                if FIT_DURING_COVERAGE or self.covered():
                    self.fit()
                else:
                    self.stuck = find_stuck(self.obs)
                self.trace.append({"probe": label(pick[0]), "score": pick[1], "spent": self.spent,
                                   "simulate_left": self.twin.left, "stuck": list(self.stuck),
                                   "support": [":".join(n) for n in self.names(self.phys, self.bias)]})
        except SimExhausted:
            note = "simulate reserve reached during exploration"
        self.reserve = 0
        try:
            self.fit()
            if POLISH:
                self.polish()
        except SimExhausted:
            note = "simulate cap reached in the final fit"
        sub = self.submission()
        self.call("submit", sub)
        st = self.status
        final = {"margins": {":".join(n): round(v, 2) for n, v in st.get("margins", {}).items()},
                 "rel_sd": {":".join(n): round(v, 3) for n, v in st.get("rel_sd", {}).items()},
                 "competitors": {":".join(n): [":".join(c) for c in v] for n, v in st.get("competitors", {}).items()},
                 "ambiguous": [":".join(c) for c in st.get("ambiguous", [])], "ssr": st.get("ssr"),
                 "n_rows": st.get("n_rows")}
        return {"submission": sub, "spent": self.spent, "simulate_used": self.twin.used, "trace": self.trace,
                "final": final, "note": note}


def label(probe: tuple) -> str:
    if probe[0] == "read":
        return f"read_sensor:{probe[1]}"
    k = probe[1]
    name = {"speed": "speed_step", "valve": "valve_step", "isolate": "isolate_branch", "static": "pump_off_static"}[k[0]]
    return ":".join([name, *[str(x) for x in k[1:] if x != ""]])


def run_oracle(call: Call) -> dict:
    return Oracle(call).run()
