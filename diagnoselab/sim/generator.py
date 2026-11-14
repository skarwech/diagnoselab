"""Plant generator (F1) by inverse design: draw a feasible design point, derive parameters from it.

Structure and numbers follow docs/sim_spec.md section 7. Validity (q_req met, pump near BEP,
static-test preconditions) holds by construction; the spec holds no solver output.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from diagnoselab.sim.model import (Branch, Junction, Link, Outlet, PlantSpec, Sensor, Source, rnd,
                                   spec_hash)

__all__ = ["generate_plant", "spec_hash", "stream", "TIER_CODE", "SIZES"]

TIER_CODE = {"easy": 0, "medium": 1, "hard": 2}
SIZES = {
    "easy": {"junctions": (6, 8), "pumps": (1, 1), "outlets": (2, 3), "chords": (0, 0)},
    "medium": {"junctions": (8, 10), "pumps": (1, 2), "outlets": (2, 4), "chords": (1, 1)},
    "hard": {"junctions": (10, 12), "pumps": (2, 2), "outlets": (3, 4), "chords": (1, 2)},
}
VALVES = (2, 4)
TANK_HEAD = 50.0
MIN_SPEED = 0.7  # lowest speed_step; must match experiments.SPEED_RANGE[0]
STRUCTURE, FAULTS, NOISE, DRIFT = 0, 1, 2, 3
REDRAWS: Counter = Counter()  # diagnostics: internal redraws (F1 'first draws valid')


def stream(seed: int, tier: str, k: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), TIER_CODE[tier], k])


def _draw(rng: np.random.Generator, lo_hi: tuple[int, int]) -> int:
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


@dataclass(frozen=True)
class _Tree:
    parent: tuple[int | None, ...]
    outlet_nodes: tuple[int, ...]


def _tree(rng: np.random.Generator, n_tree: int, out_range: tuple[int, int]) -> _Tree:
    """Random recursive tree with at most out_range[1] leaves by construction (no redraws): once the leaf
    budget is full, a new node attaches to a leaf, which keeps the count. The outlet count is drawn
    afterwards so every leaf carries an outlet."""
    parent: list[int | None] = [None]
    leaf_set = {0}
    for i in range(1, n_tree):
        pool = sorted(leaf_set) if len(leaf_set) >= out_range[1] else list(range(i))
        p = pool[int(rng.integers(len(pool)))]
        parent.append(p)
        leaf_set = (leaf_set - {p}) | {i}
    leaves = sorted(leaf_set)
    n_out = int(rng.integers(max(out_range[0], len(leaves)), out_range[1] + 1))
    others = [i for i in range(n_tree) if i not in leaves]
    extra = rng.permutation(others)[: n_out - len(leaves)].tolist()
    return _Tree(tuple(parent), tuple(leaves) + tuple(int(e) for e in extra))


def _path_to_root(parent: tuple[int | None, ...], node: int) -> list[int]:
    path = [node]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path


def generate_plant(seed: int, tier: str) -> PlantSpec:
    if tier not in TIER_CODE:
        raise ValueError(f"unknown tier {tier!r}")
    rng, size = stream(seed, tier, STRUCTURE), SIZES[tier]
    n_j, n_p = _draw(rng, size["junctions"]), _draw(rng, size["pumps"])
    n_chord, n_valve = _draw(rng, size["chords"]), _draw(rng, VALVES)
    n_src = 1 if n_p == 1 else int(rng.integers(1, 3))
    n_tree = n_j - 2 * n_p
    tree = _tree(rng, n_tree, size["outlets"])
    n_out = len(tree.outlet_nodes)

    # ids: pump k has suction J(2k+1), discharge J(2k+2); tree node i is J(2*n_p+i+1), root = node 0
    suc = [f"J{2 * k + 1}" for k in range(n_p)]
    dis = [f"J{2 * k + 2}" for k in range(n_p)]
    tn = [f"J{2 * n_p + i + 1}" for i in range(n_tree)]
    src = [f"S{k + 1}" for k in range(n_src)]
    tanks = [f"T{k + 1}" for k in range(n_out)]

    # --- design flows -------------------------------------------------------------------------
    q_out = rng.uniform(8.0, 25.0, n_out)
    sub = np.zeros(n_tree)  # outlet flow drawn through each tree node's subtree
    for o, node in enumerate(tree.outlet_nodes):
        for p in _path_to_root(tree.parent, node):
            sub[p] += q_out[o]
    tree_q = {i: sub[i] for i in range(1, n_tree)}  # flow on link parent(i) -> i

    # --- design heads ---------------------------------------------------------------------------
    adjacent = {(i, tree.parent[i]) for i in range(1, n_tree)} | {(tree.parent[i], i) for i in range(1, n_tree)}
    for attempt in range(100):  # redraw drops until enough chord endpoints differ by >= 0.5 m (small trees)
        REDRAWS["chord_heads"] += attempt > 0
        drop = {i: rng.uniform(1.0, 4.0) for i in range(1, n_tree)}
        depth = np.zeros(n_tree)
        for i in range(1, n_tree):
            depth[i] = depth[tree.parent[i]] + drop[i]
        margin = rng.uniform(1.0, 3.0, n_out)
        h_root = TANK_HEAD + max(depth[n] + m for n, m in zip(tree.outlet_nodes, margin))
        h_tree = h_root - depth
        cands = [(u, v) for u in range(n_tree) for v in range(n_tree)
                 if u < v and (u, v) not in adjacent and abs(h_tree[u] - h_tree[v]) >= 0.5]
        if len(cands) >= n_chord:
            break

    # --- chords (loops), oriented high head -> low head ------------------------------------------
    chords: list[tuple[int, int, float]] = []
    for idx in rng.permutation(len(cands))[:n_chord].tolist():
        u, v = cands[idx]
        if h_tree[u] < h_tree[v]:
            u, v = v, u
        pu, pv = _path_to_root(tree.parent, u), _path_to_root(tree.parent, v)
        lca = next(n for n in pu if n in pv)
        down_v = pv[: pv.index(lca)]  # links parent(n)->n for n on lca..v (exclusive of lca)
        down_u = pu[: pu.index(lca)]
        c = 0.2 * min(tree_q[n] for n in down_v + down_u)  # smallest tree flow on the chord path
        for n in down_v:
            tree_q[n] -= c
        for n in down_u:
            tree_q[n] += c
        chords.append((u, v, c))

    q_total = float(q_out.sum())
    split = rng.uniform(0.4, 0.6) if n_p == 2 else 1.0
    q_pump = [q_total * split, q_total * (1.0 - split)][:n_p]
    h_dis = [h_root + rng.uniform(0.5, 2.0) for _ in range(n_p)]
    h_src = [TANK_HEAD - rng.uniform(3.0, 12.0) for _ in range(n_src)]
    pump_src = [k if n_src == n_p else 0 for k in range(n_p)]
    h_suc = [h_src[pump_src[k]] - rng.uniform(0.3, 1.5) for k in range(n_p)]

    # --- links (frm, to, role, design flow, head drop) ------------------------------------------
    raw: list[tuple[str, str, str, float, float]] = []
    for k in range(n_p):
        raw.append((src[pump_src[k]], suc[k], "suction", q_pump[k], h_src[pump_src[k]] - h_suc[k]))
        raw.append((suc[k], dis[k], "pump", q_pump[k], h_dis[k] - h_suc[k]))
        raw.append((dis[k], tn[0], "discharge", q_pump[k], h_dis[k] - h_root))
    for i in range(1, n_tree):
        raw.append((tn[tree.parent[i]], tn[i], "tree", tree_q[i], h_tree[tree.parent[i]] - h_tree[i]))
    for o, node in enumerate(tree.outlet_nodes):
        raw.append((tn[node], tanks[o], "outlet", q_out[o], h_tree[node] - TANK_HEAD))
    for u, v, c in chords:
        raw.append((tn[u], tn[v], "chord", c, h_tree[u] - h_tree[v]))

    outlet_idx = [k for k, r in enumerate(raw) if r[2] == "outlet"]
    tree_idx = [k for k, r in enumerate(raw) if r[2] == "tree"]
    pref = rng.permutation(outlet_idx).tolist() + rng.permutation(tree_idx).tolist()
    valve_idx = set(pref[:n_valve])

    links: list[Link] = []
    counters = {"pipe": 0, "valve": 0, "pump": 0}
    prefix = {"pipe": "P", "valve": "V", "pump": "PU"}
    design_q: dict[str, float] = {}
    pump_design: dict[str, tuple[float, float, float]] = {}
    for k, (frm, to, role, q, dh) in enumerate(raw):
        kind = "pump" if role == "pump" else ("valve" if k in valve_idx else "pipe")
        counters[kind] += 1
        lid = f"{prefix[kind]}{counters[kind]}"
        design_q[lid] = rnd(q)
        if kind == "pump":
            # healthy pump stepped alone to the lowest speed must still beat the header (held up by the other
            # pump), or speed_step(0.7, pump=k) deadheads
            lift = h_root - h_src[pump_src[int(lid[2:]) - 1]]
            a = max(dh * rng.uniform(1.25, 1.6), (lift + 2.0) / MIN_SPEED**2)
            q_bep = q / rng.uniform(0.8, 1.1)
            links.append(Link(lid, kind, frm, to, a=rnd(a), b=rnd((a - dh) / q**2), q_bep=rnd(q_bep),
                              eta_max=rnd(rng.uniform(0.70, 0.85)), v0=rnd(rng.uniform(1.5, 3.0))))
            pump_design[lid] = (q, dh, q / q_bep)
        else:
            links.append(Link(lid, kind, frm, to, r=rnd(dh / q**2)))

    # --- nodes ------------------------------------------------------------------------------
    junctions = []
    for k in range(n_p):
        junctions.append(Junction(suc[k], rnd(h_src[pump_src[k]] - rng.uniform(2.0, 6.0)), "suction"))
        junctions.append(Junction(dis[k], rnd(TANK_HEAD - rng.uniform(2.0, 15.0)), "discharge"))
    for i in range(n_tree):
        junctions.append(Junction(tn[i], rnd(TANK_HEAD - rng.uniform(2.0, 15.0)), "discharge"))
    design_h = {suc[k]: rnd(h_suc[k]) for k in range(n_p)} | {dis[k]: rnd(h_dis[k]) for k in range(n_p)}
    design_h |= {tn[i]: rnd(h_tree[i]) for i in range(n_tree)}

    # --- design outputs ---------------------------------------------------------------------
    power, vib = {}, {}
    for ln in links:
        if ln.kind == "pump":
            q, dh, u = pump_design[ln.id]
            eta = ln.eta_max * (1.0 - (u - 1.0) ** 2)
            power[ln.id] = rnd(9.81 * q * dh / (1000.0 * eta))
            vib[ln.id] = rnd(ln.v0 * (1.0 + 3.0 * (u - 1.0) ** 2))
    se = rnd(sum(power.values()) / (3.6 * q_total))

    sensors = _sensors(rng, junctions, links, design_q, power, vib)
    spec_wo_branches = dict(
        seed=int(seed), tier=tier, tank_head=TANK_HEAD, junctions=tuple(junctions),
        sources=tuple(Source(src[k], rnd(h_src[k])) for k in range(n_src)),
        outlets=tuple(Outlet(tanks[o], rnd(0.99 * q_out[o])) for o in range(n_out)),
        links=tuple(links), sensors=sensors,
        design={"h": design_h, "q": design_q, "Q_pump": rnd(q_total), "P": power, "vib": vib, "SE": se},
    )
    return PlantSpec(**spec_wo_branches, branches=_branches(links, tn[0], set(tanks)))


def _sensors(rng, junctions, links, design_q, power, vib) -> tuple[Sensor, ...]:
    pt = [j.id for j in junctions if j.side == "suction" or any(ln.kind == "pump" and ln.to == j.id for ln in links)]
    pt += [j.id for j in junctions if j.id not in pt and rng.random() < 0.6]
    ft = [ln.id for ln in links if rng.random() < 0.5]
    touching: dict[str, list[Link]] = {}
    for ln in links:
        for n in (ln.frm, ln.to):
            touching.setdefault(n, []).append(ln)
    for j in junctions:  # series junction (exactly two non-pump links): without PT both links look identical
        ls = touching.get(j.id, [])
        if j.id not in pt and len(ls) == 2 and all(ln.kind != "pump" for ln in ls):
            pt.append(j.id)
    for ln in links:  # coverage repair: every pipe/valve within one hop of a sensor
        if ln.kind != "pump" and ln.id not in ft and ln.frm not in pt and ln.to not in pt:
            ends = [n for n in (ln.frm, ln.to) if n.startswith("J")]
            pt.append(ends[int(rng.integers(len(ends)))])
    jorder = [j.id for j in junctions]
    lorder = [ln.id for ln in links]
    out = [Sensor(f"PT-{j}", "pressure", j, 0.15) for j in sorted(set(pt), key=jorder.index)]
    out += [Sensor(f"FT-{k}", "flow", k, rnd(0.01 * max(abs(design_q[k]), 1.0)))
            for k in sorted(set(ft), key=lorder.index)]
    out += [Sensor(f"WT-{p}", "power", p, rnd(0.01 * power[p])) for p in power]
    out += [Sensor(f"VT-{p}", "vibration", p, rnd(0.05 * vib[p])) for p in vib]
    return tuple(out)


def _branches(links: list[Link], root: str, tanks: set[str]) -> tuple[Branch, ...]:
    degree: dict[str, int] = {}
    for ln in links:
        for n in (ln.frm, ln.to):
            degree[n] = degree.get(n, 0) + 1
    into = {ln.to: ln for ln in links if ln.to.startswith("J")}
    branches = []
    for ln in links:
        if ln.to not in tanks:
            continue
        chain, inner, cur = [ln.id], [], ln.frm
        while cur != root and degree.get(cur) == 2 and cur in into and into[cur].kind != "pump":
            parent_link = into[cur]
            chain.append(parent_link.id)
            inner.append(cur)
            cur = parent_link.frm
        branches.append(Branch(f"B{len(branches) + 1}", chain[-1], tuple(reversed(chain)), tuple(reversed(inner))))
    return tuple(branches)
