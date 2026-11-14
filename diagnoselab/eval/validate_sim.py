"""Measure every F1-F5 valid-when condition (plan section 8) and write results/step2_validation.json.

Run: uv run python -m diagnoselab.eval.validate_sim   (about 3-5 minutes)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2, chisquare, kstest, norm

from diagnoselab.sim import certifier as C
from diagnoselab.sim.experiments import experiment_setpoint
from diagnoselab.sim.faults import (MAGNITUDE_RANGES, PHYSICAL_TYPES, apply_faults, drift_fires, eligible_pairs,
                                    impact, sample_faults)
from diagnoselab.sim.generator import FAULTS, SIZES, generate_plant, spec_hash, stream
from diagnoselab.sim.generator import REDRAWS as GEN_REDRAWS
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Fault, Instance, PlantSpec, Setpoint
from diagnoselab.sim.sensors import observe, physical_readings
from diagnoselab.sim.solver import leak_coefficients, compile_net, outlet_flows, solve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.reference_solver import ref_solve  # noqa: E402  (independent second solver)

TIERS = ("easy", "medium", "hard")
OUT = ROOT / "results"


def plant_checks(spec: PlantSpec) -> dict[str, bool]:
    st = solve(spec)
    links = spec.links
    nodes = {j.id for j in spec.junctions} | {s.id for s in spec.sources} | {o.id for o in spec.outlets}
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    down: dict[str, set[str]] = {n: set() for n in nodes}
    for ln in links:
        adj[ln.frm].add(ln.to)
        adj[ln.to].add(ln.frm)
        down[ln.frm].add(ln.to)

    def reach(start: set[str], graph) -> set[str]:
        seen, stack = set(start), list(start)
        while stack:
            for m in graph[stack.pop()] - seen:
                seen.add(m)
                stack.append(m)
        return seen

    sources = {s.id for s in spec.sources}
    pt = {s.target for s in spec.sensors if s.kind == "pressure"}
    ft = {s.target for s in spec.sensors if s.kind == "flow"}
    delivered = outlet_flows(spec, st)
    pumps = [ln for ln in links if ln.kind == "pump"]
    return {
        "connected": reach({next(iter(nodes))}, adj) == nodes,
        "outlets_reachable_from_source": {o.id for o in spec.outlets} <= reach(sources, down),
        "design_point_reproduced": max(abs(st.h[j] - h) for j, h in spec.design["h"].items()) < 1e-6
        and max(abs(st.q[k] - q) / abs(q) for k, q in spec.design["q"].items()) < 1e-6,
        "q_req_within_2pct": all(abs(delivered[o.id] - o.q_req) / o.q_req <= 0.02 for o in spec.outlets),
        "pump_70_120pct_bep": all(0.7 <= st.q[p.id] / p.q_bep <= 1.2 for p in pumps),
        "every_pipe_one_hop_from_sensor": all(ln.id in ft or ln.frm in pt or ln.to in pt
                                              for ln in links if ln.kind != "pump"),
        "pump_pt_both_sides": all(p.frm in pt and p.to in pt for p in pumps),
        "static_precondition_tank_ge_sources": all(spec.tank_head >= s.head for s in spec.sources),
    }


PLAN_SIZES = {"junctions": (6, 12), "sources": (1, 2), "outlets": (2, 4), "pumps": (1, 2), "valves": (2, 4),
              "pipes": (8, 14)}  # plan section 4.1


def _counts(spec: PlantSpec) -> dict[str, int]:
    kinds = Counter(ln.kind for ln in spec.links)
    return {"junctions": len(spec.junctions), "sources": len(spec.sources), "outlets": len(spec.outlets),
            "pumps": kinds["pump"], "valves": kinds["valve"], "pipes": kinds["pipe"]}


def f1() -> dict:
    fails, hashes, sizes, times = Counter(), {}, {t: Counter() for t in TIERS}, []
    ranges: dict[str, dict[str, list[int]]] = {t: {} for t in TIERS}
    clean_first_draws = 0
    for tier in TIERS:
        hs = []
        for seed in range(1000):
            before = sum(GEN_REDRAWS.values())
            t0 = time.perf_counter()
            spec = generate_plant(seed, tier)
            times.append(time.perf_counter() - t0)
            clean_first_draws += sum(GEN_REDRAWS.values()) == before
            for k, v in _counts(spec).items():
                lo_hi = ranges[tier].setdefault(k, [v, v])
                ranges[tier][k] = [min(lo_hi[0], v), max(lo_hi[1], v)]
            for name, ok in plant_checks(spec).items():
                fails[name] += not ok
            hs.append(spec_hash(spec))
            sizes[tier][len(spec.junctions)] += 1
            assert spec_hash(generate_plant(seed, tier)) == hs[-1]
        hashes[tier] = hs
    cross = {}
    for name in ("hashes_linux.json",):
        p = OUT / name
        if p.exists():
            other = json.loads(p.read_text())
            cross[name] = {t: sum(a == b for a, b in zip(hashes[t], other[t])) for t in TIERS}
            cross[name]["compared_per_tier"] = len(other["easy"])
    return {"plants": 3000, "check_failures": dict(fails), "all_checks": list(plant_checks(generate_plant(0, "easy"))),
            "duplicate_hashes": {t: len(hashes[t]) - len(set(hashes[t])) for t in TIERS},
            "junction_histogram": {t: dict(sorted(sizes[t].items())) for t in TIERS},
            "junction_ranges": {t: SIZES[t]["junctions"] for t in TIERS}, "size_ranges": ranges,
            "plan_sizes": PLAN_SIZES, "gen_ms_median": 1000 * statistics.median(times),
            "first_draw_valid": clean_first_draws / 3000, "cross_build": cross}


def _spec_for(d: dict) -> PlantSpec:
    return PlantSpec.from_dict(d)


def analytic() -> dict:
    lk = dict(a=0.0, b=0.0, q_bep=0.0, eta_max=0.0, v0=0.0)
    series = _spec_for({"tank_head": 10.0, "sources": [{"id": "S1", "head": 20.0}], "outlets": [{"id": "T1", "q_req": 1}],
                        "junctions": [{"id": "J1", "z": 0.0}],
                        "links": [{"id": "P1", "kind": "pipe", "frm": "S1", "to": "J1", "r": 0.002, **lk},
                                  {"id": "P2", "kind": "pipe", "frm": "J1", "to": "T1", "r": 0.003, **lk}]})
    e_series = abs(solve(series).q["P1"] - np.sqrt(10 / 0.005)) / np.sqrt(10 / 0.005)
    par = _spec_for({"tank_head": 10.0, "sources": [{"id": "S1", "head": 20.0}], "outlets": [{"id": "T1", "q_req": 1}],
                     "junctions": [{"id": "J1", "z": 0.0}],
                     "links": [{"id": "P1", "kind": "pipe", "frm": "S1", "to": "J1", "r": 0.001, **lk},
                               {"id": "P2", "kind": "pipe", "frm": "J1", "to": "T1", "r": 0.002, **lk},
                               {"id": "P3", "kind": "pipe", "frm": "J1", "to": "T1", "r": 0.008, **lk}]})
    st = solve(par)
    e_par = abs(st.q["P2"] / st.q["P3"] - 2.0) / 2.0
    out = {"series_rel_err": e_series, "parallel_ratio_rel_err": e_par}

    def pump_net(dz: float) -> PlantSpec:
        return _spec_for({"tank_head": 30.0 + dz, "sources": [{"id": "S1", "head": 30.0}],
                          "outlets": [{"id": "T1", "q_req": 1}],
                          "junctions": [{"id": "J1", "z": 20.0, "side": "suction"}, {"id": "J2", "z": 20.0}],
                          "links": [{"id": "P1", "kind": "pipe", "frm": "S1", "to": "J1", "r": 0.001, **lk},
                                    {"id": "PU1", "kind": "pump", "frm": "J1", "to": "J2", "r": 0.0, "a": 30.0,
                                     "b": 0.004, "q_bep": 40.0, "eta_max": 0.8, "v0": 2.0},
                                    {"id": "P2", "kind": "pipe", "frm": "J2", "to": "T1", "r": 0.005, **lk}]})

    for dz in (0.0, 8.0):
        for s in (1.0, 0.8):
            exact = np.sqrt((30.0 * s * s - dz) / (0.004 + 0.001 + 0.005))
            got = solve(pump_net(dz), setpoint=Setpoint(speed={"PU1": s})).q["PU1"]
            out[f"pump_intersection_dz{dz:g}_s{s:g}_rel_err"] = abs(got - exact) / exact
    q1 = solve(pump_net(0.0)).q["PU1"]
    out["affinity_rel_err"] = max(abs(solve(pump_net(0.0), setpoint=Setpoint(speed={"PU1": s})).q["PU1"] / q1 - s) / s
                                  for s in (0.5, 0.8, 1.2))
    off = solve(pump_net(5.0), setpoint=Setpoint(speed={"PU1": 0.0}))
    out["pump_off_max_abs_flow"] = max(abs(v) for v in off.q.values())
    out["pump_off_head_err"] = max(abs(off.h["J2"] - 35.0), abs(off.h["J1"] - 30.0))
    return out


def random_case(rng: np.random.Generator):
    tier = TIERS[int(rng.integers(3))]
    spec = generate_plant(int(rng.integers(10_000)), tier)
    pick = lambda kind: [ln.id for ln in spec.links if ln.kind == kind]  # noqa: E731
    pipes, valves, pumps = pick("pipe"), pick("valve"), pick("pump")
    faults = [Fault(str(rng.choice(pipes)), "fouling", float(rng.choice([rng.uniform(0.3, 2.0), 2.0, 4.0]))),
              Fault(str(rng.choice(valves)), "stuck_valve", float(rng.choice([rng.uniform(0.2, 0.8), 0.9]))),
              Fault(pumps[0], "pump_wear", float(rng.uniform(0.1, 0.6))),
              Fault(str(rng.choice([j.id for j in spec.junctions])), "leak", float(rng.choice([0.02, 0.08, 0.15])))]
    faults = [f for f in faults if rng.random() < 0.7]
    mode = rng.integers(4)
    if mode == 0:
        sp = experiment_setpoint(spec, "pump_off_static")
    elif mode == 1 and spec.branches:
        sp = experiment_setpoint(spec, "isolate_branch", {"branch": spec.branches[int(rng.integers(len(spec.branches)))].id})
    else:
        sp = Setpoint(speed={p: float(rng.uniform(0.5, 1.2)) for p in pumps},
                      valve={v: float(rng.uniform(0.1, 1.0)) for v in valves})
    return spec, tuple(faults), sp


def f2() -> dict:
    rng = np.random.default_rng(2024)
    iters, times, resid, failures = [], [], [], 0
    for _ in range(10_000):
        spec, faults, sp = random_case(rng)
        t0 = time.perf_counter()
        try:
            st = solve(spec, apply_faults(spec, faults), sp)
        except Exception:  # noqa: BLE001 - counted, reported
            failures += 1
            continue
        times.append(time.perf_counter() - t0)
        iters.append(st.iterations)
        resid.append(st.residual)
    rng = np.random.default_rng(7)
    diff_h, diff_q = [], []
    for _ in range(1000):
        spec, faults, sp = random_case(rng)
        mine = solve(spec, apply_faults(spec, faults), sp)
        ref = ref_solve(spec.to_dict(), [vars(f) for f in faults], dict(sp.speed), dict(sp.valve), set(sp.closed))
        diff_h.append(max(abs(mine.h[j] - ref["h"][j]) for j in ref["h"]))
        diff_q.append(max(abs(mine.q[k] - ref["q"][k]) / max(abs(ref["q"][k]), 1e-3) for k in ref["q"]))
    mono_viol = 0
    for seed in range(300):
        spec = generate_plant(seed, TIERS[seed % 3])
        pipes = [ln.id for ln in spec.links if ln.kind == "pipe"]
        p = pipes[seed % len(pipes)]
        v = [ln.id for ln in spec.links if ln.kind == "valve"][0]
        base = solve(spec)
        mono_viol += abs(solve(spec, apply_faults(spec, (Fault(p, "fouling", 0.5),))).q[p]) >= abs(base.q[p])
        lo, hi = (solve(spec, setpoint=Setpoint(valve={v: t})).q[v] for t in (0.5, 0.8))
        mono_viol += abs(hi) <= abs(lo)
    return {"analytic": analytic(), "random_solves": 10_000, "failures": failures, "iter_max": max(iters),
            "iter_hist": dict(sorted(Counter(iters).items())), "residual_max": max(resid),
            "ms_median": 1000 * statistics.median(times), "ms_p99": 1000 * float(np.percentile(times, 99)),
            "times_ms": [1000 * t for t in times[:3000]], "diff_cases": 1000, "diff_h_max": max(diff_h),
            "diff_q_rel_max": max(diff_q), "diff_h": diff_h, "monotonicity_cases": 600, "monotonicity_violations": mono_viol}


def f3() -> dict:
    spec = generate_plant(7, "medium")
    links = {ln.id: ln for ln in spec.links}
    pipe = next(ln for ln in spec.links if ln.kind == "pipe" and ln.frm.startswith("J") and ln.to.startswith("J"))
    st = solve(spec, apply_faults(spec, (Fault(pipe.id, "fouling", 1.0),)))
    d = st.h[pipe.frm] - st.h[pipe.to]
    fouling_err = abs(d / st.q[pipe.id] ** 2 - 2.0 * pipe.r) / (2.0 * pipe.r)
    valve = next(ln.id for ln in spec.links if ln.kind == "valve")
    stuck = apply_faults(spec, (Fault(valve, "stuck_valve", 0.5),))
    a, b = (solve(spec, stuck, Setpoint(valve={valve: t})) for t in (1.0, 0.3))
    stuck_change = max(abs(a.h[k] - b.h[k]) for k in a.h)
    pump = next(ln for ln in spec.links if ln.kind == "pump")
    wear_err, wear_exact_err, vibs, etas, deadhead_q = 0.0, 0.0, [], [], None
    for w in np.linspace(0.0, 0.6, 13):
        s = solve(spec, apply_faults(spec, (Fault(pump.id, "pump_wear", float(w)),)))
        rise, q = s.h[pump.to] - s.h[pump.frm], s.q[pump.id]
        y = pump.b * q * q  # exact smoothed law: rise = (1-w) a - x, sp(x) = b q^2
        wear_exact_err = max(wear_exact_err, abs(rise - ((1 - w) * pump.a - (y - 1e-12 / (4 * y)))) / rise)
        if w <= 0.35 + 1e-9:  # plan identity (1-w) a s^2 - b q^2 holds while the pump delivers flow
            wear_err = max(wear_err, abs(rise - ((1 - w) * pump.a - pump.b * q * q)) / rise)
            vibs.append(s.vib[pump.id])
            etas.append(9.81 * q * rise / (1000 * s.power[pump.id]))
        elif deadhead_q is None and q < 1e-3:
            deadhead_q = (float(w), q)
    j = next(x for x in spec.junctions if x.side == "discharge").id
    leak = solve(spec, apply_faults(spec, (Fault(j, "leak", 0.05),)))
    net = compile_net(spec)
    inflow = sum(leak.q[ln.id] for ln in spec.links if ln.to == j) - sum(leak.q[ln.id] for ln in spec.links if ln.frm == j)
    c = leak_coefficients(spec, net, {j: 0.05})[net.jindex[j]]
    leak_err = abs(inflow - c * np.sqrt(leak.h[j] - next(x.z for x in spec.junctions if x.id == j)))

    sensor = spec.sensors[0]
    biased = Instance(spec, (Fault(sensor.id, "sensor_bias", 6.0, sign=-1),), None, 11)
    clean = Instance(spec, (), None, 11)
    stuck_inst = Instance(spec, (Fault(sensor.id, "sensor_stuck", None, value=3.25),), None, 11)
    rng = np.random.default_rng(3)
    bias_err, stuck_vals = 0.0, set()
    for k in range(10):
        sp = Setpoint(speed={pump.id: float(rng.uniform(0.5, 1.2))}, valve={valve: float(rng.uniform(0.1, 1))})
        phys = physical_readings(spec, solve(spec, setpoint=sp))
        delta = observe(biased, phys, [sensor.id], k)[sensor.id] - observe(clean, phys, [sensor.id], k)[sensor.id]
        bias_err = max(bias_err, abs(delta - (-6.0 * sensor.sigma)))
        stuck_vals.add(observe(stuck_inst, phys, [sensor.id], k)[sensor.id])

    fires = []
    for trial in range(1000):
        r, spent, n = np.random.default_rng(trial), 0.0, 0
        while spent < 40:
            cost = float(r.choice([1, 4, 6, 8]))
            n += drift_fires(spent, cost)
            spent += cost
        fires.append(n)

    pairs = eligible_pairs(spec)
    counts = Counter()
    for i in range(10_000):
        f = sample_faults(spec, "easy", np.random.default_rng([7, 99, i]))[0]
        counts[(f.component, f.type)] += 1
    obs = [counts[p] for p in pairs]
    chi = chisquare(obs)

    kpi_min, multi, in_range, tier_counts = 1.0, 0, True, {}
    for tier in TIERS:
        n_ok = 0
        for seed in range(100):
            sp_ = generate_plant(seed, tier)
            fs = sample_faults(sp_, tier, stream(seed, tier, FAULTS))
            comps = [f.component for f in fs]
            multi += len(comps) != len(set(comps))
            for f in fs:
                if f.type in PHYSICAL_TYPES:
                    kpi_min = min(kpi_min, impact(sp_, (f,)))
                if f.magnitude is not None:
                    lo, hi = MAGNITUDE_RANGES[f.type]
                    in_range &= lo - 1e-9 <= f.magnitude <= hi + 1e-9
            n_ok += Counter(f.type in PHYSICAL_TYPES for f in fs) == Counter({True: {"easy": 1, "medium": 2, "hard": 3}[tier],
                                                                             False: {"easy": 0, "medium": 1, "hard": 2}[tier]}) + Counter()
        tier_counts[tier] = n_ok
    return {"fouling_r_rel_err": fouling_err, "stuck_valve_max_head_change": stuck_change, "wear_head_rel_err": wear_err,
            "wear_exact_law_rel_err": wear_exact_err, "wear_deadhead": deadhead_q,
            "vib_monotone_in_w": bool(np.all(np.diff(vibs) > 0)), "eta_decreasing_in_w": bool(np.all(np.diff(etas) < 0)),
            "leak_imbalance_abs_err": float(leak_err), "bias_offset_max_err": bias_err,
            "stuck_distinct_values": len(stuck_vals), "drift_fires_counts": dict(Counter(fires)),
            "chi_square_draws": 10_000, "chi_square_cells": len(pairs), "chi_square_p": float(chi.pvalue),
            "chi_counts": obs, "kpi_min_individual_impact": kpi_min, "multi_fault_components": multi,
            "magnitudes_in_range": bool(in_range), "tier_counts_ok_of_100": tier_counts}


def f4() -> dict:
    """Plan valid-when: all statistical tests pass at p > 0.01 on 20 seeds. The bands from the plan's
    'Verifies' list are reported too; at N = 1000 the 5% std band is only about 2.2 standard errors wide,
    so roughly 1 seed in 40 misses it by chance."""
    res = {"mean_ok": 0, "std_ok": 0, "acf_ok": 0, "repeat_ok": 0, "bias_op_ok": 0, "seeds": 20,
           "p_mean": [], "p_var": [], "p_acf": []}
    z_all = []
    for seed in range(20):
        spec = generate_plant(seed, TIERS[seed % 3])
        s = spec.sensors[seed % len(spec.sensors)]
        inst = Instance(spec, (Fault(s.id, "sensor_bias", 5.0),), None, 1000 + seed)
        phys = physical_readings(spec, solve(spec))
        n = 1000
        x = np.array([observe(inst, phys, [s.id], k)[s.id] for k in range(n)])
        truth = phys[s.id] + 5.0 * s.sigma
        z = (x - truth) / s.sigma
        z_all.extend(z.tolist())
        res["mean_ok"] += abs(x.mean() - truth) <= 3 * s.sigma / np.sqrt(n)
        res["std_ok"] += abs(x.std(ddof=1) - s.sigma) <= 0.05 * s.sigma
        r1 = np.corrcoef(z[:-1], z[1:])[0, 1]
        res["acf_ok"] += abs(r1) < 0.1
        chi = (n - 1) * z.var(ddof=1)
        res["p_mean"].append(float(2 * norm.sf(abs(z.mean()) * np.sqrt(n))))
        res["p_var"].append(float(2 * min(chi2.cdf(chi, n - 1), chi2.sf(chi, n - 1))))
        res["p_acf"].append(float(2 * norm.sf(abs(r1) * np.sqrt(n))))
        res["repeat_ok"] += observe(inst, phys, [s.id], 5)[s.id] == observe(inst, phys, [s.id], 5)[s.id]
        phys2 = physical_readings(spec, solve(spec, setpoint=experiment_setpoint(spec, "pump_off_static")))
        clean = Instance(spec, (), None, 1000 + seed)
        offsets = [observe(inst, p, [s.id], 9)[s.id] - observe(clean, p, [s.id], 9)[s.id] for p in (phys, phys2)]
        res["bias_op_ok"] += abs(offsets[0] - offsets[1]) < 1e-9
    res["ks_pooled_p"] = float(kstest(z_all, "norm").pvalue)
    res["z_sample"] = z_all[::4]
    return res


def f5(n_seeds: int = 100) -> dict:
    """Adopted rule (option E): cosine <= 0.95 or >= 5 sigma separation; drift cut. Also reports the plan's
    cosine-only rule on the same draws for reference."""
    out = {"seeds_per_tier": n_seeds, "discard": {}, "discard_plan_cosine": {}, "by_type": {}, "reasons": {},
           "spend": {}, "ms": {}, "min_separation_accepted": {}}
    for tier in TIERS:
        acc = acc_plan = 0
        tot, ok, why, spends, ts, seps = Counter(), Counter(), Counter(), [], [], []
        for seed in range(n_seeds):
            spec = generate_plant(seed, tier)
            fs = sample_faults(spec, tier, stream(seed, tier, FAULTS))
            t0 = time.perf_counter()
            cert = C.certify(spec, fs)
            ts.append(time.perf_counter() - t0)
            acc += cert.accepted
            acc_plan += cert.accepted and max(cert.coherence.values()) <= C.COHERENCE_MAX
            for f in fs:
                tot[f.type] += 1
                ok[f.type] += cert.accepted
            for r in cert.reasons:
                w = r.split()
                why[w[0] + " " + (w[1].split(":")[-1] if w[0] in ("CRB", "coherence") else w[1])] += 1
            if cert.accepted:
                spends.append(sum(n * (1 if lbl.startswith("read:") else C.COSTS[lbl.split(":")[0]])
                                  for lbl, n in cert.spend.items()))
                seps += [cert.separation[k] for k, c in cert.coherence.items() if c > C.COHERENCE_MAX]
        out["discard"][tier] = 1 - acc / n_seeds
        out["discard_plan_cosine"][tier] = 1 - acc_plan / n_seeds
        out["by_type"][tier] = {t: [ok[t], tot[t]] for t in sorted(tot)}
        out["reasons"][tier] = dict(why)
        out["spend"][tier] = [float(np.median(spends)), float(max(spends))] if spends else None
        out["ms"][tier] = 1000 * statistics.median(ts)
        out["min_separation_accepted"][tier] = min(seps) if seps else None
    out["make_instance_deterministic"] = all(make_instance(s, t).faults == make_instance(s, t).faults
                                             for s in (1, 2) for t in TIERS)
    return out


def findings() -> dict:
    """Static-test leak signature (manual input): tanks backfeed a leak when the pumps stop."""
    out = {}
    for f in (0.02, 0.08):
        sag, flow = [], []
        for seed in range(30):
            spec = generate_plant(seed, "medium")
            dis = [j for j in spec.junctions if j.side == "discharge"]
            j = dis[len(dis) // 2]
            st = solve(spec, apply_faults(spec, (Fault(j.id, "leak", f),)), experiment_setpoint(spec, "pump_off_static"))
            ph, z = physical_readings(spec, st), {x.id: x for x in spec.junctions}
            sag.append(max(abs(ph[s.id] - (spec.tank_head - z[s.target].z)) / s.sigma for s in spec.sensors
                           if s.kind == "pressure" and z[s.target].side == "discharge"))
            flow.append(max(abs(ph[s.id]) / s.sigma for s in spec.sensors if s.kind == "flow"))
        out[f"leak_{f}"] = {"pt_sag_sigma_max": max(sag), "pt_sag_sigma_median": statistics.median(sag),
                            "ft_sigma_max": max(flow), "ft_sigma_median": statistics.median(flow)}
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    result = {}
    for name, fn in (("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4), ("F5", f5), ("findings", findings)):
        t0 = time.perf_counter()
        result[name] = fn()
        print(f"{name} done in {time.perf_counter() - t0:.1f}s", flush=True)
    (OUT / "step2_validation.json").write_text(json.dumps(result, default=float))
    print("wrote", OUT / "step2_validation.json")


if __name__ == "__main__":
    main()
