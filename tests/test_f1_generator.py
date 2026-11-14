"""F1 plant generator (plan section 8 F1; sim_spec.md sections 1, 2, 3, 5, 7, 8, 9).

Plant structure is read only through `spec.to_dict()` (section 8), because section 9
defines no PlantSpec attributes. The fast suite checks 200 plants (tiers cycle with the
seed); the slow test runs the plan's 1000 seeds for every tier.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from diagnoselab.sim.generator import generate_plant, spec_hash
from diagnoselab.sim.model import PlantSpec
from diagnoselab.sim.solver import kpi, solve

ROOT = Path(__file__).parents[1]
TIERS = ("easy", "medium", "hard")
N_FAST = 200  # plants in the fast suite (seed -> tier = TIERS[seed % 3])
N_FULL = 1000  # seeds per tier in the slow suite (plan: 1000 of 1000)
HEAD_TOL = 1e-6  # m
FLOW_REL = 1e-6
RESIDUAL_MAX = 1e-8
MAX_ITER = 30
GEN_MEDIAN_MAX = 0.050  # s
SENSOR_PREFIX = {"pressure": "PT", "flow": "FT", "power": "WT", "vibration": "VT"}
SIZE = {  # sim_spec section 7 structure table, inclusive
    "junctions": {"easy": (6, 8), "medium": (8, 10), "hard": (10, 12)},
    "pumps": {"easy": (1, 1), "medium": (1, 2), "hard": (2, 2)},
    "outlets": {"easy": (2, 3), "medium": (2, 4), "hard": (3, 4)},
    "chords": {"easy": (0, 0), "medium": (1, 1), "hard": (1, 2)},
    "valves": {"easy": (2, 4), "medium": (2, 4), "hard": (2, 4)},
}


class Plant(NamedTuple):
    seed: int
    tier: str
    spec: PlantSpec
    d: dict
    seconds: float


def build(pairs: Iterable[tuple[int, str]]) -> list[Plant]:
    plants = []
    for seed, tier in pairs:
        t0 = time.perf_counter()
        spec = generate_plant(seed, tier)
        plants.append(Plant(seed, tier, spec, spec.to_dict(), time.perf_counter() - t0))
    return plants


@pytest.fixture(scope="module")
def plants() -> list[Plant]:
    generate_plant(0, "easy")  # warm-up so import cost is not timed
    return build((seed, TIERS[seed % 3]) for seed in range(N_FAST))


# --- helpers -------------------------------------------------------------------------

def links_of(d: dict, kind: str) -> list[dict]:
    return [link for link in d["links"] if link["kind"] == kind]


def reachable(starts: Iterable[str], step: Callable[[str], Iterable[str]]) -> set[str]:
    seen, todo = set(starts), deque(starts)
    while todo:
        for nxt in step(todo.popleft()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def outlet_inflow(d: dict, q: dict[str, float]) -> dict[str, float]:
    flow = {o["id"]: 0.0 for o in d["outlets"]}
    for link in d["links"]:
        if link["to"] in flow:
            flow[link["to"]] += q[link["id"]]
        if link["frm"] in flow:
            flow[link["frm"]] -= q[link["id"]]
    return flow


def sizes(d: dict) -> dict[str, int]:
    """Counts of section 7 items; chords = cyclomatic number of the junction-only graph.

    Links to fixed nodes are left out, so two pumps sharing one source do not form a cycle.
    """
    junctions = {j["id"] for j in d["junctions"]}
    inner = [lk for lk in d["links"] if lk["frm"] in junctions and lk["to"] in junctions]
    adj: dict[str, set[str]] = {j: set() for j in junctions}
    for lk in inner:
        adj[lk["frm"]].add(lk["to"])
        adj[lk["to"]].add(lk["frm"])
    components, left = 0, set(junctions)
    while left:
        left -= reachable([next(iter(left))], adj.__getitem__)
        components += 1
    return {"junctions": len(junctions), "pumps": len(links_of(d, "pump")), "outlets": len(d["outlets"]),
            "chords": len(inner) - len(junctions) + components, "valves": len(links_of(d, "valve"))}


def floats(obj: Any) -> Iterable[float]:
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from floats(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from floats(value)


def close(got: float, want: float, rel: float) -> bool:
    return abs(got - want) <= rel * abs(want) + 1e-12


def content_hash(d: dict) -> str:
    """Hash of the plant without its seed/tier labels (spec_hash includes the seed)."""
    body = {k: v for k, v in d.items() if k not in ("seed", "tier")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


# --- per-plant checks (each returns a list of error strings) --------------------------

def check_connected(p: Plant) -> list[str]:
    d = p.d
    sources = {s["id"] for s in d["sources"]}
    junctions = {j["id"] for j in d["junctions"]}
    nodes = sources | junctions | {o["id"] for o in d["outlets"]}
    both: dict[str, set[str]] = {n: set() for n in nodes}
    forward: dict[str, set[str]] = {n: set() for n in nodes}  # flow-feasible: pumps only frm -> to
    for lk in d["links"]:
        both[lk["frm"]].add(lk["to"])
        both[lk["to"]].add(lk["frm"])
        forward[lk["frm"]].add(lk["to"])
        if lk["kind"] != "pump":
            forward[lk["to"]].add(lk["frm"])
    errors = []
    loose = nodes - reachable([next(iter(sorted(nodes)))], both.__getitem__)
    if loose:
        errors.append(f"graph not connected; unreachable {sorted(loose)}")
    # paths may not pass through a fixed-head node other than the starting source
    fed = reachable(sources, lambda n: forward[n] if n in sources or n in junctions else ())
    unfed = {o["id"] for o in d["outlets"]} - fed
    if unfed:
        errors.append(f"outlets not reachable from any source: {sorted(unfed)}")
    return errors


def check_structure(p: Plant) -> list[str]:
    d, errors = p.d, []
    side = {j["id"]: j["side"] for j in d["junctions"]}
    sources = {s["id"] for s in d["sources"]}
    pumps = links_of(d, "pump")
    for pu in pumps:
        if side.get(pu["frm"]) != "suction" or side.get(pu["to"]) != "discharge":
            errors.append(f"{pu['id']} must run suction junction -> discharge junction")
    for j in (j for j, s in side.items() if s == "suction"):
        touching = [lk for lk in d["links"] if j in (lk["frm"], lk["to"])]
        source_pipes = [lk for lk in touching if lk["kind"] == "pipe" and {lk["frm"], lk["to"]} - {j} <= sources]
        if sorted(lk["kind"] for lk in touching) != ["pipe", "pump"] or len(source_pipes) != 1:
            errors.append(f"suction junction {j} must connect only to its source pipe and its pump")
    if len(sources) not in (1, len(pumps)):
        errors.append(f"{len(sources)} sources for {len(pumps)} pumps (want 1 or one per pump)")
    for v in links_of(d, "valve"):
        if any(side.get(n, "discharge") != "discharge" for n in (v["frm"], v["to"])):
            errors.append(f"valve {v['id']} is not on the discharge side")
    for s in d["sensors"]:
        if s["id"] != f"{SENSOR_PREFIX[s['kind']]}-{s['target']}":
            errors.append(f"sensor id {s['id']} does not follow section 1 for {s['kind']} on {s['target']}")
    return errors


def check_sizes(p: Plant) -> list[str]:
    got = sizes(p.d)
    return [f"{item} = {got[item]} outside {rng[p.tier]}" for item, rng in SIZE.items()
            if not rng[p.tier][0] <= got[item] <= rng[p.tier][1]]


def check_design_point(p: Plant) -> list[str]:
    d, des, errors = p.d, p.d["design"], []
    state = solve(p.spec)
    if state.iterations > MAX_ITER or not state.residual < RESIDUAL_MAX:
        errors.append(f"nominal solve: {state.iterations} iterations, residual {state.residual:.2e}")
    for j in d["junctions"]:
        want = des["h"].get(j["id"])
        if want is None:
            errors.append(f"design.h lacks {j['id']}")
            continue
        if abs(state.h[j["id"]] - want) > HEAD_TOL:
            errors.append(f"h[{j['id']}] {state.h[j['id']]:.9f} vs design {want:.9f}")
        if want - j["z"] <= 0:
            errors.append(f"p_design of {j['id']} is not positive")
    for lk in d["links"]:
        if not close(state.q[lk["id"]], des["q"][lk["id"]], FLOW_REL):
            errors.append(f"q[{lk['id']}] {state.q[lk['id']]:.9f} vs design {des['q'][lk['id']]:.9f}")
    pumps = [pu["id"] for pu in links_of(d, "pump")]
    for pu in pumps:
        if not close(state.power[pu], des["P"][pu], FLOW_REL) or not close(state.vib[pu], des["vib"][pu], FLOW_REL):
            errors.append(f"{pu}: power/vibration differ from design")
    if not close(sum(des["q"][pu] for pu in pumps), des["Q_pump"], FLOW_REL):
        errors.append("design.Q_pump is not the sum of the design pump flows")
    se = sum(state.power[pu] for pu in pumps) / (3.6 * sum(outlet_inflow(d, state.q).values()))
    if not close(se, des["SE"], FLOW_REL):
        errors.append(f"SE {se} vs design {des['SE']}")
    return errors


def check_q_req(p: Plant) -> list[str]:
    state = solve(p.spec)
    inflow = outlet_inflow(p.d, state.q)
    errors = [f"outlet {o['id']} delivers {inflow[o['id']] / o['q_req']:.4f} x q_req"
              for o in p.d["outlets"] if abs(inflow[o["id"]] / o["q_req"] - 1.0) > 0.02]
    j = kpi(p.spec, state)
    if abs(j + 1.0) > 1e-6:  # section 3: no shortfall and SE == SE_design at the design point
        errors.append(f"healthy KPI {j} != -1")
    return errors


def check_pump_bep(p: Plant) -> list[str]:
    state = solve(p.spec)
    return [f"{pu['id']} runs at {state.q[pu['id']] / pu['q_bep']:.3f} x q_bep"
            for pu in links_of(p.d, "pump") if not 0.7 <= state.q[pu["id"]] / pu["q_bep"] <= 1.2]


def check_pipe_coverage(p: Plant) -> list[str]:
    pt = {s["target"] for s in p.d["sensors"] if s["kind"] == "pressure"}
    ft = {s["target"] for s in p.d["sensors"] if s["kind"] == "flow"}
    return [f"pipe {lk['id']} has no PT on an end junction and no FT"
            for lk in links_of(p.d, "pipe") if lk["id"] not in ft and not {lk["frm"], lk["to"]} & pt]


def check_pump_sensors(p: Plant) -> list[str]:
    have = {(s["kind"], s["target"]) for s in p.d["sensors"]}
    errors = []
    for pu in links_of(p.d, "pump"):
        need = {("pressure", pu["frm"]), ("pressure", pu["to"]), ("power", pu["id"]), ("vibration", pu["id"])}
        if need - have:
            errors.append(f"{pu['id']} lacks {sorted(need - have)}")
    return errors


def check_tank_head(p: Plant) -> list[str]:
    return [f"source {s['id']} head {s['head']} above tank_head {p.d['tank_head']}"
            for s in p.d["sources"] if s["head"] > p.d["tank_head"]]


def check_rounded(p: Plant) -> list[str]:
    bad = [x for x in floats(p.d) if float(f"{x:.9g}") != x]
    return [f"{len(bad)} floats not rounded to 9 significant digits, e.g. {bad[:3]}"] if bad else []


ALL_CHECKS = (check_connected, check_structure, check_sizes, check_design_point, check_q_req,
              check_pump_bep, check_pipe_coverage, check_pump_sensors, check_tank_head, check_rounded)


def run_check(plants: list[Plant], check: Callable[[Plant], list[str]]) -> None:
    errors = [f"{p.tier}/seed {p.seed}: {e}" for p in plants for e in check(p)]
    assert not errors, f"{len(errors)} failure(s); first: {errors[:5]}"


def assert_size_coverage(plants: list[Plant]) -> None:
    for tier in TIERS:
        seen = [sizes(p.d) for p in plants if p.tier == tier]
        for item, rng in SIZE.items():
            lo, hi = rng[tier]
            missing = set(range(lo, hi + 1)) - {s[item] for s in seen}
            assert not missing, f"{tier}: {item} never takes values {sorted(missing)}"


def assert_unique_hashes(plants: list[Plant]) -> None:
    hashes = [spec_hash(p.spec) for p in plants]
    assert len(set(hashes)) == len(hashes), "duplicate spec_hash"
    contents = [content_hash(p.d) for p in plants]
    assert len(set(contents)) == len(contents), "two seeds produced the same plant (seed/tier ignored)"


# --- fast tests ------------------------------------------------------------------------

def test_graph_connected_and_outlets_reachable(plants: list[Plant]) -> None:
    """F1 valid-when: graph connected; every outlet reachable from a source (pumps one-way)."""
    run_check(plants, check_connected)


def test_structure_follows_section_2_and_7(plants: list[Plant]) -> None:
    """F1 does: pumps suction->discharge, suction junction = source pipe + pump, sources 1 or per pump,
    valves on the discharge side, sensor ids per section 1."""
    run_check(plants, check_structure)


def test_nominal_solve_reproduces_design_point(plants: list[Plant]) -> None:
    """F1 valid-when: nominal solve converges and reproduces the design point
    (heads within 1e-6 m, flows/power/vibration/SE within 1e-6 relative)."""
    run_check(plants, check_design_point)


def test_outlets_meet_q_req_within_2pct(plants: list[Plant]) -> None:
    """F1 valid-when: each outlet meets q_req within 2%; healthy KPI is -1 (sim_spec section 3)."""
    run_check(plants, check_q_req)


def test_pump_flow_70_to_120pct_of_bep(plants: list[Plant]) -> None:
    """F1 valid-when: each pump runs at 70 to 120% of its best-efficiency flow."""
    run_check(plants, check_pump_bep)


def test_every_pipe_within_one_hop_of_a_sensor(plants: list[Plant]) -> None:
    """F1 valid-when: every pipe within one hop of a sensor (PT on an end junction or FT on itself)."""
    run_check(plants, check_pipe_coverage)


def test_pump_pressure_sensors_on_both_sides(plants: list[Plant]) -> None:
    """F1 valid-when: pump pressure sensors present (PT suction and discharge; WT and VT per pump)."""
    run_check(plants, check_pump_sensors)


def test_tank_head_not_below_any_source_head(plants: list[Plant]) -> None:
    """F1 / sim_spec section 2: the generator guarantees tank_head >= max(source heads)."""
    run_check(plants, check_tank_head)


def test_size_ranges_per_tier(plants: list[Plant]) -> None:
    """F1 valid-when: size histograms stay inside and cover the configured ranges (sim_spec section 7)."""
    run_check(plants, check_sizes)
    assert_size_coverage(plants)


def test_floats_rounded_to_9_significant_digits(plants: list[Plant]) -> None:
    """F1 valid-when: spec hashes identical across builds (requires every float rounded to 9 digits)."""
    run_check(plants, check_rounded)


@pytest.mark.parametrize("tier", TIERS)
def test_same_seed_and_tier_give_same_hash(tier: str) -> None:
    """F1 valid-when: identical (seed, tier) gives an identical spec hash; to_dict/from_dict round-trips."""
    a, b = generate_plant(11, tier), generate_plant(11, tier)
    assert spec_hash(a) == spec_hash(b)
    assert a.to_dict() == b.to_dict()
    assert spec_hash(PlantSpec.from_dict(a.to_dict())) == spec_hash(a)
    h = spec_hash(a)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_same_hash_across_subprocesses() -> None:
    """F1 valid-when: spec hashes identical across processes (fresh interpreter, other PYTHONHASHSEED)."""
    keys = [(3, "easy"), (4, "medium"), (5, "hard")]
    want = " ".join(spec_hash(generate_plant(s, t)) for s, t in keys)
    script = ("from diagnoselab.sim.generator import generate_plant, spec_hash\n"
              f"print(' '.join(spec_hash(generate_plant(s, t)) for s, t in {keys!r}))")
    for hash_seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=120, check=True)
        assert out.stdout.strip() == want, f"PYTHONHASHSEED={hash_seed}: hashes differ"


def test_no_duplicate_hashes_across_seeds(plants: list[Plant]) -> None:
    """F1 valid-when: distinct seeds yield no duplicate hashes (also with seed/tier removed)."""
    assert_unique_hashes(plants)


def test_generation_median_under_50ms(plants: list[Plant]) -> None:
    """F1 expect: under 50 ms per plant (median)."""
    median = statistics.median(p.seconds for p in plants)
    assert median < GEN_MEDIAN_MAX, f"median generation {median * 1e3:.1f} ms"


# --- slow: plan counts ------------------------------------------------------------------

@pytest.mark.slow
def test_full_plan_counts_1000_seeds_per_tier() -> None:
    """F1 valid-when: 1000 of 1000 emitted plants pass every check (per tier), no duplicate hashes,
    size histograms cover the configured ranges, median generation < 50 ms."""
    generate_plant(0, "easy")
    plants = build((seed, tier) for tier in TIERS for seed in range(N_FULL))
    for check in ALL_CHECKS:
        run_check(plants, check)
    assert_size_coverage(plants)
    assert_unique_hashes(plants)
    assert statistics.median(p.seconds for p in plants) < GEN_MEDIAN_MAX
