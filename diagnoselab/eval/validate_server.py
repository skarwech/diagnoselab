"""Measure the F6-F14 valid-when conditions (plan section 8) -> results/step3_validation.json.

Run: uv run python -m diagnoselab.eval.validate_server   (a few minutes; F14 needs results/isolation.json
from `make isolation`).
"""

from __future__ import annotations

import json
import math
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from diagnoselab.server import tools as T
from diagnoselab.server.episode import BUDGET, CALL_CAP, SIMULATE_CAP, Episode, ToolRejected
from diagnoselab.server.replay import replay
from diagnoselab.sim.experiments import experiment_sensors, experiment_setpoint
from diagnoselab.sim.faults import apply_faults, sample_faults
from diagnoselab.sim.generator import generate_plant, stream
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Fault, Instance
from diagnoselab.sim.sensors import physical_readings
from diagnoselab.sim.solver import solve

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("easy", "medium", "hard")


# --- helpers ------------------------------------------------------------------------------------
def twin(seed: int, tier: str) -> tuple[Instance, Instance]:
    """Two instances with the same nominal plant and different hidden faults."""
    a = make_instance(seed, tier)
    other = sample_faults(a.spec, tier, np.random.default_rng([seed, 777]))
    return a, Instance(a.spec, other, None, a.noise_seed + 1)


def safe_call(ep: Episode, tool, args):
    try:
        return True, ep.call(tool, args)
    except ToolRejected as e:
        return False, f"{e.code}: {e.message}"


def fuzz_args(rng: random.Random, spec) -> object:
    ids = [s.id for s in spec.sensors] + [ln.id for ln in spec.links] + [j.id for j in spec.junctions]
    atoms = [None, True, 0, -1, 1e308, float("nan"), float("inf"), -0.0, 0.7, 1.2, 0.69999, 1.20001, 0.1, 0.0999,
             "", "x" * 17000, rng.choice(ids), "PT-J1", "speed_step", "pump_off_static", [], {}, [1, 2], {"a": 1}]
    keys = ["sensor_id", "type", "args", "topic", "overrides", "setpoint", "sensors", "diagnosis", "repairs",
            "s", "theta", "valve", "pump", "branch", "evil", "__proto__", "component", "magnitude"]

    def deep(n):
        node = {}
        for _ in range(n):
            node = {"k": node}
        return node

    pick = rng.random()
    if pick < 0.05:
        return deep(rng.choice([50, 2000]))
    if pick < 0.10:
        return rng.choice([None, [], "str", 3, [{"a": 1}]])
    return {rng.choice(keys): rng.choice(atoms + [{"s": rng.choice(atoms)}, [rng.choice(atoms)]])
            for _ in range(rng.randint(0, 4))}


TOOLS = list(T.TOOL_SCHEMAS) + ["nope", "", "READ_SENSOR"]


# --- F6 -----------------------------------------------------------------------------------------
FREE_TOOLS = ("get_topology", "read_manual", "simulate", "budget_status")


def _same_outcome(tool, ok_a, ra, ok_b, rb) -> bool:
    """Hidden faults must not change acceptance, error text, or any free tool's answer."""
    if ok_a != ok_b:
        return False
    if not ok_a:
        return ra == rb
    if tool in FREE_TOOLS:
        strip = lambda r: {k: v for k, v in r.items() if k != "time_used_s"}  # noqa: E731
        return strip(ra) == strip(rb)
    return True  # costed tools read the true plant; their values legitimately differ


def f6(n_calls: int = 10_000) -> dict:
    a, b = twin(5, "medium")
    ea, eb = Episode(a), Episode(b)
    rng = random.Random(6)
    crashes = state_changes = diff_mismatch = rejected = 0
    for _ in range(n_calls):
        tool, args = rng.choice(TOOLS), fuzz_args(rng, a.spec)
        if tool == "submit" or (tool == "read_sensor" and rng.random() < 0.02):
            continue  # submissions are exercised in F12; keep the fuzzed episode running
        before = ea.state_hash()
        try:
            ok_a, ra = safe_call(ea, tool, args)
            ok_b, rb = safe_call(eb, tool, args)
        except Exception:  # noqa: BLE001 - any non-ToolRejected exception is a crash
            crashes += 1
            continue
        if not ok_a:
            rejected += 1
            state_changes += ea.state_hash() != before and ea.status != "FINALIZED"
        diff_mismatch += not _same_outcome(tool, ok_a, ra, ok_b, rb)
        if ea.status != "RUNNING" and ea.status != "INIT":
            ea, eb = Episode(a), Episode(b)
    # cross-session: interleaved episodes give the same results as separate runs
    seq = [("read_sensor", {"sensor_id": s.id}) for s in a.spec.sensors[:10]] + [
        ("run_experiment", {"type": "pump_off_static", "args": {}})]
    solo = [safe_call(e, t, x) for e in [Episode(a)] for t, x in seq]
    e1, e2 = Episode(a), Episode(b)
    inter = []
    for t, x in seq:
        inter.append(safe_call(e1, t, x))
        safe_call(e2, t, x)
    return {"fuzz_calls": n_calls, "rejected": rejected, "crashes": crashes,
            "state_changes_after_rejection": state_changes, "differential_mismatches": diff_mismatch,
            "cross_session_identical": solo == inter, "http": f6_http()}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def f6_http() -> dict:
    """Raw malformed HTTP bodies plus per-tool latency through the real MCP transport."""
    import anyio
    import httpx
    from mcp import Client

    port, state = _free_port(), Path(tempfile.mkdtemp())
    proc = subprocess.Popen([sys.executable, "-m", "diagnoselab.server.app", "--host", "127.0.0.1", "--port",
                             str(port), "--seed", "5", "--tier", "medium", "--state", str(state)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        url = f"http://127.0.0.1:{port}/mcp"
        bodies = ["{", "null", "[]", '{"jsonrpc":"2.0"}', '{"jsonrpc":"2.0","id":1,"method":"tools/call"}',
                  '{"jsonrpc":"1.0","id":1,"method":"initialize"}', "x" * 200_000, '{"a":' * 5000, "\x00\xff"]
        codes = []
        with httpx.Client(timeout=10) as http:
            for i in range(600):
                body = bodies[i % len(bodies)]
                r = http.post(url, content=body.encode("utf-8", "replace"),
                              headers={"content-type": "application/json", "accept": "application/json, text/event-stream"})
                codes.append(r.status_code)
        alive = proc.poll() is None
        lat: dict[str, list[float]] = {}

        async def measure():
            async with Client(url) as c:
                calls = [("get_topology", {}), ("read_manual", {"topic": "physics"}), ("budget_status", {}),
                         ("simulate", {"setpoint": {"experiment": "pump_off_static"}}),
                         ("simulate", {"overrides": [{"component": "J3", "type": "leak", "magnitude": 0.05}]}),
                         ("read_sensor", {"sensor_id": "PT-J1"})]
                for i in range(40):
                    for name, args in calls:
                        if name == "read_sensor" and i >= 30:
                            continue
                        t0 = time.perf_counter()
                        await c.call_tool(name, args)
                        lat.setdefault(name, []).append(1000 * (time.perf_counter() - t0))
                for exp in ({"type": "speed_step", "args": {"s": 0.8}}, {"type": "pump_off_static", "args": {}}):
                    t0 = time.perf_counter()
                    await c.call_tool("run_experiment", exp)
                    lat.setdefault("run_experiment", []).append(1000 * (time.perf_counter() - t0))

        anyio.run(measure)
        return {"raw_requests": len(codes), "status_5xx": sum(c >= 500 for c in codes),
                "status_codes": sorted(set(codes)), "server_alive": alive and proc.poll() is None,
                "p95_ms": {k: float(np.percentile(v, 95)) for k, v in lat.items()}}
    finally:
        proc.kill()


# --- F7 / F8 ------------------------------------------------------------------------------------
def f7_f8() -> dict:
    topo_same = manual_same = empty_same = fidelity_max = 0.0
    n = 0
    for tier in TIERS:
        for seed in range(10):
            a, b = twin(seed, tier)
            ea, eb = Episode(a), Episode(b)
            topo_same += json.dumps(ea.call("get_topology", {})) == json.dumps(eb.call("get_topology", {}))
            manual_same += all(ea.call("read_manual", {"topic": t}) == eb.call("read_manual", {"topic": t})
                               for t in T.TOPICS)
            empty_same += ea.call("simulate", {}) == eb.call("simulate", {})
            ov = [{"component": f.component, "type": f.type, **({"magnitude": f.magnitude} if f.magnitude is not None else {}),
                   **({"sign": f.sign} if f.type == "sensor_bias" else {}),
                   **({"value": f.value} if f.type == "sensor_stuck" else {})} for f in a.faults]
            sps = [{}, {"experiment": "pump_off_static"}, {"experiment": "speed_step", "s": 0.75}]
            sps += [{"experiment": "valve_step", "valve": ln.id, "theta": 0.4} for ln in a.spec.links if ln.kind == "valve"][:1]
            sps += [{"experiment": "isolate_branch", "branch": a.spec.branches[0].id}]
            for sp in sps:
                sim = ea.call("simulate", {"overrides": ov, "setpoint": sp})["readings"]
                tru = ea.true_readings(sp)
                fidelity_max = max(fidelity_max, max(abs(sim[k] - tru[k]) for k in tru))
            n += 1
    ep = Episode(make_instance(1, "easy"))
    for _ in range(SIMULATE_CAP):
        ep.call("simulate", {})
    ok301, msg = safe_call(ep, "simulate", {})
    return {"instances": n, "topology_identical": topo_same / n, "manual_identical": manual_same / n,
            "empty_simulate_identical": empty_same / n, "fidelity_max_abs": fidelity_max,
            "call_301_rejected": (not ok301) and msg.startswith("E_RATE_LIMIT"),
            "simulate_spent": ep.call("budget_status", {})["spent"]}


# --- F9 / F11 / F12 / F13 -----------------------------------------------------------------------
VIOLATIONS = {"over_budget": 0, "rejected_changed_state": 0, "free_tool_cost": 0}
RESPONSES: list = []  # response bodies of the current random episode (F12 score-text check)


def random_episode(rng: random.Random, inst: Instance, state_dir: Path | None = None, n: int = 40) -> Episode:
    """Random valid/invalid call sequence; property violations are counted in VIOLATIONS."""
    ep = Episode(inst, state_dir)
    sensors = [s.id for s in inst.spec.sensors]
    valves = [ln.id for ln in inst.spec.links if ln.kind == "valve"]
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            tool, args = "read_sensor", {"sensor_id": rng.choice(sensors + ["bogus"])}
        elif r < 0.65:
            kind = rng.choice(["speed_step", "valve_step", "isolate_branch", "pump_off_static"])
            args = {"speed_step": {"s": rng.choice([0.7, 0.9, 1.2, 1.3])}, "pump_off_static": {},
                    "valve_step": {"valve": rng.choice(valves), "theta": rng.choice([0.2, 0.5, 1.5])},
                    "isolate_branch": {"branch": inst.spec.branches[0].id}}[kind]
            tool, args = "run_experiment", {"type": kind, "args": args}
        elif r < 0.8:
            tool, args = "simulate", {"setpoint": rng.choice([{}, {"experiment": "pump_off_static"}])}
        elif r < 0.9:
            tool, args = rng.choice(["budget_status", "get_topology"]), {}
        else:
            tool, args = rng.choice(TOOLS), fuzz_args(rng, inst.spec)
            if tool == "submit":
                continue
        before, spent = ep.state_hash(), ep._state["spent"]
        ok, resp = safe_call(ep, tool, args)
        RESPONSES.append(resp)
        VIOLATIONS["over_budget"] += ep._state["spent"] > BUDGET
        VIOLATIONS["rejected_changed_state"] += (not ok and ep.status != "FINALIZED" and ep.state_hash() != before)
        VIOLATIONS["free_tool_cost"] += ok and tool in FREE_TOOLS and ep._state["spent"] != spent
    return ep


def _keys(obj) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj] + [k for v in obj.values() for k in _keys(v)]
    if isinstance(obj, list):
        return [k for v in obj for k in _keys(v)]
    return []


def _mentions_score(response, tool: str) -> bool:
    """A score field anywhere, or the words outside the manual (whose scoring section documents the formula)."""
    if any(w in k.lower() for k in _keys(response) for w in ("score", "reward")):
        return True
    if tool == "read_manual":
        return False
    text = json.dumps(response, default=str).lower()
    return "score" in text or "reward" in text


def f9_f11_f12_f13(n_sequences: int = 100_000) -> dict:
    inst = make_instance(2, "medium")
    rng = random.Random(11)
    t0, done, ledger_ok = time.perf_counter(), 0, 0
    while done < n_sequences:
        ep = random_episode(rng, inst, n=rng.randint(1, 12))
        reads = sum(1 for line in ep.trace if line["tool"] == "read_sensor" and line["ok"])
        exp_cost = sum(line["cost"] for line in ep.trace if line["tool"] == "run_experiment" and line["ok"])
        ledger_ok += ep._state["spent"] == reads + exp_cost
        done += 1
    seconds = time.perf_counter() - t0
    # call cap and time cap
    ep = Episode(inst)
    for _ in range(CALL_CAP):
        ep.call("budget_status", {})
    ok151, msg151 = safe_call(ep, "budget_status", {})
    clock = [0.0]
    ep_t = Episode(inst, clock=lambda: clock[0])
    ep_t.call("budget_status", {})
    clock[0] = 31 * 60
    ok_t, msg_t = safe_call(ep_t, "budget_status", {})
    # F12 lifecycle
    ep = Episode(inst)
    strikes = [safe_call(ep, "submit", {"diagnosis": [{"component": "nope", "type": "leak", "magnitude": 1}],
                                         "repairs": []})[1][:22] for _ in range(4)]
    status_after_strikes = ep.status
    ep = Episode(inst)
    ack = ep.call("submit", {"diagnosis": [], "repairs": []})
    second = safe_call(ep, "submit", {"diagnosis": [], "repairs": []})
    post = safe_call(ep, "read_sensor", {"sensor_id": inst.spec.sensors[0].id})
    leaked = 0
    for k in range(1000):  # every response body of every call (the independent suite also runs 10,000 episodes)
        RESPONSES.clear()
        e = random_episode(random.Random(k), inst, n=15)
        ok, resp = safe_call(e, "submit", {"diagnosis": [], "repairs": []})
        tools = [line["tool"] for line in e.trace]
        leaked += any(_mentions_score(r, t) for r, t in zip(RESPONSES + [resp], tools))
    # F13 replay on 50 traces
    mismatched = 0
    for k in range(50):
        d = Path(tempfile.mkdtemp())
        random_episode(random.Random(1000 + k), inst, state_dir=d, n=30)
        mismatched += bool(replay(d / "trace.jsonl", 2, "medium"))
    return {"sequences": done, "sequence_seconds": seconds, "ledger_exact": ledger_ok / done,
            "property_violations": dict(VIOLATIONS),
            "call_151_finalizes": (not ok151) and ep.status != "RUNNING" and msg151.startswith("E_EPISODE_OVER"),
            "time_cap_finalizes": (not ok_t) and ep_t.status == "FINALIZED",
            "strike_codes": strikes, "finalized_after_4th_invalid": status_after_strikes == "FINALIZED",
            "ack": ack, "second_submit": second[1][:14], "post_submit": post[1][:14],
            "episodes_with_score_text": leaked, "replay_traces": 50, "replay_mismatched": mismatched}


# --- F10 ----------------------------------------------------------------------------------------
def _readings(spec, faults, sp):
    phys = physical_readings(spec, solve(spec, apply_faults(spec, faults), sp))
    sig = {s.id: s.sigma for s in spec.sensors}
    for f in faults:
        if f.type == "sensor_bias":
            phys[f.component] += f.sign * f.magnitude * sig[f.component]
    return phys


def _fit(spec, make_b, target_sid, target_value, lo, hi, sp=None):
    """Magnitude of hypothesis B that reproduces A's reading on the symptom sensor/quantity."""
    g = lambda m: _readings(spec, make_b(m), sp)[target_sid] - target_value  # noqa: E731
    if g(lo) * g(hi) > 0:
        return None
    return brentq(g, lo, hi, xtol=1e-10)


def f10(n_plants: int = 60) -> dict:
    """Plan 2.1 confounds: two hypotheses matched on the symptom; the documented experiment must
    separate them by >= 5 sigma on a returned sensor in >= 95% of cases."""
    out = {}
    cases = {"bias_vs_fouling": [], "stuck_vs_fouling": [], "leak_vs_meter_bias": [], "wear_vs_fouling": []}
    mrng = np.random.default_rng(10)  # hypothesis-A magnitudes drawn over each fault type's range (sim_spec 4)
    for seed in range(n_plants):
        spec = generate_plant(seed, TIERS[seed % 3])
        sig = {s.id: s.sigma for s in spec.sensors}
        links, pts = spec.links, {s.target for s in spec.sensors if s.kind == "pressure"}
        nominal = experiment_setpoint(spec, "nominal")

        def sep(kind, args, fa, fb):
            """(best single returned sensor, whitened norm over all returned sensors), in sigma."""
            sp = experiment_setpoint(spec, kind, args)
            ra, rb = _readings(spec, fa, sp), _readings(spec, fb, sp)
            z = [abs(ra[s] - rb[s]) / sig[s] for s in experiment_sensors(spec, kind, args)]
            return (max(z), float(np.sqrt(np.sum(np.square(z)))))

        # 1. low downstream pressure: PT bias vs upstream pipe fouling -> pump_off_static
        pipe = next((ln for ln in links if ln.kind == "pipe" and ln.to in pts and ln.frm.startswith("J")), None)
        if pipe:
            sid = f"PT-{pipe.to}"
            fa = (Fault(sid, "sensor_bias", float(mrng.uniform(4.0, 10.0)), sign=-1),)
            m = _fit(spec, lambda m: (Fault(pipe.id, "fouling", m),), sid, _readings(spec, fa, nominal)[sid], 0.0, 10.0)
            if m is not None:
                cases["bias_vs_fouling"].append(sep("pump_off_static", {}, fa, (Fault(pipe.id, "fouling", m),)))
        # 2. fouling vs stuck valve in series -> valve_step
        for b in spec.branches:
            valve = next((x for x in b.links if x.startswith("V")), None)
            pipe_id = next((x for x in b.links if x.startswith("P")), None)
            if valve and pipe_id:
                fa = (Fault(valve, "stuck_valve", float(mrng.uniform(0.2, 0.8))),)
                outlet_link = b.links[-1]
                key = f"FT-{outlet_link}"
                st = solve(spec, apply_faults(spec, fa))
                target = st.q[outlet_link]
                g = lambda m: solve(spec, apply_faults(spec, (Fault(pipe_id, "fouling", m),))).q[outlet_link] - target  # noqa: E731
                if g(0.0) * g(10.0) < 0:
                    m = brentq(g, 0.0, 10.0, xtol=1e-10)
                    cases["stuck_vs_fouling"].append(max(sep("valve_step", {"valve": valve, "theta": t}, fa,
                                                             (Fault(pipe_id, "fouling", m),)) for t in (0.3, 0.6)))
                break
        # 3. flow mismatch: leak at j vs bias of a flow meter on a link out of j -> speed_step
        fts = {s.target for s in spec.sensors if s.kind == "flow"}
        meter = next((ln for ln in links if ln.id in fts and ln.frm.startswith("J") and ln.kind != "pump"), None)
        if meter:
            sid, j = f"FT-{meter.id}", meter.frm
            fa = (Fault(j, "leak", float(mrng.uniform(0.02, 0.08))),)
            target = _readings(spec, fa, nominal)[sid]
            base = _readings(spec, (), nominal)[sid]
            k = (target - base) / sig[sid]
            fb = (Fault(sid, "sensor_bias", abs(k), sign=1 if k > 0 else -1),)
            cases["leak_vs_meter_bias"].append(max(sep("speed_step", {"s": 0.7}, fa, fb),
                                                   sep("pump_off_static", {}, fa, fb)))
        # 4. pump wear vs system resistance (discharge pipe fouling) -> speed_step
        pump = next(ln for ln in links if ln.kind == "pump")
        dis = next(ln for ln in links if ln.frm == pump.to)
        fa = (Fault(pump.id, "pump_wear", float(mrng.uniform(0.10, 0.35))),)
        target = solve(spec, apply_faults(spec, fa)).q[pump.id]
        g = lambda m: solve(spec, apply_faults(spec, (Fault(dis.id, "fouling", m),))).q[pump.id] - target  # noqa: E731
        if g(0.0) * g(10.0) < 0:
            m = brentq(g, 0.0, 10.0, xtol=1e-10)
            cases["wear_vs_fouling"].append(max(sep("speed_step", {"s": s, "pump": pump.id}, fa,
                                                    (Fault(dis.id, "fouling", m),)) for s in (0.7, 1.1)))
    for name, pairs in cases.items():
        single, norm = [p[0] for p in pairs], [p[1] for p in pairs]
        out[name] = {"cases": len(pairs),
                     "separated_5sigma": float(np.mean([v >= 5 for v in single])) if pairs else None,
                     "median_sigma": float(np.median(single)) if pairs else None,
                     "min_sigma": float(min(single)) if pairs else None,
                     "separated_5sigma_whitened": float(np.mean([v >= 5 for v in norm])) if pairs else None,
                     "median_sigma_whitened": float(np.median(norm)) if pairs else None}
    out["static_leak_sag_sigma_max"] = _static_sag()
    # revert: the true plant is unchanged after experiments; costs exact
    inst = make_instance(4, "medium")
    ep = Episode(inst)
    before = ep.true_readings({})
    costs = {}
    for kind, args in (("speed_step", {"s": 0.8}), ("valve_step", {"valve": next(ln.id for ln in inst.spec.links if ln.kind == "valve"), "theta": 0.3}),
                       ("isolate_branch", {"branch": inst.spec.branches[0].id}), ("pump_off_static", {})):
        costs[kind] = ep.call("run_experiment", {"type": kind, "args": args})["cost"]
    out["revert_identical"] = ep.true_readings({}) == before
    out["costs"] = costs
    out["out_of_range_rejected"] = all(not safe_call(Episode(inst), "run_experiment", x)[0] for x in (
        {"type": "speed_step", "args": {"s": 0.69}}, {"type": "speed_step", "args": {"s": 1.21}},
        {"type": "valve_step", "args": {"valve": "V99", "theta": 0.5}}, {"type": "isolate_branch", "args": {"branch": "B9"}}))
    return out


def _static_sag() -> dict:
    """Pressure sag at discharge-side sensors in pump_off_static caused by a leak (plan F10: < 0.5 sigma)."""
    worst = {}
    for f in (0.02, 0.05, 0.08):
        sags = []
        for seed in range(30):
            spec = generate_plant(seed, TIERS[seed % 3])
            z = {x.id: x for x in spec.junctions}
            static = {x.id: spec.tank_head - x.z for x in spec.junctions if x.side == "discharge"}
            for j in spec.junctions:  # every leak position, including the leaking junction's own sensor
                if j.side != "discharge":
                    continue
                st = solve(spec, apply_faults(spec, (Fault(j.id, "leak", f),)), experiment_setpoint(spec, "pump_off_static"))
                ph = physical_readings(spec, st)
                sags.append(max(abs(ph[s.id] - static[s.target]) / s.sigma for s in spec.sensors
                                if s.kind == "pressure" and s.target in static))
        worst[str(f)] = float(max(sags))
    return worst


def f6_timing(n: int = 150) -> dict:
    """Timing side channel: latency of the padded MCP handler for run_experiment on two instances with the same
    plant and different faults (the attack review found solver iterations visible before padding)."""
    import anyio
    from scipy.stats import mannwhitneyu

    from diagnoselab.server.app import build_server

    a = make_instance(2, "medium")
    b = Instance(a.spec, sample_faults(a.spec, "medium", np.random.default_rng(999)), None, a.noise_seed)
    lat = {"A": [], "B": [], "A_unpadded": [], "B_unpadded": []}

    async def run():
        for _ in range(n):
            for key, inst in (("A", a), ("B", b)):
                for padded in (True, False):
                    server = build_server(Episode(inst), costed_latency_s=0.05 if padded else 0.0)
                    t0 = time.perf_counter()
                    await server.call_tool("run_experiment", {"type": "pump_off_static", "args": {}})
                    lat[key if padded else f"{key}_unpadded"].append(time.perf_counter() - t0)

    anyio.run(run)
    p = lambda x, y: float(mannwhitneyu(lat[x], lat[y]).pvalue)  # noqa: E731
    return {"calls_per_arm": n, "p_unpadded": p("A_unpadded", "B_unpadded"), "p_padded": p("A", "B"),
            "median_ms": {k: 1000 * float(np.median(v)) for k, v in lat.items()}}


def f7_checklist() -> dict:
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_f7_manual.py"], cwd=ROOT,
                         capture_output=True, text=True)
    return {"pytest_exit": run.returncode, "summary": run.stdout.strip().splitlines()[-1] if run.stdout else ""}


def main() -> None:
    out = {"F7_checklist": f7_checklist()}
    for name, fn in (("F6", f6), ("F6_timing", f6_timing), ("F7_F8", f7_f8), ("F9_F11_F12_F13", f9_f11_f12_f13), ("F10", f10)):
        t0 = time.perf_counter()
        out[name] = fn()
        print(f"{name} done in {time.perf_counter() - t0:.1f}s", flush=True)
    iso = ROOT / "results/isolation.json"
    out["F14"] = json.loads(iso.read_text()) if iso.exists() else None
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/step3_validation.json").write_text(json.dumps(out, default=float, indent=1))
    print("wrote results/step3_validation.json")


if __name__ == "__main__":
    main()
