"""F6 server core: strict validation, fuzzing, hidden-state isolation, episode isolation, MCP binding.

Contract: docs/server_spec.md sections 1-3 and 10; plan section 8, F6.
"""

from __future__ import annotations

import copy
import functools
import json
import re
import socket
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from diagnoselab.server.app import build_server
from diagnoselab.server.episode import Episode, ToolRejected
from diagnoselab.server.tools import TOOL_SCHEMAS
from diagnoselab.sim.faults import sample_faults
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.model import Fault, Instance, PlantSpec

TOOLS = ("read_manual", "get_topology", "simulate", "read_sensor", "run_experiment", "budget_status", "submit")
MARK = "ZQXMARK93"  # used only as a value, never as a key or tool name: errors must never echo it
NAN, INF = float("nan"), float("inf")
ROTATE = 40  # calls per fuzz episode pair, far below the 150-call cap
FUZZ_CASES = ((2, "medium"), (3, "hard"), (1, "easy"))


def nest(depth: int, as_dict: bool) -> object:
    x: object = MARK
    for _ in range(depth):
        x = {"a": x} if as_dict else [x]
    return x


DEEP = [nest(d, as_dict) for d in (64, 1000, 5000) for as_dict in (False, True)]
BAD = [None, True, False, 0, -1, 2**64, -(2**64), 1e308, -1e308, 5e-324, NAN, INF, -INF, "", MARK,
       "\x00" + MARK + "\udcff", MARK * 2000, [], {}, [MARK], {"k": MARK}, [[MARK] * 3] * 3, *DEEP]
SMALL_BAD = BAD[:16]
NON_DICT = [None, [], [{}], MARK, 0, 1.5, NAN, True, b"\x00" + MARK.encode(), (1, 2)]
UNKNOWN_TOOLS = ["", "nope", "READ_SENSOR", "read_sensor ", " submit", "true_readings", "finalize", "state_hash",
                 "__init__", "call", "run_experiment/../submit"]
BOUNDARY = [0.7, 0.7 - 1e-9, 1.2, 1.2 + 1e-9, 0.1, 0.1 - 1e-9, 1.0, 1.0 + 1e-9, 0.0, -1e-12, -0.0, 0.3, 0.3 + 1e-9,
            0.9, 0.9 + 1e-9, 10.0, 10.0 + 1e-9, 50.0, 50.0 + 1e-9, 2**53 + 1]


@functools.lru_cache(maxsize=None)
def inst(seed: int, tier: str) -> Instance:
    return make_instance(seed, tier)


@functools.lru_cache(maxsize=None)
def twin(seed: int, tier: str) -> Instance:
    """Same nominal plant (same spec object, noise seed), different hidden faults from an unrelated rng."""
    base = inst(seed, tier)
    rng = np.random.default_rng(10_000 + seed)
    for _ in range(50):
        other = sample_faults(base.spec, tier, rng)
        if {f.component for f in other} != {f.component for f in base.faults}:
            return Instance(base.spec, other, None, base.noise_seed)
    raise AssertionError("could not draw a different fault set")


def ids(spec: PlantSpec, kind: str) -> list[str]:
    return [ln.id for ln in spec.links if ln.kind == kind]


def pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    return seq[int(rng.integers(len(seq)))]


def running(instance: Instance, **kw: Any) -> Episode:
    """Episode on a fixed clock, moved INIT -> RUNNING by one accepted free call (status is in the hash)."""
    ep = Episode(instance, clock=lambda: 0.0, **kw)
    ep.call("get_topology", {})
    return ep


def outcome(ep: Episode, tool: str, args: object, label: str) -> tuple[bool, Any]:
    """(True, response) or (False, (code, message)); any other exception is a server error."""
    try:
        return True, ep.call(tool, args)
    except ToolRejected as e:
        assert isinstance(e.code, str) and e.code.startswith("E_"), label
        assert isinstance(e.message, str), label
        return False, (e.code, e.message)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"{label}: {type(e).__name__} escaped Episode.call", pytrace=False)


def strings(obj: object) -> str:
    """Every string inside obj (keys and values), without recursion (inputs can be 5000 levels deep)."""
    out, stack = [], [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, bytes):
            out.append(x.decode("latin-1"))
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, (list, tuple, set)):
            stack.extend(x)
    return " ".join(out)


def has_token(component: str, text: str) -> bool:
    """Whole-id match: J1 does not match J10 or PT-J1, P1 does not match P12."""
    return re.search(rf"(?<![A-Za-z0-9-]){re.escape(component)}(?![A-Za-z0-9])", text) is not None


def leaked(message: str, faults: tuple[Fault, ...], given: str) -> list[str]:
    return [f.component for f in faults if has_token(f.component, message) and not has_token(f.component, given)]


def templates(spec: PlantSpec) -> dict[str, list[dict]]:
    pipe, valve, pump = ids(spec, "pipe")[0], ids(spec, "valve")[0], ids(spec, "pump")[0]
    sensor, branch = spec.sensors[0].id, spec.branches[0].id
    return {
        "read_manual": [{"topic": "physics"}],
        "get_topology": [{}],
        "budget_status": [{}],
        "read_sensor": [{"sensor_id": sensor}],
        "simulate": [{"overrides": [{"component": pipe, "type": "fouling", "magnitude": 1.0}],
                      "setpoint": {"experiment": "valve_step", "valve": valve, "theta": 0.5}, "sensors": [sensor]}],
        "run_experiment": [{"type": "speed_step", "args": {"s": 0.9, "pump": pump}},
                           {"type": "valve_step", "args": {"valve": valve, "theta": 0.5}},
                           {"type": "isolate_branch", "args": {"branch": branch}},
                           {"type": "pump_off_static", "args": {}}],
        "submit": [{"diagnosis": [{"component": pipe, "type": "fouling", "magnitude": 1.0, "confidence": 0.5}],
                    "repairs": [{"action": "clean_pipe", "target": pipe}]}],
    }


def slots_of(obj: object) -> list[tuple[Any, Any]]:
    """(container, key) for every value below the root."""
    out, stack = [], [obj]
    while stack:
        x = stack.pop()
        items = x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else ()
        for k, v in items:
            out.append((x, k))
            stack.append(v)
    return out


def mutate(rng: np.random.Generator, tool: str, template: dict) -> tuple[str, object, str]:
    """One malformed (occasionally still valid) call derived from a valid template."""
    args: Any = copy.deepcopy(template)
    slots = slots_of(args)
    numeric = [(c, k) for c, k in slots if isinstance(c[k], (int, float)) and not isinstance(c[k], bool)]
    lists = [(c, k) for c, k in slots if isinstance(c[k], list)]
    kind = int(rng.integers(8))
    if kind == 0 and slots:
        c, k = pick(rng, slots)
        c[k] = pick(rng, BAD)
        return tool, args, f"{tool}:bad-value@{k}"
    if kind == 1 and slots:
        c, k = pick(rng, slots)
        del c[k]
        return tool, args, f"{tool}:drop@{k}"
    if kind == 3 and numeric:
        c, k = pick(rng, numeric)
        c[k] = pick(rng, BOUNDARY)
        return tool, args, f"{tool}:boundary@{k}={c[k]!r}"
    if kind == 4:
        return tool, pick(rng, NON_DICT), f"{tool}:non-dict"
    if kind == 5:
        return pick(rng, UNKNOWN_TOOLS), args, "unknown-tool"
    if kind == 6 and lists:
        c, k = pick(rng, lists)
        c[k] = (c[k] or [MARK]) * 21
        return tool, args, f"{tool}:oversize@{k}"
    if kind == 7:
        args["zz_pad"] = MARK * 2000
        return tool, args, f"{tool}:huge"
    dicts = [args] + [c[k] for c, k in slots if isinstance(c[k], dict)]
    pick(rng, dicts)["zz_unknown"] = pick(rng, SMALL_BAD)
    return tool, args, f"{tool}:unknown-field"


def fuzz(n_calls: int, seed: int) -> dict[str, int]:
    """Same malformed calls on two episodes whose plants differ only in hidden faults; check every invariant."""
    rng = np.random.default_rng(seed)
    stats = {"calls": 0, "running": 0, "rejected": 0}
    round_ = 0
    while stats["calls"] < n_calls:
        case = FUZZ_CASES[round_ % len(FUZZ_CASES)]
        round_ += 1
        pair = [(running(inst(*case)), inst(*case).faults), (running(twin(*case)), twin(*case).faults)]
        tpl, strikes = templates(inst(*case).spec), 0
        for _ in range(ROTATE):
            tool = pick(rng, TOOLS)
            tool = "read_sensor" if tool == "submit" and strikes >= 3 else tool
            tool, args, label = mutate(rng, tool, pick(rng, tpl[tool]))
            given = tool + " " + strings(args)
            results = []
            for ep, faults in pair:
                status, before = ep.status, ep.state_hash()
                ok, res = outcome(ep, tool, args, label)
                if not ok:
                    if tool != "submit" or status != "RUNNING":
                        assert ep.state_hash() == before, f"{label}: rejected call changed the state"
                    assert not leaked(res[1], faults, given), f"{label}: true fault id in error text"
                    assert MARK not in res[1], f"{label}: error text echoes the value given"
                results.append((ok, res, status))
            (ok_a, res_a, status_a), (ok_b, res_b, _) = results
            assert ok_a == ok_b, f"{label}: acceptance depends on hidden faults"
            if not ok_a or tool not in ("read_sensor", "run_experiment"):
                assert res_a == res_b, f"{label}: response or error text depends on hidden faults"
            strikes += tool == "submit" and not ok_a and status_a == "RUNNING"
            stats["calls"] += 1
            stats["running"] += status_a == "RUNNING"
            stats["rejected"] += not ok_a
            if pair[0][0].status != "RUNNING":  # a mutated submit stayed valid: keep fuzzing a live episode
                pair = [(running(inst(*case)), inst(*case).faults), (running(twin(*case)), twin(*case).faults)]
                strikes = 0
    return stats


def check_fuzz_stats(stats: dict[str, int]) -> None:
    assert stats["running"] >= 0.8 * stats["calls"], f"fuzz mostly hit a locked episode: {stats}"
    assert stats["rejected"] >= 0.5 * stats["calls"], f"fuzz mostly produced valid calls: {stats}"


def test_twin_instances_share_plant_but_not_faults() -> None:
    """F6 precondition: the hidden-state differential compares the same nominal plant with different faults."""
    for case in FUZZ_CASES:
        assert twin(*case).spec is inst(*case).spec and twin(*case).noise_seed == inst(*case).noise_seed
        assert {f.component for f in twin(*case).faults} != {f.component for f in inst(*case).faults}


def test_fuzz_small() -> None:
    """F6 valid-when (small count): 0 crashes (only ToolRejected escapes), 0 state changes from rejected calls,
    0 hidden-value matches in error text, identical errors for plants differing only in hidden faults."""
    check_fuzz_stats(fuzz(700, seed=0))


@pytest.mark.slow
def test_fuzz_10000() -> None:
    """F6 valid-when at plan count: 10,000 fuzz calls with the same invariants as test_fuzz_small."""
    check_fuzz_stats(fuzz(10_000, seed=1))


def test_too_large_arguments_rejected() -> None:
    """F6 'huge strings' / spec section 3: arguments whose JSON is larger than 16 KB give E_TOO_LARGE for every
    tool without touching the state; a 15 KB argument is parsed normally (unknown id -> E_INVALID_ARGUMENT)."""
    ep = running(inst(1, "easy"))
    for tool in TOOLS:
        before = ep.state_hash()
        ok, res = outcome(ep, tool, {"zz_pad": "x" * 20_000}, tool)
        assert not ok and res[0] == "E_TOO_LARGE", tool
        if tool != "submit":
            assert ep.state_hash() == before, tool
    ok, res = outcome(ep, "read_sensor", {"sensor_id": "x" * 15_000}, "15KB")
    assert not ok and res[0] == "E_INVALID_ARGUMENT"


def test_unknown_tools_and_fields_rejected() -> None:
    """F6 'unknown fields ... rejected with generic error codes': unknown tools and an unknown top-level field on
    every tool are rejected and leave the state unchanged."""
    instance = inst(2, "medium")
    ep = running(instance)
    for name in UNKNOWN_TOOLS:
        before = ep.state_hash()
        ok, _ = outcome(ep, name, {}, name)
        assert not ok and ep.state_hash() == before, name
    for tool, options in templates(instance.spec).items():
        for template in options:
            before = ep.state_hash()
            ok, res = outcome(ep, tool, {**template, "zz_unknown": 1}, tool)
            codes = {"E_INVALID_ARGUMENT", "E_INVALID_SUBMISSION"} if tool == "submit" else {"E_INVALID_ARGUMENT"}
            assert not ok and res[0] in codes, (tool, res)
            if tool != "submit":
                assert ep.state_hash() == before, tool


NAMED = [
    ("read_sensor", lambda s: {"sensor_id": 5}, "sensor_id"),
    ("read_sensor", lambda s: {"sensor_id": NAN}, "sensor_id"),
    ("read_sensor", lambda s: {}, "sensor_id"),
    ("read_manual", lambda s: {"topic": 3}, "topic"),
    ("run_experiment", lambda s: {"type": "valve_step", "args": {"valve": ids(s, "valve")[0], "theta": NAN}}, "theta"),
    ("run_experiment", lambda s: {"type": "valve_step", "args": {"valve": ids(s, "valve")[0], "theta": "0.5"}},
     "theta"),
    ("simulate", lambda s: {"overrides": [{"component": ids(s, "pipe")[0], "type": "fouling", "magnitude": INF}]},
     "magnitude"),
    ("simulate", lambda s: {"overrides": "all"}, "overrides"),
]


@pytest.mark.parametrize("tool,build,field", NAMED, ids=[f"{t}-{f}-{i}" for i, (t, _, f) in enumerate(NAMED)])
def test_wrong_type_and_nonfinite_name_the_field(tool: str, build: Callable[[PlantSpec], dict], field: str) -> None:
    """F6 / spec section 3: wrong types and NaN/Inf give E_INVALID_ARGUMENT whose message names the field and
    leaves the state unchanged."""
    instance = inst(2, "medium")
    ep = running(instance)
    before = ep.state_hash()
    ok, res = outcome(ep, tool, build(instance.spec), tool)
    assert not ok and res[0] == "E_INVALID_ARGUMENT" and field in res[1], res
    assert ep.state_hash() == before


def test_non_dict_args_rejected() -> None:
    """F6 'malformed JSON': non-object arguments to a tool that needs fields are E_INVALID_ARGUMENT."""
    ep = running(inst(1, "easy"))
    for args in NON_DICT:
        before = ep.state_hash()
        ok, res = outcome(ep, "read_sensor", args, repr(type(args)))
        assert not ok and res[0] == "E_INVALID_ARGUMENT" and ep.state_hash() == before


def script(spec: PlantSpec, rng: np.random.Generator, n: int) -> list[tuple[str, dict]]:
    sensors = [s.id for s in spec.sensors]
    menu = (lambda: ("read_sensor", {"sensor_id": pick(rng, sensors)}),
            lambda: ("read_sensor", {"sensor_id": pick(rng, sensors)}),
            lambda: ("run_experiment", {"type": "valve_step", "args": {"valve": ids(spec, "valve")[0], "theta": 0.4}}),
            lambda: ("run_experiment", {"type": "isolate_branch", "args": {"branch": spec.branches[-1].id}}),
            lambda: ("simulate", {"overrides": [], "sensors": sensors[:3]}),
            lambda: ("budget_status", {}),
            lambda: ("read_sensor", {"sensor_id": "PT-J999"}))
    calls = [pick(rng, menu)() for _ in range(n)]
    return calls + [("submit", {"diagnosis": [], "repairs": []}), ("read_sensor", {"sensor_id": sensors[0]})]


def play(ep: Episode, calls: list[tuple[str, dict]]) -> list[tuple[bool, Any]]:
    return [outcome(ep, tool, args, f"{tool}#{i}") for i, (tool, args) in enumerate(calls)]


@pytest.mark.parametrize("pair", [((1, "easy"), (1, "easy")), ((2, "medium"), (4, "medium"))], ids=["same", "diff"])
def test_episodes_do_not_interact(pair: tuple[tuple[int, str], tuple[int, str]]) -> None:
    """F6 'sessions are isolated' / '0 cross-session leaks': two Episodes called interleaved, or concurrently from
    two threads, give exactly the responses, traces and state hashes of separate runs (noise index included)."""
    rng = np.random.default_rng(7)
    insts = [inst(*p) for p in pair]
    scripts = [script(i.spec, rng, 30) for i in insts]
    alone_eps = [Episode(i, clock=lambda: 0.0) for i in insts]
    alone = [play(ep, s) for ep, s in zip(alone_eps, scripts)]

    mixed_eps = [Episode(i, clock=lambda: 0.0) for i in insts]
    mixed: list[list[tuple[bool, Any]]] = [[], []]
    for step in range(max(map(len, scripts))):
        for j in (0, 1):
            if step < len(scripts[j]):
                tool, args = scripts[j][step]
                mixed[j].append(outcome(mixed_eps[j], tool, args, f"{tool}#{step}"))
    assert mixed == alone
    assert [e.trace for e in mixed_eps] == [e.trace for e in alone_eps]
    assert [e.state_hash() for e in mixed_eps] == [e.state_hash() for e in alone_eps]

    threaded_eps = [Episode(i, clock=lambda: 0.0) for i in insts]
    threaded: list[Any] = [None, None]
    barrier = threading.Barrier(2)

    def worker(j: int) -> None:
        barrier.wait()
        threaded[j] = play(threaded_eps[j], scripts[j])

    threads = [threading.Thread(target=worker, args=(j,)) for j in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert threaded == alone


def unwrap(value: dict) -> dict:
    return value.get("inputSchema", value.get("input_schema", value))


def strict(schema: dict) -> bool:
    """The root forbids extra fields, and so does every nested schema that declares properties."""
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        return False
    stack: list[Any] = [schema]
    while stack:
        s = stack.pop()
        if isinstance(s, dict):
            if "properties" in s and s.get("additionalProperties") is not False:
                return False
            stack.extend(s.values())
        elif isinstance(s, list):
            stack.extend(s)
    return True


def test_tool_schemas_strict() -> None:
    """F6 'strict schemas' / spec section 1: exactly the seven tools, each with additionalProperties false."""
    assert set(TOOL_SCHEMAS) == set(TOOLS)
    for name, value in TOOL_SCHEMAS.items():
        assert strict(unwrap(value)), name


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mcp_streamable_http() -> None:
    """F6 'exposes the tools over MCP with strict schemas' (spec section 1): over streamable HTTP, list_tools
    advertises additionalProperties false, an unknown field is an isError result starting 'E_INVALID_ARGUMENT',
    and a valid read_sensor works and reaches the shared Episode."""
    import anyio
    import uvicorn
    from mcp import Client

    instance = inst(1, "easy")
    ep = Episode(instance)
    sensor = instance.spec.sensors[0].id
    port = free_port()
    app = build_server(ep).streamable_http_app(host="127.0.0.1")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    async def session() -> tuple[Any, Any, Any]:
        async with Client(f"http://127.0.0.1:{port}/mcp") as client:
            tools = (await client.list_tools()).tools
            bad = await client.call_tool("read_sensor", {"sensor_id": sensor, "zz_unknown": 1})
            good = await client.call_tool("read_sensor", {"sensor_id": sensor})
            return tools, bad, good

    try:
        deadline = time.monotonic() + 20
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "MCP server did not start"
            time.sleep(0.05)
        tools, bad, good = anyio.run(session)
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert {t.name for t in tools} == set(TOOL_SCHEMAS)
    for t in tools:
        assert strict(t.input_schema), t.name
    assert bad.is_error and bad.content[0].text.startswith("E_INVALID_ARGUMENT"), bad
    assert not good.is_error, good
    payload = good.structured_content if isinstance(good.structured_content, dict) else json.loads(good.content[0].text)
    payload = payload.get("result", payload) if "sensor_id" not in payload else payload
    assert payload["sensor_id"] == sensor and payload["cost"] == 1 and payload["events"] == []
    assert ep.trace[-2]["code"] == "E_INVALID_ARGUMENT" and ep.trace[-1]["ok"] is True
