"""Argument validation, advertised JSON schemas, public topology and manual (docs/server_spec.md 3-6).

Validators are hand-written so error messages name the field and never echo the value given.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from diagnoselab.sim.experiments import COSTS, SPEED_RANGE, THETA_RANGE
from diagnoselab.sim.model import PlantSpec


class ToolRejected(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def bad(field: str, why: str) -> ToolRejected:
    return ToolRejected("E_INVALID_ARGUMENT", f"field '{field}': {why}")


FAULT_KIND = {"fouling": "pipe", "stuck_valve": "valve", "pump_wear": "pump", "leak": "junction",
              "sensor_bias": "sensor", "sensor_stuck": "sensor"}
REPAIR_KIND = {"clean_pipe": "pipe", "free_valve": "valve", "replace_pump": "pump", "fix_leak": "junction",
               "recalibrate_sensor": "sensor"}
SIM_RANGES = {"fouling": (0.0, 10.0), "stuck_valve": (0.0, 0.9), "pump_wear": (0.0, 0.9), "leak": (0.0, 0.3),
              "sensor_bias": (0.0, 50.0)}
EXPERIMENT_ARGS = {"speed_step": ({"s"}, {"pump"}), "valve_step": ({"valve", "theta"}, set()),
                   "isolate_branch": ({"branch"}, set()), "pump_off_static": (set(), set())}
TOPICS = ("overview", "tools", "physics", "faults", "experiments", "sensors", "submission", "scoring")
MAX_ENTRIES = 10
NOOP_MARGIN = 0.05  # experiments may not reproduce the nominal setpoint (attack review: cheap bulk nominal reads)
MAX_OVERRIDES = 20


# --- primitive validators ----------------------------------------------------------------------
def obj(value: Any, field: str, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict):
        raise bad(field, "must be an object")
    unknown = sorted(str(k)[:40] for k in value if k not in required | optional)  # caller's own key, truncated
    if unknown:
        raise bad(f"{field}.{unknown[0]}" if field else unknown[0], "unknown field")
    missing = sorted(required - value.keys())
    if missing:
        raise bad(f"{field}.{missing[0]}" if field else missing[0], "required")
    return value


def num(value: Any, field: str, lo: float = -math.inf, hi: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise bad(field, "must be a finite number")
    try:
        if not math.isfinite(value):
            raise bad(field, "must be a finite number")
    except OverflowError:  # an int too large for a float (e.g. 10**400)
        raise bad(field, "must be a finite number") from None
    if not lo <= value <= hi:
        raise bad(field, f"must be in [{lo:g}, {hi:g}]")
    return float(value)


def choice(value: Any, field: str, allowed: dict | set | tuple, what: str = "value") -> str:
    if not isinstance(value, str) or value not in allowed:
        raise bad(field, f"unknown {what}")
    return value


def lst(value: Any, field: str, max_len: int) -> list:
    if not isinstance(value, list):
        raise bad(field, "must be a list")
    if len(value) > max_len:
        raise bad(field, f"at most {max_len} entries")
    return value


# --- plant-aware helpers -----------------------------------------------------------------------
@lru_cache(maxsize=64)
def component_kinds(spec: PlantSpec) -> dict[str, str]:
    kinds = {j.id: "junction" for j in spec.junctions} | {s.id: "sensor" for s in spec.sensors}
    return kinds | {ln.id: ln.kind for ln in spec.links}


def fault_obj(spec: PlantSpec, value: Any, field: str, *, for_sim: bool) -> dict:
    opt = {"magnitude", "sign", "value"} if for_sim else {"magnitude", "confidence"}
    f = obj(value, field, {"component", "type"}, opt)
    typ = choice(f["type"], f"{field}.type", FAULT_KIND, "fault type")
    comp = choice(f["component"], f"{field}.component", component_kinds(spec), "component")
    if component_kinds(spec)[comp] != FAULT_KIND[typ]:
        raise bad(f"{field}.component", f"fault type '{typ}' needs a {FAULT_KIND[typ]}")
    out = {"component": comp, "type": typ}
    if typ != "sensor_stuck":
        if "magnitude" not in f:
            raise bad(f"{field}.magnitude", "required")
        lo, hi = SIM_RANGES[typ] if for_sim else (0.0, math.inf)
        out["magnitude"] = num(f["magnitude"], f"{field}.magnitude", lo, hi)
    if for_sim:
        if "sign" in f:
            if typ != "sensor_bias" or f["sign"] not in (-1, 1) or isinstance(f["sign"], bool):
                raise bad(f"{field}.sign", "must be -1 or 1, sensor_bias only")
            out["sign"] = int(f["sign"])
        if "value" in f:
            if typ != "sensor_stuck":
                raise bad(f"{field}.value", "sensor_stuck only")
            out["value"] = num(f["value"], f"{field}.value", -1e6, 1e6)
    elif "confidence" in f:
        out["confidence"] = num(f["confidence"], f"{field}.confidence", 0.0, 1.0)
    return out


def repair_obj(spec: PlantSpec, value: Any, field: str) -> dict:
    r = obj(value, field, {"action", "target"})
    action = choice(r["action"], f"{field}.action", REPAIR_KIND, "repair action")
    target = choice(r["target"], f"{field}.target", component_kinds(spec), "component")
    if component_kinds(spec)[target] != REPAIR_KIND[action]:
        raise bad(f"{field}.target", f"action '{action}' needs a {REPAIR_KIND[action]}")
    return {"action": action, "target": target}


def experiment(spec: PlantSpec, kind: Any, args: Any, kind_field: str, args_field: str) -> tuple[str, dict]:
    """Validate an experiment type and its args; field names in messages use the given prefixes."""
    kind = choice(kind, kind_field, EXPERIMENT_ARGS, "experiment type")
    required, optional = EXPERIMENT_ARGS[kind]
    a = obj(args, args_field, required, optional)
    for key in ("pump", "valve", "branch"):
        if key in a:
            pool = {b.id for b in spec.branches} if key == "branch" else {ln.id for ln in spec.links if ln.kind == key}
            choice(a[key], f"{args_field}.{key}", pool, key)
    if "s" in a and abs(num(a["s"], f"{args_field}.s", *SPEED_RANGE) - 1.0) < NOOP_MARGIN:
        raise bad(f"{args_field}.s", f"must differ from 1 by at least {NOOP_MARGIN} (a no-op step is a cheap nominal read)")
    if "theta" in a and num(a["theta"], f"{args_field}.theta", *THETA_RANGE) > 1.0 - NOOP_MARGIN:
        raise bad(f"{args_field}.theta", f"must be at most {1.0 - NOOP_MARGIN:g} (theta = 1 is the nominal opening)")
    return kind, dict(a)


# --- public views ------------------------------------------------------------------------------
UNITS = {"pressure": "m", "flow": "L/s", "power": "kW", "vibration": "mm/s"}


@lru_cache(maxsize=64)
def public_topology(spec: PlantSpec) -> dict:
    """Nominal datasheet only (server_spec section 4): no seed, tier, faults or design point."""
    def link(ln):
        base = {"id": ln.id, "kind": ln.kind, "frm": ln.frm, "to": ln.to}
        if ln.kind == "pipe":
            return base | {"r": ln.r}
        if ln.kind == "valve":
            return base | {"r0": ln.r}
        return base | {"a": ln.a, "b": ln.b, "q_bep": ln.q_bep, "eta_max": ln.eta_max, "v0": ln.v0}

    return {
        "tank_head": spec.tank_head,
        "sources": [{"id": s.id, "head": s.head} for s in spec.sources],
        "outlets": [{"id": o.id, "head": spec.tank_head, "q_req": o.q_req} for o in spec.outlets],
        "junctions": [{"id": j.id, "z": j.z, "side": j.side} for j in spec.junctions],
        "links": [link(ln) for ln in spec.links],
        "sensors": [{"id": s.id, "kind": s.kind, "target": s.target, "unit": UNITS[s.kind], "sigma": s.sigma}
                    for s in spec.sensors],
        "branches": [{"id": b.id, "entry": b.entry, "links": list(b.links), "junctions": list(b.junctions)}
                     for b in spec.branches],
        "budget": 40,
        "costs": {"read_sensor": COSTS["read"]} | {k: COSTS[k] for k in EXPERIMENT_ARGS},
        "speed_range": list(SPEED_RANGE),
        "theta_range": list(THETA_RANGE),
    }


MANUAL_PATH = Path(__file__).resolve().parents[2] / "docs" / "manual.md"


@lru_cache(maxsize=1)
def manual() -> dict[str, str]:
    """Split docs/manual.md on '## <topic>' headings; text before the first heading is the contents."""
    text = MANUAL_PATH.read_text() if MANUAL_PATH.exists() else "# DiagnoseLab manual\n(manual missing)\n"
    parts = text.split("\n## ")
    out = {"contents": parts[0].strip()}
    for part in parts[1:]:
        head, _, body = part.partition("\n")
        out[head.strip().lower()] = f"## {head.strip()}\n{body}".strip()
    return out


# --- schemas advertised to the agent -----------------------------------------------------------
def _o(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


_FAULT = _o({"component": {"type": "string"}, "type": {"type": "string", "enum": list(FAULT_KIND)},
             "magnitude": {"type": "number", "minimum": 0}, "confidence": {"type": "number", "minimum": 0,
                                                                         "maximum": 1}}, ["component", "type"])
_OVERRIDE = _o({"component": {"type": "string"}, "type": {"type": "string", "enum": list(FAULT_KIND)},
                "magnitude": {"type": "number", "minimum": 0}, "sign": {"type": "integer", "enum": [-1, 1]},
                "value": {"type": "number"}}, ["component", "type"])
_REPAIR = _o({"action": {"type": "string", "enum": list(REPAIR_KIND)}, "target": {"type": "string"}},
             ["action", "target"])
_EXP_ARGS = {"type": "object", "description": "speed_step {s, pump?}; valve_step {valve, theta}; "
             "isolate_branch {branch}; pump_off_static {}", "additionalProperties": False,
             "properties": {"s": {"type": "number", "minimum": SPEED_RANGE[0], "maximum": SPEED_RANGE[1]},
                            "pump": {"type": "string"}, "valve": {"type": "string"},
                            "theta": {"type": "number", "minimum": THETA_RANGE[0], "maximum": THETA_RANGE[1]},
                            "branch": {"type": "string"}}}
_SETPOINT = {"type": "object", "additionalProperties": False, "description": "{experiment, ...args}; default nominal",
             "properties": {"experiment": {"type": "string", "enum": ["nominal", *EXPERIMENT_ARGS]},
                            **_EXP_ARGS["properties"]}}

TOOL_SCHEMAS: dict[str, dict] = {
    "read_manual": _o({"topic": {"type": "string", "enum": list(TOPICS)}}),
    "get_topology": _o({}),
    "simulate": _o({"overrides": {"type": "array", "maxItems": MAX_OVERRIDES, "items": _OVERRIDE},
                    "setpoint": _SETPOINT, "sensors": {"type": "array", "items": {"type": "string"}}}),
    "read_sensor": _o({"sensor_id": {"type": "string"}}, ["sensor_id"]),
    "run_experiment": _o({"type": {"type": "string", "enum": list(EXPERIMENT_ARGS)}, "args": _EXP_ARGS},
                         ["type", "args"]),
    "budget_status": _o({}),
    "submit": _o({"diagnosis": {"type": "array", "maxItems": MAX_ENTRIES, "items": _FAULT},
                  "repairs": {"type": "array", "maxItems": MAX_ENTRIES, "items": _REPAIR}},
                 ["diagnosis", "repairs"]),
}

TOOL_DESCRIPTIONS = {
    "read_manual": "Free. Physics, fault signatures, experiment catalog, sensors, submission format, scoring. "
                   "Call without a topic for the contents.",
    "get_topology": "Free. Network graph, nominal datasheet parameters, sensor list with noise sigma, branches, costs.",
    "simulate": "Free (max 300 calls). Noise-free predicted readings of the nominal plant with your fault "
                "overrides, at a chosen setpoint. Uses no hidden information.",
    "read_sensor": "Cost 1. One noisy reading of one sensor at the nominal operating point.",
    "run_experiment": "Cost 4 (speed_step, valve_step), 6 (isolate_branch) or 8 (pump_off_static). Noisy readings "
                      "of the documented sensor subset under a temporary setpoint; the plant then reverts.",
    "budget_status": "Free. Budget spent and remaining, calls used, time used.",
    "submit": "Terminal, one shot. Your diagnosis and repair plan. Returns an acknowledgement, never a score.",
}
