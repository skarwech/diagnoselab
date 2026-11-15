"""Experiment catalog: setpoints and returned sensor subsets. docs/sim_spec.md section 6."""

from __future__ import annotations

from typing import Any, Mapping

from diagnoselab.sim.model import PlantSpec, Setpoint

BUDGET = 40  # B: the one place the budget is set (episode, topology, reward and oracle read it)
COSTS = {"read": 1, "speed_step": 4, "valve_step": 4, "isolate_branch": 6, "pump_off_static": 8}
SPEED_RANGE = (0.7, 1.2)  # below 0.7 a healthy pump can deadhead against the static lift
THETA_RANGE = (0.1, 1.0)
KINDS = ("nominal", "speed_step", "valve_step", "isolate_branch", "pump_off_static")


def _ids(spec: PlantSpec, kind: str) -> list[str]:
    return [ln.id for ln in spec.links if ln.kind == kind]


def _in_range(x: Any, lo_hi: tuple[float, float], name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not lo_hi[0] <= v <= lo_hi[1]:
        raise ValueError(f"{name} must be in [{lo_hi[0]}, {lo_hi[1]}]")
    return v


def _member(x: Any, allowed: list[str], name: str) -> str:
    if x not in allowed:
        raise ValueError(f"unknown {name}")
    return x


def _affected_pumps(spec: PlantSpec, args: Mapping[str, Any]) -> list[str]:
    pump = args.get("pump")
    return _ids(spec, "pump") if pump is None else [_member(pump, _ids(spec, "pump"), "pump")]


def experiment_setpoint(spec: PlantSpec, kind: str, args: Mapping[str, Any] | None = None) -> Setpoint:
    args = dict(args or {})
    if kind == "nominal":
        return Setpoint()
    if kind == "speed_step":
        s = _in_range(args.get("s"), SPEED_RANGE, "s")
        return Setpoint(speed={p: s for p in _affected_pumps(spec, args)})
    if kind == "valve_step":
        v = _member(args.get("valve"), _ids(spec, "valve"), "valve")
        return Setpoint(valve={v: _in_range(args.get("theta"), THETA_RANGE, "theta")})
    if kind == "isolate_branch":
        b = {br.id: br for br in spec.branches}.get(args.get("branch"))
        if b is None:
            raise ValueError("unknown branch")
        return Setpoint(closed=frozenset({b.entry}))
    if kind == "pump_off_static":
        return Setpoint(speed={p: 0.0 for p in _ids(spec, "pump")})
    raise ValueError("unknown experiment")


def experiment_sensors(spec: PlantSpec, kind: str, args: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Sensors returned by an experiment, in spec order (args assumed valid: call experiment_setpoint first)."""
    args = dict(args or {})
    links = {ln.id: ln for ln in spec.links}
    flows = {s.id for s in spec.sensors if s.kind == "flow"}
    if kind == "speed_step":
        pumps = _affected_pumps(spec, args)
        junctions = {n for p in pumps for n in (links[p].frm, links[p].to)}
        want = flows | {f"PT-{j}" for j in junctions} | {f"{k}-{p}" for p in pumps for k in ("WT", "VT")}
    elif kind == "valve_step":
        valve = links[args["valve"]]
        ends = {n for n in (valve.frm, valve.to) if n.startswith("J")}
        touching = [ln for ln in spec.links if ln.frm in ends or ln.to in ends]
        near = ends | {n for ln in touching for n in (ln.frm, ln.to) if n.startswith("J")}
        want = {f"PT-{j}" for j in near} | {f"FT-{ln.id}" for ln in touching}
    elif kind == "isolate_branch":
        b = {br.id: br for br in spec.branches}[args["branch"]]
        want = flows | {f"PT-{j}" for j in (*b.junctions, links[b.entry].frm)}
    elif kind == "pump_off_static":
        want = {s.id for s in spec.sensors if s.kind in ("pressure", "flow")}
    else:
        want = set()
    return tuple(s.id for s in spec.sensors if s.id in want)
