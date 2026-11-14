"""Episode core (F6, F9-F13): one entry point, Episode.call(tool, raw_args), for every tool.

Validation, ledger, caps, state machine, trace and persistence live here so every call, valid or
not, is handled, counted and traced the same way. docs/server_spec.md is the contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from diagnoselab.server import tools as T
from diagnoselab.server.tools import ToolRejected, bad
from diagnoselab.sim.experiments import COSTS, experiment_sensors, experiment_setpoint
from diagnoselab.sim.faults import apply_faults
from diagnoselab.sim.model import Fault, Instance, Setpoint
from diagnoselab.sim.sensors import observe, physical_readings
from diagnoselab.sim.solver import solve

__all__ = ["Episode", "ToolRejected", "BUDGET", "CALL_CAP", "SIMULATE_CAP", "TIME_CAP_S", "MAX_STRIKES",
           "response_hash"]

BUDGET = 40
CALL_CAP = 150  # every call except simulate (server_spec section 8, default 1)
SIMULATE_CAP = 300
TIME_CAP_S = 30 * 60
MAX_STRIKES = 3
MAX_ARG_BYTES = 16 * 1024
TOO_LARGE, UNSERIALIZABLE = "__too_large__", "__unserializable__"


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def response_hash(response: dict) -> str:
    return hashlib.sha256(_canon({k: v for k, v in response.items() if k != "time_used_s"}).encode()).hexdigest()


def _round(readings: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in readings.items()}


class Episode:
    def __init__(self, instance: Instance, state_dir: Path | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.instance, self.spec, self._clock = instance, instance.spec, clock
        self._lock = threading.Lock()
        self._params = apply_faults(self.spec, instance.faults)
        self._nominal = physical_readings(self.spec, solve(self.spec, self._params))  # read_sensor never solves
        self._sensor = {s.id: s for s in self.spec.sensors}
        self._state = {"status": "INIT", "spent": 0, "costed": 0, "simulate": 0, "strikes": 0, "submission": None}
        self._calls, self._t0 = 0, None
        self.trace: list[dict] = []
        self._dir = Path(state_dir) if state_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)
            (self._dir / "trace.jsonl").touch()  # exists even for a no-op episode (Harbor collects it)
            self._save()
        self._handlers = {"read_manual": self._read_manual, "get_topology": self._get_topology,
                          "simulate": self._simulate, "read_sensor": self._read_sensor,
                          "run_experiment": self._run_experiment, "budget_status": self._budget_status,
                          "submit": self._submit}

    # --- public -----------------------------------------------------------------------------
    @property
    def status(self) -> str:
        return self._state["status"]

    def state_hash(self) -> str:
        return hashlib.sha256(_canon(self._state).encode()).hexdigest()

    def call(self, tool: Any, args: Any) -> dict:
        logged, verdict = self._loggable(args)
        return self._call(tool, args if verdict is None else logged, verdict, logged)

    def _call(self, tool: Any, args: Any, verdict: str | None, logged: Any) -> dict:
        """verdict: None, TOO_LARGE or UNSERIALIZABLE; decided from the real arguments (or by replay), so an
        agent cannot forge it by sending a marker key."""
        with self._lock:
            self._cost = 0
            try:
                result = self._dispatch(tool, args, verdict)
            except ToolRejected as e:
                self._log(tool, logged, verdict, False, e.code, {"error": f"{e.code}: {e.message}"})
                raise
            except Exception:  # noqa: BLE001 - never let an internal error escape untraced or with raw text
                e = ToolRejected("E_INTERNAL", "internal error")
                self._log(tool, logged, verdict, False, e.code, {"error": f"{e.code}: {e.message}"})
                raise e from None
            if self.status == "INIT":  # only an accepted call starts the episode (rejections leave state as is)
                self._state = {**self._state, "status": "RUNNING"}
            self._log(tool, logged, verdict, True, None, result)
            return result

    def finalize(self) -> None:
        with self._lock:
            self._finalize()

    def true_readings(self, setpoint: dict | None = None) -> dict[str, float]:
        """Privileged, test-only: noise-free readings of the true plant (sensor faults included)."""
        sp = self._setpoint(setpoint or {}, "setpoint")
        return _round(self._with_sensor_faults(physical_readings(self.spec, solve(self.spec, self._params, sp)),
                                               self.instance.faults))

    # --- dispatch, caps, persistence --------------------------------------------------------
    def _dispatch(self, tool: Any, args: Any, verdict: str | None) -> dict:
        if self.status in ("SUBMITTED", "FINALIZED"):  # every call, even an unknown tool (server_spec section 2)
            raise ToolRejected("E_EPISODE_OVER", "the episode is over")
        if self._t0 is None:  # the wall clock starts at the first call, accepted or not
            self._t0 = self._clock()
        if self._clock() - self._t0 > TIME_CAP_S:
            self._finalize()
            raise ToolRejected("E_EPISODE_OVER", "time cap reached; episode finalized")
        known = isinstance(tool, str) and tool in self._handlers
        self._count_call(tool if known else "unknown")  # every call counts toward the cap, valid or not
        if not known:
            raise ToolRejected("E_UNKNOWN_TOOL", "unknown tool")
        if verdict == TOO_LARGE:
            raise ToolRejected("E_TOO_LARGE", f"arguments larger than {MAX_ARG_BYTES} bytes")
        if args is None:
            args = {}
        if verdict == UNSERIALIZABLE or not isinstance(args, dict):
            raise bad("arguments", "must be a JSON object")
        return self._handlers[tool](args)

    def _count_call(self, tool: str) -> None:
        if tool == "simulate":
            return
        self._calls += 1
        if self._calls > CALL_CAP:
            self._finalize()
            raise ToolRejected("E_EPISODE_OVER", "call cap reached; episode finalized")

    def _loggable(self, args: Any) -> tuple[Any, str | None]:
        """(what the trace stores, verdict). Oversized or unserializable arguments are logged as markers."""
        try:
            size = len(_canon(args))
        except (TypeError, ValueError, RecursionError, OverflowError):
            return {UNSERIALIZABLE: True}, UNSERIALIZABLE
        return ({TOO_LARGE: size}, TOO_LARGE) if size > MAX_ARG_BYTES else (args, None)

    def _finalize(self) -> None:
        if self.status != "FINALIZED":
            self._state = {**self._state, "status": "FINALIZED"}
            self._save()

    def _log(self, tool: Any, args: Any, verdict: str | None, ok: bool, code: str | None, response: dict) -> None:
        name = tool[:40] if isinstance(tool, str) else repr(tool)[:40]  # bounded trace size
        line = {"i": len(self.trace), "tool": name, "args": args, "verdict": verdict,
                "ok": ok, "code": code, "cost": self._cost, "spent": self._state["spent"],
                "budget_left": BUDGET - self._state["spent"], "response_hash": response_hash(response)}
        self.trace.append(line)
        if self._dir:
            with (self._dir / "trace.jsonl").open("a") as f:
                f.write(_canon(line) + "\n")
            self._save()

    def _save(self) -> None:
        if not self._dir:
            return
        tmp = self._dir / "episode.json.tmp"
        tmp.write_text(_canon({"seed": self.spec.seed, "tier": self.spec.tier, "state": self._state,
                               "calls": self._calls}))
        os.replace(tmp, self._dir / "episode.json")  # atomic: finalize never sees a half-written file

    def _charge(self, cost: int) -> int:
        """Reserve budget for a costed call; returns the noise call index."""
        if self._state["spent"] + cost > BUDGET:
            raise ToolRejected("E_BUDGET", f"insufficient budget ({BUDGET - self._state['spent']} left)")
        index = self._state["costed"]
        self._state = {**self._state, "spent": self._state["spent"] + cost, "costed": index + 1}
        self._cost = cost
        return index

    # --- shared helpers ---------------------------------------------------------------------
    def _setpoint(self, sp: Any, field: str) -> Setpoint:
        sp = T.obj(sp, field, set(), {"experiment", "s", "pump", "valve", "theta", "branch"})
        kind = sp.get("experiment", "nominal")
        if kind == "nominal":
            if len(sp) > ("experiment" in sp):
                raise bad(field, "nominal takes no arguments")
            return Setpoint()
        kind, args = T.experiment(self.spec, kind, {k: v for k, v in sp.items() if k != "experiment"},
                                  f"{field}.experiment", field)
        return experiment_setpoint(self.spec, kind, args)

    def _with_sensor_faults(self, phys: dict[str, float], faults: tuple[Fault, ...]) -> dict[str, float]:
        out = dict(phys)
        for f in faults:
            if f.type == "sensor_bias":
                out[f.component] += f.sign * f.magnitude * self._sensor[f.component].sigma
            elif f.type == "sensor_stuck":
                out[f.component] = f.value
        return out

    # --- tools ------------------------------------------------------------------------------
    def _read_manual(self, args: dict) -> dict:
        T.obj(args, "", set(), {"topic"})
        if "topic" not in args:
            return {"topic": "contents", "text": T.manual()["contents"] + "\n\nTopics: " + ", ".join(T.TOPICS)}
        topic = T.choice(args["topic"], "topic", T.TOPICS, "topic")
        return {"topic": topic, "text": T.manual().get(topic, "")}

    def _get_topology(self, args: dict) -> dict:
        T.obj(args, "", set())
        return T.public_topology(self.spec)

    def _simulate(self, args: dict) -> dict:
        T.obj(args, "", set(), {"overrides", "setpoint", "sensors"})
        if self._state["simulate"] >= SIMULATE_CAP:
            raise ToolRejected("E_RATE_LIMIT", f"simulate is limited to {SIMULATE_CAP} calls")
        overrides = [T.fault_obj(self.spec, o, f"overrides[{i}]", for_sim=True)
                     for i, o in enumerate(T.lst(args.get("overrides", []), "overrides", T.MAX_OVERRIDES))]
        if len({o["component"] for o in overrides}) != len(overrides):
            raise bad("overrides", "at most one entry per component")
        sp = self._setpoint(args.get("setpoint", {}), "setpoint")
        wanted = T.lst(args.get("sensors", list(self._sensor)), "sensors", len(self._sensor))
        for i, sid in enumerate(wanted):
            T.choice(sid, f"sensors[{i}]", self._sensor, "sensor")
        faults = tuple(Fault(o["component"], o["type"], o.get("magnitude"), o.get("sign", 1), o.get("value"))
                       for o in overrides)
        params = apply_faults(self.spec, faults)
        if any(f.type == "sensor_stuck" and f.value is None for f in faults):  # default: nominal reading
            nominal = physical_readings(self.spec, solve(self.spec, params))
            faults = tuple(Fault(f.component, f.type, None, value=nominal[f.component])
                           if f.type == "sensor_stuck" and f.value is None else f for f in faults)
        phys = self._with_sensor_faults(physical_readings(self.spec, solve(self.spec, params, sp)), faults)
        self._state = {**self._state, "simulate": self._state["simulate"] + 1}
        return {"readings": _round({sid: phys[sid] for sid in wanted}),
                "simulate_left": SIMULATE_CAP - self._state["simulate"]}

    def _read_sensor(self, args: dict) -> dict:
        T.obj(args, "", {"sensor_id"})
        sid = T.choice(args["sensor_id"], "sensor_id", self._sensor, "sensor")
        index = self._charge(COSTS["read"])
        value = observe(self.instance, self._nominal, [sid], index)[sid]
        return {"sensor_id": sid, "value": round(value, 6), "unit": T.UNITS[self._sensor[sid].kind],
                "cost": COSTS["read"], "budget_left": BUDGET - self._state["spent"], "events": []}

    def _run_experiment(self, args: dict) -> dict:
        T.obj(args, "", {"type", "args"})
        kind, exp_args = T.experiment(self.spec, args["type"], args["args"], "type", "args")
        sp = experiment_setpoint(self.spec, kind, exp_args)
        if self._state["spent"] + COSTS[kind] > BUDGET:  # budget check before any work on the true plant
            raise ToolRejected("E_BUDGET", f"insufficient budget ({BUDGET - self._state['spent']} left)")
        phys = physical_readings(self.spec, solve(self.spec, self._params, sp))
        index = self._charge(COSTS[kind])
        sensors = experiment_sensors(self.spec, kind, exp_args)
        readings = observe(self.instance, phys, sensors, index)
        return {"type": kind, "readings": _round(readings), "cost": COSTS[kind],
                "budget_left": BUDGET - self._state["spent"], "events": []}

    def _budget_status(self, args: dict) -> dict:
        T.obj(args, "", set())
        s = self._state
        return {"status": "RUNNING" if s["status"] == "INIT" else s["status"], "budget": BUDGET, "spent": s["spent"], "remaining": BUDGET - s["spent"],
                "calls_used": self._calls, "call_cap": CALL_CAP, "simulate_used": s["simulate"],
                "simulate_cap": SIMULATE_CAP, "time_used_s": round(self._clock() - self._t0, 1),
                "time_cap_s": TIME_CAP_S}

    def _submit(self, args: dict) -> dict:
        try:
            T.obj(args, "", {"diagnosis", "repairs"})
            diagnosis = [T.fault_obj(self.spec, f, f"diagnosis[{i}]", for_sim=False)
                         for i, f in enumerate(T.lst(args["diagnosis"], "diagnosis", T.MAX_ENTRIES))]
            repairs = [T.repair_obj(self.spec, r, f"repairs[{i}]")
                       for i, r in enumerate(T.lst(args["repairs"], "repairs", T.MAX_ENTRIES))]
        except ToolRejected as e:
            strikes = self._state["strikes"] + 1
            self._state = {**self._state, "strikes": strikes}
            if strikes > MAX_STRIKES:
                self._finalize()
                raise ToolRejected("E_EPISODE_OVER", "too many invalid submissions; finalized with an empty "
                                   "submission") from None
            raise ToolRejected("E_INVALID_SUBMISSION", f"{e.message} (invalid submission {strikes} of "
                               f"{MAX_STRIKES} allowed; the next one after that finalizes the episode)") from None
        self._state = {**self._state, "status": "SUBMITTED",
                       "submission": {"diagnosis": diagnosis, "repairs": repairs}}
        return {"status": "submitted"}
