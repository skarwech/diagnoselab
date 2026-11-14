"""Sensor model (F4): physical readings, sensor faults and seeded noise. docs/sim_spec.md section 5."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from diagnoselab.sim.model import Fault, Instance, PlantSpec
from diagnoselab.sim.solver import State

SENSOR_TYPES = ("sensor_bias", "sensor_stuck")


def physical_readings(spec: PlantSpec, state: State) -> dict[str, float]:
    """Noise-free physical value of every sensor (pressure head h - z, flow, power, vibration)."""
    z = {j.id: j.z for j in spec.junctions}
    source = {"flow": state.q, "power": state.power, "vibration": state.vib}
    return {s.id: (state.h[s.target] - z[s.target]) if s.kind == "pressure" else source[s.kind][s.target]
            for s in spec.sensors}


def observe(instance: Instance, physical: Mapping[str, float], sensor_ids: Iterable[str], call_index: int,
            faults: tuple[Fault, ...] | None = None) -> dict[str, float]:
    """Apply sensor faults and N(0, sigma) noise keyed by (noise_seed, call_index, sensor index)."""
    spec = instance.spec
    by_sensor = {f.component: f for f in (instance.faults if faults is None else faults) if f.type in SENSOR_TYPES}
    index = {s.id: i for i, s in enumerate(spec.sensors)}
    sigma = {s.id: s.sigma for s in spec.sensors}
    out = {}
    for sid in sensor_ids:
        fault = by_sensor.get(sid)
        # always draw the noise, even for a stuck sensor, so the work done does not depend on the faults
        noise = np.random.default_rng([instance.noise_seed, int(call_index), index[sid]]).standard_normal()
        if fault is not None and fault.type == "sensor_stuck":
            out[sid] = float(fault.value)
            continue
        bias = fault.sign * fault.magnitude * sigma[sid] if fault is not None else 0.0
        out[sid] = float(physical[sid] + bias + sigma[sid] * noise)
    return out
