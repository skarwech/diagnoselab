"""Certified instances: plant + faults for (seed, tier). Any integer seed works."""

from __future__ import annotations

from diagnoselab.sim.certifier import certify
from diagnoselab.sim.faults import sample_faults
from diagnoselab.sim.generator import FAULTS, NOISE, generate_plant, stream
from diagnoselab.sim.model import Instance

MAX_ATTEMPTS = 60


def make_instance(seed: int, tier: str) -> Instance:
    """Resample faults from the fault stream until the certifier accepts.

    ponytail: mid-episode drift is cut (user decision, option E; plan cut list item 1), so drift is
    always None. faults.sample_drift/apply_drift/drift_fires stay tested so drift can return.
    """
    spec = generate_plant(seed, tier)
    rng_faults = stream(seed, tier, FAULTS)
    noise_seed = int(stream(seed, tier, NOISE).integers(2**62))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        faults = sample_faults(spec, tier, rng_faults)
        cert = certify(spec, faults)
        if cert.accepted:
            return Instance(spec, faults, None, noise_seed, cert, attempt)
    raise RuntimeError(f"no certifiable fault set for seed={seed} tier={tier} in {MAX_ATTEMPTS} attempts")
