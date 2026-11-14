"""Plant data model (docs/sim_spec.md sections 1, 8, 9). All records are immutable."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

SIG_DIGITS = 9


def rnd(x: float) -> float:
    """Round to SIG_DIGITS significant digits (canonical spec floats)."""
    return float(f"{x:.{SIG_DIGITS}g}")


@dataclass(frozen=True)
class Junction:
    id: str
    z: float
    side: str = "discharge"  # "suction" | "discharge"


@dataclass(frozen=True)
class Source:
    id: str
    head: float


@dataclass(frozen=True)
class Outlet:
    id: str
    q_req: float


@dataclass(frozen=True)
class Link:
    id: str
    kind: str  # "pipe" | "valve" | "pump"
    frm: str
    to: str
    r: float = 0.0
    a: float = 0.0
    b: float = 0.0
    q_bep: float = 0.0
    eta_max: float = 0.0
    v0: float = 0.0


@dataclass(frozen=True)
class Sensor:
    id: str
    kind: str  # "pressure" | "flow" | "power" | "vibration"
    target: str
    sigma: float


@dataclass(frozen=True)
class Branch:
    id: str
    entry: str
    links: tuple[str, ...]
    junctions: tuple[str, ...]


@dataclass(frozen=True, eq=False)  # identity hash: compiled-network caches key on the object
class PlantSpec:
    seed: int
    tier: str
    tank_head: float
    junctions: tuple[Junction, ...]
    sources: tuple[Source, ...]
    outlets: tuple[Outlet, ...]
    links: tuple[Link, ...]
    sensors: tuple[Sensor, ...] = ()
    branches: tuple[Branch, ...] = ()
    design: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["branches"] = [{**b, "links": list(b["links"]), "junctions": list(b["junctions"])}
                         for b in d["branches"]]
        return json.loads(json.dumps(d))  # deep copy into plain JSON types

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> PlantSpec:
        return PlantSpec(
            seed=int(d.get("seed", 0)),
            tier=str(d.get("tier", "custom")),
            tank_head=float(d["tank_head"]),
            junctions=tuple(Junction(**j) for j in d["junctions"]),
            sources=tuple(Source(**s) for s in d.get("sources", ())),
            outlets=tuple(Outlet(**o) for o in d.get("outlets", ())),
            links=tuple(Link(**ln) for ln in d["links"]),
            sensors=tuple(Sensor(**s) for s in d.get("sensors", ())),
            branches=tuple(Branch(b["id"], b["entry"], tuple(b["links"]), tuple(b["junctions"]))
                           for b in d.get("branches", ())),
            design=json.loads(json.dumps(d.get("design", {}))),
        )


def spec_hash(spec: PlantSpec) -> str:
    canonical = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Fault:
    component: str
    type: str
    magnitude: float | None
    sign: int = 1
    value: float | None = None  # stuck-sensor constant
    onset: str = "start"  # "start" | "drift" (F3 ground-truth record)


@dataclass(frozen=True)
class Drift:
    kind: str  # "grow" | "new"
    index: int | None = None
    factor: float | None = None
    fault: Fault | None = None


@dataclass(frozen=True)
class Setpoint:
    speed: Mapping[str, float] = field(default_factory=dict)
    valve: Mapping[str, float] = field(default_factory=dict)
    closed: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Params:
    """Physical fault state, sparse by component id (healthy = all empty)."""

    fouling: Mapping[str, float] = field(default_factory=dict)  # pipe -> m
    stuck: Mapping[str, float] = field(default_factory=dict)  # valve -> actual theta
    wear: Mapping[str, float] = field(default_factory=dict)  # pump -> w
    leak: Mapping[str, float] = field(default_factory=dict)  # junction -> f


@dataclass(frozen=True)
class CertResult:
    accepted: bool
    crb_rel: Mapping[str, float]
    coherence: Mapping[str, float]
    reasons: tuple[str, ...]
    spend: Mapping[str, int]
    separation: Mapping[str, float] = field(default_factory=dict)  # sigma, at the true magnitude


@dataclass(frozen=True)
class Instance:
    spec: PlantSpec
    faults: tuple[Fault, ...]
    drift: Drift | None
    noise_seed: int
    cert: CertResult | None = None
    attempts: int = 1
