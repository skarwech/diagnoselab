# DiagnoseLab simulator spec (F1–F5 contract)

This file is the contract that both solvers, the tests and the manual follow. If the code and this file disagree, the code is the bug.

## 1. Units and identifiers

| Quantity | Unit |
|---|---|
| Head, elevation, pressure head | m |
| Flow | L/s |
| Power | kW |
| Vibration | mm/s |

Every identifier is a string, with the following prefixes.

| Kind | Prefix | Example |
|---|---|---|
| Source reservoir (fixed head) | `S` | `S1` |
| Outlet tank (fixed head, common for all tanks) | `T` | `T1` |
| Junction | `J` | `J3` |
| Pipe | `P` | `P4` |
| Control valve | `V` | `V1` |
| Pump with check valve | `PU` | `PU1` |
| Pressure sensor | `PT-<junction>` | `PT-J3` |
| Flow meter | `FT-<link>` | `FT-P4` |
| Power sensor | `WT-<pump>` | `WT-PU1` |
| Vibration sensor | `VT-<pump>` | `VT-PU1` |
| Branch | `B` | `B1` |

## 2. Network

- **Fixed-head nodes.** Sources `S` each have their own `head`. Outlet tanks `T` all sit at `tank_head`. The generator guarantees `tank_head >= max(source heads)`.
- **Junctions.** Each junction has an elevation `z` and a `side`, either `suction` or `discharge`. A suction junction connects only to its source pipe and to its pump.
- **Links.** Each link runs `frm -> to`, and positive flow means flow in that direction. Unknowns are the heads `h` at junctions; fixed nodes have known heads.

### Link laws

Let `D = h[frm] - h[to]`. The smoothing constants are `EPS = 1e-4` (m) and `DELTA = 1e-6`.

**Pipe**

```
q = D / sqrt(r * sqrt(D^2 + EPS^2))
```

**Valve**

The same law as a pipe, with `r_eff = r / theta^2`.
- `theta` is the actual opening, in `[0.1, 1]`.
- `theta` equals the commanded opening (nominal 1). A stuck valve ignores the command and holds `theta = 1 - m`.

**Pump**

`frm` is the suction junction and `to` is the discharge junction.

```
sp(x) = (x + sqrt(x^2 + DELTA^2)) / 2         # smooth max(x, 0)
x     = a * (1 - w) * s^2 - (h[to] - h[frm])
q     = sqrt(sp(x) / b)
```

- `s` is the speed fraction. `s = 0` means the pump is stopped, and the check valve then makes `q` effectively 0.
- `w` is the pump wear.

**Closed link**

`q = 0`. It is removed from the system.

**Numerics.** For `x < 0`, evaluate `sp(x)` as `DELTA^2 / (2 * (sqrt(x^2 + DELTA^2) - x))`; the literal form cancels. A stopped pump still passes about `DELTA / (2 * sqrt(b * lift))`, around 1e-5 L/s, so zero-flow checks use an absolute tolerance of 1e-4 L/s.

### Leak and continuity

**Leak at junction j**

```
q_leak = C_j * sqrt(sp(h_j - z_j))
C_j    = f_j * Q_design / sqrt(p_design_j)
```

- `f_j` is the leak magnitude.
- `Q_design` is the sum of the pump flows at the design point (`design.Q_pump`).
- `p_design_j = h_design_j - z_j`.

**Continuity at every junction**

```
sum(q into j) - sum(q out of j) - q_leak_j = 0
```

### Pump outputs

These do not enter the hydraulics. Let `H = h[to] - h[frm]` and `u = q / (s * q_bep)`.

```
eta = clip(eta_max * (1 - 0.5*w) * (1 - (u - 1)^2), 0.05, eta_max)
P   = 9.81 * q * H / (1000 * eta)                 [kW]
vib = v0 * s^2 * (1 + 2*w) * (1 + 3*(u - 1)^2)    [mm/s]
```

When `s = 0`, `P = 0` and `vib = 0`.

## 3. KPI

```
SE = sum(P_pumps) / (3.6 * sum(q into outlet tanks))      [kWh/m^3]
J  = -( sum_o max(0, q_req_o - q_o) / q_req_o  +  SE / SE_design )
```

The delivered flow in the `SE` denominator is floored at `0.1 * Q_design`, so a deadheaded pump or tank backfeed keeps `J` finite.

- The weights are `w1 = 1` and `w2 = 1 / SE_design`. `SE_design` is the specific energy at the design point, so a healthy plant has `J = -1` or slightly above.
- **Relative degradation:** `(J_healthy - J_faulty) / |J_healthy|`.

## 4. Faults

Each fault is `Fault(component, type, magnitude, sign=1, value=None)`. The magnitude is always the **excess quantity** and is always `>= 0`.

| Type | Component | Effect | Magnitude range | Physical max (after drift) |
|---|---|---|---|---|
| `fouling` | pipe | `r <- r * (1 + m)` | [0.3, 2.0] | 4.0 |
| `stuck_valve` | valve | `theta = 1 - m`; commands ignored | [0.2, 0.8] | 0.9 |
| `pump_wear` | pump | `w = m` (head, efficiency, vibration) | [0.10, 0.35] | 0.6 |
| `leak` | junction | `f = m` | [0.02, 0.08] | 0.15 |
| `sensor_bias` | sensor | reading `+= sign * m * sigma` | [4, 10] | 14 |
| `sensor_stuck` | sensor | reading `= value` (constant, no noise) | none | none |

- **Stuck value.** For `sensor_stuck`, `value` is the noise-free physical reading of the faulted plant (before drift) at the nominal setpoint.
- **Sampling.**
  - At most one fault per component.
  - Physical faults are drawn uniformly over the **eligible** (component, type) pairs.
  - A pair is eligible if applying it alone to the healthy plant at its maximum range magnitude degrades the KPI by at least 5%.
  - The magnitude is drawn uniformly from `[max(lo, m*), hi]`, where `m*` is the smallest magnitude that reaches 5% (found by bisection; the KPI is monotone in magnitude).
  - Sensor faults are drawn uniformly over all (sensor, type) pairs with type in {`sensor_bias`, `sensor_stuck`}. The bias sign is ±1 with equal probability.
- **Faults per tier.**

  | Tier | Physical | Sensor | Drift |
  |---|---|---|---|
  | easy | 1 | 0 | no |
  | medium | 2 | 1 | no |
  | hard | 3 | 2 | no (cut, option E) |

- **Drift: cut (user decision, option E; plan cut list item 1).** Instances carry no drift. The primitives below stay implemented and tested so drift can return. When enabled, it is pre-drawn at generation, with each kind equally likely:
  - `grow`: an existing physical fault's magnitude is multiplied by `1 + g`, with `g` in [0.4, 0.8], then clipped to the physical max.
  - `new`: an eligible physical pair on a healthy component is added, with a magnitude drawn as above.

  The trigger is pure. A costed call fires the drift when `spent_before < 0.4*B <= spent_before + cost`. When the drift fires is a Step 3 decision.

## 5. Sensors

```
reading = physical + bias_effect + sigma * N(0, 1)
```

- **Stuck sensors** return `value` and have no noise.
- **Noise.** `N(0, 1)` is drawn from `numpy.random.default_rng([noise_seed, call_index, sensor_index])`, where `sensor_index` is the sensor's position in `spec.sensors`.

The noise level `sigma` depends on the sensor kind.

| Kind | sigma |
|---|---|
| pressure | 0.15 m |
| flow | `0.01 * max(abs(q_design_link), 1.0)` L/s |
| power | `0.01 * P_design` |
| vibration | `0.05 * vib_design` |

Physical readings are:
- pressure: `h - z`
- flow: `q`
- power: `P`
- vibration: `vib`

## 6. Experiments (catalog in `sim/experiments.py`)

| Kind | Args | Cost | Setpoint | Returned sensors |
|---|---|---|---|---|
| `nominal` | none | (read 1 each) | s = 1 for every pump, valve commands = 1 | the one sensor read |
| `speed_step` | `s` in [0.7, 1.2], `pump` (optional; default all) | 4 | the chosen pump(s) at `s` | PT on the affected pumps' suction and discharge junctions, their WT and VT, all FT |
| `valve_step` | `valve`, `theta` in [0.1, 1.0] | 4 | commanded opening `theta` | PT on the valve's end junctions and their junction neighbours, FT on links touching the end junctions |
| `isolate_branch` | `branch` | 6 | the branch entry link closed | all FT, PT on the branch junctions and on the upstream junction of the entry link |
| `pump_off_static` | none | 8 | every pump at s = 0 | all PT and all FT |

- **Static-test expectations.** With no leak, every discharge-side junction has `h = tank_head` and every suction junction has `h = its source head`, so flows are 0. A leak lets the tanks (or the source) backfeed the leak, which is a documented leak signature.
- **Branches.** A branch is the maximal chain from an outlet tank back towards the root, through junctions of degree 2. Its entry link is the chain's link nearest the root. Closing the entry link never disconnects a junction from every fixed-head node.

## 7. Generator (inverse design)

Given `(seed, tier)`, independent RNG streams `default_rng([seed, tier_code, stream])` are used, with streams `structure=0`, `faults=1`, `noise=2` and `drift=3`. The `tier_code` values are easy 0, medium 1 and hard 2.

### Structure

| Item | easy | medium | hard |
|---|---|---|---|
| Junctions | 6–8 | 8–10 | 10–12 |
| Pumps | 1 | 1–2 | 2 |
| Outlets | 2–3 | 2–4 | 3–4 |
| Loop chords | 0 | 1 | 1–2 |
| Valves | 2–4 | 2–4 | 2–4 |

- **Sources:** 1, or one per pump.
- **Suction side:** each pump gets one suction junction and one discharge junction. The suction side is `source -> pipe -> suction junction -> pump -> discharge junction`.
- **Discharge side:** each discharge junction feeds a header (the root) through a pipe. The remaining junctions form a random recursive tree under the root; once it has the tier's maximum outlet count of leaves, new nodes attach to a leaf, so no redraw is needed. The outlet count is then drawn from [max(min, leaves), max]. Outlet tanks hang from the leaves, plus extra outlets on random junctions. Loop chords link pairs of discharge junctions that are not adjacent.
- **Valves:** discharge-side links (preferring outlet links) are converted to valves.

### Design point (chosen first; parameters derived from it)

1. **Outlet flows** `q_o` are drawn from U[8, 25].
2. **Chord flows** are 0.2 × the smallest tree flow on the chord path, oriented high head to low head.
3. **Tree flows** come from continuity. Pump split: U[0.4, 0.6] when there are two pumps.
4. **Heads**:
   - `tank_head = 50`.
   - Tree drops are U[1, 4] per link, measured down from the root.
   - Each outlet link drop is at least U[1, 3].
   - Source heads are `tank_head - U[3, 12]`.
   - Suction pipe drops are U[0.3, 1.5].
5. **Derived parameters**:
   - Pipe and valve resistance: `r = D / q^2`, with valves at `theta = 1`.
   - Pump: `rise = h_d - h_s`, `a = max(rise * U[1.25, 1.6], (lift + 2) / 0.7^2)`, where lift = h_root - source head, so a healthy pump stepped alone to the lowest speed never deadheads, `b = (a - rise) / q^2`, `q_bep = q / U[0.8, 1.1]`, `eta_max` from U[0.70, 0.85], `v0` from U[1.5, 3.0].
6. **Elevations**:
   - Discharge junctions: `z = tank_head - U[2, 15]`.
   - Suction junctions: `z = source_head - U[2, 6]`.
7. **Requirements:** `q_req = 0.99 * q_o`.

Every float in the spec is rounded to 9 significant digits. The spec hash is the sha256 of the canonical JSON of the rounded spec (sorted keys, `repr` floats).

### Sensors

- PT on every pump suction and discharge junction, plus about 60% of the other junctions.
- FT on about 50% of links.
- WT and VT on every pump.
- Series junctions: every junction that joins exactly two non-pump links gets a PT. Without one, faults on either side are indistinguishable (cosine 1).
- Coverage repair: every pipe must have a PT on one of its end junctions or an FT on itself.
- Chords: if a small tree has no non-adjacent pair with end heads at least 0.5 m apart, the head drops are redrawn.

## 8. Spec dict format (`PlantSpec.to_dict()` / `PlantSpec.from_dict()`)

```json
{"seed": 1, "tier": "easy", "tank_head": 50.0,
 "junctions": [{"id": "J1", "z": 40.0, "side": "suction"}],
 "sources":   [{"id": "S1", "head": 45.0}],
 "outlets":   [{"id": "T1", "q_req": 12.3}],
 "links":     [{"id": "P1", "kind": "pipe", "frm": "S1", "to": "J1", "r": 0.002,
                "a": 0.0, "b": 0.0, "q_bep": 0.0, "eta_max": 0.0, "v0": 0.0}],
 "sensors":   [{"id": "PT-J1", "kind": "pressure", "target": "J1", "sigma": 0.15}],
 "branches":  [{"id": "B1", "entry": "V1", "links": ["V1", "P5"], "junctions": ["J7"]}],
 "design":    {"h": {"J1": 44.0}, "q": {"P1": 30.0}, "Q_pump": 30.0, "P": {"PU1": 9.1},
               "vib": {"PU1": 2.0}, "SE": 0.08}}
```

`from_dict` accepts hand-built specs for analytic tests. In that case `sensors`, `branches` and `design` may be empty; `Q_pump` and `p_design` are only needed for leaks.

## 9. Python API

```python
from diagnoselab.sim.model import PlantSpec, Fault, Setpoint, Instance
from diagnoselab.sim.generator import generate_plant, spec_hash
from diagnoselab.sim.solver import solve, kpi
from diagnoselab.sim.faults import (apply_faults, sample_faults, eligible_pairs,
                                    sample_drift, apply_drift, drift_fires, MAGNITUDE_RANGES)
from diagnoselab.sim.sensors import physical_readings, observe
from diagnoselab.sim.experiments import experiment_setpoint, experiment_sensors, COSTS
from diagnoselab.sim.instance import make_instance
from diagnoselab.sim.certifier import certify
```

| Function | Signature | Behaviour |
|---|---|---|
| `generate_plant` | `(seed: int, tier: str) -> PlantSpec` | Builds the plant structure and design point (section 7). |
| `spec_hash` | `(spec) -> str` | sha256 of the canonical rounded spec. |
| `Setpoint` | `(speed: dict[str, float] = {}, valve: dict[str, float] = {}, closed: frozenset[str] = frozenset())` | Missing entries mean nominal: s = 1, theta = 1. |
| `apply_faults` | `(spec, faults: tuple[Fault, ...] = ()) -> Params` | Applies faults to the spec's parameters. |
| `solve` | `(spec, params: Params \| None = None, setpoint: Setpoint \| None = None) -> State` | `State` fields: `.h` (dict over junctions and fixed nodes), `.q` (dict over links), `.leak` (dict over junctions), `.power` (dict over pumps), `.vib` (dict over pumps), `.iterations: int`, `.residual: float` (max absolute junction residual). |
| `kpi` | `(spec, state) -> float` | The KPI `J` of section 3. |
| `physical_readings` | `(spec, state) -> dict[sensor_id, float]` | Noise-free physical values for every sensor. |
| `observe` | `(instance, physical: dict, sensor_ids, call_index: int) -> dict[sensor_id, float]` | Applies sensor faults and noise. |
| `make_instance` | `(seed: int, tier: str) -> Instance` | Returns `Instance(spec, faults, drift, noise_seed, cert)`, whose faults are certified (section 10). |
| `Fault` | `(component: str, type: str, magnitude: float \| None, sign: int = 1, value: float \| None = None, onset: str = "start")` | Frozen dataclass. `onset` is `"drift"` for drift faults. |
| `Drift` | `(kind: "grow" \| "new", index: int \| None, factor: float \| None, fault: Fault \| None)` | Frozen dataclass. `grow` uses `index` (position in faults) and `factor` (1+g); `new` uses `fault`. |
| `eligible_pairs` | `(spec) -> tuple[tuple[str, str], ...]` | Eligible physical (component, type) pairs, in spec order. |
| `sample_faults` | `(spec, tier, rng: numpy.random.Generator) -> tuple[Fault, ...]` | Physical faults first, then sensor faults. |
| `sample_drift` | `(spec, faults, rng) -> Drift` | Pre-draws the drift. |
| `apply_drift` | `(spec, faults, drift) -> tuple[Fault, ...]` | Returns the post-drift fault set. |
| `drift_fires` | `(spent_before: float, cost: float, budget: float = 40) -> bool` | The pure trigger of section 4. |
| `MAGNITUDE_RANGES`, `PHYSICAL_MAX` | `dict[type, (lo, hi)]`, `dict[type, float]` | Ranges and drift caps from the section 4 table. |
| `certify` | `(spec, faults, budget: int = BUDGET) -> CertResult` | `CertResult(accepted: bool, crb_rel: dict[str, float], coherence: dict[str, float], reasons: tuple[str, ...], spend: dict[str, int], separation: dict[str, float])`. `crb_rel` and `coherence` are keyed by `"component:type"`; `spend` is the greedy allocation (probe label to repeat count). |

## 10. Certifier (F5)

**Candidate parameters**

| Component | Parameter |
|---|---|
| pipe | `fouling` |
| valve | `stuck_valve` |
| pump | `pump_wear` |
| junction | `leak` |
| sensor | `sensor_bias` |

**Columns**
- A column is the whitened finite-difference derivative of every probe's readings, `d reading / d param / sigma`.
- The derivative is taken at the true plant, at the parameter's true value for true faults and at 0 for the others.
- For a healthy valve, 0 means stuck at `theta = 1`.

**Probe menu**
- One nominal read per sensor.
- `pump_off_static`.
- `speed_step` at s in {0.7, 0.85, 1.1} for each pump.
- `valve_step` at theta in {0.3, 0.6} for each valve.
- `isolate_branch` for each branch.

**Accept an instance only if all of the following hold.**
1. **CRB rule.** Repeated probes are allowed, with total cost at most B = 40, chosen by greedy allocation. With that allocation, the Cramér–Rao standard deviation of each continuous true-fault magnitude is at most 10% of its magnitude. The Fisher information is restricted to the true support.
2. **Coherence rule (option E).** On the full menu (each probe once), a true-fault column whose absolute cosine similarity with another candidate column exceeds 0.95 is accepted only if it is still at least 5σ from the best-fitting competitor at its true magnitude: `m * ||col|| * sqrt(1 - cos^2) >= 5`. A stuck valve is a discrete change, so its signature is the secant `(y(F) - y(F without it)) / m`, compared with candidate columns taken in the world without it.
3. **Stuck-sensor observability.** Some menu probe moves the healthy reading at least 5σ away from the stuck value.
4. **Stuck-valve observability.** `valve_step(v, theta)` for theta in {0.3, 0.6} moves some returned reading at least 5σ away from what a healthy valve would give.

Rejected fault sets are resampled from the same fault stream; the structure is kept. Acceptance rates are reported per fault type.
