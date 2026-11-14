# DiagnoseLab manual

Call `read_manual` with `topic` set to one of the keys below to read that section. Everything you need is in this manual or in `get_topology`.

| Topic | Contents |
|---|---|
| `overview` | The plant, your goal, the budget and the caps |
| `tools` | Every tool: arguments, cost, response, error codes |
| `physics` | The steady-state equations used by the plant and by `simulate` |
| `faults` | The six fault types, their magnitudes and their signatures |
| `experiments` | The experiment catalog, the sensors each returns, and how to separate confounded causes |
| `sensors` | Sensor kinds, ids, units and noise |
| `submission` | Diagnosis and repair formats |
| `scoring` | The reward formula and the KPI `J` |

## overview

**The plant.** A water pumping network in steady state.

- Source reservoirs `S*`, each at its own fixed head.
- Pumps `PU*`, each with a check valve. Each pump draws from a source through a suction pipe: `source -> pipe -> suction junction -> pump -> discharge junction`.
- A discharge-side network of junctions `J*`, pipes `P*` and control valves `V*`. It may contain loops.
- Outlet tanks `T*`, which all sit at one common fixed head `tank_head`. Every source head is at or below `tank_head`, so the pumps must lift the water. Each outlet has a required flow `q_req`.
- Nominal operation: every pump at speed `s = 1`, every valve commanded fully open (`theta = 1`).

**Your goal.** The plant has one or more physical faults, and possibly faulty sensors. Find them and call `submit` once, with a diagnosis (component, fault type, magnitude) and a repair plan.

**Hidden-fault rules.**
- At most one fault per component.
- Physical faults are `fouling` (pipe), `stuck_valve` (valve), `pump_wear` (pump) and `leak` (junction). Each one, on its own, degrades the KPI `J` (topic `scoring`) by at least 5%.
- Sensor faults are `sensor_bias` and `sensor_stuck`, on any sensor.

**Resources.**
- Budget B = 26 units. `read_sensor` costs 1. `run_experiment` costs 4, 4, 6 or 8 depending on the experiment. `read_manual`, `get_topology`, `simulate`, `budget_status` and `submit` are free.
- Caps: 150 tool calls (every tool except `simulate` counts, rejected calls included), 300 `simulate` calls, and 30 minutes of wall clock from your first call. Hitting the call cap or the time cap ends the episode with no diagnosis, which scores 0.
- `simulate` is a free digital twin. It solves exactly the plant equations (topic `physics`) for any faults you propose, at any setpoint, without noise. It knows nothing about the real faults.

**What you see.** Noisy readings (`read_sensor`, `run_experiment`), the datasheet (`get_topology`) and predictions (`simulate`). You never see a score: `submit` returns only an acknowledgement.

## tools

Every tool returns a JSON object. A rejected call returns an error result whose text is `"<code>: <message>"`. The message names the offending field and never echoes your value.

| Code | When |
|---|---|
| `E_INVALID_ARGUMENT` | Unknown field, wrong type, NaN or Inf, value out of range, or unknown id |
| `E_TOO_LARGE` | The JSON encoding of the arguments is larger than 16 KB |
| `E_BUDGET` | A costed call would exceed the remaining budget. Nothing runs and nothing is charged |
| `E_RATE_LIMIT` | `simulate` beyond its 300-call cap. The episode continues |
| `E_INVALID_SUBMISSION` | `submit` failed validation (topic `submission`). Counts as one strike |
| `E_EPISODE_OVER` | Any call after a valid `submit` or after the episode has ended. Also the call beyond the 150-call cap or past the 30-minute deadline, which ends the episode |
| `E_INTERNAL` | an unexpected internal error; the call had no effect |
| `E_UNKNOWN_TOOL` | No tool has that name |

A rejected call changes nothing, with three exceptions: it counts toward the 150-call cap (unless it is a `simulate` call), an invalid `submit` adds a strike, and a call that hits a cap ends the episode.

**Submit strikes.** Three invalid `submit` calls are tolerated. Each is rejected with `E_INVALID_SUBMISSION` and a field-level message. A fourth invalid `submit` ends the episode with an empty submission.

| Tool | Arguments | Cost | Response |
|---|---|---|---|
| `read_manual` | `topic`: optional, one of `overview`, `tools`, `physics`, `faults`, `experiments`, `sensors`, `submission`, `scoring` | 0 | `{"topic", "text"}`. With no topic: the table of contents |
| `get_topology` | none | 0 | The nominal datasheet (below) |
| `simulate` | `overrides`, `setpoint` (optional), `sensors` (optional) | 0, cap 300 | `{"readings": {sensor_id: value}, "simulate_left": n}` |
| `read_sensor` | `sensor_id` | 1 | `{"sensor_id", "value", "unit", "cost", "budget_left", "events"}` |
| `run_experiment` | `type`, `args` | 4, 4, 6 or 8 | `{"type", "readings": {sensor_id: value}, "cost", "budget_left", "events"}` |
| `budget_status` | none | 0 | `{"status", "budget", "spent", "remaining", "calls_used", "call_cap", "simulate_used", "simulate_cap", "time_used_s", "time_cap_s"}` |
| `submit` | `diagnosis`, `repairs` | terminal | `{"status": "submitted"}`. Never a score |

**`get_topology`** returns the nominal datasheet. It contains no faults, no noise and no hidden state.

| Field | Content |
|---|---|
| `tank_head` | Common head of every outlet tank (m) |
| `sources` | `{"id", "head"}` |
| `outlets` | `{"id", "head", "q_req"}`: `head` equals `tank_head`; `q_req` in L/s |
| `junctions` | `{"id", "z", "side"}`: elevation `z` (m) and `side`, `suction` or `discharge` |
| `links` | `{"id", "kind", "frm", "to", ...}`. Pipes carry `r`. Valves carry `r0`, the resistance at full opening. Pumps carry `a`, `b`, `q_bep`, `eta_max`, `v0`. Positive flow runs `frm -> to`; a pump's `frm` is its suction junction |
| `sensors` | `{"id", "kind", "target", "unit", "sigma"}` |
| `branches` | `{"id", "entry", "links", "junctions"}` (topic `experiments`) |
| `budget`, `costs` | 26, and the cost of each costed call |
| `speed_range`, `theta_range` | `[0.7, 1.2]` and `[0.1, 1.0]`; an experiment may not reproduce the nominal setpoint, so `s` must differ from 1 by at least 0.05 and `theta` must be at most 0.95 |

**`simulate`** takes `{"overrides": [...], "setpoint": {...}, "sensors": [...]}`.
- `overrides`: a list of at most 20 fault objects, at most one per component. Pass `[]` for the healthy plant. Each object is `{"component", "type", "magnitude"}`, plus optionally `"sign"` (-1 or 1, for `sensor_bias`, default 1) or `"value"` (the frozen reading, for `sensor_stuck`, default the nominal reading). `magnitude` is required except for `sensor_stuck`, and must lie in these ranges: fouling [0, 10], stuck_valve [0, 0.9], pump_wear [0, 0.9], leak [0, 0.3], sensor_bias [0, 50]. `confidence` is not accepted here.
- `setpoint`: optional, default nominal. It is a flat object: `{"experiment": "nominal"}`, `{"experiment": "speed_step", "s": 0.8}` (optionally with `"pump": "PU1"`), `{"experiment": "valve_step", "valve": "V1", "theta": 0.5}`, `{"experiment": "isolate_branch", "branch": "B1"}` or `{"experiment": "pump_off_static"}`. The argument ranges are those of `run_experiment`.
- `sensors`: optional list of the sensor ids to return. Omit it to get every sensor.
- The response holds noise-free readings of the nominal plant with your overrides applied, sensor overrides included. With the true faults as overrides it reproduces the real plant's noise-free readings exactly.

**`read_sensor`** takes `{"sensor_id": "PT-J3"}` and returns one noisy reading at the nominal setpoint. Every call draws fresh noise.

**`run_experiment`** takes `{"type": "<experiment>", "args": {...}}`. Note that `args` is nested here, unlike the flat `simulate` setpoint.

| `type` | `args` | Cost |
|---|---|---|
| `speed_step` | `{"s": number in [0.7, 1.2] with abs(s - 1) >= 0.05, "pump": optional pump id (default: all pumps)}` | 4 |
| `valve_step` | `{"valve": valve id, "theta": number in [0.1, 0.95]}` | 4 |
| `isolate_branch` | `{"branch": branch id}` | 6 |
| `pump_off_static` | `{}` | 8 |

It returns noisy readings of that experiment's sensor subset (topic `experiments`) under the temporary setpoint. The plant then reverts to nominal.

**`submit`** takes `{"diagnosis": [...], "repairs": [...]}`. A valid submit ends the episode. Format: topic `submission`.

**Common.** Numbers in responses are floats rounded to 6 decimals. Units: pressure head in m, flow in L/s, power in kW, vibration in mm/s. `events` is reserved and is always `[]`.

## physics

The plant is in steady state. `simulate` implements exactly these equations: it is a free, noise-free digital twin of the plant carrying whatever faults you give it.

**Units.** Head, elevation and pressure head in m; flow in L/s; power in kW; vibration in mm/s.

**Heads.** The unknowns are the junction heads `h`. Each source has its own fixed head, and every outlet tank is at `tank_head`. The pressure head at junction `j` is `p_j = h_j - z_j`. For a link `frm -> to`, `D = h[frm] - h[to]`, and positive flow means `frm -> to`. Constants: `EPS = 1e-4` m, `DELTA = 1e-6`.

```
pipe     q = D / sqrt(r * sqrt(D^2 + EPS^2))          (smoothed D = r * q * |q|)
         fouling m: r -> r * (1 + m)

valve    same law with r_eff = r0 / theta^2,  theta = actual opening in [0.1, 1]
         healthy: theta = commanded opening (nominal 1)
         stuck:   theta = 1 - m, whatever the command

pump     frm = suction junction, to = discharge junction, s = speed, w = wear
         sp(x) = (x + sqrt(x^2 + DELTA^2)) / 2          (smooth max(x, 0))
         x     = a * (1 - w) * s^2 - (h[to] - h[frm])
         q     = sqrt(sp(x) / b)
         => pump curve while it delivers:  H = h[to] - h[frm] = a*(1-w)*s^2 - b*q^2
         => the check valve blocks reverse flow; if the required rise exceeds
            a*(1-w)*s^2 the pump deadheads (q close to 0)

closed   q = 0 (the entry link closed by isolate_branch)

leak     q_leak_j = C_j * sqrt(sp(h_j - z_j))
         C_j      = f_j * Q_design / sqrt(p_design_j)

continuity at every junction j:
         sum(q into j) - sum(q out of j) - q_leak_j = 0
```

**Leak constants.** `f_j` is the leak magnitude. `Q_design` is the total pump flow and `p_design_j` the pressure head at `j`, both taken from the healthy plant at the nominal setpoint (the state `simulate` returns with `overrides = []`). At its design pressure a leak of magnitude `f` therefore draws `f * Q_design`; at another pressure it scales with `sqrt(p / p_design)`. `simulate` computes `C_j` internally, so you only handle `f`.

**Stopped pump.** At `s = 0` the check valve passes at most about 1e-5 L/s. Treat it as zero (tolerance 1e-4 L/s).

**Pump outputs.** They do not enter the hydraulics. With `H = h[to] - h[frm]` and `u = q / (s * q_bep)`:

```
eta = clip(eta_max * (1 - 0.5*w) * (1 - (u - 1)^2), 0.05, eta_max)
P   = 9.81 * q * H / (1000 * eta)                  [kW]
vib = v0 * s^2 * (1 + 2*w) * (1 + 3*(u - 1)^2)     [mm/s]
```

When `s = 0`, `P = 0` and `vib = 0`.

**Healthy reference.** The healthy plant at nominal is the design point. It meets every `q_req`, and its flows, power and vibration set the sensor sigmas (topic `sensors`) and `SE_design` (topic `scoring`).

## faults

The magnitude `m` is always the excess quantity and is always `>= 0`. The typical ranges below are where real faults lie.

| Type | Component | Effect | Magnitude `m` | Typical range |
|---|---|---|---|---|
| `fouling` | pipe | `r -> r * (1 + m)` | relative resistance increase | 0.3 to 2.0 |
| `stuck_valve` | valve | `theta = 1 - m`; commands ignored | closure `1 - theta_stuck` | 0.2 to 0.8 |
| `pump_wear` | pump | `w = m`: head coefficient `a*(1 - w)`, efficiency times `(1 - 0.5*w)`, vibration times `(1 + 2*w)` | wear fraction | 0.10 to 0.35 |
| `leak` | junction | leak law with `f = m` | leak flow as a fraction of `Q_design` at design pressure | 0.02 to 0.08 |
| `sensor_bias` | sensor | reading `+= sign * m * sigma`, sign -1 or +1 | offset in units of that sensor's sigma | 4 to 10 |
| `sensor_stuck` | sensor | reading = a constant, with no noise | none | none |

A stuck sensor's constant is the noise-free reading of the faulty plant at the nominal setpoint, so at nominal it looks plausible.

**Signatures.** The directions below were observed on generated plants at the nominal setpoint. Use `simulate` to get the exact size on your plant. "The other pump" applies only to plants with two pumps.

**`fouling` (pipe)**
- Nominal: more head loss across that pipe, so its flow falls and the pressure upstream of it rises relative to the pressure downstream. Total outlet flow falls; outlets fed through the fouled pipe lose flow, while outlets on other paths can gain flow.
- Fouling on the discharge-side network: every pump's flow falls, its pressure rise increases (the operating point moves up the healthy pump curve) and its power rises.
- Fouling of a suction pipe: that pump's suction pressure and flow fall, its rise and power increase; the other pump picks up flow with a lower rise.
- `pump_off_static`: invisible, because there is no flow and so no head loss.
- `speed_step`: the pipe's head loss scales with `q^2`. The pump's (flow, rise) points stay on the datasheet curve `a*s^2 - b*q^2`.
- `valve_step`: unaffected by the command. Nearby valves respond as `simulate` predicts with the fouling included.
- `isolate_branch`: inside the isolated branch, its effect disappears (no flow); elsewhere it persists.

**`stuck_valve` (valve)**
- Nominal: extra resistance on the valve, `r0 / (1 - m)^2`, with the same pattern as discharge-side fouling: the valve's flow falls, every pump's flow falls while its rise and power increase, total outlet flow falls, and outlets on other paths can gain flow.
- `valve_step` on that valve: no response. Its readings stay at their nominal values whatever `theta` you command. A healthy valve responds as `simulate` predicts.
- `pump_off_static`: invisible.
- `speed_step`: behaves as a fixed resistance, like fouling.
- `isolate_branch`: inside the isolated branch, its effect disappears.

**`pump_wear` (pump)**
- Nominal: that pump's flow, pressure rise, discharge pressure and power fall; its vibration rises; every outlet's flow falls. The other pump picks up flow with a lower rise.
- `speed_step` on that pump: its (flow, rise) points lie on the reduced curve `a*(1 - w)*s^2 - b*q^2`, below the datasheet curve at every speed. Vibration carries the factor `(1 + 2*w)`.
- `pump_off_static`: invisible (the pump is stopped).
- `valve_step`, `isolate_branch`: the pump's operating point moves along the reduced curve.

**`leak` (junction)**
- Nominal: extra outflow at the junction. Pressures near the leak fall and every outlet's flow falls. Continuity is broken at the leaking junction: where every incident link is metered, metered inflow minus metered outflow equals `q_leak`.
- Discharge-side leak: every pump's flow rises while its pressure rise, its discharge pressure and its power fall.
- Suction-side leak: the suction pipe's flow rises, but that pump's flow falls while its rise and power increase (its suction pressure drops); the other pump picks up flow with a lower rise.
- `pump_off_static`: the tanks (discharge-side leak) or the source (suction-side leak) backfeed the leak through the pipes. Meters on that path read flow towards the leak, and pressures sag slightly (topic `experiments`).
- `speed_step`: `q_leak` follows `sqrt(p)` at the leak, so the imbalance changes with the operating point.
- `isolate_branch`: a leak inside the isolated branch is fed backwards from the branch's own tank, so the branch meters read flow towards the leak. With no leak inside, branch flows are zero.

**`sensor_bias` (sensor)**
- A constant offset `sign * m * sigma` at every operating point. Its residual (reading minus the `simulate` prediction) is the same in every experiment, while residuals caused by physical faults change with the operating point. The sensor disagrees with its neighbours and with continuity.
- `pump_off_static`: a biased pressure sensor reads its static value plus the offset; a biased flow meter reads the offset instead of about 0.

**`sensor_stuck` (sensor)**
- It returns the same value exactly (all 6 decimals) on every call and at every operating point, with no noise. A healthy sensor's noise differs on every call. When an experiment moves the true value, a stuck sensor does not move.

## experiments

| Experiment | `args` | Cost | Temporary setpoint | Returned sensors (installed ones only) |
|---|---|---|---|---|
| nominal (`read_sensor`) | `sensor_id` | 1 per read | every pump at `s = 1`, every valve commanded to 1 | the sensor read |
| `speed_step` | `s` in [0.7, 1.2] with abs(s - 1) >= 0.05, `pump` optional (default: all pumps) | 4 | the chosen pump(s) at speed `s` | PT on the affected pumps' suction and discharge junctions, their WT and VT, all FT |
| `valve_step` | `valve`, `theta` in [0.1, 0.95] | 4 | that valve commanded to `theta` | PT on the valve's end junctions and on their junction neighbours, FT on the links touching the end junctions |
| `isolate_branch` | `branch` | 6 | the branch's entry link closed | all FT, PT on the branch junctions and on the upstream (`frm`) junction of the entry link |
| `pump_off_static` | none | 8 | every pump at `s = 0` | all PT and all FT |

Readings are noisy (topic `sensors`), and the plant reverts to nominal after each experiment. `simulate` with the same setpoint predicts every returned reading under any hypothesis.

**Branches.** A branch is the maximal chain from an outlet tank back towards the root, through junctions that have exactly two links. `entry` is the chain's link nearest the root; `links` and `junctions` list the chain. Closing the entry leaves the branch hanging from its own tank, which makes a local static test, and never disconnects a junction from every fixed-head node.

**What each experiment separates.**
- `speed_step`: pump wear vs system resistance; leak vs flow-meter bias; stuck sensors (the true values move, a stuck reading does not).
- `valve_step`: stuck valve vs fouling vs healthy valve.
- `isolate_branch`: localises a leak or fouling to a branch; gives a mass balance around the branch; with no leak inside, branch pressure sensors read their static value `tank_head - z`, a local bias check.
- `pump_off_static`: a zero-flow test. Pressures must read their static values and flow meters must read zero, except for leak backfeed. Sensor biases and meter offsets read out directly, and leaks show as backfeed.

**Confounds and the experiment that separates them.**

| Symptom at nominal | Candidate causes | Disambiguating experiment | Why it works |
|---|---|---|---|
| Low pressure downstream of a pipe | fouling, stuck valve, sensor bias, upstream leak | `pump_off_static` | At zero flow, fouling and valves have no effect and every pressure must read its static value, so a deviation is a sensor bias (or a stuck sensor, fact (g)); a leak shows as backfeed flow |
| Low pressure downstream of a pipe next to a valve | fouling vs stuck valve | `valve_step` on that valve | A healthy valve responds as modeled, a stuck valve does not respond, and fouling is unaffected by the command |
| Flow mismatch between meters | leak vs flow-meter bias | `speed_step` (mass balance at two operating points); `pump_off_static` reads meter offsets directly | Leak flow scales with the square root of pressure head; a meter bias is constant |
| Low pump flow, changed pump power | pump wear vs fouling (system resistance) | `speed_step` (speed sweep of that pump) | Pump rise vs flow traces the pump curve `a*(1-w)*s^2 - b*q^2`: wear lowers the curve, fouling leaves it unchanged and moves the operating point along it |

**Measured facts.**
- **(a) Static test, no leak.** Every discharge-side junction head equals `tank_head` and every suction junction head equals its source head (to about 1e-8 m), and all flows are zero. A pressure sensor must therefore read `tank_head - z` (discharge side) or `source head - z` (suction side), and a flow meter must read 0. A pressure sensor that deviates from its static value by several sigma is faulty: biased, or stuck at its nominal value. Fact (g) tells the two apart.
- **(b) Static test with a leak.** The tanks (or the source, for a suction-side leak) backfeed the leak through the pipes, so flow meters on that path read nonzero flow directed towards the leak. For a leak of magnitude 0.08, the largest such reading is typically several sigma up to about 25 sigma, and observed up to about 36 sigma; it scales with the magnitude. Pressures sag slightly: under 1 sigma at every junction other than the leaking one, and up to about 1.6 sigma at the leaking junction itself (typically about 0.2 sigma). A continuity-consistent pattern across meters (flows that add up along the path to one junction) points to a leak; a lone offset on one meter, with its neighbours at zero, points to a faulty meter (biased, or stuck at its nominal value; fact (g)). If only one meter lies on the path, compare both hypotheses with `simulate` and use a speed step. If no meter lies on the path, the static test shows no flow signature; use the nominal mass balance, a speed step or `isolate_branch`. When you suspect a leak, read a pressure bias against `simulate` at the static setpoint with the leak included, not against the plain static value.
- **(c) Stopped pumps.** A stopped pump passes about 1e-5 L/s at most through its check valve, which is negligible.
- **(d) Valve steps.** A stuck valve does not respond to `valve_step`, a healthy valve does, and fouling is unaffected by the command. A shallow step (for example `theta = 0.9`) can move readings by only a few sigma, and a deeper step moves them much more; check the predicted response with `simulate` before you spend.
- **(e) Pump wear vs system resistance.** Compare the pump's rise against its flow across speed steps. Pressure sensors read pressure head, so the rise is `H = (p_dis + z_dis) - (p_suc + z_suc)`. A healthy pump satisfies `H = a*s^2 - b*q^2`; a worn pump satisfies `H = a*(1-w)*s^2 - b*q^2`. The pump flow comes from a meter on the pump, on its suction pipe (equal unless the suction junction leaks) or on its discharge pipe (equal only if the discharge junction has no other link and no leak); otherwise fit it with `simulate`, using WT and VT as well.
- **(f) Leak vs meter bias.** Leak flow scales with the square root of the pressure head at the leak, while a flow-meter bias is constant across operating points: the meter's residual against `simulate` is the same at every setpoint. This is the weakest of the four separations. For a typical leak, one speed step or one static test changes the leak's contribution to a single meter by only a few sigma. Combine the evidence: the leak also moves the meters upstream of it and the nearby pressures, while a bias moves only its own meter. Repeated reads average down the noise.
- **(g) Stuck sensors.** A stuck sensor returns the same value exactly, whatever the operating point. A healthy sensor never repeats a value exactly.

**Planning with `simulate`.** For two candidate explanations, simulate the readings each predicts under a candidate experiment. The experiment is informative where the predictions differ by several sigma on the sensors it returns. Fit magnitudes by adjusting the overrides until the simulated readings match the measured ones within noise.

## sensors

| Kind | Id | Physical reading | Unit | sigma |
|---|---|---|---|---|
| pressure | `PT-<junction>` | pressure head `h - z` | m | 0.15 |
| flow | `FT-<link>` | link flow `q`, signed (positive `frm -> to`) | L/s | `0.01 * max(abs(q_design_link), 1.0)` |
| power | `WT-<pump>` | pump power `P` | kW | `0.01 * P_design` |
| vibration | `VT-<pump>` | pump vibration `vib` | mm/s | `0.05 * vib_design` |

Design values come from the healthy plant at nominal. Each sensor's sigma is listed in `get_topology`, as are the installed sensors: PT on every pump's suction and discharge junctions and on some other junctions, FT on some links, WT and VT on every pump.

**Noise model.**

```
reading = physical + bias_effect + sigma * N(0, 1)
```

- `N(0, 1)` is independent for every sensor and every costed call. Repeated reads of one sensor are independent, so averaging `n` reads reduces the noise by `sqrt(n)`. Sensors returned by the same experiment carry independent noise.
- `bias_effect` is `sign * m * sigma` for a biased sensor and 0 otherwise. It is the same at every operating point.
- A stuck sensor returns its constant, noise-free, on every call.
- `simulate` returns noise-free readings.
- Values are rounded to 6 decimals.

## submission

```json
{"diagnosis": [{"component": "P3", "type": "fouling", "magnitude": 1.2, "confidence": 0.7}],
 "repairs":   [{"action": "clean_pipe", "target": "P3"}]}
```

**Diagnosis.** At most 10 fault objects.
- `component`: a component id from `get_topology`.
- `type`: must match the component's kind.
- `magnitude`: any finite number `>= 0`, the excess quantity below. Required except for `sensor_stuck`; omit it there.
- `confidence`: optional, in [0, 1]. It is logged and not scored.
- `sign` and `value` are `simulate`-only fields and are not accepted here.
- Duplicate entries for one component are allowed, but each extra entry counts as a false positive.

| Type | Component kind | Magnitude to submit |
|---|---|---|
| `fouling` | pipe | `r_actual / r - 1` |
| `stuck_valve` | valve | `1 - theta_stuck` |
| `pump_wear` | pump | `w` |
| `leak` | junction | `f` |
| `sensor_bias` | sensor | `abs(offset) / sigma` (unsigned) |
| `sensor_stuck` | sensor | none |

**Repairs.** At most 10 objects `{"action", "target"}`. The action must match the target's kind.

| Action | Target kind |
|---|---|
| `clean_pipe` | pipe |
| `free_valve` | valve |
| `replace_pump` | pump |
| `fix_leak` | junction |
| `recalibrate_sensor` | sensor |

A repair on a component that is not faulty is unneeded and is penalised (topic `scoring`). Recalibrating a faulty sensor is not penalised, although sensor faults do not change `J`.

**Lifecycle.** Exactly one valid `submit` is accepted. It ends the episode and returns `{"status": "submitted"}`. An invalid one is rejected with `E_INVALID_SUBMISSION` and a field-level message; three are tolerated, and a fourth ends the episode with an empty submission. An episode that ends without a valid `submit` is scored as an empty submission.

## scoring

Scoring runs server-side, after the episode, against the hidden truth. You never see the score.

```
R    = 0.45 * D + 0.35 * S + 0.20 * g(D) * (1 - spent / B)        B = 26
g(D) = clip((D - 0.4) / 0.3, 0, 1)
```

**D (diagnosis).**
- A submitted fault matches a true fault when component and type agree.
- Magnitude credit: with `e = abs(m_sub - m_true) / m_true` rounded to 6 decimals, credit is 1.0 if `e <= 0.30`, 0.5 if `e <= 0.60`, else 0. An error of exactly 0.30 or 0.60 is inside the band. A matched `sensor_stuck` earns 1.0. `sensor_bias` magnitudes are unsigned (`abs(offset) / sigma`).
- Each true fault is matched by at most one entry, the one with the highest credit. The right component and type with a magnitude more than 60% off is still a match: it earns credit 0, and it is not a false positive.
- `FP` counts submitted faults that match no true fault, including any extra entry for a fault that is already matched. Each costs 0.5.
- `D = clip(sum(credit) - 0.5 * FP, 0, n_true) / n_true`, where `n_true` counts all true faults, physical and sensor. The order of entries does not matter.

**S (restoration).**
- Your repairs are applied to the true plant; each repair removes the fault on its target. Sensor repairs do not change the plant, so they do not move `J`.
- `restoration = (J_after - J_before) / (J_healthy - J_before)`, with `J` evaluated at the nominal setpoint.
- `S = clip(restoration - 0.15 * n_unneeded, 0, 1)`. A repair is unneeded when its target is not a faulty component, or when it repeats an earlier repair of the same target. Recalibrating a truly faulty sensor is needed: it costs nothing, though it does not move `J`.
- Repairs interact: in rare cases, repairing only some of the faults can lower `J`, for example by sending freed flow to an outlet that already had enough.

**Efficiency.** `g(D) * (1 - spent / B)`. It pays nothing while `D <= 0.4` and gets its full weight once `D >= 0.7`.

**KPI J.**

```
SE = sum(P_pumps) / (3.6 * sum(q into outlet tanks))       [kWh/m^3]
J  = -( sum_o max(0, q_req_o - q_o) / q_req_o  +  SE / SE_design )
```

- The delivered flow in the `SE` denominator is floored at `0.1 * Q_design`.
- `SE_design` is the `SE` of the healthy plant at nominal, so a healthy plant has `J` of about -1.

**Not scored.** `confidence` is not scored. An empty submission gives `D = 0`, `S = 0` and `R = 0`.
