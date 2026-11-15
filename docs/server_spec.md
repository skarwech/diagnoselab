# DiagnoseLab server spec (F6–F14 contract)

This file is the contract for the episode core, the MCP binding, the manual and the tests. If the code and this file disagree, the code is the bug. The physics comes from `docs/sim_spec.md`.

## 1. Architecture

- **Episode core.** `diagnoselab/server/episode.py` holds all logic: validation, the ledger, caps, the state machine, the trace and the tool behaviour.
  - Everything enters through one entry point, `Episode.call(tool: str, args: object) -> dict`, which raises `ToolRejected(code, message)`.
  - Raw arguments go straight to it, so every call is validated, traced and counted the same way, valid or not.
  - A single lock serialises calls, because mcp 2.3 runs sync tools in worker threads.
- **MCP binding.** `diagnoselab/server/app.py` subclasses `MCPServer` and overrides `list_tools` and `call_tool`.
  - `list_tools` advertises the strict JSON schemas of section 3 (`additionalProperties: false`).
  - `call_tool` passes the raw `arguments` dict to `Episode.call` and turns `ToolRejected` into an `isError` result whose text is `"<code>: <message>"`.
  - The library's own validation never runs.
- **One episode per sim container.** State is global to the process, never keyed by MCP session. Several MCP connections, for example a reconnect, see the same episode.
- **Instance.** The instance comes from `make_instance(seed, tier)` at server start. `seed` and `tier` arrive only through the sim container's environment (`DL_SEED`, `DL_TIER`) and are never returned by any tool.
- **Constant timing.** At start the server solves the true plant at the nominal setpoint, so `read_sensor` never runs the solver. The MCP binding pads every `read_sensor` and `run_experiment` reply, accepted or not, to a constant 50 ms, so solver work on the true plant does not show in response time. A stuck sensor still draws its noise, so the work done is the same.
- **Internal errors.** Any unexpected exception inside a tool becomes `E_INTERNAL: internal error`. It is traced, and the state it leaves is the state before the tool ran: `run_experiment` solves before charging.

## 2. State machine, ledger and caps

**States**
- `INIT`: no call yet.
- `RUNNING`: entered on the first accepted call. The wall clock starts at the first call, accepted or not, so a rejected first call leaves the state (and its hash) unchanged.
- `SUBMITTED`: entered on a valid `submit`.
- `FINALIZED`: entered through `Episode.finalize()` (the Harbor collect hook), or when a cap is hit.
- In `SUBMITTED` and `FINALIZED` every call, including one to an unknown tool, is rejected with `E_EPISODE_OVER`.

**Budget**
- B = 40, set once in `diagnoselab/sim/experiments.py`.
- Costs: `read_sensor` 1, `speed_step` 4, `valve_step` 4, `isolate_branch` 6, `pump_off_static` 8.
- `read_manual`, `get_topology`, `simulate`, `budget_status` and `submit` cost 0.
- A costed call that would exceed the remaining budget is rejected with `E_BUDGET` before anything happens.

**Caps.** Defaults; both open points are listed for the user in section 8.
- **Call cap.** 150 tool calls in total, counting every call except `simulate`, and counting rejected calls too.
  - Call number 151 finalizes the episode with the current, possibly empty, submission, and is rejected with `E_EPISODE_OVER`.
- **Simulate cap.** `simulate` has its own cap of 300.
  - Call 301 is rejected with `E_RATE_LIMIT`; the episode keeps running.
- **Time cap.** 30 minutes of wall clock from the first call.
  - The first call after the deadline finalizes the episode and is rejected with `E_EPISODE_OVER`.
  - The clock is injected (`Episode(..., clock=time.monotonic)`) so tests can move time.

**Invalid submissions**
- A `submit` that fails validation is rejected with `E_INVALID_SUBMISSION` and a field-level message, and counts as one strike.
- After 3 strikes, the next invalid `submit` finalizes the episode with an empty submission and is answered with `E_EPISODE_OVER` (plan F12).
- Arguments too large to parse (`E_TOO_LARGE`) are rejected before validation and add no strike.

**State hash.** `Episode.state_hash()` is the sha256 of the canonical JSON of the episode state:
- status
- spent
- costed-call count
- simulate count
- strike count
- submission

It excludes the trace, the total-call counter and the clock. A rejected call must leave `state_hash()` unchanged, with two deliberate exceptions:
- an invalid `submit` adds a strike;
- a call that hits a cap finalizes the episode.

**Noise key.** Readings from a costed call use `call_index` = the number of costed calls accepted before it (0-based). Every sensor in one experiment shares that index; their noise differs by sensor index (sim_spec section 5).

**Drift.** Drift is cut (option E). `events` is always `[]`, and the field is kept so drift can return without changing the response shape.

## 3. Tools

An unknown tool name is rejected with `E_UNKNOWN_TOOL`; it still counts toward the call cap. Unknown fields, wrong types, NaN or Inf, out-of-range values and unknown ids are all rejected with `E_INVALID_ARGUMENT`, except inside `submit`, where any validation failure is `E_INVALID_SUBMISSION` and counts as a strike. The message names the field, never echoes the value given, and never reveals hidden state. Arguments whose JSON encoding is larger than 16 KB are rejected with `E_TOO_LARGE` before they are parsed further; the call still counts toward the call cap. Numbers too large for a float (for example `10**400`) are rejected as non-finite. Tool names are logged truncated to 40 characters.

| Tool | Arguments | Cost | Response |
|---|---|---|---|
| `read_manual` | `topic`: optional, one of `overview`, `tools`, `physics`, `faults`, `experiments`, `sensors`, `submission`, `scoring` | 0 | `{"topic", "text"}`. With no topic: the table of contents. |
| `get_topology` | none | 0 | Nominal datasheet only (section 4). Never seed, tier, faults, noise or design point. |
| `simulate` | `overrides`: optional list of at most 20 fault objects (section 5; default `[]`)<br>`setpoint`: optional (section 6)<br>`sensors`: optional list of sensor ids | 0 (cap 300) | `{"readings": {sensor_id: value}, "simulate_left": n}`. Noise-free predicted readings of the nominal plant with the overrides applied. Uses no hidden state. |
| `read_sensor` | `sensor_id` | 1 | `{"sensor_id", "value", "unit", "cost", "budget_left", "events"}`. The reading is taken at the nominal setpoint. |
| `run_experiment` | `type`: `speed_step`, `valve_step`, `isolate_branch` or `pump_off_static`<br>`args`: object (section 6) | 4, 4, 6, 8 | `{"type", "readings": {sensor_id: value}, "cost", "budget_left", "events"}`. Noisy readings of the documented sensor subset (sim_spec section 6). The plant reverts to nominal afterwards. |
| `budget_status` | none | 0 | `{"status", "budget", "spent", "remaining", "calls_used", "call_cap", "simulate_used", "simulate_cap", "time_used_s", "time_cap_s"}` |
| `submit` | `diagnosis`: list of at most 10 fault objects<br>`repairs`: list of at most 10 repair objects | terminal | `{"status": "submitted"}` only. Never a score. |

Values are floats rounded to 6 decimals. Units: pressure head in `m`, flow in `L/s`, power in `kW`, vibration in `mm/s`.

## 4. `get_topology` response

```json
{"tank_head": 50.0,
 "sources":   [{"id": "S1", "head": 41.2}],
 "outlets":   [{"id": "T1", "head": 50.0, "q_req": 12.3}],
 "junctions": [{"id": "J1", "z": 37.0, "side": "suction"}],
 "links":     [{"id": "P1", "kind": "pipe", "frm": "S1", "to": "J1", "r": 0.002},
               {"id": "V1", "kind": "valve", "frm": "J7", "to": "T1", "r0": 0.01},
               {"id": "PU1", "kind": "pump", "frm": "J1", "to": "J2", "a": 31.2, "b": 0.004, "q_bep": 30.0,
                "eta_max": 0.8, "v0": 2.1}],
 "sensors":   [{"id": "PT-J1", "kind": "pressure", "target": "J1", "unit": "m", "sigma": 0.15}],
 "branches":  [{"id": "B1", "entry": "V1", "links": ["V1"], "junctions": []}],
 "budget": 40,
 "costs": {"read_sensor": 1, "speed_step": 4, "valve_step": 4, "isolate_branch": 6, "pump_off_static": 8},
 "speed_range": [0.7, 1.2],
 "theta_range": [0.1, 1.0]}
```

The response for two instances with the same nominal plant and different faults is byte-identical (F7). Leak coefficients use `C = f * Q_design / sqrt(p_design)`. `simulate` computes this internally, so the agent works only with the magnitude `f`.

## 5. Fault objects (shared by `simulate` overrides and `submit` diagnosis)

```json
{"component": "P3", "type": "fouling", "magnitude": 1.2, "confidence": 0.7}
```

**Types and the component kind each needs**

| Type | Component kind |
|---|---|
| `fouling` | pipe |
| `stuck_valve` | valve |
| `pump_wear` | pump |
| `leak` | junction |
| `sensor_bias` | sensor |
| `sensor_stuck` | sensor |

**Magnitude**
- It is the excess quantity of sim_spec section 4, and is required except for `sensor_stuck`.
- In `submit`: any finite number ≥ 0.
- In `simulate` it must lie within these ranges:

| Type | Range |
|---|---|
| fouling | [0, 10] |
| stuck_valve | [0, 0.9] |
| pump_wear | [0, 0.9] |
| leak | [0, 0.3] |
| sensor_bias | [0, 50] |

**Override-only fields** (`simulate`, not `submit`)
- `sign`: -1 or 1 for `sensor_bias`, default 1.
- `value`: the frozen reading for `sensor_stuck`; default is the nominal reading.

**Confidence**
- `confidence` is optional, in [0, 1], and accepted in `submit` only.
- It is logged and never scored.

**Duplicates**
- At most one entry per component in `simulate` overrides.
- In `submit`, duplicates are allowed; scoring counts them as false positives.

**Repair objects**
- Shape: `{"action": "...", "target": "<component id>"}`.
- The action must match the target's kind:

| Action | Target kind |
|---|---|
| `clean_pipe` | pipe |
| `free_valve` | valve |
| `replace_pump` | pump |
| `fix_leak` | junction |
| `recalibrate_sensor` | sensor |

## 6. Setpoints and experiment arguments

| Experiment | `args` |
|---|---|
| `speed_step` | `{"s": number in [0.7, 1.2] with abs(s - 1) >= 0.05, "pump": optional pump id (default: all pumps)}` |
| `valve_step` | `{"valve": valve id, "theta": number in [0.1, 0.95]}` |
| `isolate_branch` | `{"branch": branch id}` |
| `pump_off_static` | `{}` |

An experiment may not reproduce the nominal setpoint: `abs(s - 1) < 0.05` or `theta > 0.95` is rejected with `E_INVALID_ARGUMENT`. Otherwise a no-op step would be a cheap bulk nominal read. `simulate` setpoints follow the same rule.

A `simulate` `setpoint` is `{"experiment": "<type or nominal>", ...args}`, for example `{"experiment": "valve_step", "valve": "V1", "theta": 0.5}`. If it is missing, the setpoint is nominal.

## 7. Trace, persistence, replay and finalize

**Persistence**
- `episode.json` holds the state, `seed` and `tier`, and is rewritten atomically after every call (temp file plus `os.replace`).
- It lives in the sim container under a directory that only the `sim` user can read.

**Trace (`trace.jsonl`)**
- Append-only and flushed once per call.
- Fields per line:

| Field | Meaning |
|---|---|
| `i` | 0-based call index |
| `tool` | tool name |
| `args` | raw arguments, or `{"__too_large__": n_bytes}` / `{"__unserializable__": true}` |
| `verdict` | null, `"__too_large__"` or `"__unserializable__"`: decided from the real arguments, so an agent cannot forge it |
| `ok` | whether the call was accepted |
| `code` | error code, or null |
| `cost` | budget units charged |
| `spent` | budget spent after the call |
| `budget_left` | budget remaining after the call |
| `response_hash` | hash of the response (below) |

- `response_hash` is the sha256 of the canonical JSON of the response, or of `{"error": "<code>: <message>"}`. The `time_used_s` field is excluded.

**Replay**
- `diagnoselab.server.replay.replay(trace_path, seed, tier) -> list[int]` feeds the logged arguments, in order, to a fresh `Episode`.
- The episode runs on a fixed clock that never reaches the time cap.
- It returns the indices whose response hash differs; an empty list means a perfect replay.

**Finalize**
- `python -m diagnoselab.server.finalize --state DIR --out FILE` runs as root in the sim container (Harbor `verifier.collect`).
- It rebuilds the instance from `seed` and `tier` in `episode.json`, then writes `{"reward": float, ...}`.
- Until Step 4 the reward is a placeholder: the fraction of true faults matched on component and type, minus 0.5 per false positive, clipped to [0, 1]. Empty or missing submission scores 0.

## 8. Defaults that need the user's confirmation

1. **Caps.** The 150-call cap excludes `simulate`, which has its own 300 cap. If `simulate` counted, the two caps could not both hold.
2. **Invalid submissions.** The fourth invalid `submit` finalizes the episode with an empty submission. Plan F12 is silent on what follows the third rejection, and F22 wants no episode lost to format errors.
3. **Stuck sensors.** They read an exact constant, as plan F3 says.
4. **No tier or fault counts are disclosed.** The manual says "one or more physical faults, and possibly faulty sensors".

## 9. Isolation (F14)

- **Agent image.** It contains only bash, curl, Python and the mcp client. It has no simulator package, no truth and no shared volumes.
- **Network.** Harbor's `no-network` and `allowlist` modes would block internet egress, which an agent could use to fetch the public repository and regenerate plants.
  - Harbor enforces them with nftables (`CONFIG_NFT_FIB_INET`). Docker Desktop's VM kernel lacks it, so on that setup Harbor rejects either mode and only `public` runs.
  - Per-phase switching (`[agent] network_mode`) is also unavailable there.
  - Judged runs should use a Linux Docker host with a static `[environment] network_mode = "allowlist"` limited to the model API (plus npm, unless OpenCode is baked into the agent image). This is untested.
  - With a non-public mode, the MCP URL becomes `http://127.0.0.1:8000/mcp`, because the containers then share a network namespace with Harbor's egress sidecar.
- **Sim surface.** The sim exposes exactly one listener: MCP on port 8000. Finalize runs only through `docker compose exec`, and no route returns truth or a score.

## 10. Python API (for tests)

```python
from diagnoselab.server.episode import Episode, ToolRejected, BUDGET, CALL_CAP, SIMULATE_CAP, TIME_CAP_S, MAX_STRIKES
from diagnoselab.server.tools import TOOL_SCHEMAS          # name -> JSON schema advertised to the agent
from diagnoselab.server.replay import replay
from diagnoselab.server.app import build_server              # build_server(episode) -> MCPServer subclass
from diagnoselab.sim.instance import make_instance
```

| API | Behaviour |
|---|---|
| `Episode(instance, state_dir: Path \| None = None, clock: Callable[[], float] = time.monotonic)` | With `state_dir=None` nothing is written to disk; the trace is kept in memory only. |
| `ep.call(tool, args) -> dict` | Raises `ToolRejected` with `.code` and `.message`. |
| `ep.status` | `"INIT"`, `"RUNNING"`, `"SUBMITTED"` or `"FINALIZED"`. |
| `ep.state_hash() -> str` | Hash of the episode state (section 2). |
| `ep.trace -> list[dict]` | In-memory copy of the trace lines. |
| `ep.finalize() -> None` | Idempotent. |
| `ep.true_readings(setpoint_dict) -> dict` | Privileged and test-only: the noise-free readings of the true plant, for the F8 fidelity check. Not exposed as a tool. |
