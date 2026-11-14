# DiagnoseLab: hidden-fault diagnosis under a probe budget

You are the maintenance engineer for an industrial pump network. Some components are faulty, and some sensors may be lying. Find the faults, then submit a diagnosis and a repair plan. Use the `diagnoselab` MCP server.

## Tools

| Tool | Cost | What it does |
|---|---|---|
| `read_manual(topic?)` | free | Physics, fault signatures, experiment catalog, sensors, submission format and scoring. Start here. |
| `get_topology()` | free | Network graph, nominal datasheet parameters, and the sensor list with each sensor's noise sigma. |
| `simulate(overrides, setpoint?, sensors?)` | free, max 300 calls | A digital twin of the nominal plant, built on exactly the same physics as the real one. It returns noise-free predicted readings with your hypothesised faults applied. |
| `read_sensor(sensor_id)` | 1 | One noisy reading at the nominal operating point. |
| `run_experiment(type, args)` | 4 to 8 | `speed_step` (4), `valve_step` (4), `isolate_branch` (6) or `pump_off_static` (8). Returns noisy readings of the documented sensor subset under a temporary setpoint; the plant then reverts. |
| `budget_status()` | free | Budget, calls and time used. |
| `submit(diagnosis, repairs)` | final | One shot. Returns an acknowledgement, never a score. |

## Limits

- **Budget:** 26 units. A call that would exceed it is rejected.
- **Calls:** at most 150 tool calls, not counting `simulate`, which has its own limit of 300.
- **Time:** 30 minutes.
- **What happens at a limit:** the episode ends with whatever you have submitted, which may be nothing.

## Submission

```json
{"diagnosis": [{"component": "P3", "type": "fouling", "magnitude": 1.2}],
 "repairs":   [{"action": "clean_pipe", "target": "P3"}]}
```

- **Fault types:** `fouling` (pipe), `stuck_valve` (valve), `pump_wear` (pump), `leak` (junction), `sensor_bias` (sensor), `sensor_stuck` (sensor).
- **Magnitude:** the excess quantity. The manual defines it for each type. `sensor_stuck` takes no magnitude.
- **Repair actions:** `clean_pipe`, `free_valve`, `replace_pump`, `fix_leak`, `recalibrate_sensor`.
- **Size:** at most 10 entries per list.
- **Confidence:** an optional `confidence` field in [0, 1] is logged but never scored.

## Scoring

Scoring is server-side, against the hidden truth and a simulated repair of the true plant:

```
R = 0.45 * D + 0.35 * S + 0.20 * g(D) * (1 - spent / 26)
g(D) = clip((D - 0.4) / 0.3, 0, 1)
```

| Term | What it measures |
|---|---|
| D | Diagnosis quality: component, type and magnitude match; each false positive costs 0.5. |
| S | How much your repairs restore plant performance; each unneeded repair costs 0.15. |
| Efficiency | Counts only once the diagnosis is decent, so stopping early earns nothing. |

The manual (`read_manual("scoring")`) has the full definitions.
