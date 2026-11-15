# DiagnoseLab oracle spec (F19 contract)

This spec is fixed before the oracle is tuned or measured.

## 1. What the oracle is

The oracle is the reference run, and doubles as the fairness proof that the task is solvable with the information and tools an agent has. It is one self-contained file, `diagnoselab/oracle/oracle.py`, with entry point `run_oracle(call)`.

- `call(tool, args)` returns a tool result dict, and raises on rejection.

**What it may use**

- The tool responses.
- The manual's constants: costs, typical magnitude ranges, override ranges, the returned-sensor rules and sigma (sigma comes from `get_topology`).
- The Python standard library.

**What it may not use**

- Any `diagnoselab` import (checked statically), so it runs unchanged in the agent container as Harbor's `solve.sh`.
- `numpy`.
- Any private physics model. The digital twin is `simulate`, within the episode's 300-call cap.
- Fault counts inferred from the tier. A fault needs evidence, never a quota.
- Repairs of undiagnosed components. Repair gating would therefore leave its score unchanged.

**Determinism**

- Every score that decides an argmax is rounded to 1e-9, and ties go to a fixed candidate order.
- The tools round readings to 6 decimals, and the arithmetic is pure Python IEEE.
- The same seed therefore gives the same call sequence on any machine.

## 2. Algorithm (as built; settings tuned on dev seeds 0–49)

**Parameters**

- Physical candidates: every link with its kind's fault type, and every junction with `leak`.
- One bias per sensor. A bias is analytic (sigma units) and needs no `simulate` call.

**Screening (free columns).** At three operating points every candidate gets a whitened secant column per unit magnitude, step = the manual's typical mid magnitude, at the healthy plant: nominal, the all-pump `speed_step` at s = 1.2, and `pump_off_static` (leaks only: at zero flow, fouling, valves and wear are invisible). Hard plants: about 70 `simulate` calls. Local operating points (valve steps at theta = 0.3, branch isolations, pump-local steps: all pumps at s = 0.75 on one-pump plants; each pump at s = 0.75 and at s = 1.2 on two-pump plants) are screened lazily, only for parameters in play.

**Loop**

1. **Coverage phase.** Until every live sensor has two readings, probes are chosen by Bayesian D-optimal gain per cost over every candidate and every bias, with prior variance 0.1 × MID². No fit runs in this phase. Two readings make a bias a testable constant offset and expose a stuck sensor, which repeats a value exactly (manual fact (g)). Fitting earlier let biases on once-read sensors absorb real faults (measured: hard D 0.91 vs 0.94).
2. **Fit after every later probe.**
   - Greedy additions: a candidate joins if its real SSR drop is ≥ 25 (5 sigma); physical magnitudes are ≥ 0, within the simulate ranges.
   - Bounded Levenberg-Marquardt on the support, with per-parameter finite-difference columns re-anchored after 15% drift.
   - Pruning: an element is removed if its explicit removal costs < 9 (3 sigma).
   - Swaps: up to 2 single swaps tested with real refits.
   - Each fit has a soft allowance of 35 `simulate` calls for optional work.
3. **Stop rule inputs.** For each element: its swap margin against the best rival and its relative sd. Candidates whose drop is in [9, 25) count as ambiguous. An element is unresolved if its margin is < 25 or its relative sd is > 0.15.
4. **Focus phase.** Probes are chosen by D-optimal gain per cost over the unresolved elements, their competitors and ambiguous candidates only, including local probes that touch them.
5. **Stop** when nothing is unresolved, when the spend would exceed 70% of the budget `get_topology` reports (28 of 40; B = 26 was probed in Step 6 and rejected), when no probe gains ≥ 0.05 nats per unit, or when `simulate` runs down to the reserve (10 + (support + 2) per operating point in use).
6. **Final polish** (no allowance, every improvement committed at once):
   - Borderline elements (linear margin < 50) are refit against their 2 leading rivals.
   - One-for-two pair search when the fit is poor (SSR z > 1.5) or ambiguous candidates remain: pairs among an element's 6 leading rivals are scored by two-column least squares, and a pair that beats the element by ≥ 9 gets a real refit. This fixes the common local minimum of one leak standing in for a neighbouring leak plus a fouled pipe.
   - Remove-one-and-regrow when the SSR z-score is > 3.
7. **Submit.** Every supported fault with its magnitude, plus a repair for each diagnosed physical component. Sensor repairs never move J, so none are made.

**Tuning log (dev seeds 0–49).** These variants were tried and rejected:
- confirming each fitted bias with an immediate read (medium D −0.05);
- requiring two readings before a bias may be fitted, while fitting during coverage (hard D −0.13);
- a coverage bonus of 0.5 or 0.25 nats per needed read (worse than 0);
- swap margin 36.

Kept: no coverage bonus, prior inclusion 0.1, swap margin 25.

## 3. Information-efficiency metric (evaluation side; may use the truth)

**Information after each costed call.** After each costed call k, compute:

```
info_k = logdet(P + J_{1..k}^T J_{1..k}) - logdet(P)
```

- `J` holds the whitened finite-difference rows of every sensor returned by calls 1..k, at that call's setpoint, evaluated at the true plant.
- The columns are the true fault support only: continuous magnitudes, with stuck sensors excluded.
- `P = diag(1 / (10 * m_i)^2)` is the certifier's weak prior.

The curve plots `info_k` against cumulative spend.

**Comparator.** A random-probe policy makes uniformly random costed calls among the same probe kinds until its spend reaches the oracle's final spend. It uses 5 random sequences per instance.

**Pass condition (F19).** The oracle's `info` at its final spend exceeds the median of the 5 random sequences' `info` at the same spend, on at least 90% of instances.

## 4. Measurement protocol

- **Seeds.** Tune on dev seeds 0–49 and report on dev seeds 50–99. F19 rows use seeds 0–29 per tier (the plan's 30), F5 and the cache use 0–199. Seeds 1000 and above are never used here.
- **Plan thresholds:**

  | Item | Threshold |
  |---|---|
  | D | ≥ 0.9 on easy and medium, ≥ 0.8 on hard |
  | Spend | ≤ 70% of B |
  | Runtime | < 2 min per episode |
  | Determinism | 3 repeats give the same call sequence and score |
  | Tool diversity | each experiment type in ≥ 10% of episodes, reads in ≥ 50% |
  | Gap to lazy heuristic | `R_oracle ≥ R_lazy + 0.3` |

- **Usage table.** Tool usage is reported as measured. A tool the oracle never chooses is a finding, not something to engineer around.
- **Gap condition on easy.** `R_oracle ≥ R_lazy + 0.3` cannot hold on easy, because R_lazy = 0.82 and the maximum reward is 1. It is reported, not chased.
- **Cache.** Oracle scores are cached in `diagnoselab/scoring/oracle_scores.json`, keyed `"tier:seed"`, generated in the Linux image (`make oracle-cache`) so they reproduce under Harbor; the macOS run is compared entry by entry.
