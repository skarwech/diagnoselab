# DiagnoseLab scoring spec (F15–F18 contract)

This file is the contract for the reward, the baselines and the tests. If the code and this file disagree, the code is the bug.

- The reward follows plan section 5.
- Where the plan contradicts itself, this file states the choice and the reason.
- Scoring runs server-side in the sim container, with only numpy available: no scipy.

## 1. Inputs

- **Truth.** The certified instance's faults, rebuilt from `(seed, tier)` by `make_instance`. Drift is cut, so the truth never changes during an episode.
- **Submission.** The validated submission stored by the episode (server_spec section 5).
  - `diagnosis` holds `{component, type, magnitude?, confidence?}` entries.
  - `repairs` holds `{action, target}` entries.
  - Either list may be empty.
  - With no submission, the score is R = 0.
- **Spend.** `spent`: the budget spent, in [0, 40].

## 2. Diagnosis score D (F15)

**Matching.** A submitted fault *matches* a true fault when `component` and `type` are both equal.
- Each true fault can be matched by at most one submitted entry: the one with the highest credit.
- Ties go to the earliest entry, which does not change the score.
- Every true fault sits on a different component, so this greedy matching is optimal and does not depend on order.

**Magnitude credit** for a matched pair.
- The relative error is `e = round(abs(m_sub - m_true) / m_true, 6)`. Rounding to 1e-6 follows plan decision 8, so an error of exactly 0.30 or 0.60 lands inside the band.
- `m_true` is the excess quantity and is always > 0.
- Credit is 1.0 if e ≤ 0.30, 0.5 if e ≤ 0.60, and 0 otherwise.
- `sensor_stuck` has no magnitude, so a component-and-type match earns credit 1.0.
- `sensor_bias` magnitudes are unsigned (|offset| / sigma); the sign is ignored.

**False positives.** FP is the number of submitted entries that match no true fault.
- That includes duplicates beyond the best entry for a true fault.
- **Choice:** a correct component and type with a wildly wrong magnitude is a match with credit 0, *not* a false positive. This follows plan section 5.2. The plan's Appendix A sketch counts it as a false positive, which breaks F15's own property that adding a correct fault never lowers D.

**Score**

```
D = clip(sum(credit) - 0.5 * FP, 0, n_true) / n_true
```

- `matched_credit` is the sum of the best entry's credit for each true fault.
- D itself is not rounded; only the relative error `e` is.
- Every instance has at least one true fault, so `n_true` is never 0.
- **Monotonicity (F15).** Adding a correct fault never lowers D, provided its (component, type) is not already in the submission. A resubmitted fault is a duplicate, which counts as a false positive by design.

## 3. Restoration score S (F16)

**KPI terms.** All are evaluated on the true plant at the nominal setpoint (sim_spec section 3).
- `J_before`: the KPI with every true physical fault present.
- `J_healthy`: the KPI with no fault.
- `J_after`: the KPI after removing every physical fault whose component is the `target` of a submitted repair.
- Repair actions always match the target's kind (server_spec section 5), so a repair of a faulty component removes its fault.
- Sensor repairs do not change the plant, so they do not move J.

**Unneeded repairs.** A repair is unneeded if either:
- its target is not a faulty component (physical or sensor); or
- it repeats an earlier repair of the same target. A second repair of the same component fixes nothing, so it costs the same 0.15.

Recalibrating a truly faulty sensor is *needed*, so it costs nothing, but it does not move J.

**Score**

```
restoration = (J_after - J_before) / (J_healthy - J_before)
S = clip(restoration - 0.15 * n_unneeded, 0, 1)
```

- `J_healthy - J_before` > 0 because every physical fault degrades J by 5% or more on its own. Combined faults have always kept it positive: the smallest value seen over 10,000 draws is 0.050.
- `restoration_score` uses only the instance's `spec` and `faults`, so uncertified instances work too.
- **Interactions (measured).** Repairing only some of the faults can lower J, in about 1% of instances. Two causes have been seen: pumping harder against a remaining restriction, and sending freed flow to an outlet that already had enough. Plan F16's "never lowers J" does not hold for this KPI; the decision is open (see the Step 4 report).
- With no repairs, S = 0.
- Repairing exactly the true physical faults gives S = 1, and sensor recalibrations of truly faulty sensors do not lower it.

## 4. Efficiency, total reward and normalization (F17)

```
g(D)       = clip((D - 0.4) / 0.3, 0, 1)
efficiency = g(D) * (1 - spent / 40)
R          = 0.45 * D + 0.35 * S + 0.20 * efficiency
score_norm = min(R / R_oracle(seed, tier), 1)
```

The uncapped ratio is kept in the raw logs, and any seed where it exceeds 1 is flagged.

- **No oracle before Step 5.** `R_oracle` comes from the cached oracle scores (`results/oracle_scores.json`, keyed `"tier:seed"`), which arrive at Step 5.
- **Until then:**
  - `score_norm` is left out.
  - Baselines report raw R.
  - The Harbor `reward` is raw R; F20 asks for normalized.

## 5. Finalize output (`/var/lib/sim/out/score.json`, relayed as `reward.json`)

A flat JSON object of finite numbers only:

| Key | Meaning |
|---|---|
| `reward` | R (raw until Step 5) |
| `D`, `S`, `efficiency` | the three reward terms |
| `spent` | budget spent |
| `submitted` | 1 if a submission exists, else 0 |
| `n_true` | number of true faults |
| `matched_credit` | sum of magnitude credits |
| `false_positives` | FP count |
| `unneeded_repairs` | unneeded-repair count |
| `restoration` | restoration before the unneeded penalty and clipping |

## 6. Python API

```python
from diagnoselab.scoring.reward import diagnosis_score, restoration_score, efficiency, total_reward, score_episode
```

| Function | Returns |
|---|---|
| `diagnosis_score(true_faults, diagnosis) -> dict` | `D`, `matched_credit`, `false_positives` |
| `restoration_score(instance, repairs) -> dict` | `S`, `restoration`, `unneeded_repairs`, `J_before`, `J_after`, `J_healthy` |
| `efficiency(D, spent) -> float` | the efficiency term |
| `total_reward(D, S, spent) -> float` | R |
| `score_episode(instance, state) -> dict` | the finalize object of section 5. `state` is the episode state dict, with `submission` and `spent`. |

- `true_faults` is a sequence of `diagnoselab.sim.model.Fault`.
- `diagnosis` is a list of dicts as stored by the episode.

## 7. Baselines (F18)

**Interface.** Each baseline is a function `policy(call) -> None`.
- `call(tool, args)` returns the tool's result dict and raises `diagnoselab.server.tools.ToolRejected` on rejection.
- The same policy therefore runs in-process (`Episode.call`) and over MCP.
- Baselines import nothing from `diagnoselab.sim` or `diagnoselab.server`, except `ToolRejected`.
- Each baseline draws its randomness from `random.Random(seed)` passed in at construction.

| Baseline | Behaviour |
|---|---|
| `noop` | Makes no call and never submits. |
| `stop_early` | Submits an empty diagnosis and no repairs at once. |
| `everything_faulty` | Submits 10 distinct components (at most 10 entries), each with its kind's fault type at a mid-range magnitude, and repairs all 10. |
| `blanket_repair` | Submits an empty diagnosis and repairs 10 distinct components. |
| `random_guess` | Spends about half the budget on random reads, then submits 1–3 random (component, matching type) faults with random magnitudes and matching repairs. |
| `prior_guess` | Submits the most common fault type on two random eligible components, at a mid-range magnitude, with repairs. No probes. |
| `lazy_heuristic` | Reads every sensor once at nominal, then fits greedy single-fault explanations using linearised `simulate` sensitivities (one simulate call per candidate pair, well under the 300 cap). It adds faults while the whitened residual drops by at least 25 (5 sigma), then submits them with repairs. |

The plan's expected normalized scores for these baselines can be checked only once the Step 5 oracle exists. Until then raw R is reported, and only no-op and stop-early are exact: R = 0 for both.
