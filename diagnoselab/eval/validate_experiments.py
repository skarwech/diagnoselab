"""Step 7 (F21 statistics, F22 failure modes) from a Harbor agent job on the frozen test seeds.

    python -m diagnoselab.eval.validate_experiments jobs/<agent-job> [--out results/step7_validation.json]

Every number comes from stored raw results (Harbor reward.json and the server trace.jsonl per trial), so a rerun
reproduces the output exactly (bootstrap seeded). Failure flags use thresholds fixed before the test runs were read:
  anchoring        D < 1 and the first non-empty simulate hypothesis (component set) is inside the submission,
                   with at most 3 distinct hypotheses tried
  no_disambiguate  never ran pump_off_static nor valve_step
  budget_myopia    10 or more read_sensor calls before the first experiment
  trusts_liar      a true sensor fault exists and the submission misses it
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("easy", "medium", "hard")
EXPERIMENTS = ("speed_step", "valve_step", "isolate_branch", "pump_off_static")
MYOPIA_READS, ANCHOR_MAX_HYP = 10, 3


def bootstrap_ci(values: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% interval of the mean (seeded, so reruns are identical)."""
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def parse_trial(trial: Path) -> dict | None:
    config = (trial / "config.json").read_text() if (trial / "config.json").exists() else ""
    m = re.search(r"diagnoselab-(easy|medium|hard)-(\d+)", config) or re.search(r"diagnoselab-(easy|medium|hard)-(\d+)", trial.name)
    if not m:
        return None
    reward = json.loads((trial / "verifier" / "reward.json").read_text()) if (trial / "verifier" / "reward.json").exists() else {}
    result = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    trace_file = next(trial.glob("artifacts/**/trace.jsonl"), None)
    calls = [json.loads(x) for x in trace_file.read_text().splitlines() if x.strip()] if trace_file else []
    exc = (result.get("exception_info") or {}).get("exception_type")
    return {"trial": trial.name, "tier": m.group(1), "seed": int(m.group(2)), "started": result.get("started_at") or "",
            "reward": reward.get("reward", 0.0), "reward_raw": reward.get("reward_raw", 0.0), "D": reward.get("D", 0.0),
            "S": reward.get("S", 0.0), "spent": reward.get("spent", 0.0), "submitted": reward.get("submitted", 0.0),
            "normalized": reward.get("normalized", 0.0), "exception": exc, "calls": calls,
            "tokens_in": (result.get("agent_result") or {}).get("n_input_tokens"),
            "tokens_out": (result.get("agent_result") or {}).get("n_output_tokens")}


def failure_metrics(t: dict, truth: list[dict]) -> dict:
    calls = [c for c in t["calls"] if c.get("ok")]
    first_exp = next((i for i, c in enumerate(calls) if c["tool"] == "run_experiment"), len(calls))
    reads_before = sum(1 for c in calls[:first_exp] if c["tool"] == "read_sensor")
    exps = [c["args"].get("type") for c in calls if c["tool"] == "run_experiment"]
    hyps = []
    for c in calls:
        if c["tool"] == "simulate":
            comps = tuple(sorted({o.get("component") for o in (c["args"] or {}).get("overrides", []) if isinstance(o, dict)}))
            if comps and comps not in hyps:
                hyps.append(comps)
    sub = next((c["args"] for c in reversed(calls) if c["tool"] == "submit"), None)
    sub_items = [(d.get("component"), d.get("type")) for d in (sub or {}).get("diagnosis", [])]
    sub_comps = {c for c, _ in sub_items}
    sensor_truth = [(f["component"], f["type"]) for f in truth if f["type"].startswith("sensor_")]
    missed_sensor = [x for x in sensor_truth if x not in sub_items]
    points = {"nominal"} if any(c["tool"] == "read_sensor" for c in calls) else set()
    points |= {json.dumps(c["args"], sort_keys=True) for c in calls if c["tool"] == "run_experiment"}
    conf = [(d.get("confidence"), (d.get("component"), d.get("type")) in {(f["component"], f["type"]) for f in truth})
            for d in (sub or {}).get("diagnosis", []) if d.get("confidence") is not None]
    flags = {
        "anchoring": t["D"] < 1 and bool(hyps) and set(hyps[0]) <= sub_comps and len(hyps) <= ANCHOR_MAX_HYP,
        "no_disambiguate": "pump_off_static" not in exps and "valve_step" not in exps,
        "budget_myopia": reads_before >= MYOPIA_READS,
        "trusts_liar": bool(missed_sensor),
    }
    return {"reads_before_first_experiment": reads_before, "experiments": len(exps),
            "used": {k: k in exps for k in EXPERIMENTS}, "distinct_points": len(points),
            "distinct_hypotheses": len(hyps), "simulate_calls": sum(1 for c in t["calls"] if c["tool"] == "simulate"),
            "sensor_faults": len(sensor_truth), "sensor_faults_missed": len(missed_sensor),
            "brier": statistics.fmean((p - float(ok)) ** 2 for p, ok in conf) if conf else None,
            "n_confidence": len(conf), "flags": flags}


def f21(trials: list[dict]) -> dict:
    """Paired gap oracle - agent in normalised units (the oracle is 1 on every shipped seed by construction)."""
    by_seed: dict[tuple, list[dict]] = {}
    for t in trials:
        by_seed.setdefault((t["tier"], t["seed"]), []).append(t)
    for v in by_seed.values():
        v.sort(key=lambda t: (t["started"], t["trial"]))
    n_rep = min(len(v) for v in by_seed.values())
    seed_gap = {k: 1.0 - statistics.fmean(t["reward"] for t in v[:n_rep]) for k, v in by_seed.items()}
    rep_gap = [statistics.fmean(1.0 - v[r]["reward"] for v in by_seed.values()) for r in range(n_rep)]
    out = {"repeats": n_rep, "seeds": len(by_seed), "overall": {}}
    for scope, keys in [("overall", list(seed_gap))] + [(t, [k for k in seed_gap if k[0] == t]) for t in TIERS]:
        if not keys:
            continue
        gaps = [seed_gap[k] for k in keys]
        lo, hi = bootstrap_ci(gaps)
        reps = [statistics.fmean(1.0 - by_seed[k][r]["reward"] for k in keys) for r in range(n_rep)]
        out[scope] = {"n": len(gaps), "agent_mean": 1.0 - statistics.fmean(gaps), "gap": statistics.fmean(gaps),
                      "ci": [lo, hi], "per_repeat_gap": reps, "run_sd": statistics.stdev(reps) if n_rep > 1 else None}
    o = out["overall"]
    out["valid"] = {"gap_ge_025": o["gap"] >= 0.25, "ci_excludes_0": o["ci"][0] > 0,
                    "each_repeat_ge_015": all(g >= 0.15 for g in rep_gap),
                    "sd_lt_half_gap": o["run_sd"] is not None and o["run_sd"] < o["gap"] / 2}
    out["pass"] = all(out["valid"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "step7_validation.json")
    ap.add_argument("--allow-dev", action="store_true", help="dry runs on dev-seed jobs only")
    a = ap.parse_args()
    from diagnoselab.sim.instance import make_instance

    trials = [t for t in (parse_trial(p) for p in sorted(a.job.iterdir()) if p.is_dir()) if t]
    truth_cache: dict = {}
    rows = []
    for t in trials:
        key = (t["tier"], t["seed"])
        if key not in truth_cache:
            truth_cache[key] = [{"component": f.component, "type": f.type} for f in make_instance(t["seed"], t["tier"]).faults]
        fm = failure_metrics(t, truth_cache[key])
        rows.append({k: v for k, v in t.items() if k != "calls"} | {"n_calls": len(t["calls"]),
                    "rejected": sum(1 for c in t["calls"] if not c.get("ok")), "metrics": fm})
    if any(t["seed"] < 1000 for t in trials) and not a.allow_dev:
        raise SystemExit("dev seeds found in a test job")
    stats = f21(trials)
    flags = {f: {tier: statistics.fmean(r["metrics"]["flags"][f] for r in rows if r["tier"] == tier) for tier in TIERS}
             for f in ("anchoring", "no_disambiguate", "budget_myopia", "trusts_liar")}
    usage = {k: {tier: statistics.fmean(r["metrics"]["used"][k] for r in rows if r["tier"] == tier) for tier in TIERS}
             for k in EXPERIMENTS}
    sensor = {tier: (sum(r["metrics"]["sensor_faults_missed"] for r in rows if r["tier"] == tier),
                     sum(r["metrics"]["sensor_faults"] for r in rows if r["tier"] == tier)) for tier in TIERS}
    calls = sum(r["n_calls"] for r in rows)
    out = {"job": a.job.name, "episodes": len(rows), "F21": stats, "flags": flags, "experiment_use": usage,
           "sensor_fault_miss": sensor,
           "reads_before_first_experiment": {tier: statistics.fmean(r["metrics"]["reads_before_first_experiment"]
                                                                   for r in rows if r["tier"] == tier) for tier in TIERS},
           "distinct_hypotheses": {tier: statistics.fmean(r["metrics"]["distinct_hypotheses"] for r in rows if r["tier"] == tier)
                                   for tier in TIERS},
           "friction": {"calls": calls, "rejected": sum(r["rejected"] for r in rows),
                        "exceptions": [r["exception"] for r in rows if r["exception"]],
                        "no_submission": sum(1 for r in rows if not r["submitted"])},
           "tokens": {"in": sum(r["tokens_in"] or 0 for r in rows), "out": sum(r["tokens_out"] or 0 for r in rows)},
           "trials": rows}
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    o = stats["overall"]
    print(f"episodes {len(rows)}, agent mean {o['agent_mean']:.3f}, gap {o['gap']:.3f} CI {o['ci']}, per repeat "
          f"{[round(g, 3) for g in o['per_repeat_gap']]}, F21 pass {stats['pass']}")


if __name__ == "__main__":
    main()
