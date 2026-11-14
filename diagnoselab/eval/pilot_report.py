"""Summarise a Harbor agent job: tool friction first (from the server traces), then scores.

    python -m diagnoselab.eval.pilot_report jobs/<job-dir> [--out results/pilot_<name>.json]

Friction (plan Step 6 / F22): rejected-call rate, error codes, invalid submits, episodes with no submission,
agent exceptions and timeouts. Scores: normalised reward, D, S, spend, tool usage, per tier.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

TIERS = ("easy", "medium", "hard")
COSTED = ("read_sensor", "run_experiment")


def trial_info(trial: Path) -> dict:
    config = (trial / "config.json").read_text() if (trial / "config.json").exists() else ""
    m = re.search(r"diagnoselab-(easy|medium|hard)-(\d+)", config) or re.search(r"diagnoselab-(easy|medium|hard)-(\d+)", trial.name)
    reward_file = trial / "verifier" / "reward.json"
    reward = json.loads(reward_file.read_text()) if reward_file.exists() else {}
    result = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    trace = next(trial.glob("artifacts/**/trace.jsonl"), None)
    calls = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()] if trace else []
    codes = Counter(c.get("code") or "ok" for c in calls)
    tools = Counter(c.get("tool") for c in calls)
    experiments = Counter(c["args"].get("type") for c in calls
                          if c.get("tool") == "run_experiment" and c.get("ok") and isinstance(c.get("args"), dict))
    exc = result.get("exception_info") or {}
    usage = (result.get("agent_result") or {})
    return {"trial": trial.name, "tier": m.group(1) if m else None, "seed": int(m.group(2)) if m else None,
            "reward": reward.get("reward"), "reward_raw": reward.get("reward_raw"), "normalized": reward.get("normalized"),
            "D": reward.get("D"), "S": reward.get("S"), "spent": reward.get("spent"), "submitted": reward.get("submitted"),
            "n_calls": len(calls), "rejected": sum(v for k, v in codes.items() if k != "ok"), "codes": dict(codes),
            "tools": dict(tools), "experiments": dict(experiments), "exception": exc.get("exception_type"),
            "input_tokens": usage.get("n_input_tokens"), "output_tokens": usage.get("n_output_tokens"),
            "cache_tokens": usage.get("n_cache_tokens"), "cost_usd": usage.get("cost_usd")}


def mean(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.mean(xs) if xs else None


def summarise(rows: list[dict]) -> dict:
    out = {}
    for tier in TIERS:
        rs = [r for r in rows if r["tier"] == tier]
        if not rs:
            continue
        n_calls = sum(r["n_calls"] for r in rs)
        out[tier] = {"n": len(rs), "reward_norm": mean([r["reward"] for r in rs]), "D": mean([r["D"] for r in rs]),
                     "S": mean([r["S"] for r in rs]), "spent": mean([r["spent"] for r in rs]),
                     "no_submission": sum(1 for r in rs if not r["submitted"]),
                     "exceptions": sum(1 for r in rs if r["exception"]),
                     "rejected_rate": sum(r["rejected"] for r in rs) / n_calls if n_calls else None,
                     "invalid_submits": sum(r["codes"].get("E_INVALID_SUBMISSION", 0) for r in rs),
                     "experiment_use": {k: sum(1 for r in rs if r["experiments"].get(k)) / len(rs)
                                        for k in ("speed_step", "valve_step", "isolate_branch", "pump_off_static")},
                     "input_tokens": sum(r["input_tokens"] or 0 for r in rs),
                     "output_tokens": sum(r["output_tokens"] or 0 for r in rs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = [trial_info(t) for t in sorted(p for p in a.job.iterdir() if p.is_dir())]
    out = {"job": a.job.name, "trials": rows, "summary": summarise(rows)}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    for tier, s in out["summary"].items():
        print(tier, json.dumps(s))


if __name__ == "__main__":
    main()
