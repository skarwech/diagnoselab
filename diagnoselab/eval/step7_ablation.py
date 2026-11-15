"""Step 7 hint ablation (F22): one instruction line against budget myopia, medium and hard test seeds.

Paired per seed: hinted score minus the mean of the same seed's 3 unhinted runs (results/step7_validation.json).
Writes results/step7_ablation.json.

    python -m diagnoselab.eval.step7_ablation jobs/<hinted-job> [jobs/<rerun-job> ...]

Trials that failed with AgentAuthenticationError (the model API rejected the key, so the agent never ran) are
infrastructure failures and are excluded; any other failure counts against the agent.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from diagnoselab.eval.validate_experiments import bootstrap_ci, failure_metrics, parse_trial

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("medium", "hard")


def main() -> None:
    from diagnoselab.sim.instance import make_instance

    jobs = [Path(a) for a in sys.argv[1:]]
    base = json.loads((ROOT / "results" / "step7_validation.json").read_text())["trials"]
    parsed = [t for job in jobs for t in (parse_trial(p) for p in sorted(job.iterdir()) if p.is_dir()) if t]
    excluded = [f"{t['tier']}:{t['seed']}" for t in parsed if t["exception"] == "AgentAuthenticationError"]
    hinted = [t for t in parsed if t["exception"] != "AgentAuthenticationError"]
    if len({(t["tier"], t["seed"]) for t in hinted}) != len(hinted):
        raise SystemExit("a seed has more than one valid hinted episode")
    rows = []
    for t in hinted:
        truth = [{"component": f.component, "type": f.type} for f in make_instance(t["seed"], t["tier"]).faults]
        m = failure_metrics(t, truth)
        before = [b for b in base if b["tier"] == t["tier"] and b["seed"] == t["seed"]]
        rows.append({"tier": t["tier"], "seed": t["seed"], "hinted": t["reward"], "unhinted_mean": statistics.fmean(b["reward"] for b in before),
                     "lift": t["reward"] - statistics.fmean(b["reward"] for b in before),
                     "spent": t["spent"], "unhinted_spent": statistics.fmean(b["spent"] for b in before),
                     "reads_before": m["reads_before_first_experiment"],
                     "unhinted_reads_before": statistics.fmean(b["metrics"]["reads_before_first_experiment"] for b in before),
                     "myopia": m["flags"]["budget_myopia"], "submitted": t["submitted"], "exception": t["exception"]})
    out = {"episodes": len(rows), "jobs": [j.name for j in jobs], "excluded_auth_failures": len(excluded), "rows": rows}
    for scope in ("overall", *TIERS):
        rs = [r for r in rows if scope == "overall" or r["tier"] == scope]
        lifts = [r["lift"] for r in rs]
        lo, hi = bootstrap_ci(lifts)
        out[scope] = {"n": len(rs), "hinted": statistics.fmean(r["hinted"] for r in rs),
                      "unhinted": statistics.fmean(r["unhinted_mean"] for r in rs), "lift": statistics.fmean(lifts), "ci": [lo, hi],
                      "spent": statistics.fmean(r["spent"] for r in rs), "unhinted_spent": statistics.fmean(r["unhinted_spent"] for r in rs),
                      "reads_before": statistics.fmean(r["reads_before"] for r in rs),
                      "unhinted_reads_before": statistics.fmean(r["unhinted_reads_before"] for r in rs),
                      "myopia_rate": statistics.fmean(r["myopia"] for r in rs)}
    out["pass"] = out["overall"]["lift"] >= 0.15
    (ROOT / "results" / "step7_ablation.json").write_text(json.dumps(out, indent=1))
    for s in ("overall", *TIERS):
        o = out[s]
        print(s, f"hinted {o['hinted']:.3f} vs unhinted {o['unhinted']:.3f}, lift {o['lift']:+.3f} CI [{o['ci'][0]:+.3f}, "
                 f"{o['ci'][1]:+.3f}], spend {o['spent']:.1f} vs {o['unhinted_spent']:.1f}, reads before first experiment "
                 f"{o['reads_before']:.1f} vs {o['unhinted_reads_before']:.1f}, myopia {o['myopia_rate']:.0%}")


if __name__ == "__main__":
    main()
