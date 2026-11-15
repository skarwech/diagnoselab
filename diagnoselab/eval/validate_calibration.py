"""Step 6 (F24) aggregation: frontier-agent pilot and dial probes -> results/step6_validation.json.

Inputs are the per-trial records written by pilot_report.py for each Harbor job, plus the oracle measurements
in results/step6_oracle_probes.json (recorded under each probe's code). Means carry a 10,000-resample
bootstrap 95% interval over seeds.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
TIERS = ("easy", "medium", "hard")
BAND = (0.35, 0.65)
RUNS = {"B40": ["pilot_episode1.json", "pilot_rest.json"], "B26": ["probe_b26.json", "probe_b26_more.json"],
        "B26_subtle": ["probe_subtle.json"], "B26_meters": ["probe_meters.json"]}
EXPERIMENTS = ("speed_step", "valve_step", "isolate_branch", "pump_off_static")


def bootstrap(values: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def trials(files: list[str]) -> list[dict]:
    return [t for f in files for t in json.loads((R / f).read_text())["trials"]]


def tier_summary(rows: list[dict]) -> dict:
    v = [r["reward"] or 0.0 for r in rows]
    lo, hi = bootstrap(v)
    calls = sum(r["n_calls"] for r in rows)
    return {"n": len(rows), "mean": statistics.mean(v), "ci": [lo, hi], "D": statistics.mean(r["D"] or 0 for r in rows),
            "S": statistics.mean(r["S"] or 0 for r in rows), "spent": statistics.mean(r["spent"] or 0 for r in rows),
            "seeds": sorted(r["seed"] for r in rows), "per_seed": {str(r["seed"]): r["reward"] for r in rows},
            "rejected": sum(r["rejected"] for r in rows), "calls": calls,
            "exceptions": [r["exception"] for r in rows if r["exception"]],
            "no_submission": sum(1 for r in rows if not r["submitted"]),
            "experiment_use": {k: sum(1 for r in rows if r["experiments"].get(k)) / len(rows) for k in EXPERIMENTS},
            "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
            "output_tokens": sum(r["output_tokens"] or 0 for r in rows)}


def main() -> None:
    oracle = json.loads((R / "step6_oracle_probes.json").read_text())
    step5 = json.loads((R / "step5_validation.json").read_text())  # official F19 seeds 0-29 and F5 seeds 0-199, frozen config
    configs = {}
    for name, files in RUNS.items():
        rows = trials(files)
        per = {t: tier_summary([r for r in rows if r["tier"] == t]) for t in TIERS if any(r["tier"] == t for r in rows)}
        allv = [r["reward"] or 0.0 for r in rows]
        configs[name] = {**oracle["configs"][name], "tiers": per, "overall": statistics.mean(allv),
                         "overall_ci": list(bootstrap(allv)), "episodes": len(rows)}
    b40 = {(r["tier"], r["seed"]): r["reward"] for r in trials(RUNS["B40"])}
    b26 = {(r["tier"], r["seed"]): r["reward"] for r in trials(RUNS["B26"])}
    paired = sorted(k for k in b40 if k in b26 and k[0] != "easy")
    final = configs["B40"]  # the frozen configuration
    out = {
        "band": BAND, "configs": configs,
        "paired_B40_B26": {"pairs": [{"tier": t, "seed": s, "B40": b40[(t, s)], "B26": b26[(t, s)]} for t, s in paired],
                           "mean_B40": statistics.mean(b40[k] for k in paired), "mean_B26": statistics.mean(b26[k] for k in paired),
                           "note": "same fault sets at both budgets (checked: 10 of 10)"},
        "spend_cap_curve_B40": oracle["spend_cap_curve_B40"], "D_targets": oracle["D_targets"],
        "chosen": "B40",  # frozen by the user after the probes
        "official_F19": {t: {"D": step5["F19"]["tiers"][t]["D"], "D_pass": step5["F19"]["tiers"][t]["D_pass"],
                             "F5": step5["F5"][t]["D_ge_085"]} for t in TIERS},
        "F24": {"oracle_ok": all(step5["F19"]["tiers"][t]["D_pass"] and step5["F5"][t]["pass"] for t in TIERS),
                "agent_in_band": BAND[0] <= final["overall"] <= BAND[1], "agent_overall": final["overall"],
                "human_audit": "skipped (user decision)", "pass": False},
        "totals": {"episodes": sum(c["episodes"] for c in configs.values()),
                   "input_tokens": sum(t["input_tokens"] for c in configs.values() for t in c["tiers"].values()),
                   "output_tokens": sum(t["output_tokens"] for c in configs.values() for t in c["tiers"].values()),
                   "rejected": sum(t["rejected"] for c in configs.values() for t in c["tiers"].values()),
                   "calls": sum(t["calls"] for c in configs.values() for t in c["tiers"].values())},
    }
    (R / "step6_validation.json").write_text(json.dumps(out, indent=1))
    for name, c in configs.items():
        print(name, f"overall {c['overall']:.3f}", {t: round(s["mean"], 3) for t, s in c["tiers"].items()})
    print("paired", round(out["paired_B40_B26"]["mean_B40"], 3), "->", round(out["paired_B40_B26"]["mean_B26"], 3), out["totals"])


if __name__ == "__main__":
    main()
