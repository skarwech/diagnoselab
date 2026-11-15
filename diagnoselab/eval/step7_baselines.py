"""Step 7: every F18 baseline on the frozen test seeds, normalised by the cached oracle score.

Writes results/step7_baselines.json. In-process scoring equals MCP scoring (Step 4 cross-check, 18 of 18).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("easy", "medium", "hard")


def test_seeds() -> dict[str, list[int]]:
    """The tasks generated for Step 7 (results/step7_test_seeds.txt, written by make_tasks)."""
    out = {}
    for line in (ROOT / "results" / "step7_test_seeds.txt").read_text().splitlines():
        tier, _, seeds = line.partition(":")
        out[tier.strip()] = json.loads(seeds)
    return out


def _job(args: tuple[str, str, int]) -> dict:
    from diagnoselab.baselines.policies import BASELINES
    from diagnoselab.eval.validate_scoring import run_inprocess
    from diagnoselab.scoring.reward import normalization
    from diagnoselab.sim.instance import make_instance

    name, tier, seed = args
    score = run_inprocess(BASELINES[name], make_instance(seed, tier), seed)
    norm = normalization(score["reward"], seed, tier)
    return {"policy": name, "tier": tier, "seed": seed, "reward_raw": score["reward"], "D": score["D"],
            "score_norm": norm.get("score_norm"), "normalized": norm["normalized"]}


def main() -> None:
    from diagnoselab.baselines.policies import BASELINES

    seeds = test_seeds()
    jobs = [(n, t, s) for n in BASELINES for t in TIERS for s in seeds[t]]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(_job, jobs))
    (ROOT / "results" / "step7_baselines.json").write_text(json.dumps(rows, indent=1))
    for n in BASELINES:
        print(n, {t: round(sum(r["score_norm"] for r in rows if r["policy"] == n and r["tier"] == t) / len(seeds[t]), 3)
                  for t in TIERS})


if __name__ == "__main__":
    main()
