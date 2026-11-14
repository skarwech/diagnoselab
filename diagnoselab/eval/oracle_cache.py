"""Oracle score cache (F17): run the oracle per (tier, seed) and emit {"tier:seed": record} as JSON on stdout.

Generated in the Linux image (`make oracle-cache`) so it matches what Harbor's sim container recomputes; the
reward normaliser reads only the "reward" field (diagnoselab/scoring/oracle_scores.json).

    python -m diagnoselab.eval.oracle_cache --seeds 0-199 [--tiers easy,medium,hard] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys

from diagnoselab.eval.run_oracle import TIERS, parse_seeds, run_many


def cache_records(results: list[dict]) -> dict[str, dict]:
    return {f"{r['tier']}:{r['seed']}": {"reward": r["score"]["reward"], "D": r["score"]["D"],
                                         "spent": r["score"]["spent"], "calls_hash": r["calls_hash"]}
            for r in sorted(results, key=lambda r: (TIERS.index(r["tier"]), r["seed"]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-199")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    results = run_many(parse_seeds(a.seeds), a.tiers.split(","), a.workers)
    bad = [f"{r['tier']}:{r['seed']}" for r in results if r["error"]]
    if bad:
        sys.exit(f"oracle errors on {bad}")
    json.dump(cache_records(results), sys.stdout, indent=0, sort_keys=False)


if __name__ == "__main__":
    main()
