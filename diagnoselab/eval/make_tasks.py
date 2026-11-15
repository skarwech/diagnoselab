"""Generate one Harbor task per (tier, seed) from harbor_task/ into a dataset directory (gitignored).

Acceptance rule (user decision, Step 6): a seed ships only if the oracle scores at least NORM_FLOOR on it, so
every shipped seed is normalised. Seeds missing from the oracle cache are refused: cache them first.

    python -m diagnoselab.eval.make_tasks --out tasks/pilot --tier easy,medium,hard --seeds 50-99 --per-tier 5
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from diagnoselab.eval.run_oracle import TIERS, parse_seeds
from diagnoselab.scoring.reward import NORM_FLOOR, oracle_scores

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "harbor_task"


def accepted(tier: str, seeds: list[int], per_tier: int | None) -> list[int]:
    cache = oracle_scores()
    missing = [s for s in seeds if f"{tier}:{s}" not in cache]
    if missing:
        raise SystemExit(f"{tier} seeds not in the oracle cache: {missing[:10]}; run make oracle-cache first")
    ok = [s for s in seeds if cache[f"{tier}:{s}"] >= NORM_FLOOR]
    return ok[:per_tier] if per_tier else ok


def write_task(out: Path, tier: str, seed: int) -> Path:
    dest = out / f"diagnoselab-{tier}-{seed}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(TEMPLATE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    compose = dest / "environment" / "docker-compose.yaml"
    text = compose.read_text()
    text, n_seed = re.subn(r'DL_SEED: "\d+"', f'DL_SEED: "{seed}"', text)
    text, n_tier = re.subn(r'DL_TIER: "\w+"', f'DL_TIER: "{tier}"', text)
    if n_seed != 1 or n_tier != 1:
        raise SystemExit("docker-compose.yaml no longer has exactly one DL_SEED and one DL_TIER")
    compose.write_text(text)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", default=",".join(TIERS))
    ap.add_argument("--seeds", default="50-99")
    ap.add_argument("--per-tier", type=int)
    ap.add_argument("--test", action="store_true", help="allow test seeds (1000+): only once the configuration is frozen")
    a = ap.parse_args()
    if any(s >= 1000 for s in parse_seeds(a.seeds)) and not a.test:
        raise SystemExit("test seeds (1000+) are reported once after the configuration is frozen: pass --test")
    a.out.mkdir(parents=True, exist_ok=True)
    for tier in a.tier.split(","):
        seeds = accepted(tier, parse_seeds(a.seeds), a.per_tier)
        for s in seeds:
            write_task(a.out, tier, s)
        print(f"{tier}: {seeds}")


if __name__ == "__main__":
    main()
