"""Install results/oracle_cache_linux.json as the shipped normaliser, refusing a cache built under other code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from diagnoselab.config_hash import config_hash
from diagnoselab.scoring.reward import ORACLE_META, ORACLE_SCORES

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cache = json.loads((ROOT / "results" / "oracle_cache_linux.json").read_text())
    if cache["config_hash"] != config_hash():
        sys.exit(f"cache built under {cache['config_hash'][:12]}, code is {config_hash()[:12]}: run `make oracle-cache`")
    ORACLE_SCORES.write_text(json.dumps({k: v["reward"] for k, v in cache["records"].items()}, indent=0))
    ORACLE_META.write_text(json.dumps({"config_hash": cache["config_hash"], "entries": len(cache["records"]),
                                       "generated_by": "make oracle-cache (python:3.12-slim, numpy 2.5.4)"}, indent=1))
    print(f"installed {len(cache['records'])} oracle scores under config {cache['config_hash'][:12]}")


if __name__ == "__main__":
    main()
