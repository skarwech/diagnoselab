"""F20: compare oracle trials from a Harbor job with the oracle cache; writes results/f20_harbor.json.

    python -m diagnoselab.eval.f20_check jobs/<job-dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from diagnoselab.config_hash import config_id
from diagnoselab.scoring.reward import oracle_scores

ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-12


def trial_key(trial: Path) -> str | None:
    compose = next(trial.glob("**/docker-compose.yaml"), None)
    config = trial / "config.json"
    text = config.read_text() if config.exists() else ""
    m = re.search(r"diagnoselab-(easy|medium|hard)-(\d+)", text or trial.name)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    if compose:
        c = compose.read_text()
        return f"{re.search(r'DL_TIER: \"(\\w+)\"', c).group(1)}:{re.search(r'DL_SEED: \"(\\d+)\"', c).group(1)}"
    return None


def check(job: Path) -> dict:
    cache, rows = oracle_scores(), []
    for reward_file in sorted(job.glob("*/verifier/reward.json")):
        trial = reward_file.parents[1]
        key = trial_key(trial)
        reward = json.loads(reward_file.read_text())
        cached = cache.get(key)
        rows.append({"trial": trial.name, "key": key, "reward": reward.get("reward"), "reward_raw": reward.get("reward_raw"),
                     "cached": cached, "normalized": reward.get("normalized"), "config_id": reward.get("config_id"),
                     "match": cached is not None and abs(reward.get("reward_raw", -1) - cached) <= TOL
                     and reward.get("config_id") == config_id()})
    result = json.loads((job / "result.json").read_text())
    errors = result["stats"].get("n_errored_trials", 0)
    return {"job": job.name, "trials": len(rows), "errored": errors, "all_match": bool(rows) and all(r["match"] for r in rows)
            and errors == 0, "rows": rows}


def main() -> None:
    out = check(Path(sys.argv[1]))
    (ROOT / "results" / "f20_harbor.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "rows"})
    for r in out["rows"]:
        print(r["key"], r["reward_raw"], r["cached"], r["reward"], r["match"])


if __name__ == "__main__":
    main()
