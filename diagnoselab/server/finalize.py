"""Server-side finalize: rebuild the truth from (seed, tier) and score the persisted episode (F15-F17).

Runs inside the sim container via Harbor's [[verifier.collect]] hook after the agent container stops.
Writes the flat finite-number object of docs/scoring_spec.md section 5. `reward` is score_norm when the seed is in
the oracle cache (R_oracle >= 0.5), else raw R; `reward_raw` always holds raw R.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnoselab.scoring.reward import normalization, score_episode
from diagnoselab.sim.instance import make_instance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    saved = json.loads((a.state / "episode.json").read_text())
    result = score_episode(make_instance(saved["seed"], saved["tier"]), saved["state"])
    result |= normalization(result["reward"], saved["seed"], saved["tier"])
    if result["normalized"]:
        result["reward"] = result["score_norm"]
    a.out.write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
