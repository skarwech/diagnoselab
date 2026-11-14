"""Server-side finalize: rebuild the truth from (seed, tier) and score the persisted episode (F15-F17).

Runs inside the sim container via Harbor's [[verifier.collect]] hook after the agent container stops.
Writes the flat finite-number object of docs/scoring_spec.md section 5; `reward` is raw R until the
Step 5 oracle provides the normalizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnoselab.scoring.reward import score_episode
from diagnoselab.sim.instance import make_instance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    saved = json.loads((a.state / "episode.json").read_text())
    result = score_episode(make_instance(saved["seed"], saved["tier"]), saved["state"])
    a.out.write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
