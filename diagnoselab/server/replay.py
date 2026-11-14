"""Trace replay (F13): feed a logged episode's calls to a fresh Episode and compare response hashes."""

from __future__ import annotations

import json
from pathlib import Path

from diagnoselab.server.episode import TOO_LARGE, UNSERIALIZABLE, Episode, ToolRejected, response_hash
from diagnoselab.sim.instance import make_instance


def replay(trace_path: Path, seed: int, tier: str) -> list[int]:
    """Return the call indices whose response hash differs (empty list = perfect replay).

    The fresh episode runs on a fixed clock, so an original episode that hit the time cap differs.
    """
    episode = Episode(make_instance(seed, tier), None, clock=lambda: 0.0)
    mismatches = []
    for line in Path(trace_path).read_text().splitlines():
        rec = json.loads(line)
        args, verdict = rec["args"], rec.get("verdict")
        try:
            if verdict in (TOO_LARGE, UNSERIALIZABLE):  # the original arguments were not stored verbatim
                h = response_hash(episode._call(rec["tool"], args, verdict, args))
            else:
                h = response_hash(episode.call(rec["tool"], args))
        except ToolRejected as e:
            h = response_hash({"error": f"{e.code}: {e.message}"})
        if h != rec["response_hash"]:
            mismatches.append(rec["i"])
    return mismatches
