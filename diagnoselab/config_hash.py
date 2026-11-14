"""Configuration hash: one fingerprint of everything that can change a score.

It covers the simulator (every dial: sizes, fault counts, magnitudes, noise, budget, costs), the reward, the
episode server and the oracle. The oracle cache records the hash it was generated under, and finalize only
normalises against a cache with the same hash, so a dial change can never be scored against a stale oracle.
The hash is over file bytes, so the host and the sim container (an rsync copy) agree.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SCORED_SOURCES = ("sim/*.py", "scoring/reward.py", "server/episode.py", "server/tools.py", "oracle/oracle.py")


@lru_cache(maxsize=1)
def config_hash() -> str:
    h = hashlib.sha256()
    for pattern in SCORED_SOURCES:
        for path in sorted(PACKAGE.glob(pattern)):
            h.update(path.relative_to(PACKAGE).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def config_id(digest: str | None = None) -> int:
    """The first 48 bits as an integer: exact as a float, so it can travel in the numbers-only reward.json."""
    return int((digest or config_hash())[:12], 16)
