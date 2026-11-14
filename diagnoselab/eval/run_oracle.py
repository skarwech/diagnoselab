"""Run the oracle in-process on dev seeds (evaluation side: may use the truth to score).

    uv run python -m diagnoselab.eval.run_oracle --tier hard --seeds 0-49 [--out results/x.json] [--workers 8]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from diagnoselab.oracle.oracle import run_oracle
from diagnoselab.scoring.reward import score_episode
from diagnoselab.server.episode import Episode
from diagnoselab.sim.instance import make_instance

TIERS = ("easy", "medium", "hard")


def run_one(seed: int, tier: str) -> dict:
    inst = make_instance(seed, tier)
    ep = Episode(inst)
    calls: list[tuple[str, str]] = []
    usage: dict[str, int] = {}

    def call(tool: str, args: dict) -> dict:
        calls.append((tool, json.dumps(args, sort_keys=True)))
        if tool == "run_experiment":
            usage[args["type"]] = usage.get(args["type"], 0) + 1
        elif tool in ("read_sensor", "simulate"):
            usage[tool] = usage.get(tool, 0) + 1
        return ep.call(tool, args)

    t0 = time.perf_counter()
    out, error = {}, None
    try:
        out = run_oracle(call)
    except Exception:  # noqa: BLE001 - recorded, the episode is scored on what was submitted
        error = traceback.format_exc(limit=4)
    runtime = time.perf_counter() - t0
    score = score_episode(inst, ep._state)
    truth = [{"component": f.component, "type": f.type, "magnitude": f.magnitude} for f in inst.faults]
    return {"seed": seed, "tier": tier, "score": score, "runtime_s": runtime, "usage": usage,
            "calls_hash": hashlib.sha256(json.dumps(calls).encode()).hexdigest(), "n_calls": len(calls),
            "truth": truth, "submission": out.get("submission"), "trace": out.get("trace", []),
            "simulate_used": out.get("simulate_used"), "final": out.get("final"), "note": out.get("note"),
            "error": error}


def _job(args: tuple[int, str]) -> dict:
    return run_one(*args)


def _patch(settings: dict) -> None:
    """Tuning only: override oracle module constants in a worker process."""
    from diagnoselab.oracle import oracle
    for k, v in settings.items():
        setattr(oracle, k, v)


def run_many(seeds: list[int], tiers: list[str], workers: int = 8, settings: dict | None = None) -> list[dict]:
    jobs = [(s, t) for t in tiers for s in seeds]
    with ProcessPoolExecutor(max_workers=workers, initializer=_patch, initargs=(settings or {},)) as pool:
        return list(pool.map(_job, jobs))


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def summary(results: list[dict]) -> dict:
    out = {}
    for tier in TIERS:
        rs = [r for r in results if r["tier"] == tier]
        if not rs:
            continue
        n = len(rs)
        out[tier] = {"n": n, "R": sum(r["score"]["reward"] for r in rs) / n, "D": sum(r["score"]["D"] for r in rs) / n,
                     "S": sum(r["score"]["S"] for r in rs) / n, "spent": sum(r["score"]["spent"] for r in rs) / n,
                     "max_spent": max(r["score"]["spent"] for r in rs),
                     "sim_max": max(r["simulate_used"] or 0 for r in rs),
                     "runtime_max": max(r["runtime_s"] for r in rs),
                     "errors": sum(1 for r in rs if r["error"]), "notes": sum(1 for r in rs if r["note"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--set", action="append", default=[], help="tuning: oracle constant override NAME=python-literal")
    a = ap.parse_args()
    tiers = list(TIERS) if a.tier == "all" else a.tier.split(",")
    settings = {k: ast.literal_eval(v) for k, _, v in (x.partition("=") for x in a.set)}
    results = run_many(parse_seeds(a.seeds), tiers, a.workers, settings)
    print(json.dumps(summary(results), indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
