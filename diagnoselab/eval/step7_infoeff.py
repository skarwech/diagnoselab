"""Step 7, hypothesis 5: information gained about the true faults per budget unit, agent vs oracle vs random.

Same metric as docs/oracle_spec.md section 3 (whitened Fisher information on the true support, weak prior).
Writes results/step7_infoeff.json. Evaluation side: uses the truth.
"""

from __future__ import annotations

import json
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def agent_probes(trial_dir: Path) -> list[tuple[str, dict]]:
    trace = next(trial_dir.glob("artifacts/**/trace.jsonl"))
    out = []
    for line in trace.read_text().splitlines():
        c = json.loads(line)
        if not c.get("ok"):
            continue
        if c["tool"] == "read_sensor":
            out.append(("read", {"sensor": c["args"]["sensor_id"]}))
        elif c["tool"] == "run_experiment":
            out.append((c["args"]["type"], dict(c["args"].get("args") or {})))
    return out


def _one(args: tuple[str, int, str]) -> dict:
    from diagnoselab.eval.run_oracle import run_one
    from diagnoselab.eval.validate_oracle import TrueInfo, cost, parse_probe, random_probes
    from diagnoselab.sim.instance import make_instance

    trial, seed, tier = args
    inst = make_instance(seed, tier)
    ti = TrueInfo(inst)
    a = agent_probes(Path(trial))
    a_spend = int(sum(cost(k) for k, _ in a))
    o = [parse_probe(t["probe"]) for t in run_one(seed, tier)["trace"]]
    o_spend = int(sum(cost(k) for k, _ in o))
    rand = [ti.info(random_probes(inst.spec, a_spend, random.Random(f"{tier}:{seed}:{k}"))) for k in range(5)]
    a_info, o_info = ti.info(a), ti.info(o)
    return {"trial": Path(trial).name, "tier": tier, "seed": seed, "agent_info": a_info, "agent_spend": a_spend,
            "oracle_info": o_info, "oracle_spend": o_spend, "random_median": statistics.median(rand),
            "agent_per_unit": a_info / max(a_spend, 1), "oracle_per_unit": o_info / max(o_spend, 1),
            "agent_above_random": a_info > statistics.median(rand)}


def main() -> None:
    val = json.loads((ROOT / "results" / "step7_validation.json").read_text())
    job = Path((ROOT / "results" / "step7_agent_job.txt").read_text().strip())
    jobs = [(str(job / r["trial"]), r["seed"], r["tier"]) for r in val["trials"]]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(_one, jobs))
    summary = {}
    for tier in ("easy", "medium", "hard"):
        rs = [r for r in rows if r["tier"] == tier]
        summary[tier] = {k: statistics.fmean(r[k] for r in rs) for k in
                         ("agent_info", "oracle_info", "random_median", "agent_spend", "oracle_spend", "agent_per_unit",
                          "oracle_per_unit")}
        summary[tier]["agent_above_random"] = statistics.fmean(r["agent_above_random"] for r in rs)
    (ROOT / "results" / "step7_infoeff.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    for t, s in summary.items():
        print(t, {k: round(v, 2) for k, v in s.items()})


if __name__ == "__main__":
    main()
