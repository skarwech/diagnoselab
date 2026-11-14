#!/bin/bash
# Placeholder reference run (the D-optimal oracle replaces it at plan Step 5).
# Uses only the public MCP tools, exactly like the agent: static test -> flag biased pressure sensors.
set -euo pipefail
python3 - <<'PY'
import anyio
from mcp import Client

URL = "http://sim:8000/mcp"

def data(r):
    if r.is_error:
        raise SystemExit(r.content[0].text)
    return r.structured_content

async def main():
    async with Client(URL) as c:
        data(await c.call_tool("read_manual", {"topic": "experiments"}))
        topo = data(await c.call_tool("get_topology", {}))
        heads = {s["id"]: s["head"] for s in topo["sources"]}
        feed = {l["to"]: heads[l["frm"]] for l in topo["links"] if l["frm"] in heads}
        static = {j["id"]: (feed.get(j["id"], topo["tank_head"]) if j["side"] == "suction" else topo["tank_head"]) - j["z"]
                  for j in topo["junctions"]}
        exp = data(await c.call_tool("run_experiment", {"type": "pump_off_static", "args": {}}))
        sigma = {s["id"]: s["sigma"] for s in topo["sensors"]}
        diagnosis, repairs = [], []
        for sid, value in exp["readings"].items():
            if sid.startswith("PT-"):
                dev = (value - static[sid[3:]]) / sigma[sid]
                if abs(dev) > 5:
                    diagnosis.append({"component": sid, "type": "sensor_bias", "magnitude": abs(dev)})
                    repairs.append({"action": "recalibrate_sensor", "target": sid})
        print("budget:", data(await c.call_tool("budget_status", {}))["spent"], "flagged:", [d["component"] for d in diagnosis])
        print("submit:", data(await c.call_tool("submit", {"diagnosis": diagnosis, "repairs": repairs})))

anyio.run(main)
PY
