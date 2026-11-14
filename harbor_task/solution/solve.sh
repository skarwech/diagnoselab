#!/bin/bash
# Reference run (plan Step 5): the D-optimal oracle, using only the public MCP tools exactly like an agent.
# oracle.py is a verbatim copy of diagnoselab/oracle/oracle.py (`make sync`); standard library only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 - "$HERE" <<'PY'
import json
import sys

import anyio
from mcp import Client

sys.path.insert(0, sys.argv[1])
from oracle import run_oracle  # noqa: E402

URL = "http://sim:8000/mcp"


async def main() -> None:
    async with Client(URL) as client:
        def call(tool: str, args: dict) -> dict:
            result = anyio.from_thread.run(client.call_tool, tool, args)
            if result.is_error:
                raise RuntimeError(result.content[0].text)
            return result.structured_content

        out = await anyio.to_thread.run_sync(run_oracle, call)
        print(json.dumps({"spent": out["spent"], "simulate_used": out["simulate_used"], "note": out["note"],
                          "submission": out["submission"]}))


anyio.run(main)
PY
