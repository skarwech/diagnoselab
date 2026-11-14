"""DiagnoseLab MCP server: a thin binding over the episode core (docs/server_spec.md section 1).

The agent sees strict JSON schemas (additionalProperties: false); raw arguments go straight to
Episode.call, so the library's own argument validation never runs and every call is traced.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent, Tool

from diagnoselab.server.episode import Episode, ToolRejected
from diagnoselab.server.tools import TOOL_DESCRIPTIONS, TOOL_SCHEMAS
from diagnoselab.sim.instance import make_instance

COSTED_TOOLS = ("read_sensor", "run_experiment")
COSTED_LATENCY_S = 0.05  # costed calls answer after a constant delay: solver work on the true plant must not show

INSTRUCTIONS = ("DiagnoseLab: diagnose hidden faults in a pump network under a probe budget of 40. "
                "Start with read_manual and get_topology. simulate is a free digital twin.")


class DiagnoseLabServer(MCPServer):
    def __init__(self, episode: Episode, costed_latency_s: float = COSTED_LATENCY_S):
        super().__init__("diagnoselab", instructions=INSTRUCTIONS)
        self.episode, self.costed_latency_s = episode, costed_latency_s

    async def list_tools(self) -> list[Tool]:
        return [Tool(name=n, description=TOOL_DESCRIPTIONS[n], input_schema=s) for n, s in TOOL_SCHEMAS.items()]

    def _tool_input_schema(self, name: str) -> dict[str, Any] | None:
        return TOOL_SCHEMAS.get(name)

    async def call_tool(self, name: str, arguments: dict[str, Any], context: Any = None) -> CallToolResult:
        start = time.monotonic()
        try:
            result = self.episode.call(name, arguments)
            reply = CallToolResult(content=[TextContent(type="text", text=json.dumps(result))],
                                   structured_content=result, is_error=False)
        except ToolRejected as e:
            reply = CallToolResult(content=[TextContent(type="text", text=f"{e.code}: {e.message}")], is_error=True)
        if name in COSTED_TOOLS:  # pad every costed call, accepted or not, to the same wall time
            await anyio.sleep(max(0.0, self.costed_latency_s - (time.monotonic() - start)))
        return reply


def build_server(episode: Episode, costed_latency_s: float = COSTED_LATENCY_S) -> DiagnoseLabServer:
    return DiagnoseLabServer(episode, costed_latency_s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")  # 0.0.0.0 so Host: sim:8000 passes mcp's localhost guard
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--seed", type=int, default=int(os.environ.get("DL_SEED", "0")))
    p.add_argument("--tier", default=os.environ.get("DL_TIER", "easy"))
    a = p.parse_args()
    episode = Episode(make_instance(a.seed, a.tier), a.state)
    build_server(episode).run("streamable-http", host=a.host, port=a.port, streamable_http_path="/mcp")


if __name__ == "__main__":
    main()
