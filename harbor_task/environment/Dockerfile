FROM ubuntu:24.04
# Agent image ("main"): bash, curl for agent installers, and an MCP client only.
# Build context is all of environment/ (incl. sim/ and the truth): never COPY . here.
# .dockerignore excludes sim/ as a second guard.
RUN apt-get update \
 && apt-get install -y --no-install-recommends bash ca-certificates curl python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/mcp && /opt/mcp/bin/pip install --no-cache-dir mcp==2.3.0
ENV PATH=/opt/mcp/bin:$PATH
WORKDIR /workspace
