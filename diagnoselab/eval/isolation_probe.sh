#!/bin/bash
# F14 isolation probes, run INSIDE the agent container during the agent phase (as the oracle's solve.sh),
# so they see exactly the agent's filesystem, environment and network policy.
# Prints one "PROBE <name> PASS|FAIL <detail>" line per probe and a summary line.
SIM_HOST="${SIM_HOST:-sim}"
pass=0; fail=0
report() { if [ "$2" = PASS ]; then pass=$((pass+1)); else fail=$((fail+1)); fi; echo "PROBE $1 $2 $3"; }

# 1. environment: no seed, tier or simulator configuration visible
if env | grep -qiE 'DL_SEED|DL_TIER|noise_seed'; then report env FAIL "seed/tier variable visible"; else report env PASS "no seed/tier variables"; fi

# 2. filesystem: no truth, state, trace, simulator source or spec anywhere on the agent's disk
hits=$(find / -xdev \( -name 'episode.json' -o -name 'trace.jsonl' -o -name 'sim_spec.md' -o -name 'certifier.py' \
        -o -path '*diagnoselab*' -o -name '*truth*' \) 2>/dev/null | grep -v '^/proc' | head -5)
if [ -n "$hits" ]; then report filesystem FAIL "found: $(echo $hits | tr '\n' ' ')"; else report filesystem PASS "no truth/state/simulator files"; fi
if grep -rqsE 'make_instance|noise_seed|sample_faults' /usr/lib/python3* /opt /workspace /root /home 2>/dev/null; then
  report grep_source FAIL "simulator source strings found"; else report grep_source PASS "no simulator source strings"; fi

# 3. the simulator package cannot be imported
if python3 -c "import diagnoselab" 2>/dev/null; then report import FAIL "diagnoselab importable"; else report import PASS "diagnoselab not importable"; fi

# 4. sim network surface: full TCP scan (1-65535); only the MCP port may be open
open_ports=$(python3 - "$SIM_HOST" <<'PY'
import socket, sys
host = socket.gethostbyname(sys.argv[1])
found = []
for port in range(1, 65536):
    s = socket.socket(); s.settimeout(0.2)
    if s.connect_ex((host, port)) == 0:
        found.append(port)
    s.close()
print(" ".join(map(str, found)))
PY
)
if [ "$open_ports" = "8000" ]; then report ports PASS "full scan 1-65535: only 8000 open on $SIM_HOST"; else report ports FAIL "open on $SIM_HOST: $open_ports"; fi

# 5. no admin, score or truth routes next to MCP
bad_routes=""
for path in finalize score admin state truth episode trace reward health debug; do
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://$SIM_HOST:8000/$path")
  [ "$code" != "404" ] && bad_routes="$bad_routes /$path=$code"
done
if [ -z "$bad_routes" ]; then report routes PASS "all candidate routes 404"; else report routes FAIL "$bad_routes"; fi

# 6. internet egress (lets an agent fetch the public repo and regenerate plants): must be blocked
if curl -s -o /dev/null -m 8 https://pypi.org/simple/ ; then report egress FAIL "internet reachable"; else report egress PASS "internet blocked"; fi

# 7. MCP itself still works from here (the agent's legitimate path)
if python3 - "$SIM_HOST" <<'PY'
import sys, anyio
from mcp import Client
async def main():
    async with Client(f"http://{sys.argv[1]}:8000/mcp") as c:
        r = await c.call_tool("budget_status", {})
        assert not r.is_error
anyio.run(main)
PY
then report mcp PASS "MCP reachable at $SIM_HOST:8000"; else report mcp FAIL "MCP unreachable"; fi

echo "SUMMARY pass=$pass fail=$fail"
