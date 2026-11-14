#!/bin/bash
# Relays the sim's server-side score into Harbor's reward file. Computes nothing.
# Never reads /logs/artifacts: that content is agent-controlled.
python3 - /var/lib/sim/out/score.json /logs/verifier/reward.json <<'PY'
import json, math, sys
src, dst = sys.argv[1], sys.argv[2]
reward = {"reward": 0}
try:
    with open(src) as f:
        data = json.load(f)
    ok = isinstance(data, dict) and "reward" in data and all(
        isinstance(k, str) and type(v) in (int, float) and math.isfinite(v)
        for k, v in data.items())
    if ok:
        reward = data
    else:
        print(f"bad score shape in {src}: {data!r}", file=sys.stderr)
except FileNotFoundError:
    print(f"missing {src}: collect hook or artifact transfer failed", file=sys.stderr)
except Exception as e:
    print(f"unreadable {src}: {e!r}", file=sys.stderr)
with open(dst, "w") as f:
    json.dump(reward, f)
print(json.dumps(reward))
PY
