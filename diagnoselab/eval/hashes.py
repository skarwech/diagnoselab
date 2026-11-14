"""Print spec hashes for seeds 0..N-1 per tier as JSON (cross-OS determinism check, plan F1)."""

import json
import sys

from diagnoselab.sim.generator import generate_plant, spec_hash

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
print(json.dumps({t: [spec_hash(generate_plant(s, t)) for s in range(N)] for t in ("easy", "medium", "hard")}))
