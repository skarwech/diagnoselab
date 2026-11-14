"""F14: run the isolation probes inside the agent container through real Harbor trials.

Builds two task variants in a temp dir (public network, and the hardened agent-phase no-network
variant), swaps solve.sh for the probe script, runs the oracle agent, and collects the probe lines.
Run: uv run python -m diagnoselab.eval.run_isolation   (needs Docker)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"public": "sim", "no-network": "127.0.0.1", "allowlist": "127.0.0.1"}
ALLOWED = ["api.anthropic.com", "registry.npmjs.org"]  # what an OpenCode run needs


def make_variant(base: Path, name: str, host: str) -> Path:
    task = base / f"isolation_{name.replace('-', '_')}"
    shutil.copytree(ROOT / "harbor_task", task, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "diagnoselab/eval/isolation_probe.sh", task / "solution/solve.sh")
    toml = (task / "task.toml").read_text()
    toml = toml.replace('url = "http://sim:8000/mcp"', f'url = "http://{host}:8000/mcp"')
    toml = toml.replace('command = "exec 3<>/dev/tcp/sim/8000"', f'command = "exec 3<>/dev/tcp/{host}/8000"')
    if name != "public":  # static baseline policy: Docker here cannot switch policy per phase
        extra = f"\nallowed_hosts = {ALLOWED!r}".replace("'", '"') if name == "allowlist" else ""
        toml = toml.replace('network_mode = "public"', f'network_mode = "{name}"{extra}')
    toml += f'\n[solution.env]\nSIM_HOST = "{host}"\n' if "[solution.env]" not in toml else ""
    (task / "task.toml").write_text(toml)
    return task


CRASH_SCRIPT = """#!/bin/bash
# simulated agent crash: one costed call, then a non-zero exit with no submission
python3 - <<'PY'
import anyio
from mcp import Client
async def main():
    async with Client("http://sim:8000/mcp") as c:
        await c.call_tool("run_experiment", {"type": "pump_off_static", "args": {}})
anyio.run(main)
PY
echo "agent crashed"; exit 3
"""
LAYER_STRINGS = [b"make_instance", b"noise_seed", b"diagnoselab", b"sample_faults", b"episode.json"]


def image_layer_scan() -> dict:
    """Build the agent image exactly as Harbor does and grep every layer for simulator/truth strings."""
    tag = "dl-agent-scan"  # neutral tag: the saved tar contains its own tag name
    subprocess.run(["docker", "build", "-q", "-t", tag, str(ROOT / "harbor_task/environment")], check=True,
                   capture_output=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "img.tar"
        subprocess.run(["docker", "save", "-o", str(tar), tag], check=True)
        data = tar.read_bytes()
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
    hits = [s.decode() for s in LAYER_STRINGS if s in data]
    return {"bytes_scanned": len(data), "hits": hits, "status": "PASS" if not hits else "FAIL"}


def crash_trial(base: Path) -> dict:
    task = base / "isolation_crash"
    shutil.copytree(ROOT / "harbor_task", task, ignore=shutil.ignore_patterns("__pycache__"))
    (task / "solution/solve.sh").write_text(CRASH_SCRIPT)
    jobs = ROOT / "jobs" / "isolation"
    subprocess.run(["uv", "run", "harbor", "run", "-p", str(task), "-a", "oracle", "-k", "1", "-o", str(jobs)],
                   cwd=ROOT, capture_output=True, text=True)
    rewards = sorted(jobs.glob(f"*/{task.name}__*/verifier/reward.json"), key=lambda p: p.stat().st_mtime)
    logs = sorted(jobs.glob(f"*/{task.name}__*/agent/oracle.txt"), key=lambda p: p.stat().st_mtime)
    return {"reward": json.loads(rewards[-1].read_text()) if rewards else None,
            "agent_log_tail": logs[-1].read_text()[-200:] if logs else None}


def main() -> None:
    subprocess.run(["make", "sync"], cwd=ROOT, check=True)
    results = {"image_layer_scan": image_layer_scan()}
    print("image layer scan:", results["image_layer_scan"], flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, host in VARIANTS.items():
            task = make_variant(Path(tmp), name, host)
            jobs = ROOT / "jobs" / "isolation"
            run = subprocess.run(["uv", "run", "harbor", "run", "-p", str(task), "-a", "oracle", "-k", "1",
                                  "-o", str(jobs)], cwd=ROOT, capture_output=True, text=True)
            logs = sorted(jobs.glob(f"*/{task.name}__*/agent/oracle.txt"), key=lambda p: p.stat().st_mtime)
            text = logs[-1].read_text() if logs else run.stdout[-2000:] + run.stderr[-2000:]
            probes = re.findall(r"^PROBE (\S+) (PASS|FAIL) (.*)$", text, re.M)
            unsupported = "is not supported by" in text
            results[name] = {"probes": [{"name": n, "status": s, "detail": d} for n, s, d in probes],
                             "harbor_exit": run.returncode, "raw_tail": "" if probes or unsupported else text[-600:],
                             "unsupported_here": ("Harbor enforces this mode with nftables (CONFIG_NFT_FIB_INET); "
                                                  "Docker Desktop's VM kernel lacks it, so Harbor rejects the task. "
                                                  "Expected to work on a Linux Docker host, but not verified here.")
                             if unsupported else None}
            print(name, [(n, s) for n, s, _ in probes] or text[-400:], flush=True)
        results["crash"] = crash_trial(Path(tmp))
        print("crash trial:", results["crash"], flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/isolation.json").write_text(json.dumps(results, indent=1))
    sys.exit(0)


if __name__ == "__main__":
    main()
