# Harbor facts (Step 0 recon)

Read from the source of `harbor==0.25.0` and `mcp==2.3.0`, then confirmed by real trials on Docker Desktop 4.57 (engine 29.1.3, arm64) on 2026-11-13.
"Run" means a real Harbor trial exercised it.

## Confirmed

| # | Fact | How confirmed |
|---|---|---|
| 1 | The task layout is `task.toml`, `instruction.md`, `environment/` (Dockerfile plus optional `docker-compose.yaml`), `tests/test.sh` and `solution/solve.sh`. The schema is version 1.4. | `harbor init`, `harbor lint`; run |
| 2 | Each task declares its MCP servers under `[[environment.mcp_servers]]` with `transport = "streamable-http"` and a `url`. Harbor does not check that the server is reachable. | Source only: neither oracle nor nop uses this config, so it is untested until opencode runs |
| 3 | The sidecar defined in `environment/docker-compose.yaml` resolves as `sim` from the agent container `main` (public network mode). | Run |
| 4 | The verifier mode defaults to `shared`. We set `environment_mode = "separate"`, and the verifier image comes from `tests/Dockerfile`, which contains no simulator and no truth. | Source; run |
| 5 | `[[verifier.collect]]` runs `docker compose exec sim sh -c ...` after the agent container is stopped. It is best-effort: a failure is only logged. | Source; run (`trial.log` records "Main service stopped" before "Running collect hook in service 'sim'") |
| 6 | A sidecar `[[artifacts]]` entry is copied from sim to `<trial>/artifacts/<abs path>` on the host and then re-created at the same absolute path inside the separate verifier container. It never passes through `main`. | Run: `score.json` is identical in sim, the artifact and `reward.json` |
| 7 | `reward.json` must be a flat object of finite numbers, and it takes precedence over `reward.txt`. Extra keys show up as columns in the job table. | Source; run |
| 8 | The oracle agent uploads `solution/` and runs `solve.sh` inside `main`. It does **not** receive the MCP config, so `solve.sh` speaks MCP itself. | Source; run |
| 9 | The `nop` agent does nothing. Its reward is 0 because no submission exists. | Run |
| 10 | opencode installs itself with npm, so `main` needs internet while the agent is being set up. It maps streamable-http to `{"type":"remote","url":...}` and does not support MCP HTTP headers. | Source |
| 11 | opencode takes its model as `-m provider/model` and its key from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (via `--ae` or the host environment). | Source, `harbor run --print-config` |
| 12 | `mcp` 2.x: the server is `mcp.server.mcpserver.MCPServer` and the client is `mcp.Client(url)`. The server must bind `0.0.0.0`, because a localhost bind rejects `Host: sim`. Never add a `custom_route`: it would share the `/mcp` app and become reachable by the agent. | Source; run |
| 13 | Unknown `task.toml` keys are silently ignored. A typo fails without any error, so `harbor lint` and real runs are the only guard. | Source |
| 14 | Harbor injects `[environment.env]` into `main` only, so that section must never hold truth or a seed. | Source |

## Open (needs the starter kit, mentors, or a decision)

- Whether the organisers' starter kit differs from `harbor==0.25.0` on PyPI, for example a fork or a pinned version.
- Model credentials and the credit budget for the frontier agent. No key is set on this machine, so opencode has not been run yet.
- Concurrency available for `make eval-fast` (`-n`), and per-trial wall-clock limits.
- Whether the judges' scoring skeleton expects a scalar or a breakdown. Ours writes `reward` plus breakdown keys.
- GPU availability for the stretch goal (F25).
- Network hardening (measured at Step 3): Harbor's `no-network` and `allowlist` modes need nftables fib support in the Docker host kernel. Docker Desktop's VM lacks it, so Harbor rejects them on this Mac, including the per-phase `[agent] network_mode`. A static allowlist on a Linux Docker host is the candidate for judged runs. It is untested.
- F20 decision: `tests/test.sh` writes reward 0 when `score.json` is missing, and the collect hook is best-effort. A broken pipeline would therefore look like a legitimate zero. The fix is to make a missing score fail the trial (raise an exception) instead of scoring 0.
- How multiple seeds become Harbor tasks: one task directory per seed, or a dataset. To be decided at F1/F21.
