"""Render results/step3_validation.json into docs/proof/step3_report.html.

Run: uv run python -m diagnoselab.eval.report_server
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagnoselab.eval.report_sim import AXIS, BADGE, BLUE, CRITICAL, GOOD, GRID, ORANGE, SURFACE, TEXT, TEXT2, png

BADGE = {**BADGE, "NOT APPLICABLE": ("&#8212;", "#8a8983")}

ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"simulate": 300.0}


def charts(d: dict) -> dict[str, str]:
    out = {}
    p95 = d["F6"]["http"]["p95_ms"]
    names = sorted(p95, key=p95.get)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
    ax = axs[0]
    xs = np.arange(len(names))
    ax.barh(xs, [p95[n] for n in names], color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
    for x, n in zip(xs, names):
        ax.annotate(f"{p95[n]:.1f} ms", (p95[n], x), textcoords="offset points", xytext=(4, -3), fontsize=8,
                    color=TEXT2)
    ax.axvline(100, color=CRITICAL, lw=1.5, ls="--")
    ax.annotate("limit 100 ms (simulate 300)", (100, len(names) - 0.5), textcoords="offset points",
                xytext=(-4, 0), ha="right", fontsize=8, color=TEXT2)
    ax.set_yticks(xs, names)
    ax.set(xlabel="p95 latency over the real MCP transport (ms)", xlim=(0, 115))
    ax.set_title("Tool latency (F6)", fontsize=10, loc="left")
    ax = axs[1]
    f10 = d["F10"]
    pairs = ["bias_vs_fouling", "stuck_vs_fouling", "wear_vs_fouling", "leak_vs_meter_bias"]
    labels = ["sensor bias\nvs fouling", "stuck valve\nvs fouling", "pump wear\nvs fouling", "leak vs\nmeter bias"]
    single = [100 * f10[p]["separated_5sigma"] for p in pairs]
    whitened = [100 * f10[p]["separated_5sigma_whitened"] for p in pairs]
    xs = np.arange(len(pairs))
    ax.bar(xs - 0.19, single, 0.36, color=BLUE, edgecolor=SURFACE, linewidth=2, label="best single sensor (plan)")
    ax.bar(xs + 0.19, whitened, 0.36, color=ORANGE, edgecolor=SURFACE, linewidth=2,
           label="all returned sensors combined")
    for x, s, w in zip(xs, single, whitened):
        ax.annotate(f"{s:.0f}", (x - 0.19, s), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
        ax.annotate(f"{w:.0f}", (x + 0.19, w), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.axhline(95, color=CRITICAL, lw=1.5, ls="--")
    ax.set_xticks(xs, labels)
    ax.set(ylabel="cases separated at >= 5 sigma (%)", ylim=(0, 112))
    ax.set_title("Confounds separated (F10); dashed: 95% target", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    out["latency_f10"] = png(fig)
    return out


def rows(d: dict) -> list[tuple[str, str, str, str, str]]:
    f6, f78, fx, f10, f14 = d["F6"], d["F7_F8"], d["F9_F11_F12_F13"], d["F10"], d["F14"]
    http = f6["http"]
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    tot = sum(f10[p]["cases"] for p in ("bias_vs_fouling", "stuck_vs_fouling", "wear_vs_fouling", "leak_vs_meter_bias"))
    sep = sum(f10[p]["cases"] * f10[p]["separated_5sigma"] for p in
              ("bias_vs_fouling", "stuck_vs_fouling", "wear_vs_fouling", "leak_vs_meter_bias")) / tot
    sep_w = sum(f10[p]["cases"] * f10[p]["separated_5sigma_whitened"] for p in
                ("bias_vs_fouling", "stuck_vs_fouling", "wear_vs_fouling", "leak_vs_meter_bias")) / tot
    sag = f10["static_leak_sag_sigma_max"]
    viol = fx["property_violations"]
    lat_ok = all(v < LIMITS.get(k, 100.0) for k, v in http["p95_ms"].items())
    pub = {p["name"]: p["status"] for p in f14["public"]["probes"]} if f14 else {}
    container = [n for n in ("env", "filesystem", "grep_source", "import", "ports", "routes", "mcp")]
    return [
        ("F6", "Fuzzing: no crash, no server error",
         f"{f6['fuzz_calls']:,} calls ({f6['rejected']:,} rejected): {f6['crashes']} crashes; {http['raw_requests']} "
         f"malformed HTTP bodies: {http['status_5xx']} server errors, server alive {http['server_alive']} (all answered "
         f"{http['status_codes']} at the transport, before any tool code)",
         "0 crashes, 0 server errors", ok(f6["crashes"] == 0 and http["status_5xx"] == 0 and http["server_alive"])),
        ("F6", "Invalid calls change no state", f"{f6['state_changes_after_rejection']} state changes", "0",
         ok(f6["state_changes_after_rejection"] == 0)),
        ("F6", "No hidden value in error text (differential across fault sets)",
         f"{f6['differential_mismatches']} differences between two instances with the same plant and different faults",
         "0", ok(f6["differential_mismatches"] == 0)),
        ("F6", "Sessions isolated", f"interleaved episodes identical to separate runs: {f6['cross_session_identical']}",
         "identical", ok(f6["cross_session_identical"])),
        ("F6", "Response time does not depend on the hidden faults (timing side channel)",
         f"run_experiment on two instances, same plant, different faults: Mann-Whitney p = "
         f"{d['F6_timing']['p_padded']:.2f} with the 50 ms padding (p = {d['F6_timing']['p_unpadded']:.0e} without it)",
         "p > 0.01", ok(d["F6_timing"]["p_padded"] > 0.01)),
        ("F6", "Latency p95", ", ".join(f"{k} {v:.1f} ms" for k, v in sorted(http["p95_ms"].items())),
         "< 100 ms (simulate < 300 ms)", ok(lat_ok)),
        ("F7", "Every confound pair has its experiment documented",
         f"tests/test_f7_manual.py: {d['F7_checklist']['summary']}", "100%", ok(d["F7_checklist"]["pytest_exit"] == 0)),
        ("F7", "Manual and topology leak nothing (differential)",
         f"topology identical {100 * f78['topology_identical']:.0f}%, manual identical "
         f"{100 * f78['manual_identical']:.0f}% over {f78['instances']} instance pairs", "byte-identical",
         ok(f78["topology_identical"] == 1 and f78["manual_identical"] == 1)),
        ("F7", "Blind human audit: two teammates, 3 medium episodes each", "needs two people who did not write the manual",
         "mean >= 0.7 normalized", "NOT YET TESTABLE"),
        ("F8", "simulate with the true faults equals the true plant",
         f"max |difference| {f78['fidelity_max_abs']:.1e} over nominal, static, speed, valve and branch setpoints",
         "< 1e-9", ok(f78["fidelity_max_abs"] < 1e-9)),
        ("F8", "Empty overrides identical across fault sets; 301st call rejected; budget unchanged",
         f"identical {100 * f78['empty_simulate_identical']:.0f}%; 301st rejected {f78['call_301_rejected']}; "
         f"spent {f78['simulate_spent']}", "all", ok(f78["empty_simulate_identical"] == 1 and f78["call_301_rejected"]
                                                   and f78["simulate_spent"] == 0)),
        ("F9", "Ledger equals calls x cost", f"{100 * fx['ledger_exact']:.0f}% of {fx['sequences']:,} random sequences",
         "100%", ok(fx["ledger_exact"] == 1)),
        ("F9", "Deviation event fires exactly once", "drift is cut (option E): events is always []", "n/a",
         "NOT APPLICABLE"),
        ("F10", "Experiments revert; costs exact; out-of-range args rejected",
         f"revert {f10['revert_identical']}; costs {f10['costs']}; rejected {f10['out_of_range_rejected']}", "all",
         ok(f10["revert_identical"] and f10["out_of_range_rejected"]
            and f10["costs"] == {"speed_step": 4, "valve_step": 4, "isolate_branch": 6, "pump_off_static": 8})),
        ("F10", "Confound pairs separated at >= 5 sigma on one returned sensor",
         f"{100 * sep:.0f}% of {tot} cases (leak vs meter bias {100 * f10['leak_vs_meter_bias']['separated_5sigma']:.0f}%, "
         f"median {f10['leak_vs_meter_bias']['median_sigma']:.1f} sigma); all returned sensors combined: {100 * sep_w:.0f}%",
         ">= 95%", ok(sep >= 0.95)),
        ("F10", "Static-test pressure sag from a leak",
         f"max {sag['0.02']:.2f} / {sag['0.05']:.2f} / {sag['0.08']:.2f} sigma at leak 0.02 / 0.05 / 0.08", "< 0.5 sigma",
         ok(max(sag.values()) < 0.5)),
        ("F11", "Budget never exceeded; rejected calls change nothing; free tools cost 0",
         f"{fx['sequences']:,} random sequences: over-budget {viol['over_budget']}, changed state "
         f"{viol['rejected_changed_state']}, free-tool cost {viol['free_tool_cost']}", "0 / 0 / 0",
         ok(not any(viol.values()))),
        ("F11", "Caps finalize the episode", f"call 151 finalizes {fx['call_151_finalizes']}; 30-min cap finalizes "
         f"{fx['time_cap_finalizes']}", "both", ok(fx["call_151_finalizes"] and fx["time_cap_finalizes"])),
        ("F11", "Cap-hit rate in the frontier pilot", "needs the frontier agent (Step 6 pilot)", "<= 5%",
         "NOT YET TESTABLE"),
        ("F12", "One-shot submit, acknowledgement only, locked afterwards",
         f"ack {fx['ack']}; second submit {fx['second_submit']}; post-submit call {fx['post_submit']}", "all",
         ok(fx["ack"] == {"status": "submitted"} and fx["second_submit"].startswith("E_EPISODE_OVER")
            and fx["post_submit"].startswith("E_EPISODE_OVER"))),
        ("F12", "Score never appears in any response", f"{fx['episodes_with_score_text']} of 1,000 random episodes have a score/reward field in any response, or the "
         "words outside the manual's documentation (the independent suite also checks 10,000 episodes)", "0",
         ok(fx["episodes_with_score_text"] == 0)),
        ("F12", "Invalid format: 3 field-level rejections, then finalized",
         f"codes {', '.join(c.split(':')[0] for c in fx['strike_codes'])}; finalized {fx['finalized_after_4th_invalid']}",
         "3 x E_INVALID_SUBMISSION, then E_EPISODE_OVER", ok(fx["finalized_after_4th_invalid"])),
        ("F12", "A crashed agent still gets a verifier result",
         f"Harbor trial whose agent spends 8 units then exits 3 without submitting: reward.json "
         f"{(f14 or {}).get('crash', {}).get('reward')}", "reward written",
         ok(bool((f14 or {}).get("crash", {}).get("reward")))),
        ("F13", "Replay reproduces every response hash", f"{fx['replay_traces'] - fx['replay_mismatched']} / "
         f"{fx['replay_traces']} traces", "100%", ok(fx["replay_mismatched"] == 0)),
        ("F14", "Container isolation probes (inside the agent container, agent phase)",
         ", ".join(f"{n} {pub.get(n, '?')}" for n in container), "all PASS",
         ok(all(pub.get(n) == "PASS" for n in container))),
        ("F14", "Agent image layers hold no simulator or truth",
         f"{(f14 or {}).get('image_layer_scan', {}).get('bytes_scanned', 0) / 1e6:.0f} MB scanned, hits "
         f"{(f14 or {}).get('image_layer_scan', {}).get('hits')}", "no hits",
         ok((f14 or {}).get("image_layer_scan", {}).get("status") == "PASS")),
        ("F14", "Internet egress blocked (stops fetching the public repo to regenerate plants)",
         f"public network: {pub.get('egress', '?')}; Harbor's no-network and allowlist modes are rejected on this "
         "machine (Docker Desktop kernel lacks nftables fib); untested on Linux", "blocked",
         ok(pub.get("egress") == "PASS")),
    ]


def render(d: dict, figs: dict[str, str]) -> str:
    r = rows(d)
    table = "".join(f"<tr><td>{f}</td><td>{html.escape(c)}</td><td>{html.escape(m)}</td><td>{html.escape(t)}</td>"
                    f"<td class='st'><span style='color:{BADGE[s][1]}'>{BADGE[s][0]}</span> {s}</td></tr>"
                    for f, c, m, t, s in r)
    counts = {s: sum(1 for *_, x in r if x == s) for s in BADGE}
    iso = d["F14"] or {}
    probe_names = [p["name"] for p in iso.get("public", {}).get("probes", [])]
    iso_rows = "".join(f"<tr><td>{n}</td>" + "".join(
        f"<td>{next((p['status'] + ': ' + html.escape(p['detail']) for p in iso[v]['probes'] if p['name'] == n), 'not run')}</td>"
        for v in ("public",)) + "</tr>" for n in probe_names)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Step 3 Server Validation</title>
<style>
body {{ background:{SURFACE}; color:{TEXT}; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1040px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 8px; }}
.sub {{ color:{TEXT2}; margin:0 0 20px; }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
.tile {{ border:1px solid {GRID}; border-radius:8px; padding:10px 16px; min-width:120px; }}
.tile b {{ display:block; font-size:26px; }} .tile span {{ color:{TEXT2}; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid {GRID}; vertical-align:top; }}
th {{ color:{TEXT2}; font-weight:600; }} td.st {{ white-space:nowrap; font-weight:600; }}
.wrap {{ overflow-x:auto; }} img {{ width:100%; height:auto; margin:8px 0; }} li {{ margin:6px 0; }}
code {{ font-size:13px; }}
</style></head><body>
<h1>DiagnoseLab Step 3: MCP server (F6 to F14)</h1>
<p class="sub">Every "valid when" condition from the plan, measured by <code>diagnoselab/eval/validate_server.py</code>
and the isolation probes (<code>make isolation</code>). One episode core handles every call, valid or not; the MCP
binding passes raw arguments straight to it.</p>
<div class="tiles">
<div class="tile"><b style="color:{GOOD}">{counts['PASS']}</b><span>pass</span></div>
<div class="tile"><b style="color:{CRITICAL}">{counts['FAIL']}</b><span>fail</span></div>
<div class="tile"><b>{counts['NOT YET TESTABLE']}</b><span>not yet testable</span></div></div>
<h2>Valid-when conditions</h2>
<div class="wrap"><table><thead><tr><th>Feature</th><th>Condition</th><th>Measured</th><th>Threshold</th><th>Status</th>
</tr></thead><tbody>{table}</tbody></table></div>
<h2>Latency and confound separation</h2>{''.join(f"<img src='data:image/png;base64,{figs[k]}' alt='{k}'>" for k in figs)}
<h2>Isolation probes (inside the agent container, public network)</h2>
<div class="wrap"><table><thead><tr><th>Probe</th><th>Result</th></tr></thead><tbody>{iso_rows}</tbody></table></div>
<h2>Findings and open decisions</h2><ul>{d.get("findings_html", "")}</ul>
</body></html>"""


def main() -> None:
    d = json.loads((ROOT / "results/step3_validation.json").read_text())
    findings = ROOT / "docs/proof/step3_findings.html"
    d["findings_html"] = findings.read_text() if findings.exists() else ""
    out = ROOT / "docs/proof/step3_report.html"
    out.write_text(render(d, charts(d)))
    print("wrote", out)


if __name__ == "__main__":
    main()
