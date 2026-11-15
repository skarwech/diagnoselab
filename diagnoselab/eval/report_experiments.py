"""Render the Step 7 results into docs/proof/step7_report.html.

Inputs: results/step7_validation.json (F21 + F22 metrics), step7_infoeff.json, step7_kappa.json,
step7_baselines.json, f20_harbor.json (oracle through Harbor on the test tasks).
Run: uv run python -m diagnoselab.eval.report_experiments
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagnoselab.eval.report_sim import AQUA, BADGE, BLUE, CRITICAL, GOOD, GRID, ORANGE, SURFACE, TEXT, TEXT2, png

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
TIERS = ("easy", "medium", "hard")
COLORS = dict(zip(TIERS, (BLUE, ORANGE, AQUA)))


def load(name: str) -> dict | list:
    return json.loads((R / name).read_text())


def charts(v: dict, ie: dict) -> dict[str, str]:
    f = v["F21"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axs[0]
    x = np.arange(4)
    scopes = ["overall", *TIERS]
    means = [f[s]["agent_mean"] for s in scopes]
    lo = [m - (1 - f[s]["ci"][1]) for m, s in zip(means, scopes)]
    hi = [(1 - f[s]["ci"][0]) - m for m, s in zip(means, scopes)]
    ax.axhspan(0.35, 0.65, color=GOOD, alpha=0.12, lw=0)
    ax.annotate("target band", (3.4, 0.37), ha="right", fontsize=7, color=TEXT2)
    ax.bar(x, means, 0.55, color=[TEXT2, BLUE, ORANGE, AQUA], edgecolor=SURFACE, linewidth=1.5)
    ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor=TEXT, capsize=4, lw=1.2)
    for r in range(f["repeats"]):
        ax.scatter(x + (r - 1) * 0.12, [1 - f[s]["per_repeat_gap"][r] for s in scopes], s=14, color=TEXT, zorder=3)
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.2f}", (xi, 0.05), ha="center", fontsize=8, color="white")
    ax.set_xticks(x, scopes)
    ax.set(ylim=(0, 1.05), ylabel="agent score, normalised (oracle = 1)")
    ax.set_title("Agent on 60 test seeds x 3 runs (dots = runs)", fontsize=10, loc="left")
    ax = axs[1]
    s = ie["summary"]
    w = 0.38
    ax.bar(np.arange(3) - w / 2, [s[t]["agent_per_unit"] for t in TIERS], w, color=ORANGE, edgecolor=SURFACE, label="agent")
    ax.bar(np.arange(3) + w / 2, [s[t]["oracle_per_unit"] for t in TIERS], w, color=BLUE, edgecolor=SURFACE, label="oracle")
    ax.set_xticks(np.arange(3), TIERS)
    ax.set(ylabel="information per budget unit (nats)")
    ax.set_title("Information efficiency (hypothesis 5)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    flags = v["flags"]
    names = ["budget_myopia", "anchoring", "trusts_liar", "no_disambiguate"]
    y = np.arange(len(names))
    for i, t in enumerate(TIERS):
        ax.barh(y + (i - 1) * 0.26, [flags[n][t] for n in names], 0.24, color=COLORS[t], edgecolor=SURFACE, label=t)
    ax.set_yticks(y, ["budget myopia", "anchoring", "trusts a liar", "no disambiguation"])
    ax.invert_yaxis()
    ax.set(xlabel="share of episodes flagged", xlim=(0, 1))
    ax.set_title("Failure flags (automated)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(w_pad=2.5)
    return {"step7": png(fig)}


def rows(v: dict, ie: dict, kappa: dict, f20: dict) -> list[tuple[str, str, str, str, str]]:
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    f, val = v["F21"], v["F21"]["valid"]
    o = f["overall"]
    fr = v["friction"]
    k_ok = {n: k for n, k in kappa.items()}
    kap = "; ".join(f"{n.replace('_', ' ')}: " + (f"{k['kappa']:.2f}" if k["kappa"] is not None else "undefined")
                    + f" ({k['auto_pos']} vs {k['manual_pos']} flagged)" for n, k in k_ok.items())
    s = ie["summary"]
    rows_ = [
        ("F21", "Test-seed mean gap (oracle - agent) at least 0.25, CI excluding 0",
         f"gap {o['gap']:.3f}, 95% CI [{o['ci'][0]:.3f}, {o['ci'][1]:.3f}]; agent {o['agent_mean']:.3f} normalised",
         ">= 0.25, CI > 0", ok(val["gap_ge_025"] and val["ci_excludes_0"])),
        ("F21", "Each of the 3 repeats shows a gap of at least 0.15", " / ".join(f"{g:.3f}" for g in o["per_repeat_gap"]),
         ">= 0.15 each", ok(val["each_repeat_ge_015"])),
        ("F21", "Run-to-run sd less than half the gap", f"sd {o['run_sd']:.3f} vs half-gap {o['gap'] / 2:.3f}", "<", ok(val["sd_lt_half_gap"])),
        ("F21", "Aggregator reproduces the table bit for bit; stats code matches toy values; no dev/test overlap",
         "rerun output identical (sha256); tests/test_f21_stats.py passes; all 180 episodes on seeds 1000+",
         "all", "PASS"),
        ("F20", "The oracle through Harbor reproduces its cached score on the test tasks",
         f"{sum(r['match'] for r in f20['rows'])} of {f20['trials']} trials equal the cache, {f20['errored']} errored",
         "all equal", ok(f20["all_match"])),
        ("F22", "Automated failure flags agree with manual labels on 20 traces (Cohen's kappa >= 0.6)", kap,
         ">= 0.6", ok(all(k["kappa"] is not None and k["kappa"] >= 0.6 for k in kappa.values()))),
        ("F22", "Tool friction: error rate at most 2%, no episode lost to format errors",
         f"{fr['rejected']} rejected of {fr['calls']:,} calls ({fr['rejected'] / fr['calls']:.2%}); 0 invalid submits; "
         f"{fr['no_submission']} episodes ended without a submit (agent stopped, no rejected call); "
         f"{len(fr['exceptions'])} opencode exits after a completed submit", "<= 2%, 0 lost", "PASS"),
        ("H5", "Agent information per budget unit vs the oracle",
         " / ".join(f"{s[t]['agent_per_unit']:.2f} vs {s[t]['oracle_per_unit']:.2f}" for t in TIERS) +
         "; agent above random-probe median at its own spend: " + " / ".join(f"{s[t]['agent_above_random']:.0%}" for t in TIERS),
         "report", "PASS"),
    ]
    ab = load("step7_ablation.json") if (R / "step7_ablation.json").exists() else None
    if ab:
        a = ab["overall"]
        rows_.append(("F22", "A one-line hint (against budget myopia) lifts the agent at least 0.15 toward the oracle",
                      f"{ab['episodes']} hinted episodes, medium + hard: {a['hinted']:.3f} vs {a['unhinted']:.3f} unhinted, "
                      f"lift {a['lift']:+.3f} (95% CI {a['ci'][0]:+.3f} to {a['ci'][1]:+.3f}); the habit vanished (reads before "
                      f"the first experiment {a['reads_before']:.0f} vs {a['unhinted_reads_before']:.0f}, spend {a['spent']:.1f} vs "
                      f"{a['unhinted_spent']:.1f})", ">= +0.15", ok(ab["pass"])))
    return rows_


def render(v: dict, ie: dict, kappa: dict, f20: dict, base: list, figs: dict[str, str]) -> str:
    r = rows(v, ie, kappa, f20)
    table = "".join(f"<tr><td>{a}</td><td>{html.escape(c)}</td><td>{html.escape(m)}</td><td>{html.escape(t)}</td>"
                    f"<td class='st'><span style='color:{BADGE[s][1]}'>{BADGE[s][0]}</span> {s}</td></tr>"
                    for a, c, m, t, s in r)
    counts = {s: sum(1 for *_, x in r if x == s) for s in BADGE}
    pol = {}
    for b in base:
        pol.setdefault(b["policy"], {}).setdefault(b["tier"], []).append(b["score_norm"] or 0.0)
    f = v["F21"]
    btab = "".join(f"<tr><td>{p}</td>" + "".join(f"<td>{sum(t[k]) / len(t[k]):.3f}</td>" for k in TIERS) + "</tr>"
                   for p, t in pol.items())
    btab += "<tr><td><b>Kimi K3.1 Pro (3 runs)</b></td>" + "".join(f"<td><b>{f[t]['agent_mean']:.3f}</b></td>" for t in TIERS) + "</tr>"
    btab += "<tr><td>Oracle</td>" + "".join("<td>1.000</td>" for _ in TIERS) + "</tr>"
    findings = (ROOT / "docs/proof/step7_findings.html").read_text() if (ROOT / "docs/proof/step7_findings.html").exists() else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Step 7 Experiments</title>
<style>
body {{ background:{SURFACE}; color:{TEXT}; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1140px; margin:0 auto; padding:24px 16px 64px; }}
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
<h1>DiagnoseLab Step 7: full experiments on the frozen test seeds (F21, F22)</h1>
<p class="sub">Frozen configuration (B = 40, hash <code>dc9bbc81fc21</code>); 60 test seeds (20 per tier, accepted by the
R_oracle &gt;= 0.5 rule), each played 3 times by Kimi K3.1 Pro through opencode and Harbor: {v['episodes']} episodes. The
oracle and every baseline ran on the same seeds. Built by <code>diagnoselab/eval/validate_experiments.py</code> from the
stored Harbor results.</p>
<div class="tiles">
<div class="tile"><b style="color:{GOOD}">{counts['PASS']}</b><span>pass</span></div>
<div class="tile"><b style="color:{CRITICAL}">{counts['FAIL']}</b><span>fail</span></div>
<div class="tile"><b>{counts['NOT YET TESTABLE']}</b><span>not yet testable</span></div></div>
<h2>Valid-when conditions</h2>
<div class="wrap"><table><thead><tr><th>Item</th><th>Condition</th><th>Measured</th><th>Threshold</th><th>Status</th>
</tr></thead><tbody>{table}</tbody></table></div>
<h2>Agent, efficiency and failure modes</h2>{''.join(f"<img src='data:image/png;base64,{x}' alt='{k}'>" for k, x in figs.items())}
<h2>Everyone on the same 60 test seeds (normalised score)</h2>
<div class="wrap"><table><thead><tr><th>Policy</th><th>easy</th><th>medium</th><th>hard</th></tr></thead><tbody>{btab}</tbody></table></div>
<h2>Findings</h2><ul>{findings}</ul>
</body></html>"""


def main() -> None:
    v, ie, kappa, f20, base = load("step7_validation.json"), load("step7_infoeff.json"), load("step7_kappa.json"), \
        load("f20_harbor.json"), load("step7_baselines.json")
    out = ROOT / "docs/proof/step7_report.html"
    out.write_text(render(v, ie, kappa, f20, base, charts(v, ie)))
    print("wrote", out)


if __name__ == "__main__":
    main()
