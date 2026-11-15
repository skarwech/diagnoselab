"""Render results/step6_validation.json into docs/proof/step6_report.html.

Run: uv run python -m diagnoselab.eval.report_calibration
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagnoselab.eval.report_sim import AQUA, BADGE, BLUE, CRITICAL, GOOD, GRID, ORANGE, SURFACE, TEXT, TEXT2, png

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("easy", "medium", "hard")
COLORS = dict(zip(TIERS, (BLUE, ORANGE, AQUA)))
ORDER = ("B40", "B26", "B26_subtle", "B26_meters")
SHORT = {"B40": "B = 40\n(frozen)", "B26": "B = 26", "B26_subtle": "B = 26\n+ subtle faults",
         "B26_meters": "B = 26\n+ half the meters"}


def charts(d: dict) -> dict[str, str]:
    c = d["configs"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.5, 1, 1.1]})
    ax = axs[0]
    x = np.arange(len(ORDER))
    ax.axhspan(*d["band"], color=GOOD, alpha=0.12, lw=0)
    ax.annotate("target band 0.35-0.65", (len(ORDER) - 0.5, 0.36), ha="right", fontsize=7, color=TEXT2)
    for i, t in enumerate(TIERS):
        m = [c[k]["tiers"][t]["mean"] for k in ORDER]
        lo = [m[j] - c[k]["tiers"][t]["ci"][0] for j, k in enumerate(ORDER)]
        hi = [c[k]["tiers"][t]["ci"][1] - m[j] for j, k in enumerate(ORDER)]
        ax.errorbar(x + (i - 1) * 0.2, m, yerr=[lo, hi], fmt="o", ms=6, color=COLORS[t], capsize=3, lw=1.5, label=t)
    ax.plot(x, [c[k]["overall"] for k in ORDER], color=TEXT, lw=1.5, ls="--", marker="D", ms=5, label="overall")
    ax.set_xticks(x, [SHORT[k] for k in ORDER], fontsize=8)
    ax.set(ylim=(0, 1.05), ylabel="agent score, normalised by the oracle")
    ax.set_title("Frontier agent (Kimi K3.1 Pro) per configuration, 95% CI", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=4)
    ax = axs[1]
    curve = d["spend_cap_curve_B40"]
    caps = [p["cap"] for p in curve]
    for t in TIERS:
        ax.plot(caps, [p[t] for p in curve], color=COLORS[t], lw=2, marker="o", ms=5, label=t)
        ax.axhline(d["D_targets"][t], color=COLORS[t], lw=1, ls=":")
    ax.axvline(18, color=CRITICAL, lw=1.2, ls="--")
    ax.annotate("cap 18 = 70% of B = 26", (18.3, 0.835), fontsize=7, color=TEXT2)
    ax.set(xlabel="oracle spend cap (units)", ylabel="oracle D (seeds 50-99)", ylim=(0.82, 1.01))
    ax.invert_xaxis()
    ax.set_title("How little budget the oracle needs", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax = axs[2]
    for t in TIERS:
        ax.scatter([c[k]["oracle_D"][t] for k in ORDER], [c[k]["tiers"][t]["D"] for k in ORDER], s=40,
                   color=COLORS[t], edgecolor=SURFACE, label=t, zorder=3)
    ax.plot([0.6, 1.0], [0.6, 1.0], color=TEXT2, lw=1, ls=":")
    ax.annotate("agent = oracle", (0.78, 0.75), fontsize=7, color=TEXT2, rotation=38)
    ax.set(xlabel="oracle D", ylabel="agent D", xlim=(0.6, 1.02), ylim=(0.6, 1.02))
    ax.set_title("Agent accuracy vs oracle accuracy", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(w_pad=2.5)
    return {"calibration": png(fig)}


def rows(d: dict) -> list[tuple[str, str, str, str, str]]:
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    f = d["F24"]
    chosen = d["configs"][d["chosen"]]
    tot = d["totals"]
    p = d["paired_B40_B26"]
    return [
        ("F24", "Oracle thresholds hold in the frozen configuration (B = 40)",
         "F19 seeds 0-29: D " + " / ".join(f"{d['official_F19'][t]['D']:.3f}" for t in TIERS) + "; F5 (D >= 0.85): " +
         " / ".join(f"{d['official_F19'][t]['F5']:.1%}" for t in TIERS) + "; held-out 50-99: D " +
         " / ".join(f"{chosen['oracle_D'][t]:.3f}" for t in TIERS),
         "D >= 0.9 / 0.9 / 0.8; F5 >= 90%", ok(f["oracle_ok"])),
        ("F24", "Frontier agent mean normalised score in the band",
         f"{f['agent_overall']:.3f} overall at B = 40 (" + " / ".join(f"{chosen['tiers'][t]['mean']:.2f}" for t in TIERS) +
         "), 5 seeds per tier; best probe B = 26: " + f"{d['configs']['B26']['overall']:.3f}", "0.35-0.65", ok(f["agent_in_band"])),
        ("F24", "Blind human audit reaches at least 0.7", "skipped (your decision in Step 6)", ">= 0.7", "FAIL"),
        ("Pilot", "Tool friction is not what the scores measure",
         f"{tot['rejected']} rejected calls of {tot['calls']:,} over {tot['episodes']} episodes; no invalid submits; "
         "1 timeout (the agent deliberating past 30 minutes)", "low", "PASS"),
        ("Probe", "Tighter budget moves the agent on the same instances",
         f"10 paired medium/hard instances: {p['mean_B40']:.3f} at B = 40 -> {p['mean_B26']:.3f} at B = 26", "lower", ok(p["mean_B26"] < p["mean_B40"])),
        ("Probe", "Subtler faults widen the gap", f"overall {d['configs']['B26_subtle']['overall']:.3f} vs "
         f"{d['configs']['B26']['overall']:.3f} at B = 26 alone", "lower",
         ok(d["configs"]["B26_subtle"]["overall"] < d["configs"]["B26"]["overall"])),
        ("Probe", "Fewer flow meters widen the gap", f"overall {d['configs']['B26_meters']['overall']:.3f} vs "
         f"{d['configs']['B26']['overall']:.3f} at B = 26 alone", "lower",
         ok(d["configs"]["B26_meters"]["overall"] < d["configs"]["B26"]["overall"])),
    ]


def render(d: dict, figs: dict[str, str]) -> str:
    r = rows(d)
    table = "".join(f"<tr><td>{f}</td><td>{html.escape(c)}</td><td>{html.escape(m)}</td><td>{html.escape(t)}</td>"
                    f"<td class='st'><span style='color:{BADGE[s][1]}'>{BADGE[s][0]}</span> {s}</td></tr>"
                    for f, c, m, t, s in r)
    counts = {s: sum(1 for *_, x in r if x == s) for s in BADGE}
    c = d["configs"]
    conf = "".join(
        f"<tr><td>{html.escape(c[k]['label'])}</td><td>{c[k]['episodes']}</td>" + "".join(
            f"<td>{c[k]['tiers'][t]['mean']:.2f} <span class='mut'>[{c[k]['tiers'][t]['ci'][0]:.2f}, {c[k]['tiers'][t]['ci'][1]:.2f}]"
            f"<br>D {c[k]['tiers'][t]['D']:.2f} &middot; spend {c[k]['tiers'][t]['spent']:.1f} &middot; oracle D {c[k]['oracle_D'][t]:.2f}</span></td>"
            for t in TIERS) + f"<td><b>{c[k]['overall']:.2f}</b></td></tr>" for k in ORDER)
    val = d.get("revalidation_html", "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Step 6 Calibration</title>
<style>
body {{ background:{SURFACE}; color:{TEXT}; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1140px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 8px; }}
.sub {{ color:{TEXT2}; margin:0 0 20px; }} .mut {{ color:{TEXT2}; font-size:12px; }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
.tile {{ border:1px solid {GRID}; border-radius:8px; padding:10px 16px; min-width:120px; }}
.tile b {{ display:block; font-size:26px; }} .tile span {{ color:{TEXT2}; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid {GRID}; vertical-align:top; }}
th {{ color:{TEXT2}; font-weight:600; }} td.st {{ white-space:nowrap; font-weight:600; }}
.wrap {{ overflow-x:auto; }} img {{ width:100%; height:auto; margin:8px 0; }} li {{ margin:6px 0; }}
code {{ font-size:13px; }}
</style></head><body>
<h1>DiagnoseLab Step 6: pilot and difficulty calibration (F24)</h1>
<p class="sub">A real frontier agent (opencode with Kimi K3.1 Pro on Verda) played {d['totals']['episodes']} episodes
through Harbor on held-out dev seeds, under four configurations. Scores are normalised by the oracle (Step 5). Built by
<code>diagnoselab/eval/validate_calibration.py</code> from the Harbor jobs' per-trial records.</p>
<div class="tiles">
<div class="tile"><b style="color:{GOOD}">{counts['PASS']}</b><span>pass</span></div>
<div class="tile"><b style="color:{CRITICAL}">{counts['FAIL']}</b><span>fail</span></div></div>
<h2>Conditions and probe outcomes</h2>
<div class="wrap"><table><thead><tr><th>Item</th><th>Condition</th><th>Measured</th><th>Threshold</th><th>Status</th>
</tr></thead><tbody>{table}</tbody></table></div>
<h2>Agent vs oracle across configurations</h2>{''.join(f"<img src='data:image/png;base64,{v}' alt='{k}'>" for k, v in figs.items())}
<div class="wrap"><table><thead><tr><th>Configuration</th><th>episodes</th><th>easy</th><th>medium</th><th>hard</th><th>overall</th></tr>
</thead><tbody>{conf}</tbody></table></div>
<h2>Re-validation of Steps 2-5</h2>{val}
<h2>Findings and open decisions</h2><ul>{d.get("findings_html", "")}</ul>
</body></html>"""


def main() -> None:
    d = json.loads((ROOT / "results/step6_validation.json").read_text())
    for key, name in (("findings_html", "step6_findings.html"), ("revalidation_html", "step6_revalidation.html")):
        p = ROOT / "docs/proof" / name
        d[key] = p.read_text() if p.exists() else ""
    out = ROOT / "docs/proof/step6_report.html"
    out.write_text(render(d, charts(d)))
    print("wrote", out)


if __name__ == "__main__":
    main()
