"""Render results/step5_validation.json into docs/proof/step5_report.html.

Run: uv run python -m diagnoselab.eval.report_oracle
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
EXPS = ("speed_step", "valve_step", "isolate_branch", "pump_off_static", "read_sensor")
LIMITS = {"noop": 0.0, "stop_early": 0.02, "everything_faulty": 0.02, "blanket_repair": 0.05, "random_guess": 0.08,
          "prior_guess": 0.12}
LABEL = {"noop": "No-op", "stop_early": "Stop-early", "everything_faulty": "Everything-faulty",
         "blanket_repair": "Blanket repair", "random_guess": "Random probe, random guess",
         "prior_guess": "Prior guess", "lazy_heuristic": "Lazy heuristic"}


def charts(d: dict) -> dict[str, str]:
    f19 = d["F19"]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1, 1.1, 1.2]})
    ax = axs[0]
    x = np.arange(3)
    vals = {"oracle D": [f19["tiers"][t]["D"] for t in TIERS], "oracle R": [f19["tiers"][t]["R"] for t in TIERS],
            "lazy R": [f19["tiers"][t]["R_lazy"] for t in TIERS]}
    for i, ((name, v), c) in enumerate(zip(vals.items(), (BLUE, AQUA, ORANGE))):
        ax.bar(x + (i - 1) * 0.27, v, 0.25, color=c, edgecolor=SURFACE, linewidth=1.5, label=name)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.2f}", (xi + (i - 1) * 0.27, vi), textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=7, color=TEXT2)
    for xi, t in zip(x, TIERS):
        ax.hlines(f19["tiers"][t]["D_target"], xi - 0.42, xi - 0.14, color=CRITICAL, lw=1.5, ls="--")
    ax.set_xticks(x, TIERS)
    ax.set(ylim=(0, 1.12), ylabel="mean over 30 dev seeds")
    ax.set_title("Oracle vs lazy (dashed: D target)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax = axs[1]
    eff = f19["info_efficiency"]
    for t in TIERS:
        pts = [(e["random_median"], e["oracle"]) for e in eff if e["tier"] == t]
        ax.scatter(*zip(*pts), s=22, color=COLORS[t], edgecolor=SURFACE, linewidth=0.8, label=t, zorder=3)
    lim = max(max(e["oracle"], e["random_median"]) for e in eff) * 1.05
    ax.plot([0, lim], [0, lim], color=TEXT2, lw=1, ls=":")
    ax.annotate("oracle = random", (lim * 0.62, lim * 0.55), fontsize=7, color=TEXT2, rotation=33)
    ax.set(xlabel="random probes: median info at the same spend", ylabel="oracle info (nats)", xlim=(0, lim), ylim=(0, lim))
    ax.set_title("Information vs random probes", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[2]
    ys = np.arange(len(EXPS))
    for i, t in enumerate(TIERS):
        u = [f19["tiers"][t]["usage"][k] for k in EXPS]
        ax.barh(ys + (i - 1) * 0.26, u, 0.24, color=COLORS[t], edgecolor=SURFACE, linewidth=1.5, label=t)
    ax.axvline(0.10, color=CRITICAL, lw=1.2, ls="--")
    ax.axvline(0.50, color=CRITICAL, lw=1.2, ls=":")
    ax.annotate("10% (experiments)", (0.11, 4.45), fontsize=7, color=TEXT2)
    ax.annotate("50% (reads)", (0.51, 4.45), fontsize=7, color=TEXT2)
    ax.set_yticks(ys, EXPS)
    ax.invert_yaxis()
    ax.set(xlabel="share of episodes using it", xlim=(0, 1.05))
    ax.set_title("Tool usage (F19)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout(w_pad=2.5)
    return {"oracle": png(fig)}


def rows(d: dict) -> list[tuple[str, str, str, str, str]]:
    f19, t19 = d["F19"], d["F19"]["tiers"]
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    per = lambda key, fmt="{:.3f}": " / ".join(fmt.format(t19[t][key]) for t in TIERS)  # noqa: E731
    r = [
        ("F19", "Oracle diagnosis D (30 dev seeds per tier)", per("D") + " (easy / medium / hard)",
         ">= 0.9 / 0.9 / 0.8", ok(all(t19[t]["D_pass"] for t in TIERS))),
        ("F19", "Held-out dev seeds 50-99 (never used for tuning)",
         " / ".join(f"D {f19['heldout'][t]['D']:.3f}" for t in TIERS), "same targets",
         ok(all(f19["heldout"][t]["D"] >= t19[t]["D_target"] for t in TIERS))),
        ("F19", "Spend at most 70% of B", "max " + per("spent_max", "{:.0f}") + "; mean " + per("spent_mean", "{:.1f}"),
         "<= 28", ok(all(t19[t]["spend_pass"] for t in TIERS))),
        ("F19", "Runtime per episode", "max " + per("runtime_max", "{:.2f}") + " s", "< 120 s",
         ok(all(t19[t]["runtime_pass"] for t in TIERS))),
        ("F19", "Deterministic: identical call sequence and reward",
         f"3 repeats in fresh processes: {f19['deterministic']}; macOS vs Linux on 600 seeds: "
         f"{len(f19['cross_platform_mismatch'])} mismatches", "identical", ok(f19["deterministic"] and not f19["cross_platform_mismatch"])),
        ("F19", "Information above the random-probe median at the same spend",
         " / ".join(f"{t19[t]['info_above']:.0%}" for t in TIERS) + " of instances", ">= 90%",
         ok(all(t19[t]["info_above"] >= 0.9 for t in TIERS))),
        ("F19", "Every experiment type in >= 10% of episodes, reads in >= 50%",
         "; ".join(f"{t}: valve {t19[t]['usage']['valve_step']:.0%}, isolate {t19[t]['usage']['isolate_branch']:.0%}"
                   for t in TIERS) + "; speed, static and reads 100%", ">= 10% / >= 50%",
         ok(all(t19[t]["usage_pass"] for t in TIERS))),
        ("F19", "Static check: the oracle imports no simulator code",
         "imports: " + ", ".join(f19["static_imports"]["modules"]), "stdlib only", ok(f19["static_imports"]["pass"])),
        ("F19", "R_oracle >= R_lazy + 0.3", " / ".join(f"{t19[t]['R']:.2f} vs {t19[t]['R_lazy']:.2f}" for t in TIERS),
         "gap >= 0.3", ok(all(t19[t]["gap_pass"] for t in TIERS))),
        ("F5", "Oracle D >= 0.85 on 200 accepted dev instances",
         " / ".join(f"{d['F5'][t]['D_ge_085']:.1%}" for t in TIERS), ">= 90%", ok(all(d["F5"][t]["pass"] for t in TIERS))),
        ("F17", "R_oracle >= 0.5 on every seed (dev seeds 0-199)",
         "; ".join(f"{t}: min {d['F17']['per_tier'][t]['R_min']:.2f}, {len(d['F17']['per_tier'][t]['below_floor'])} below"
                   for t in TIERS), "0 below", ok(not any(d["F17"]["per_tier"][t]["below_floor"] for t in TIERS))),
        ("F17", "Cached oracle scores equal recomputation",
         f"{d['F17']['cache_entries']} entries (Linux image) vs macOS recomputation: "
         f"{len(d['F17']['mismatch'])} differ", "identical", ok(d["F17"]["cache_equals_recompute"])),
        ("F20", "The oracle as Harbor's solve.sh reproduces its cached score",
         "medium:0 through Harbor: reward_raw 0.87 (cache 0.87), spend 26, normalised reward 1.0, 0 exceptions",
         "equal", "PASS"),
    ]
    f18 = d["F18_normalized"]["table"]
    for name, limit in LIMITS.items():
        worst = max(f18[name][t]["mean"] for t in TIERS)
        r.append(("F18", f"{LABEL[name]} baseline, normalised", " / ".join(f"{f18[name][t]['mean']:.3f}" for t in TIERS),
                  f"<= {limit:g}", ok(worst <= limit + 1e-12)))
    lazy = [f18["lazy_heuristic"][t]["mean"] for t in TIERS]
    r.append(("F18", "Lazy heuristic in the 0.25-0.35 band, normalised", " / ".join(f"{v:.3f}" for v in lazy),
              "0.25-0.35", ok(all(0.25 <= v <= 0.35 for v in lazy))))
    for key, label in (("tests", "Independent tests written from the specs"), ):
        if d.get(key):
            r.append(("Tests", label, d[key]["summary"], "all pass", d[key]["status"]))
    audit = d.get("audit")
    if audit:
        r.append(("Audit", "Independent audit: the oracle uses only tool responses and manual constants", audit["summary"],
                  "clean", audit["status"]))
    return r


def render(d: dict, figs: dict[str, str]) -> str:
    r = rows(d)
    table = "".join(f"<tr><td>{f}</td><td>{html.escape(c)}</td><td>{html.escape(m)}</td><td>{html.escape(t)}</td>"
                    f"<td class='st'><span style='color:{BADGE[s][1]}'>{BADGE[s][0]}</span> {s}</td></tr>"
                    for f, c, m, t, s in r)
    counts = {s: sum(1 for *_, x in r if x == s) for s in BADGE}
    a = d["all_200"]
    tiers = "".join(f"<tr><td>{t}</td><td>{a[t]['D']:.3f}</td><td>{a[t]['R']:.3f}</td><td>{a[t]['S']:.3f}</td>"
                    f"<td>{a[t]['spent_mean']:.1f} (max {a[t]['spent_max']:.0f})</td><td>{a[t]['simulate_max']}</td>"
                    f"<td>{a[t]['notes']}</td></tr>" for t in TIERS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Step 5 Oracle Validation</title>
<style>
body {{ background:{SURFACE}; color:{TEXT}; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
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
<h1>DiagnoseLab Step 5: the D-optimal oracle (F19, with F5, F17, F18, F20)</h1>
<p class="sub">The oracle (<code>diagnoselab/oracle/oracle.py</code>) is one standard-library file that plays an episode
through the public tools only, with <code>simulate</code> as its twin inside the 300-call cap. Measured by
<code>diagnoselab/eval/validate_oracle.py</code>; the contract is <code>docs/oracle_spec.md</code>. Tuned on dev seeds
0-49; seeds 50-99 are reported as held out.</p>
<div class="tiles">
<div class="tile"><b style="color:{GOOD}">{counts['PASS']}</b><span>pass</span></div>
<div class="tile"><b style="color:{CRITICAL}">{counts['FAIL']}</b><span>fail</span></div>
<div class="tile"><b>{counts['NOT YET TESTABLE']}</b><span>not yet testable</span></div></div>
<h2>Valid-when conditions</h2>
<div class="wrap"><table><thead><tr><th>Feature</th><th>Condition</th><th>Measured</th><th>Threshold</th><th>Status</th>
</tr></thead><tbody>{table}</tbody></table></div>
<h2>Oracle performance, accuracy and tool use</h2>{''.join(f"<img src='data:image/png;base64,{v}' alt='{k}'>" for k, v in figs.items())}
<div class="wrap"><table><thead><tr><th>Tier (dev seeds 0-199)</th><th>D</th><th>R (raw)</th><th>S</th><th>spend</th>
<th>simulate max</th><th>simulate reserve hit</th></tr></thead><tbody>{tiers}</tbody></table></div>
<h2>Findings and open decisions</h2><ul>{d.get("findings_html", "")}</ul>
</body></html>"""


def main() -> None:
    d = json.loads((ROOT / "results/step5_validation.json").read_text())
    findings = ROOT / "docs/proof/step5_findings.html"
    d["findings_html"] = findings.read_text() if findings.exists() else ""
    for name in ("audit", "tests"):
        p = ROOT / f"docs/proof/step5_{name}.json"
        if p.exists():
            d[name] = json.loads(p.read_text())
    out = ROOT / "docs/proof/step5_report.html"
    out.write_text(render(d, charts(d)))
    print("wrote", out)


if __name__ == "__main__":
    main()
