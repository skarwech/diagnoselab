"""Render results/step2_validation.json into docs/proof/step2_report.html (charts embedded as PNG).

Run: uv run python -m diagnoselab.eval.report_sim
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from diagnoselab.sim.generator import generate_plant  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# reference palette (dataviz skill): surface, text, grid, categorical slots 1-3, status
SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
                     "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plant_figure(seed: int = 3, tier: str = "hard") -> str:
    spec = generate_plant(seed, tier)
    pt = {s.target for s in spec.sensors if s.kind == "pressure"}
    ft = {s.target for s in spec.sensors if s.kind == "flow"}
    col: dict[str, int] = {s.id: 0 for s in spec.sources}
    for _ in range(len(spec.links)):  # longest-path column from the sources (graph is a DAG by design)
        for ln in spec.links:
            if ln.frm in col and ln.kind != "pipe" or ln.frm in col:
                col[ln.to] = max(col.get(ln.to, 0), col[ln.frm] + 1)
    tanks = {o.id for o in spec.outlets}
    by_col: dict[int, list[str]] = {}
    for n, c in col.items():
        if n not in tanks:
            by_col.setdefault(c, []).append(n)
    pos = {n: (c, (i - (len(ns) - 1) / 2) * 1.3) for c, ns in by_col.items() for i, n in enumerate(sorted(ns))}
    for ln in spec.links:  # each tank sits just right of the junction feeding it, so no line crosses a node
        if ln.to in tanks:
            x, y = pos[ln.frm]
            pos[ln.to] = (x + 0.7, y - 0.45)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.grid(False)
    ax.axis("off")
    for ln in spec.links:
        (x0, y0), (x1, y1) = pos[ln.frm], pos[ln.to]
        color = {"pipe": AXIS, "valve": ORANGE, "pump": BLUE}[ln.kind]
        ax.plot([x0, x1], [y0, y1], color=color, lw=2 if ln.kind != "pipe" else 1.6, zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if ln.kind != "pipe":
            ax.scatter([mx], [my], marker=">" if ln.kind == "pump" else "D", s=70, color=color, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
        ax.annotate(ln.id, (mx, my), textcoords="offset points", xytext=(0, -14 if ln.kind == "valve" else 6),
                    ha="center", fontsize=7, color=TEXT2)
        if ln.id in ft:
            ax.scatter([x0 + 0.3 * (x1 - x0)], [y0 + 0.3 * (y1 - y0)], marker="s", s=22, color=AQUA, zorder=3,
                       edgecolors=SURFACE, linewidths=1)
    for n, (x, y) in pos.items():
        fixed = not n.startswith("J")
        face = TEXT if (n in pt or fixed) else SURFACE
        ax.scatter([x], [y], marker="s" if fixed else "o", s=140 if fixed else 90, color=face, edgecolors=TEXT,
                   linewidths=1.4, zorder=4)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, color=TEXT)
    handles = [plt.Line2D([], [], color=AXIS, lw=1.6, label="pipe"),
               plt.Line2D([], [], color=ORANGE, marker="D", lw=2, label="control valve"),
               plt.Line2D([], [], color=BLUE, marker=">", lw=2, label="pump + check valve"),
               plt.Line2D([], [], color=TEXT, marker="o", ls="", label="junction with pressure sensor"),
               plt.Line2D([], [], color=TEXT, marker="o", ls="", markerfacecolor=SURFACE, label="junction, no sensor"),
               plt.Line2D([], [], color=AQUA, marker="s", ls="", label="flow meter"),
               plt.Line2D([], [], color=TEXT, marker="s", ls="", label="source (S) / outlet tank (T)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=8)
    ax.set_title(f"Generated plant: seed {seed}, {tier} tier ({len(spec.junctions)} junctions, "
                 f"{len(spec.sensors)} sensors, {len(spec.branches)} branches)", fontsize=10, color=TEXT, loc="left")
    return png(fig)


def bars(ax, labels, values, color, fmt="{:.0f}"):
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=color, width=0.7, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(xs, labels)
    for x, v in zip(xs, values):
        ax.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=TEXT2)


def charts(d: dict) -> dict[str, str]:
    out = {"plant": plant_figure()}
    hist = {int(k): v for k, v in d["F2"]["iter_hist"].items()}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axs[0]
    ks = list(range(0, max(hist) + 1))
    ax.bar(ks, [hist.get(k, 0) for k in ks], color=BLUE, width=0.8, edgecolor=SURFACE, linewidth=1)
    ax.axvline(30, color=CRITICAL, lw=1.5, ls="--")
    ax.annotate("plan limit: 30", (30, ax.get_ylim()[1] * 0.85), xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=8, color=TEXT2)
    ax.set(xlabel="Newton iterations", ylabel="solves", title="Solver convergence, 10,000 random faulted solves")
    ax.title.set_fontsize(10)
    ax = axs[1]
    dh = np.log10(np.maximum(d["F2"]["diff_h"], 1e-16))
    ax.hist(dh, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(-6, color=CRITICAL, lw=1.5, ls="--")
    ax.annotate("plan limit: 1e-6", (-6, ax.get_ylim()[1] * 0.85), xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=8, color=TEXT2)
    ax.set(xlabel="log10 max |head difference| (m)", ylabel="cases",
           title="Agreement with the independent solver, 1,000 cases")
    ax.title.set_fontsize(10)
    out["solver"] = png(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axs[0]
    obs = d["F3"]["chi_counts"]
    ax.bar(range(len(obs)), obs, color=BLUE, width=0.75, edgecolor=SURFACE, linewidth=1.5)
    ax.axhline(np.mean(obs), color=TEXT, lw=1.2, ls="--")
    ax.set_ylim(0, max(obs) * 1.18)
    ax.annotate(f"dashed: expected {np.mean(obs):.0f} per pair", (0.99, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color=TEXT2)
    ax.set(xlabel="eligible (component, type) pair", ylabel="draws",
           title=f"First-fault uniformity, 10,000 draws (chi-square p = {d['F3']['chi_square_p']:.2f})")
    ax.title.set_fontsize(10)
    ax = axs[1]
    z = np.array(d["F4"]["z_sample"])
    ax.hist(z, bins=40, density=True, color=BLUE, edgecolor=SURFACE, linewidth=1)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), color=TEXT, lw=1.5)
    ax.annotate("N(0, 1)", (1.2, 0.3), fontsize=8, color=TEXT2)
    ax.set(xlabel="(reading - truth - bias) / sigma", ylabel="density",
           title=f"Sensor noise, {len(z) * 4:,} reads on 20 seeds")
    ax.title.set_fontsize(10)
    out["faults_noise"] = png(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axs[0]
    tiers = ["easy", "medium", "hard"]
    xs = np.arange(3)
    strict = [100 * d["F5"]["discard_plan_cosine"][t] for t in tiers]
    alt = [100 * d["F5"]["discard"][t] for t in tiers]
    ax.bar(xs - 0.19, strict, 0.36, color=BLUE, edgecolor=SURFACE, linewidth=2, label="plan rule: cos <= 0.95 (reference)")
    ax.bar(xs + 0.19, alt, 0.36, color=ORANGE, edgecolor=SURFACE, linewidth=2,
           label="adopted (option E): cos <= 0.95 or >= 5 sigma apart")
    for x, s, a in zip(xs, strict, alt):
        ax.annotate(f"{s:.0f}%", (x - 0.19, s), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.annotate(f"{a:.0f}%", (x + 0.19, a), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.axhline(35, color=CRITICAL, lw=1.5, ls="--")
    ax.annotate("plan limit: 35%", (-0.45, 35), textcoords="offset points", xytext=(0, 4), ha="left", fontsize=8,
                color=TEXT2)
    ax.set_xticks(xs, tiers)
    ax.set(ylabel="discard rate (%)", ylim=(0, 75), title="Certifier discard rate, 100 dev seeds per tier")
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axs[1]
    bt = d["F5"]["by_type"]["hard"]
    types = sorted(bt, key=lambda t: bt[t][0] / bt[t][1])
    names = {"leak": "leak", "sensor_stuck": "sensor\nstuck", "stuck_valve": "valve\nstuck",
             "sensor_bias": "sensor\nbias", "pump_wear": "pump\nwear", "fouling": "fouling"}
    bars(ax, [names[t] for t in types],
         [100 * bt[t][0] / bt[t][1] for t in types], BLUE, "{:.0f}%")
    ax.set(ylabel="first-draw acceptance (%)", ylim=(0, 100), title="Hard tier: acceptance by fault type (adopted rule)")
    ax.title.set_fontsize(10)
    out["certifier"] = png(fig)
    return out


def rows(d: dict) -> list[tuple[str, str, str, str, str]]:
    f1, f2, f3, f4, f5 = d["F1"], d["F2"], d["F3"], d["F4"], d["F5"]
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    a = f2["analytic"]
    analytic_max = max(v for k, v in a.items() if k.endswith("rel_err"))
    cross = f1["cross_build"].get("hashes_linux.json")
    hist_ok = all(set(range(lo, hi + 1)) <= {int(k) for k in f1["junction_histogram"][t]}
                  for t, (lo, hi) in f1["junction_ranges"].items())
    n_fail = sum(f1["check_failures"].values())
    dev = [f"{k} {t} {lo}-{hi}" for t, kinds in f1["size_ranges"].items() for k, (lo, hi) in kinds.items()
           if k != "junctions" and (lo < f1["plan_sizes"][k][0] or hi > f1["plan_sizes"][k][1])]
    r = [
        ("F1", "Every emitted plant passes every check (8 checks)", f"{3000 - n_fail} / 3000 plants", "all", ok(n_fail == 0)),
        ("F1", "Spec hash identical across two OS/Python builds",
         f"{sum(cross[t] for t in ('easy', 'medium', 'hard'))} / 3000 identical (macOS py3.12.13 vs Linux py3.12.16)"
         if cross else "not run", "all identical", ok(cross and sum(cross[t] for t in ('easy', 'medium', 'hard')) == 3000)),
        ("F1", "No duplicate hashes across 1000 seeds", f"{sum(f1['duplicate_hashes'].values())} duplicates", "0",
         ok(sum(f1["duplicate_hashes"].values()) == 0)),
        ("F1", "Junction histograms cover the per-tier ranges", "every junction count in each tier's range hit",
         "every value hit", ok(hist_ok)),
        ("F1", "Component counts within plan section 4.1", "; ".join(dev) if dev else "all within range",
         "; ".join(f"{k} {lo}-{hi}" for k, (lo, hi) in f1["plan_sizes"].items() if k != "junctions"), ok(not dev)),
        ("F1", "First draws valid",
         f"{100 * f1['first_draw_valid']:.1f}% of seeds need no internal redraw (redraws: tree with more leaves than "
         "outlets; cheap and deterministic)", ">= 95%", ok(f1["first_draw_valid"] >= 0.95)),
        ("F1", "Generation time per plant", f"{f1['gen_ms_median']:.2f} ms median", "< 50 ms", ok(f1["gen_ms_median"] < 50)),
        ("F2", "Analytic cases: series, parallel, pump-system, affinity", f"max rel error {analytic_max:.1e}", "< 1e-6",
         ok(analytic_max < 1e-6)),
        ("F2", "Pump-off: zero flow, static heads",
         f"|q| <= {a['pump_off_max_abs_flow']:.1e} L/s, head error {a['pump_off_head_err']:.1e} m",
         "flow << sigma (0.01 L/s)", ok(a["pump_off_max_abs_flow"] < 1e-4 and a["pump_off_head_err"] < 1e-6)),
        ("F2", "Agreement with independent second solver",
         f"heads {f2['diff_h_max']:.1e} m, flows {f2['diff_q_rel_max']:.1e} rel (1000 cases)", "< 1e-6",
         ok(f2["diff_h_max"] < 1e-6 and f2["diff_q_rel_max"] < 1e-6)),
        ("F2", "Junction residuals", f"max {f2['residual_max']:.1e} L/s", "< 1e-8", ok(f2["residual_max"] < 1e-8)),
        ("F2", "10,000 random plants + faults converge in <= 30 iterations",
         f"{f2['failures']} failures, max {f2['iter_max']} iterations", "0 failures, <= 30",
         ok(f2["failures"] == 0 and f2["iter_max"] <= 30)),
        ("F2", "Monotonicity", f"{f2['monotonicity_violations']} / {f2['monotonicity_cases']} violations", "0",
         ok(f2["monotonicity_violations"] == 0)),
        ("F2", "Solve time", f"median {f2['ms_median']:.2f} ms, p99 {f2['ms_p99']:.2f} ms", "< 5 ms, p99 < 20 ms",
         ok(f2["ms_median"] < 5 and f2["ms_p99"] < 20)),
        ("F3", "Fouling multiplies r exactly", f"rel error {f3['fouling_r_rel_err']:.1e}", "smoothing only",
         ok(f3["fouling_r_rel_err"] < 1e-6)),
        ("F3", "Stuck valve ignores commands", f"max head change {f3['stuck_valve_max_head_change']:.1e} m", "0",
         ok(f3["stuck_valve_max_head_change"] == 0)),
        ("F3", "Pump wear: head (1-w)a s^2 - b q^2; efficiency falls; vibration rises",
         f"rel error {f3['wear_head_rel_err']:.1e}; eta falls {f3['eta_decreasing_in_w']}; vib rises {f3['vib_monotone_in_w']}",
         "all", ok(f3["wear_head_rel_err"] < 1e-9 and f3["eta_decreasing_in_w"] and f3["vib_monotone_in_w"])),
        ("F3", "Leak imbalance equals C sqrt(h - z)", f"abs error {f3['leak_imbalance_abs_err']:.1e} L/s", "< 1e-9",
         ok(f3["leak_imbalance_abs_err"] < 1e-9)),
        ("F3", "Sensor bias offset at 10 setpoints; stuck sensor constant",
         f"bias error {f3['bias_offset_max_err']:.1e}; stuck values {f3['stuck_distinct_values']} distinct", "0; 1",
         ok(f3["bias_offset_max_err"] < 1e-12 and f3["stuck_distinct_values"] == 1)),
        ("F3", "Drift fires exactly once when spend crosses 0.4 B",
         f"{f3['drift_fires_counts'].get('1', 0)} / 1000 sequences fired once", "all", ok(f3["drift_fires_counts"] == {"1": 1000})),
        ("F3", "Uniform (component, type) draws, chi-square",
         f"p = {f3['chi_square_p']:.3f} ({f3['chi_square_cells']} pairs, 10,000 draws)", "p > 0.01", ok(f3["chi_square_p"] > 0.01)),
        ("F3", "Every physical fault degrades KPI >= 5%", f"min individual impact {100 * f3['kpi_min_individual_impact']:.2f}%",
         ">= 5%", ok(f3["kpi_min_individual_impact"] >= 0.05)),
        ("F3", "At most one fault per component; magnitudes in range",
         f"{f3['multi_fault_components']} violations; in range {f3['magnitudes_in_range']}", "0; True",
         ok(f3["multi_fault_components"] == 0 and f3["magnitudes_in_range"])),
        ("F4", "All statistical tests pass at p > 0.01 on 20 seeds (mean, variance, lag-1 autocorrelation)",
         f"lowest p: mean {min(f4['p_mean']):.3f}, variance {min(f4['p_var']):.3f}, autocorrelation "
         f"{min(f4['p_acf']):.3f} (N = 1000 per seed); pooled 20,000 reads vs N(0, 1): KS p = {f4['ks_pooled_p']:.2f}. "
         f"Bands: mean {f4['mean_ok']:.0f}/20, std within 5% {f4['std_ok']:.0f}/20 (5% is about 2.2 standard "
         f"errors), |r| < 0.1 {f4['acf_ok']:.0f}/20", "p > 0.01 each",
         ok(min(min(f4["p_mean"]), min(f4["p_var"]), min(f4["p_acf"])) > 0.01)),
        ("F4", "Same key gives same value; bias independent of operating point",
         f"{f4['repeat_ok']}/20, {f4['bias_op_ok']}/20", "20/20", ok(f4["repeat_ok"] == f4["bias_op_ok"] == 20)),
    ]
    for t in ("easy", "medium", "hard"):
        r.append(("F5", f"Discard rate, {t} (adopted rule, option E)",
                  f"{100 * f5['discard'][t]:.0f}% (plan cosine rule: {100 * f5['discard_plan_cosine'][t]:.0f}%)",
                  "< 35%", ok(f5["discard"][t] < 0.35)))
    r += [("F5", "Certifier decisions reproducible", f"make_instance deterministic: {f5['make_instance_deterministic']}",
           "identical", ok(f5["make_instance_deterministic"])),
          ("F5", "Oracle D >= 0.85 on >= 90% of 200 accepted dev instances", "needs the F19 oracle (Step 5)", ">= 90%",
           "NOT YET TESTABLE")]
    return r


BADGE = {"PASS": ("&#10003;", GOOD), "FAIL": ("&#10007;", CRITICAL), "NOT YET TESTABLE": ("&#9711;", "#8a8983")}


def render(d: dict, figs: dict[str, str]) -> str:
    table = "".join(
        f"<tr><td>{f}</td><td>{html.escape(c)}</td><td>{html.escape(m)}</td><td>{html.escape(t)}</td>"
        f"<td class='st'><span style='color:{BADGE[s][1]}'>{BADGE[s][0]}</span> {s}</td></tr>"
        for f, c, m, t, s in rows(d))
    counts = {s: sum(1 for *_, x in rows(d) if x == s) for s in BADGE}
    fd = d["findings"]
    img = lambda k, alt: f"<img src='data:image/png;base64,{figs[k]}' alt='{html.escape(alt)}'>"  # noqa: E731
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Step 2 Physics Validation</title>
<style>
:root {{ --surface:{SURFACE}; --text:{TEXT}; --text2:{TEXT2}; --grid:{GRID}; }}
body {{ background:var(--surface); color:var(--text); font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1040px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 8px; }}
.sub {{ color:var(--text2); margin:0 0 20px; }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
.tile {{ border:1px solid var(--grid); border-radius:8px; padding:10px 16px; min-width:120px; }}
.tile b {{ display:block; font-size:26px; }} .tile span {{ color:var(--text2); font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--grid); vertical-align:top; }}
th {{ color:var(--text2); font-weight:600; }} td.st {{ white-space:nowrap; font-weight:600; }}
.wrap {{ overflow-x:auto; }} img {{ width:100%; height:auto; margin:8px 0; }}
li {{ margin:6px 0; }} code {{ font-size:13px; }}
</style></head><body>
<h1>DiagnoseLab Step 2: physics core (F1 to F5)</h1>
<p class="sub">Every "valid when" condition from the plan, measured by <code>diagnoselab/eval/validate_sim.py</code>
at the plan's counts. Regenerate with <code>make validate-sim</code>; run the tests with <code>FULL=1 make test</code>. The F1 and F2 tests and the second
solver were written by separate agents from the spec; their transcripts show they never opened the implementation.</p>
<div class="tiles">
<div class="tile"><b style="color:{GOOD}">{counts['PASS']}</b><span>pass</span></div>
<div class="tile"><b style="color:{CRITICAL}">{counts['FAIL']}</b><span>fail</span></div>
<div class="tile"><b>{counts['NOT YET TESTABLE']}</b><span>not yet testable</span></div></div>
<h2>A generated plant</h2>{img("plant", "network diagram of a generated hard-tier plant")}
<h2>Valid-when conditions</h2>
<div class="wrap"><table><thead><tr><th>Feature</th><th>Condition</th><th>Measured</th><th>Threshold</th><th>Status</th></tr></thead>
<tbody>{table}</tbody></table></div>
<h2>Solver (F2)</h2>{img("solver", "histograms of solver iterations and differential error")}
<h2>Fault draws and sensor noise (F3, F4)</h2>{img("faults_noise", "chi-square counts and noise histogram")}
<h2>Certifier (F5)</h2>{img("certifier", "discard rates by rule and acceptance by fault type")}
<h2>Findings that change the plan or the manual</h2><ul>
<li><b>Decision: option E adopted.</b> The coherence rule now accepts a true fault whose signature has cosine above
0.95 with a competitor only if it is still at least 5 sigma from the best-fitting competitor at its true magnitude, and
mid-episode drift is cut (plan cut list item 1): the hard tier keeps 3 physical and 2 sensor faults without drift.
Discard rates (easy / medium / hard): {100 * d['F5']['discard']['easy']:.0f} / {100 * d['F5']['discard']['medium']:.0f} /
{100 * d['F5']['discard']['hard']:.0f}%, against {100 * d['F5']['discard_plan_cosine']['easy']:.0f} /
{100 * d['F5']['discard_plan_cosine']['medium']:.0f} / {100 * d['F5']['discard_plan_cosine']['hard']:.0f}% for the plan's
cosine rule on the same draws. Both rules use the full probe menu, which costs more than B, so they are necessary
conditions, not proof of solvability within budget; the oracle (Step 5) checks sufficiency. The drift code stays in
<code>faults.py</code> with its tests, ready to return if time allows.</li>
<li><b>F1 deviation: pipe counts.</b> The plan's ranges cannot all hold at once: easy plants have 6-8 junctions and
2-3 outlets, which leaves 3-8 pipes once 2-4 links are valves, while hard plants reach 15. Measured:
{"; ".join(f"{t} {lo}-{hi}" for t, (lo, hi) in ((t, d['F1']['size_ranges'][t]['pipes']) for t in ('easy', 'medium', 'hard')))}
pipes. Counting all links (pipes, valves and pumps) would be the natural reading if the plan meant network size.</li>
<li><b>Generator first draws.</b> Trees are now built with at most the tier's maximum number of leaves, so
{100 * d['F1']['first_draw_valid']:.1f}% of seeds need no internal redraw (plan: at least 95%).</li>
<li><b>Static test with a leak is not "flows read zero".</b> When the pumps stop, the tanks backfeed the leak. Flow
meters on that path read up to {fd['leak_0.08']['ft_sigma_max']:.0f} sigma (median {fd['leak_0.08']['ft_sigma_median']:.0f})
at an 8% leak and up to {fd['leak_0.02']['ft_sigma_max']:.1f} sigma at 2%. This is a leak signature for the manual.
Pressure sag reaches {fd['leak_0.08']['pt_sag_sigma_max']:.2f} sigma at 8%, above the plan's 0.5 sigma assumption (F10);
at 2% it is {fd['leak_0.02']['pt_sag_sigma_max']:.2f} sigma.</li>
<li><b>Open decision: stuck sensors read an exact constant.</b> The plan says "the reading is constant", so two
reads that match bit for bit identify a stuck sensor with certainty for 2 budget units, and each read is a noise-free
measurement. Alternative: stuck value plus normal noise, so detection needs an experiment that should move it.</li>
<li><b>Open decision: every physical fault must degrade the KPI by at least 5%.</b> Plan section 4.2 asks for
"at least one".
It keeps every fault worth repairing, but it excludes suction-side leaks and most loop-chord fouling (fouling 82%,
leaks 83% of pairs eligible; pump wear and stuck valves 100%).</li>
<li><b>Fixed: low-speed steps deadheaded healthy pumps.</b> At s = 0.5, 72% of healthy plants could not beat the static
lift, which made a speed step a cheap static test (plan decision 12). The speed step is now limited to [0.7, 1.2], and
the generator guarantees a healthy pump overcomes the lift at 0.7 with 2 m to spare. Worn pumps at low speed can still
deadhead; that is a real wear signature, and the KPI floors delivered flow so it stays
finite.</li>
<li><b>Fixed after independent review:</b> the stuck-valve check now tests both valve-step openings (a valve stuck at 0.3
looked healthy under a 0.3 command); the truth record has an <code>onset</code> field; the generator redraws heads when a small tree has no valid loop chord
(the independent F1 test caught 3 in 1000 medium seeds). An independent code review then found that in two-pump plants a
single pump stepped to 0.7 could still deadhead against the header held up by the other pump (149 of 622 plants); the
lift is now measured against the header and 0 of 1,244 single-pump steps deadhead. The review cross-checked the
certifier against its own implementation of spec section 10 and found no bugs. When every pump is stopped the solver
now starts from the known static solution.</li>
<li><b>The CRB rule is weak with known support.</b> Accepted instances need a median of
{d['F5']['spend']['easy'][0]:.0f} / {d['F5']['spend']['medium'][0]:.1f} / {d['F5']['spend']['hard'][0]:.1f} budget units
(easy / medium / hard) to pin magnitudes to 10%. Coherence does almost all the rejecting. Real difficulty sits in
finding the support, which the oracle (Step 5) will measure.</li>
<li><b>A stopped pump leaks about 1e-5 L/s</b> through the smoothed check valve, three orders below the smallest
flow sigma (0.01 L/s). Static-test checks use an absolute tolerance.</li>
<li><b>Sensor placement rule added.</b> Every junction joining exactly two pipes or valves in series now carries a
pressure sensor. Without it, fouling on either side is truly indistinguishable (cosine exactly 1.0).</li></ul>
</body></html>"""


def main() -> None:
    d = json.loads((ROOT / "results/step2_validation.json").read_text())
    out = ROOT / "docs/proof/step2_report.html"
    out.write_text(render(d, charts(d)))
    print("wrote", out)


if __name__ == "__main__":
    main()
