"""Draw a DiagnoseLab plant from its public topology (what an agent sees), with optional fault highlights."""

from __future__ import annotations

from collections import defaultdict, deque

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, DEEP, ORANGE, GRAY, RED, GREEN = "#1b4073", "#132e59", "#ed7d20", "#9aa3b5", "#c0392b", "#2e8b57"
KIND_STYLE = {"pipe": (GRAY, 2.0, "-"), "valve": (ORANGE, 3.0, "-"), "pump": (BLUE, 4.0, "-")}


def layout(topo: dict) -> dict[str, tuple[float, float]]:
    """Layered left-to-right layout: depth = longest link distance from a source; tanks in the last column."""
    out = defaultdict(list)
    for ln in topo["links"]:
        out[ln["frm"]].append(ln["to"])
    tanks = {o["id"] for o in topo["outlets"]}
    depth = {s["id"]: 0 for s in topo["sources"]}
    queue = deque(depth)
    while queue:
        n = queue.popleft()
        for m in out[n]:
            if m in tanks:
                continue
            if depth.get(m, -1) < depth[n] + 1 and depth[n] + 1 < 40:
                depth[m] = depth[n] + 1
                queue.append(m)
    for j in topo["junctions"]:
        depth.setdefault(j["id"], 1)
    last = max(depth.values()) + 1
    for t in tanks:
        depth[t] = last
    columns = defaultdict(list)
    order = [s["id"] for s in topo["sources"]] + [j["id"] for j in topo["junctions"]] + [o["id"] for o in topo["outlets"]]
    for n in order:
        columns[depth[n]].append(n)
    pos = {}
    for d, nodes in columns.items():
        for i, n in enumerate(nodes):
            pos[n] = (float(d), (len(nodes) - 1) / 2.0 - i)
    return pos


def draw(topo: dict, truth: dict[str, str] | None = None, diagnosed: dict[str, str] | None = None,
         probed: set[str] | None = None) -> plt.Figure:
    """truth / diagnosed: component -> fault type. Sensors are drawn as small rings on their junction or link."""
    truth, diagnosed, probed = truth or {}, diagnosed or {}, probed or set()
    pos = layout(topo)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sensors_on = defaultdict(list)
    for s in topo["sensors"]:
        sensors_on[s["target"]].append(s["id"])
    for ln in topo["links"]:
        (x1, y1), (x2, y2) = pos[ln["frm"]], pos[ln["to"]]
        color, width, style = KIND_STYLE[ln["kind"]]
        if ln["id"] in truth:
            color, width = RED, width + 2
        ax.plot([x1, x2], [y1, y2], color=color, lw=width, ls=style, solid_capstyle="round", zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        label = ln["id"] + (" ✓" if ln["id"] in diagnosed and ln["id"] in truth else " ✗" if ln["id"] in diagnosed else "")
        ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=8, color=DEEP, zorder=4)
        if any(s.startswith("FT-") for s in sensors_on[ln["id"]]):
            ax.scatter([mx], [my], s=60, facecolor="white", edgecolor=ORANGE if f"FT-{ln['id']}" in probed else DEEP,
                       linewidth=2, zorder=3)
    tanks = {o["id"] for o in topo["outlets"]}
    sources = {s["id"] for s in topo["sources"]}
    for n, (x, y) in pos.items():
        if n in tanks or n in sources:
            ax.scatter([x], [y], s=520, marker="s", color=DEEP, zorder=2)
            ax.text(x, y, n, ha="center", va="center", fontsize=8, color="white", zorder=5)
            continue
        face = RED if n in truth else BLUE
        ax.scatter([x], [y], s=420, color=face, zorder=2)
        ax.text(x, y, n, ha="center", va="center", fontsize=8, color="white", zorder=5)
        if f"PT-{n}" in sensors_on[n]:
            ax.scatter([x], [y], s=760, facecolor="none", edgecolor=ORANGE if f"PT-{n}" in probed else GRAY,
                       linewidth=2, zorder=1)
    sensor_faults = [f"{c} ({t.replace('sensor_', '')})" for c, t in truth.items() if t.startswith("sensor_")]
    if sensor_faults:
        ax.text(0.01, 0.02, "Lying sensors: " + ", ".join(sensor_faults), transform=ax.transAxes, fontsize=9, color=RED)
    handles = [Line2D([0], [0], color=GRAY, lw=2, label="pipe"), Line2D([0], [0], color=ORANGE, lw=3, label="valve"),
               Line2D([0], [0], color=BLUE, lw=4, label="pump"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=GRAY, markersize=12,
                      lw=0, label="pressure sensor (ring)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=DEEP, markersize=8,
                      lw=0, label="flow meter")]
    if truth:
        handles.append(Line2D([0], [0], color=RED, lw=4, label="true fault"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    ax.set_axis_off()
    ax.margins(0.08)
    fig.tight_layout()
    return fig
