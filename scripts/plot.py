import csv
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
IDEAL = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# the five phases of the breakdown take the first five categorical colors, in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# n is an ordered size, so the lines get one blue from light to dark, darker for a larger n
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

TIMES = ["t_load", "t_count", "t_ppmi", "t_sim", "t_fw", "t_fw_comm", "t_edges", "t_total"]
# name in the csv, legend label, line style, marker
VARIANTS = [("perk", "V1 (per-k)", "-", "o"), ("blocked", "V2 (blocked)", "--", "s")]
PANELS = [("t_fw", "Floyd-Warshall (t_fw)"), ("t_total", "whole run (t_total)")]
PHASES = ["load + count + ppmi", "similarity", "fw compute", "fw comm", "edges"]


def setup_plots():
    matplotlib.use("Agg")  # no display on the cluster
    plt.rcParams.update({
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.75,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.handlelength": 2.8,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
    })


def read_medians(path):
    groups = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["program"], row["variant"], int(row["n"]), int(row["p"]))
            groups.setdefault(key, []).append(row)
    medians = []
    for (program, variant, n, p), rows in groups.items():
        m = {"program": program, "variant": variant, "n": n, "p": p}
        for c in TIMES:
            m[c] = float(np.median([float(r[c]) for r in rows]))
        medians.append(m)
    return medians


def pick(rows, variant, n=None):
    # rows of one variant (and one n, if given) in order of p
    keep = [r for r in rows if r["variant"] == variant and (n is None or r["n"] == n)]
    return sorted(keep, key=lambda r: r["p"])


def find(rows, variant, n, p):
    for r in rows:
        if r["variant"] == variant and r["n"] == n and r["p"] == p:
            return r
    return None


def ratios(rows, col):
    # T(1) / T(p), for rows in order of p with p = 1 first
    return [rows[0][col] / r[col] for r in rows]


def draw(ax, x, y, color, style, marker):
    # the fill is 8 px (6 pt) across, inside a 2 px ring in the surface color, so markers
    # that overlap stay apart
    ax.plot(x, y, color=color, linestyle=style, marker=marker, markersize=7.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, solid_capstyle="round")


def key(label, color, style="-", marker=None, width=1.5):
    return Line2D([], [], color=color, linestyle=style, marker=marker, markersize=6,
                  linewidth=width, label=label)


def save(fig, handles, path):
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=len(handles))
    fig.savefig(path, bbox_inches="tight")


def strong_figure(strong, what, path):
    ns = sorted({r["n"] for r in strong})
    ps = sorted({r["p"] for r in strong})
    steps = np.linspace(0, len(RAMP) - 1, len(ns)).round().astype(int)
    shades = [RAMP[i] for i in steps]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.5, 3.0))
    for ax, (col, title) in zip(axes, PANELS):
        ideal = [1, ps[-1]] if what == "speedup" else [1, 1]
        ax.plot([1, ps[-1]], ideal, color=IDEAL, linewidth=1, zorder=1)
        for n, color in zip(ns, shades):
            for variant, _, style, marker in VARIANTS:
                rows = pick(strong, variant, n)
                x = [r["p"] for r in rows]
                y = ratios(rows, col)
                if what == "efficiency":
                    y = [s / p for s, p in zip(y, x)]
                draw(ax, x, y, color, style, marker)
        ax.set_title(title, loc="left")
        ax.set_xlabel("p (MPI ranks, one GPU each)")
        ax.set_xticks(ps)
        ax.set_xlim(0, ps[-1] + 1)
        ax.set_ylim(bottom=0)
    if what == "speedup":
        axes[0].set_ylabel("speedup S = T(1) / T(p)")
    else:
        axes[0].set_ylabel("efficiency E = S / p")
    handles = [key(f"n = {n}", color) for n, color in zip(ns, shades)]
    handles += [key(label, INK, style, marker) for _, label, style, marker in VARIANTS]
    handles.append(key("ideal", IDEAL, width=1))
    save(fig, handles, path)


def weak_figure(weak, path):
    # n grows with p, so the points sit evenly spaced and the tick labels give p and n
    rows = pick(weak, "perk")
    labels = [f"{r['p']}\nn={r['n']}" for r in rows]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.5, 3.0))
    for ax, (col, title) in zip(axes, PANELS):
        ax.plot([0, len(rows) - 1], [1, 1], color=IDEAL, linewidth=1, zorder=1)
        for variant, _, style, marker in VARIANTS:
            y = ratios(pick(weak, variant), col)
            draw(ax, range(len(y)), y, INK, style, marker)
        ax.set_title(title, loc="left")
        ax.set_xlabel("p (MPI ranks, one GPU each)")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("weak efficiency T(1) / T(p)")
    handles = [key(label, INK, style, marker) for _, label, style, marker in VARIANTS]
    handles.append(key("ideal", IDEAL, width=1))
    save(fig, handles, path)


def phases(r):
    # the five parts add up to t_total
    return [r["t_load"] + r["t_count"] + r["t_ppmi"], r["t_sim"],
            r["t_fw"] - r["t_fw_comm"], r["t_fw_comm"], r["t_edges"]]


def seconds(t):
    return f"{t:.0f}" if t >= 10 else f"{t:.1f}"


def breakdown_figure(strong, path):
    n = max(r["n"] for r in strong)
    ps = sorted({r["p"] for r in strong if r["n"] == n})
    # V1 and V2 side by side for each p
    bars = []
    for g, p in enumerate(ps):
        for b, v in enumerate(VARIANTS):
            bars.append((g + (b - 0.5) * 0.34, phases(find(strong, v[0], n, p))))
    top = max(sum(heights) for x, heights in bars) * 1.1

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.set_ylim(0, top)
    for x, heights in bars:
        bottom = 0
        for h, color in zip(heights, SERIES):
            # the surface-colored edge leaves a thin gap between the segments
            ax.bar(x, h, width=0.22, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.5)
            if h > 0.07 * top:  # shorter segments cannot hold the number
                ax.text(x, bottom + h / 2, seconds(h), ha="center", va="center", fontsize=7,
                        color=INK if color == SERIES[3] else "#ffffff")
            bottom += h
        ax.text(x, bottom, seconds(bottom), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks([x for x, heights in bars])
    ax.set_xticklabels(["V1", "V2"] * len(ps))
    ax.tick_params(axis="x", length=0)
    for g, p in enumerate(ps):
        ax.text(g, -0.1, f"p = {p}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=INK2)
    ax.set_ylabel("time (s)")
    ax.set_title(f"time per phase at n = {n}", loc="left")
    # the legend lists the phases in the same order as the stack, top first
    handles = [Patch(facecolor=c, label=name) for c, name in zip(SERIES, PHASES)]
    ax.legend(handles=handles[::-1], loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def serial_table(serial, strong, path):
    def cell(rows, variant, n, col):
        r = find(rows, variant, n, 1)
        return "-" if r is None else f"{r[col]:.3f}"

    with open(path, "w") as f:
        print(r"\begin{tabular}{rrrrrrr}", file=f)
        print(r"\toprule", file=f)
        print(r" & \multicolumn{3}{c}{fw time (s)} & \multicolumn{3}{c}{total time (s)} \\", file=f)
        print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}", file=f)
        print(r"$n$ & serial & V1 & V2 & serial & V1 & V2 \\", file=f)
        print(r"\midrule", file=f)
        for n in sorted({r["n"] for r in serial}):
            row = [str(n)]
            for col in ["t_fw", "t_total"]:
                row.append(cell(serial, "serial", n, col))
                row.append(cell(strong, "perk", n, col))
                row.append(cell(strong, "blocked", n, col))
            print(" & ".join(row) + r" \\", file=f)
        print(r"\bottomrule", file=f)
        print(r"\end{tabular}", file=f)


def main():
    results_dir, out_dir = sys.argv[1:3]
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    serial = read_medians(os.path.join(results_dir, "serial.csv"))
    strong = read_medians(os.path.join(results_dir, "strong.csv"))
    weak = read_medians(os.path.join(results_dir, "weak.csv"))

    setup_plots()
    strong_figure(strong, "speedup", os.path.join(fig_dir, "strong_speedup.pdf"))
    strong_figure(strong, "efficiency", os.path.join(fig_dir, "strong_efficiency.pdf"))
    weak_figure(weak, os.path.join(fig_dir, "weak_efficiency.pdf"))
    breakdown_figure(strong, os.path.join(fig_dir, "breakdown.pdf"))
    serial_table(serial, strong, os.path.join(tab_dir, "serial_vs_gpu.tex"))


if __name__ == "__main__":
    main()
