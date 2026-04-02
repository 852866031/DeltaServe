#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

CSV     = "bottleneck.csv"
CONFIGS = [(2, 256), (2, 2048), (2, 8192)]
LABELS  = ["512 total tokens\n(bs=2, sl=256)",
           "4 096 total tokens\n(bs=2, sl=2 048)",
           "16 384 total tokens\n(bs=2, sl=8 192)"]

C_GPU0 = "#D32F2F"   # red   – GPU 0 slower
C_GPU1 = "#1565C0"   # blue  – GPU 1 slower
C_ZERO = "#212121"
HIGHLIGHT_LAYERS = {23, 35}

# ── load ────────────────────────────────────────────────────────────────────
data = defaultdict(list)
with open(CSV) as f:
    for row in csv.DictReader(f):
        key = (int(row["batch_size"]), int(row["seq_len"]), int(row["layer_idx"]))
        data[key].append(float(row["expert_time_ratio"]))

layers = np.arange(48)

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(22, 14), sharex=True,
                         gridspec_kw={"hspace": 0.55})

fig.patch.set_facecolor("white")

for ax, (bs, sl), label in zip(axes, CONFIGS, LABELS):
    means = np.array([np.mean(data[(bs, sl, l)]) for l in layers])
    devs  = means - 1.0

    # bar colours: highlighted layers get a saturated edge
    bar_colours = []
    edge_colours = []
    edge_widths = []
    for i, d in enumerate(devs):
        base = C_GPU0 if d >= 0 else C_GPU1
        bar_colours.append(base)
        if i in HIGHLIGHT_LAYERS:
            edge_colours.append("#FFD600")   # gold outline for known hot layers
            edge_widths.append(1.8)
        else:
            edge_colours.append("none")
            edge_widths.append(0)

    ax.bar(layers, devs, color=bar_colours, edgecolor=edge_colours,
           linewidth=edge_widths, width=0.75, zorder=2)

    # zero baseline
    ax.axhline(0, color=C_ZERO, lw=0.9, zorder=3)

    # light horizontal gridlines
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # y-axis: symmetric, with extra headroom for rotated labels
    ylim = max(abs(devs.max()), abs(devs.min())) * 1.55 + 0.05
    ax.set_ylim(-ylim, ylim)
    ax.set_ylabel("GPU0/GPU1 ratio − 1", fontsize=8.5, labelpad=6)

    # label every bar with its ratio
    for i, (d, m) in enumerate(zip(devs, means)):
        sign = 1 if d >= 0 else -1
        pad  = ylim * 0.03 * sign
        va   = "bottom" if sign > 0 else "top"
        fw   = "bold" if i in HIGHLIGHT_LAYERS else "normal"
        ax.text(i, d + pad, f"{m:.2f}",
                ha="center", va=va, fontsize=6.5, fontweight=fw,
                color="#111111", rotation=90)

    # panel label (left side, inside)
    ax.text(0.01, 0.96, label, transform=ax.transAxes,
            fontsize=9, va="top", ha="left", color="#444444",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

axes[-1].set_xlabel("Layer index", fontsize=10)
axes[-1].set_xticks(range(0, 48, 4))
axes[-1].tick_params(axis="x", labelsize=8.5)

# ── shared title & legend ────────────────────────────────────────────────────
fig.suptitle("Qwen3-30B-A3B EP — Per-layer expert compute imbalance (GPU0 / GPU1)",
             fontsize=12, fontweight="bold", y=1.01)

legend_handles = [
    mpatches.Patch(color=C_GPU0, label="GPU 0 slower  (ratio > 1)"),
    mpatches.Patch(color=C_GPU1, label="GPU 1 slower  (ratio < 1)"),
    mpatches.Patch(facecolor="white", edgecolor="#FFD600", linewidth=1.8,
                   label="Known hot layers (23, 35)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.03),
           frameon=True, framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout(rect=[0, 0.04, 1, 1])
for ext in ("pdf", "png"):
    out = f"layer_imbalance_bars.{ext}"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")
