#!/usr/bin/env python3
"""Full sweep summary: TTFT + latency vs FT% across tight/loose timelines."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "output"


def stats(path):
    rows = list(csv.DictReader(open(path)))
    ttft = sorted(float(r["ttft_s"])*1000 for r in rows if r["ttft_s"])
    lat = sorted(float(r["latency_s"])*1000 for r in rows if r["latency_s"])
    return {
        "ttft_mean": np.mean(ttft),
        "ttft_p95": ttft[int(len(ttft)*0.95)],
        "lat_mean": np.mean(lat),
        "lat_p95": lat[int(len(lat)*0.95)],
        "lat_p99": lat[int(len(lat)*0.99)],
    }


configs = {
    "tight": [
        (0, "output/timeline_results_tight_inf_g.csv"),
        (10, "output/timeline_results_tight_co10_g.csv"),
        (25, "output/timeline_results_tight_co25_g.csv"),
        (50, "output/timeline_results_tight_co50_g.csv"),
    ],
    "loose": [
        (0, "output/timeline_results_loose_inf_g.csv"),
        (10, "output/timeline_results_loose_co10_g.csv"),
        (25, "output/timeline_results_loose_co25_g.csv"),
    ],
}

data = {}
for shape, runs in configs.items():
    series = []
    for ft, path in runs:
        p = _HERE / path
        if not p.exists():
            continue
        s = stats(p)
        series.append({"ft": ft, **s})
    data[shape] = series


fig, axs = plt.subplots(1, 2, figsize=(14, 5))

colors = {"tight": "tab:red", "loose": "tab:blue"}
for shape, series in data.items():
    if not series:
        continue
    fts = [s["ft"] for s in series]
    ttft_means = [s["ttft_mean"] for s in series]
    ttft_p95 = [s["ttft_p95"] for s in series]
    lat_means = [s["lat_mean"] for s in series]
    lat_p95 = [s["lat_p95"] for s in series]
    lat_p99 = [s["lat_p99"] for s in series]

    axs[0].plot(fts, ttft_means, "-o", color=colors[shape], label=f"{shape} mean", lw=2)
    axs[0].plot(fts, ttft_p95, "--^", color=colors[shape], label=f"{shape} p95", lw=1.5, alpha=0.7)
    axs[1].plot(fts, lat_means, "-o", color=colors[shape], label=f"{shape} mean", lw=2)
    axs[1].plot(fts, lat_p95, "--^", color=colors[shape], label=f"{shape} p95", lw=1.5, alpha=0.7)
    axs[1].plot(fts, lat_p99, ":s", color=colors[shape], label=f"{shape} p99", lw=1.5, alpha=0.7)

for ax, ylabel, title in [
    (axs[0], "TTFT (ms)", "TTFT vs FT load"),
    (axs[1], "E2E latency (ms)", "End-to-end latency vs FT load"),
]:
    ax.set_xlabel("FT fraction (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("sglang DeltaServe v046 — co-serving sweep (Llama-3.2-1B / H200)",
             fontsize=12, y=1.0)
plt.tight_layout()
out = _HERE / "plots" / "sweep_summary.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
