#!/usr/bin/env python3
"""Plot the 3-way co-serving comparison (inf vs co10 vs co25)."""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "output"


def load(path):
    rows = list(csv.DictReader(open(path)))
    ttft = np.array([float(r["ttft_s"])*1000 for r in rows if r["ttft_s"]])
    lat = np.array([float(r["latency_s"])*1000 for r in rows if r["latency_s"]])
    sent = np.array([float(r["sent_t"]) for r in rows])
    is_ft = np.array([int(r["is_ft"]) for r in rows], dtype=bool)
    return {"ttft": ttft, "lat": lat, "sent": sent, "is_ft": is_ft}


runs = {
    "inf-only":      ("output/timeline_results_tight_inf.csv", "tab:blue"),
    "co-serve 10%":  ("output/timeline_results_tight_co10.csv", "tab:orange"),
    "co-serve 25%":  ("output/timeline_results_tight_co25.csv", "tab:red"),
}
data = {label: load(_HERE / path) for label, (path, _) in runs.items()}

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# --- 1. TTFT CDF ---
ax = axs[0]
for label, (_, color) in runs.items():
    ttft_sorted = np.sort(data[label]["ttft"])
    cdf = np.arange(1, len(ttft_sorted)+1) / len(ttft_sorted)
    ax.plot(ttft_sorted, cdf, label=label, color=color, lw=2)
ax.set_xlabel("TTFT (ms)")
ax.set_ylabel("CDF")
ax.set_title("TTFT CDF — inf vs co-serve")
ax.set_ylim(0, 1.02)
ax.legend()
ax.grid(alpha=0.3)

# --- 2. Latency CDF ---
ax = axs[1]
for label, (_, color) in runs.items():
    lat_sorted = np.sort(data[label]["lat"])
    cdf = np.arange(1, len(lat_sorted)+1) / len(lat_sorted)
    ax.plot(lat_sorted, cdf, label=label, color=color, lw=2)
ax.set_xlabel("End-to-end latency (ms)")
ax.set_ylabel("CDF")
ax.set_title("Latency CDF — inf vs co-serve")
ax.set_ylim(0, 1.02)
ax.legend()
ax.grid(alpha=0.3)

# --- 3. Latency over time (scatter) ---
ax = axs[2]
for label, (_, color) in runs.items():
    d = data[label]
    ax.scatter(d["sent"], d["lat"], c=color, alpha=0.5, s=10, label=label)
ax.set_xlabel("Send time (s, since timeline start)")
ax.set_ylabel("Latency (ms)")
ax.set_title("Latency vs send time")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = _OUT / "co_serving_comparison.png"
plt.savefig(out_path, dpi=120)
print(f"Saved: {out_path}")
