#!/usr/bin/env python3
"""Apples-to-apples: sglang DeltaServe port vs DSV-vLLM on the same H200.

Both ran with: Llama-3-8B, tight timeline, 224 reqs, max_new_tokens=80,
batch_size=80 prefill.

DSV-vLLM bench used factor=1, api_server_count=1, gpu-mem 0.5.
Our sglang bench used cuda-graph enabled.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "output"


def load(path):
    rows = list(csv.DictReader(open(path)))
    # Two slightly different CSV schemas; normalize
    if "status" in rows[0]:
        ok = [r for r in rows if r["status"] == "ok"]
    else:
        ok = [r for r in rows if not r.get("error")]
    return {
        "ttft": np.array([float(r["ttft_s"])*1000 for r in ok if r.get("ttft_s") and r["ttft_s"] != ""]),
        "lat":  np.array([float(r["latency_s"])*1000 for r in ok if r.get("latency_s") and r["latency_s"] != ""]),
        "n": len(ok),
    }


configs = {
    "sglang inf-only":  (_OUT / "timeline_results_tight_inf_8b.csv", "tab:blue", "-"),
    "sglang co-serve":  (_OUT / "timeline_results_tight_co25_8b.csv", "tab:red", "-"),
    "DSV-vLLM inf-only":(_OUT / "dsv_vllm_inf_8b.csv", "tab:cyan", "--"),
    "DSV-vLLM co-serve":(_OUT / "dsv_vllm_co_8b.csv", "tab:orange", "--"),
}
data = {label: load(p) for label, (p, _, _) in configs.items()}

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# Panel 1: TTFT CDF
for label, (_, color, ls) in configs.items():
    d = data[label]
    if d["ttft"].size:
        x = np.sort(d["ttft"]); y = np.arange(1, x.size+1) / x.size
        p50 = float(np.percentile(d["ttft"], 50))
        p95 = float(np.percentile(d["ttft"], 95))
        axs[0].plot(x, y, label=f"{label} (p50 {p50:.0f}ms · p95 {p95:.0f}ms)",
                    color=color, linestyle=ls, lw=2)
axs[0].set_xlabel("TTFT (ms)"); axs[0].set_ylabel("CDF"); axs[0].set_title("TTFT CDF — sglang vs DSV-vLLM")
axs[0].set_xlim(0, max(80, max(d["ttft"].max() for d in data.values())*1.1))
axs[0].set_ylim(0, 1.02); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)

# Panel 2: Latency CDF
for label, (_, color, ls) in configs.items():
    d = data[label]
    if d["lat"].size:
        x = np.sort(d["lat"]); y = np.arange(1, x.size+1) / x.size
        p50 = float(np.percentile(d["lat"], 50))
        p95 = float(np.percentile(d["lat"], 95))
        axs[1].plot(x, y, label=f"{label} (p50 {p50:.0f}ms · p95 {p95:.0f}ms)",
                    color=color, linestyle=ls, lw=2)
axs[1].set_xlabel("E2E latency (ms)"); axs[1].set_ylabel("CDF"); axs[1].set_title("Latency CDF — sglang vs DSV-vLLM")
axs[1].set_ylim(0, 1.02); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)

fig.suptitle("Apples-to-apples co-serving: sglang DeltaServe port vs DSV-vLLM\n"
             "Both: H200 / Llama-3-8B / tight timeline / 224 reqs",
             fontsize=11, y=0.99)
plt.tight_layout()
out = _HERE / "plots" / "sglang_vs_dsv_vllm_apples.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
