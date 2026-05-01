# DeltaServe — Revised Evaluation Plan

> Target: re-submission as a co-serving paper for LLM inference + LoRA SFT.
> This plan supersedes `osdi_eval.md`. Key changes from the OSDI version:
> S-LoRA dropped, dataset switched to Alpaca, primary hardware moved to a
> 4-GPU multi-tenant setting, **LLMStation added as the direct competitor**.
> The eval keeps our **loose / tight + Company X** workload framing — we do
> not mirror LLMStation's RPS-sweep or TPOT-sweep figures. Our story is
> driven by *workload shape*, not by a steady-rate sweep.

---

## 0. What's different from the OSDI plan

| Item | OSDI version | Now |
|---|---|---|
| Primary baseline | S-LoRA | **LLMStation** + **vLLM-3GPU + PEFT-1GPU split pool**. S-LoRA dropped — pre-graph era, not apples-to-apples. |
| FT dataset | Emotion (~20-tok samples) | **Alpaca-1000** (real instruction-tuning lengths). Already the live path (`serving_config_finetuning_alpaca.yaml`). |
| FT throughput unit | tokens/s | **tokens/s** (kept). Backward batching is token-bounded; tokens/s is the natural unit. We convert LLMStation's samples/s numbers to tokens/s using their reported avg seq length when citing their results. |
| Primary hardware | 1× RTX 5090 + 4× A100 | **4× A100 for the head-to-head** (LLMStation does not run on 5090). **4× RTX 5090 used for mechanism studies** that don't require LLMStation. |
| Workload framing | Synthetic burst patterns + Company X | **loose / tight + Company X** (`timeline_loose.csv`, `timeline_tight.csv` — already wired into `auto_benchmark.py --loose / --tight`). Current code already shows we beat LLMStation on FT throughput in the loose case. |
| SLO definition | Per-experiment (100 ms, 0.1 s, 0.3 s) | **P99 TTFT and P99 TPOT, dual constraint, percentile-based.** Targets fixed across plots. |
| Single-GPU deep dive | Headline (Fig 3) | Demoted to ablation context. The contested deployment is multi-GPU. |
| Model | Llama-7B (llama1) | **Llama-3-8B** primary (active dev path; GQA; matches `eval/llama3/` tooling and `packed_kv` allocator path). |

---

## 1. Guiding questions

1. **Q1 — Head-to-head on our workloads.** Across loose, tight, and the
   Company X trace, how do DeltaServe, LLMStation, and the split-pool
   baseline compare on PEFT throughput at the same dual P99 TTFT/TPOT
   SLO?
2. **Q2 — Where DeltaServe wins, and why.** The loose workload — long
   inference valleys with bursty peaks — is where our prefill-fusion +
   forward/backward CUDA-graph capture compound into a measurable FT
   throughput advantage over LLMStation. Establish this and explain it
   with utilization counters and FT-tokens-during-valley measurements.
3. **Q3 — Tight-workload SLO behavior.** Under high-pressure tight
   traffic, does DeltaServe still hold the dual SLO (yielding FT as
   needed)? Does LLMStation? Does the split-pool baseline?
4. **Q4 — Mechanism contributions.** How much does each DeltaServe
   mechanism contribute to the headline result: CUDA graph capture
   (forward + backward, treated as one mechanism), the two-regime SLO
   estimator, and the packed-KV allocator?
5. **Q5 — Predictor accuracy.** Is the two-regime SLO estimator accurate
   enough across batch composition and hardware to drive admission
   correctly?
6. **Q6 — Cold-start sensitivity.** How much does offline profiling
   buy us at startup vs. relying purely on live refit? Quantifies a
   mechanism the OSDI version didn't have.
7. **Q7 — Allocator capacity headroom.** Does the GQA-packed KV pool
   translate into more concurrent inference at the same SLO, or just
   a different layout?

---

## 2. Common setup

### 2.1 Hardware
- **4 × NVIDIA A100 (40 GB)** — **head-to-head platform**. All
  three-system comparisons (DeltaServe vs LLMStation vs split-pool)
  run here. LLMStation's release does not run on RTX 5090, so this is
  the only platform where the head-to-head is feasible.
- **4 × NVIDIA RTX 5090 (32 GB)** — **mechanism-study platform**.
  Used for experiments where LLMStation isn't required: predictor
  cross-hardware fit, allocator capacity headroom (32 GB makes the KV
  pool binding, which is where packed_kv pays off), and any
  ablation that benefits from showing a consumer-GPU number.
- Both servers have intra-node NVLink. ≥ 256 GB host RAM. MPS daemon
  active on both.

### 2.2 Models
- **Llama-3-8B** (only model). LoRA rank **r=16**, α=32, dropout=0.05,
  applied to all attention projections.

### 2.3 Workloads
Three traces, kept consistent across every experiment. **Inference
arrivals are pre-recorded traces, not steady RPS sweeps** — this is
deliberate; we want to show robustness on workload *shape*, not a
synthetic dial.

| Trace | Source | Pattern | Where we expect to win |
|---|---|---|---|
| **loose** | `timeline_loose.csv` | Long valleys + sharp peaks; ample co-serving slack | Yes — current results already show DeltaServe > LLMStation on FT throughput here. **This is the headline workload.** |
| **tight** | `timeline_tight.csv` | Dense peaks, minimal slack | SLO compliance > FT throughput. We expect DeltaServe to hold dual SLO and yield FT. |
| **Company X** | 20-min production trace | Multi-scale bursts, irregular | Realism check; the trace LLMStation can't claim to have seen. |

PEFT inputs across all traces: **Alpaca-1000** (`load_alpaca.py` →
`_p95.txt`), pinned so `attn_l_max` is sized to avoid the monolithic
attention-bwd fallback.

### 2.4 Baselines (3 systems compared)
1. **DeltaServe (full)** — current main with all mechanisms on:
   `--enable-cuda-graph --enable-prefill-cuda-graph --enable-bwd-cuda-graph
   --packed-kv --alpaca`. TP=4 across the 4 GPUs.
2. **LLMStation** — author release, MPS thread % and deferral bound
   tuned per workload as in their paper. TP=4. Direct competitor.
3. **vLLM (3 GPU) + PEFT (1 GPU) split pool** — what production
   actually deploys. vLLM serves inference (TP=2 + 1 idle, or 3 single-
   GPU replicas — see contingencies §6); torchtune/PEFT on the 4th GPU.

> Notes:
> - **DeltaServe-Inf** (FT disabled) retained for ablation context only.
> - **chunked-training / FineInfer** intentionally skipped. They were
>   LLMStation's strawmen, not ours; if a reviewer asks we can add a
>   single-cell comparison row to Tab 1.

### 2.5 SLOs
- **P99 TTFT SLO = 500 ms.**
- **P99 TPOT SLO = 50 ms** (Llama-3-8B, LLMStation default).
- Both enforced **simultaneously**. PEFT throughput is reported subject
  to the dual constraint. All TTFT/TPOT/E2E numbers also reported as
  CDFs and P50/P99 in tables.

### 2.6 Metrics
- **PEFT throughput:** tokens/s. (When citing LLMStation paper numbers,
  convert their samples/s using the reported avg input length so the
  comparison is in our unit.)
- **Latency:** P99 TTFT, P99 TPOT, P50 E2E.
- **Utilization:** raw NSight Compute counters (SM Active %, Compute
  Warps in Flight, DRAM R/W BW). nvidia-smi util dropped as primary
  signal.
- **Allocator occupancy:** used-pages over time, 1 Hz CSV from the
  existing `_OccupancyTracker`.
- **System overheads:** scheduler decision time per iteration;
  predictor inference time.

---

## 3. Experiments

### Exp 1 — Headline head-to-head: loose, tight, Company X (Q1)
**The main result figure. Single platform: 4×A100. One row per
workload, three systems per panel.**

- Three rows: loose / tight / Company X.
- Four columns per row, mirroring the existing `plot_loose_tight.py`
  layout but with three system traces overlaid:
  1. Inference arrival pattern (shared across systems; one trace).
  2. P99 TTFT over 30-s windows, dashed at 500 ms.
  3. Per-request E2E latency scatter.
  4. Cumulative FT tokens over time (with tokens/s annotation).
- **Driver:** extend `auto_benchmark.py` to emit a per-system suffix and
  add an `--llmstation` / `--vllm-split` mode that launches the
  appropriate baseline harness. Then a new `compare_systems_plot.py`
  reads the per-system CSVs and produces the overlay.
- **Headline numbers** reported as **Tab 1** (one row per
  workload-system, six metrics: P99 TTFT, P99 TPOT, P50 E2E, total FT
  tokens, FT tokens/s, avg SM Active %).

**Expected story:**
- Loose: DeltaServe ≫ LLMStation on FT tokens/s (this is the result
  current data already supports). Both meet TTFT/TPOT.
- Tight: DeltaServe and LLMStation comparable on PEFT (or DeltaServe
  slightly behind, intentionally yielding); DeltaServe holds dual SLO,
  the split-pool baseline doesn't (only 3 GPUs serve inference under
  the same arrivals). Whether LLMStation holds is the open question.
- Company X: DeltaServe > LLMStation on PEFT, both hold SLO.

### Exp 2 — Loose-workload deep dive: *why* we win (Q2)
**The "innovation showcase" plot. Anchored on the loose workload from
Exp 1 because that's where the gap is largest and most explainable.**

Single 4-panel figure, one platform (4×A100), all three systems:
1. Inference arrival pattern.
2. **Cumulative FT tokens over time** (the headline trace) — the
   gap opens up during long valleys.
3. **SM Active %** time series (NSight Compute) — DeltaServe should
   stay high through valleys while LLMStation oscillates because its
   PEFT path has more launch overhead per kernel.
4. **Per-iteration kernel-launch count** for the FT path — pulled from
   our profiler vs. LLMStation profiling. Quantifies the graph-capture
   advantage directly.

Companion text in the paper attributes the gap to two design choices:
- Forward FT fused into **prefill** (vs LLMStation's decode-fusion):
  during loose valleys, decode batches are small/empty and
  decode-fusion has nothing to ride on; prefill-fusion still admits
  FT tokens on the next inference burst.
- **Full-graph capture of forward + backward**: removes per-iteration
  kernel-launch overhead that LLMStation's PyTorch-Autograd-based PEFT
  path pays at every layer.

### Exp 3 — Tight-workload SLO behavior (Q3)
**Single-platform figure, 4×A100, three systems, the tight workload from
Exp 1.**

Two-panel layout:
1. **TTFT CDF and TPOT CDF**, 500 ms / 50 ms reference lines. Goal:
   show DeltaServe at ≥ P99 compliance, and read off where LLMStation
   and split-pool fall.
2. **Cumulative FT tokens vs. time** (so the reader sees DeltaServe
   yields FT during peaks rather than violating SLO).

Reported numbers fold into Tab 1; this figure is for the visual story
of *yielding* under pressure.

### Exp 4 — Mechanism ablations (Q4)
**4×A100, Llama-3-8B, loose workload (the regime where each mechanism
is most exercised).**

Single bar plot: FT tokens/s subject to dual P99 SLO, one bar per
configuration. Three ablations, deliberately compressed:

- **Full DeltaServe** (all mechanisms on).
- **A1. All CUDA graphs off** — disable `enable_decode_cuda_graph`,
  `enable_prefill_cuda_graph`, *and* `enable_bwd_cuda_graph` together.
  One bar instead of three: the graph mechanism is the same idea
  applied to forward and backward, and reviewers don't need to see
  fwd-only vs bwd-only — only "graphs vs no graphs" matters.
- **A2. Two-regime estimator → single-regime** (patch `predict_*` to
  always use `_eager_params`). Tests whether the regime split is
  load-bearing or just bookkeeping.
- **A3. `packed_kv` → `unified` allocator.** Tests whether the GQA-
  packed pool layout matters at this workload scale.

Each ablation expected to lose ≥10% on FT tokens/s or inflate P99 TTFT
past the SLO floor (then the bar collapses to 0).

### Exp 5 — SLO estimator accuracy + scheduler overhead (Q5)
**Three panels (predictor) + one inline number (scheduler timing).**

Predictor accuracy panels:
- 4×5090 — predicted vs. measured prefill latency, with `_graph_params`
  and `_eager_params` fits overlaid.
- 4×A100 — same.
- 30-s slice from the Company X replay on 4×A100 — predicted vs.
  actual prefill duration over time.

Layout sweep: the OSDI `(#inf_reqs, #ft_samples)` set, plus
graph-bucket-aligned variants so we hit `_graph_params` for some shapes
and `_eager_params` for others. Reported metrics: relative error and
R² per regime. Compare side-by-side to LLMStation's predictor R²
(0.73 / 0.69 on held-out hardware).

Scheduler overhead: a single inline number (one sentence, not a panel).
Measure per-iteration decision time inside `_co_serving_step` over a
1-minute window of the loose workload. LLMStation reports < 1 µs cached
/ 18 µs uncached — we should be in the same ballpark and just say so.
A bar chart isn't earned here.

### Exp 6 — Cold-start: value of offline profiling
**4×A100, Llama-3-8B, Company X trace replay. Single 2-panel figure.**

Motivation: the system's two-regime predictor and `GraphEligibility`
mirror are seeded by `profiling_batch_generator.py` *before* live
serving. The OSDI version had nothing equivalent. The natural reviewer
question is "do you actually need the offline profiler, or would live
refit catch up fast enough?" — answer with data.

Two configurations replayed against the same trace:
- **DeltaServe-warm:** standard startup (offline profiling runs, then
  serving begins).
- **DeltaServe-cold:** offline profiling skipped; predictor starts
  with default params (or the cold-start fallback in `data_fit`).
  Live refit fires every 256 batches, as today.

Two panels:
1. **P99 TTFT over rolling 30-s windows for the first 5 minutes**, both
   configurations overlaid. Cold-start is expected to over-admit FT
   while the predictor is uncalibrated, blowing TTFT until the first
   refits land.
2. **Cumulative FT tokens** for the same window. Cold-start may
   actually *appear* ahead in early FT throughput — that's the whole
   problem (it's stealing from the SLO budget it can't yet measure).

Reported: time-to-SLO-compliance (first 30-s window where P99 TTFT
falls under 500 ms) for each configuration. We expect warm to start in
compliance and cold to take 1–3 minutes.

### Exp 7 — Allocator capacity headroom (NEW)
**4×5090, Llama-3-8B, synthetic load curve. Single panel, two
allocators.**

Motivation: the packed-KV allocator's value isn't just "different
layout" — it should translate into *more concurrent inference at the
same SLO*. 5090 is the right platform here because 32 GB HBM makes the
KV-pool size a binding constraint; on A100-40 GB the pool rarely runs
out, so the result is muted.

Methodology: at fixed FT load, sweep the inference RPS upward (steady
synthetic, no trace) until P99 TTFT crosses the SLO floor. Plot:
- x-axis: inference RPS.
- left y-axis: P99 TTFT (with 500 ms reference line).
- right y-axis: peak observed allocator occupancy (used pages /
  total pages, from the `_OccupancyTracker` CSV).

Two curves: `unified` vs. `packed_kv`. Read off the SLO-violation knee
for each — the gap between knees is the headroom packed_kv buys.

Companion number: at the unified knee, packed_kv is operating at
~`1/F` of its pool capacity, so the design has substantial additional
headroom even past the unified curve.

### Exp 8 — Convergence sanity (optional)
**Train an Alpaca LoRA to a fixed step count under DeltaServe and under
torchtune (split-pool baseline). Final loss + a small downstream eval
score (MMLU subset or AlpacaEval). One small table, no figure.**

Not a quality argument — a one-line refutation of "but does the
interrupted backward actually train?". LLMStation doesn't do this
either; matching is sufficient.

---

## 4. Plot/table inventory

| # | Artifact | Platform | Driver |
|---|---|---|---|
| Fig 1 | Exp 1 — head-to-head (loose / tight / Company X × 3 systems × 4 panels) | 4×A100 | extend `auto_benchmark.py` + new `compare_systems_plot.py` |
| Fig 2 | Exp 2 — loose deep dive (4 panels) | 4×A100 | `compare_systems_plot.py` w/ NSight overlays |
| Fig 3 | Exp 3 — tight SLO behavior (CDFs + FT-tokens timeline) | 4×A100 | new `tight_cdf_plot.py` |
| Fig 4 | Exp 4 — ablation bars (3 bars) | 4×A100 | new `ablation_plot.py` |
| Fig 5 | Exp 5 — predictor accuracy (3 panels) | 4×5090 + 4×A100 | extend tracker dump + plot script |
| Fig 6 | Exp 6 — cold-start (warm vs cold, 2 panels) | 4×A100 | new `cold_start_plot.py` |
| Fig 7 | Exp 7 — allocator capacity headroom | 4×5090 | new `headroom_plot.py` |
| Tab 1 | Exp 1 — headline numbers (workload × system) | 4×A100 | derive from CSVs |
| Tab 2 | Exp 8 — convergence (optional) | — | manual |

---

## 5. Execution order

1. **Get LLMStation running on 4×A100** with Llama-3-8B end-to-end.
   Single biggest schedule risk; verify on a known LLMStation workload
   before integrating. (No 5090 port — confirmed unsupported.)
2. **Get the vLLM-3GPU + PEFT-1GPU baseline running** on 4×A100.
3. **Exp 1 (loose + tight + Company X) on 4×A100** — the headline. If
   the loose-workload FT-tokens/s advantage doesn't survive the dual-
   SLO reframing, the rest of the paper has to be re-narrativized
   before continuing.
4. **Exp 2 (loose deep dive)** — uses Exp 1's CSVs + adds NSight
   captures. Only re-runs the system whose counters we want.
5. **Exp 3 (tight SLO).** Same Exp 1 CSVs; mostly just re-plotting.
6. **Exp 4 (ablations) on 4×A100, Exp 5 (predictor) on both, Exp 6
   (cold-start) on 4×A100, Exp 7 (capacity headroom) on 4×5090.**
   Parallelizable across the two boxes — A100 runs the trace replays,
   5090 runs the headroom sweep concurrently.
7. **Exp 8 (convergence)** if there's slack.

---

## 6. Risks and contingencies

- **LLMStation does not run on RTX 5090** (confirmed). Hence the
  head-to-head is A100-only, and 5090 is reserved for mechanism
  studies. We say so explicitly in the eval section rather than
  pretending we tried.
- **vLLM TP=3 unsupported.** Fallback for the split-pool baseline:
  vLLM TP=2 on 2 GPUs + 1 idle inference replica + 1 PEFT GPU; or
  3 vLLM TP=1 replicas + 1 PEFT GPU. Pick whichever matches what
  production actually deploys at the model size.
- **Loose-workload PEFT advantage doesn't survive the dual-SLO
  reframing.** Existing data is at our prior SLO settings; the dual
  P99 500 ms / 50 ms reframe is stricter. Mitigation: run Exp 1 on
  4×A100 first as a load-bearing calibration before committing the
  rest.
- **NSight Compute counter capture under MPS** is fragile. Mitigation:
  if NSight refuses under our MPS setup, fall back to NVML SM-active
  sampling (`pynvml.nvmlDeviceGetUtilizationRates`) — coarser but
  works under MPS.
- **Cold-start experiment (Exp 6)** depends on having a clean way to
  bypass `profiling_batch_generator`. Mitigation: a config flag
  (`finetune.skip_offline_profiling=true`, ~1-line plumbing) gating
  the call site in `manager.estimate_finetuning_overhead`. Verify the
  cold-start path actually exercises `data_fit`'s fallback before
  committing to the experiment.
- **Capacity-headroom experiment (Exp 7) needs the SLO knee to actually
  appear within the 5090 KV pool size.** If even at high inference RPS
  the 5090 hasn't saturated KV under `unified`, we won't see a knee.
  Mitigation: pre-check by running a single short sweep — if the knee
  isn't visible, reduce `unified_mem_manager_max_size_gb` to force the
  binding constraint and document the change in the figure caption.
- **Convergence experiment (Exp 8)** is new. If it doesn't fit the
  schedule, drop it and add one paragraph noting the omission is
  symmetric with LLMStation's.

---

## 7. What we will explicitly *not* claim

- We do not claim better PEFT *quality* than LLMStation. Exp 8 is a
  sanity check, not a quality argument.
- We do not claim multi-adapter serving — codebase serves one adapter
  at a time.
- We do not claim multi-node — all hardware is single-node.
- We do not claim Llama-70B — fits LLMStation's 8-H100 setup, not our
  4-GPU testbed at TP=4 with Alpaca-length sequences.
- We do not run an RPS-sweep or TPOT-sweep figure mirroring
  LLMStation's. Our story is workload-shape (loose/tight/real), not
  steady-rate sensitivity.
