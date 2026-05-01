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
| FT throughput unit | tokens/s | Both: **samples/s** (LLMStation's metric, makes head-to-head legible) + **tokens/s** (our internal metric — backward batching is token-bounded). |
| Primary hardware | 1× RTX 5090 + 4× A100 | **4× RTX 5090 + 4× A100**. Multi-GPU is the contested setting. |
| Workload framing | burst-light / burst-dense / Company X | **loose / tight / Company X** (`timeline_loose.csv`, `timeline_tight.csv` — already wired into `auto_benchmark.py --loose / --tight` and the current code already shows we beat LLMStation on FT throughput in the loose case). |
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
   forward/backward CUDA-graph capture compound into a measurable PEFT
   advantage over LLMStation. Establish this and explain it with
   utilization counters and FT-throughput-during-valley measurements.
3. **Q3 — Tight-workload SLO behavior.** Under high-pressure tight
   traffic, does DeltaServe still hold the dual SLO (yielding FT as
   needed)? Does LLMStation? Does the split-pool baseline?
4. **Q4 — Mechanism contributions.** How much does each DeltaServe
   mechanism contribute to the headline result: forward graph capture
   (decode + prefill), backward graph capture (FFN + padded attn), the
   two-regime SLO estimator, and the packed-KV allocator?
5. **Q5 — Predictor accuracy.** Is the two-regime SLO estimator accurate
   enough across batch composition and hardware to drive admission
   correctly?

---

## 2. Common setup

### 2.1 Hardware
- **4 × NVIDIA RTX 5090 (32 GB)** — primary consumer-GPU result.
- **4 × NVIDIA A100 (40 GB)** — primary datacenter-GPU result.
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
- **PEFT throughput:** samples/s and tokens/s.
- **Latency:** P99 TTFT, P99 TPOT, P50 E2E.
- **Utilization:** raw NSight Compute counters (SM Active %, Compute
  Warps in Flight, DRAM R/W BW). nvidia-smi util dropped as primary
  signal.
- **Allocator occupancy:** used-pages over time, 1 Hz CSV from the
  existing `_OccupancyTracker`.
- **System overheads:** scheduler decision time per iteration;
  `_maybe_pause` cost; predictor inference time.

---

## 3. Experiments

### Exp 1 — Headline head-to-head: loose, tight, Company X (Q1)
**The main result figure. One row per workload, three systems per panel,
two hardware platforms → two figures with the same shape.**

- 4×5090 figure and 4×A100 figure.
- Three rows: loose / tight / Company X.
- Four columns per row, mirroring the existing `plot_loose_tight.py`
  layout but with three system traces overlaid:
  1. Inference arrival pattern (shared across systems; one trace).
  2. P99 TTFT over 30-s windows, dashed at 500 ms.
  3. Per-request E2E latency scatter.
  4. Cumulative PEFT samples over time (with samples/s annotation).
- **Driver:** extend `auto_benchmark.py` to emit a per-system suffix and
  add an `--llmstation` / `--vllm-split` mode that launches the
  appropriate baseline harness. Then a new `compare_systems_plot.py`
  reads the per-system CSVs and produces the overlay.
- **Headline numbers** reported as **Tab 1** (one row per
  workload-platform-system, six metrics: P99 TTFT, P99 TPOT, P50 E2E,
  total PEFT samples, PEFT samples/s, avg SM Active %).

**Expected story:**
- Loose: DeltaServe ≫ LLMStation on PEFT samples/s (this is the result
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
2. **Cumulative PEFT samples over time** (the headline trace) — the
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
  FT samples on the next inference burst.
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
2. **PEFT samples vs. time** (so the reader sees DeltaServe yields FT
   during peaks rather than violating SLO).

Reported numbers fold into Tab 1; this figure is for the visual story
of *yielding* under pressure.

### Exp 4 — Mechanism ablations (Q4)
**4×A100, Llama-3-8B, loose workload (the regime where each mechanism
is most exercised).**

Single bar plot: PEFT samples/s subject to dual P99 SLO, one bar per
configuration:
- **Full DeltaServe** (all mechanisms on).
- **A1.** Forward graph capture off (`enable_decode_cuda_graph=false`,
  `enable_prefill_cuda_graph=false`).
- **A2.** Backward graph capture off (`enable_bwd_cuda_graph=false`).
- **A3.** Padded-attention graph off (`use_graphed_bwd_attention=false`
  → forces monolithic per-sample attention bwd).
- **A4.** Two-regime estimator → single-regime (patch
  `predict_*` to always use `_eager_params`).
- **A5.** `packed_kv` → `unified` allocator.

Each ablation expected to lose 5–25% on PEFT samples/s, or inflate P99
TTFT past the SLO floor (then the bar collapses to 0).

### Exp 5 — SLO estimator accuracy (Q5)
**Direct successor to OSDI Fig 5; updated for the two-regime
estimator.**

Three panels:
- 5090 — predicted vs. measured prefill latency, with `_graph_params`
  and `_eager_params` fits overlaid.
- A100 — same.
- 30-s slice from the Company X replay on 4×A100 — predicted vs.
  actual prefill duration over time.

Layout sweep: the OSDI `(#inf_reqs, #ft_samples)` set, plus
graph-bucket-aligned variants so we hit `_graph_params` for some shapes
and `_eager_params` for others. Reported metrics: relative error and
R² per regime. Compare side-by-side to LLMStation's predictor R²
(0.73 / 0.69 on held-out hardware).

### Exp 6 — Microbenchmarks: scheduler + pause overhead (Q4)
**Match LLMStation §6.4. One figure, two panels.**

- Panel 1: per-iteration scheduler decision time at the manager —
  measured inside `_co_serving_step` with cached vs. uncached predictor
  paths. LLMStation reports < 1 µs cached / 18 µs uncached.
- Panel 2: backward-pass overhead from `_maybe_pause()` — vary the
  number of pause points (0, 8, 16, 32, all-layers) and report % added
  latency on a fixed FT batch. LLMStation reports < 0.5% single-GPU /
  up to 18% multi-GPU; ours should be lower (stream-sync, not
  coroutine context-switch).

### Exp 7 — Convergence sanity (optional)
**Train an Alpaca LoRA to a fixed step count under DeltaServe and under
torchtune (split-pool baseline). Final loss + a small downstream eval
score (MMLU subset or AlpacaEval). One small table, no figure.**

Not a quality argument — a one-line refutation of "but does the
interrupted backward actually train?". LLMStation doesn't do this
either; matching is sufficient.

---

## 4. Plot/table inventory

| # | Artifact | Driver |
|---|---|---|
| Fig 1 | Exp 1 — head-to-head, 4×5090 (loose / tight / Company X × 3 systems × 4 panels) | extend `auto_benchmark.py` + new `compare_systems_plot.py` |
| Fig 2 | Exp 1 — head-to-head, 4×A100 | same |
| Fig 3 | Exp 2 — loose deep dive (4 panels, 4×A100) | `compare_systems_plot.py` w/ NSight overlays |
| Fig 4 | Exp 3 — tight SLO behavior (CDFs + PEFT timeline, 4×A100) | new `tight_cdf_plot.py` |
| Fig 5 | Exp 4 — ablation bars (loose, 4×A100) | new `ablation_plot.py` |
| Fig 6 | Exp 5 — predictor accuracy (3 panels) | extend tracker dump + plot script |
| Fig 7 | Exp 6 — scheduler / pause microbenchmarks (2 panels) | new `microbench.py` |
| Tab 1 | Exp 1 — headline numbers (workload × platform × system) | derive from CSVs |
| Tab 2 | Exp 7 — convergence (optional) | manual |

---

## 5. Execution order

1. **Get LLMStation running** on both 4×5090 and 4×A100 with Llama-3-8B
   end-to-end. Single biggest schedule risk; verify on a known
   LLMStation workload before integrating.
2. **Get the vLLM-3GPU + PEFT-1GPU baseline running** on both platforms.
3. **Exp 1 (loose + tight + Company X) on 4×A100 first** — this is the
   headline figure; if the loose-workload PEFT advantage doesn't hold
   under the new dual-SLO definition, the rest of the paper has to be
   re-narrativized before continuing.
4. **Exp 1 on 4×5090.** Long single-run pass; can run overnight.
5. **Exp 2 (loose deep dive)** — uses Exp 1's CSVs + adds NSight
   captures. Only re-runs the system whose counters we want; the
   arrival/PEFT timeline data is already on disk.
6. **Exp 3 (tight SLO).** Same Exp 1 CSVs; mostly just re-plotting.
7. **Exp 4–6 (ablations, predictor, microbench).** Parallelizable,
   ~1 day of GPU time each.
8. **Exp 7 (convergence)** if there's slack.

---

## 6. Risks and contingencies

- **LLMStation may not run on RTX 5090** (artifacts likely not tested
  there). Fallback: 4×A100 head-to-head as primary; 4×5090 figure
  becomes DeltaServe vs. split-pool only, with a sentence noting the
  LLMStation port is left to future work.
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
- **Convergence experiment (Exp 7)** is new. If it doesn't fit the
  schedule, drop it and add one paragraph noting the omission is
  symmetric with LLMStation's.

---

## 7. What we will explicitly *not* claim

- We do not claim better PEFT *quality* than LLMStation. Exp 7 is a
  sanity check, not a quality argument.
- We do not claim multi-adapter serving — codebase serves one adapter
  at a time.
- We do not claim multi-node — all hardware is single-node.
- We do not claim Llama-70B — fits LLMStation's 8-H100 setup, not our
  4-GPU testbed at TP=4 with Alpaca-length sequences.
- We do not run an RPS-sweep or TPOT-sweep figure mirroring
  LLMStation's. Our story is workload-shape (loose/tight/real), not
  steady-rate sensitivity.
