# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**DeltaServe** — an extension of [S-LoRA](https://arxiv.org/abs/2311.03285) that adds **co-serving**: running LoRA inference and LoRA fine-tuning simultaneously on the same GPU, with SLO-aware scheduling. The core package is `slora` inside `S-LoRA/`.

## Setup

```bash
conda create -n dserve python=3.9
conda activate dserve
# Requires CUDA > 12.6
pip install torch==2.8.0
pip install uvloop==0.22.0
cd S-LoRA
pip install -e . --no-build-isolation   # compiles custom CUDA kernels (bgmv/)
pip install triton==3.4.0
```

## Running the Llama3 Experiments

**Step 1: Generate toy LoRA adapters** (requires HuggingFace access to `meta-llama/Meta-Llama-3-8B`):
```bash
cd S-LoRA/test/llama3
python adapter_train.py
cp -r adapters/llama3-toy-lora adapters/llama3-toy-lora-ft
```

**Step 2: Start NVIDIA MPS** (required by launcher):
```bash
sudo nvidia-cuda-mps-control -d
```

**Step 3: Launch server**:
```bash
cd S-LoRA/test/llama3
python launch_llama3.py                     # inference only
python launch_llama3.py --enable-finetuning # co-serving mode
```

**Step 4: Run benchmark** (auto-launches server, runs warmup, replays timeline CSV):
```bash
python auto_benchmark.py          # inference only
python auto_benchmark.py --co     # co-serving (inference + finetuning)
```
Results written to `timeline_results.csv` with columns: `idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s`.

### Config files to check
- `S-LoRA/test/llama3/launch_llama3.py` — `CONFIG["online"]` adapter/base model paths
- `S-LoRA/test/llama3/auto_benchmark.py` — `--timeline_csv`, `--lora_dir` defaults
- `S-LoRA/test/llama3/config/finetuning_config.json` — `finetuning_data_path`, `finetuning_lora_path`
- `S-LoRA/test/llama3/config/no_finetuning_config.json` — same fields

### HTTP API (when server is running on port 8000)
```bash
# Inference request
curl -X POST http://localhost:8000/generate -d '{"model_dir": "...", "lora_dir": "...", "inputs": "...", "parameters": {...}}'
# Start finetuning
curl -X POST http://localhost:8000/start_finetuning
# Stop finetuning
curl -X POST http://localhost:8000/exit_finetuning
# Check finetuning status
curl http://localhost:8000/get_finetuning_status
```

## Architecture

### Process Model
The server runs as three separate processes communicating via ZMQ and rpyc:
1. **API Server** (`slora/server/api_server.py`) — FastAPI/uvicorn HTTP frontend. Receives requests, forwards to `HttpServerManager` via ZMQ PUSH/PULL.
2. **Router** (`slora/server/router/manager.py`) — Scheduler process. Batches inference and fine-tuning requests, drives the model RPC loop via `_co_serving_step`.
3. **ModelRpcServer** (`slora/server/router/model_infer/model_rpc.py`) — GPU worker process (rpyc service). Owns the model weights, KV cache, and LoRA adapter memory.

A separate **detokenization process** (`slora/server/detokenization/`) decodes token IDs asynchronously.

### Scheduler Selection (`slora/server/router/manager.py: get_scheduler`)
The `--scheduler` flag picks the queue implementation:
- `slora` — base S-LoRA FCFS queue (`req_queue.py`)
- `slora_plus` — **DeltaServe's** `Mixed_ReqQueue` (`mixed_req_queue.py`): interleaves inference and SFT backward passes, gated by SLO thresholds. When `finetuning_type == "SFT Profile"`, uses `Profile_ReqQueue` instead.
- `vtc_fair` — VTC fairness scheduler
- `pets`, `peft`, `cluster` — other research schedulers

### Co-serving Flow (`Mixed_ReqQueue` / `finetuning_store.py`)
- `FinetuningManager` (`finetuning_store.py`) loads training data, tokenizes it, and uses a length-bucketed deque for efficient `pop_best_under(max_tokens)` selection.
- `Mixed_ReqQueue` holds `waiting_req_list` (inference) and calls `FinetuningManager` to build FT sub-batches.
- SFT forward passes run in the prefill step (FT tokens are included in the batch). Activations are saved in `LlamaSFTBackwardService` for later backward.
- SFT backward passes run **asynchronously** via `_start_back_batch_threading` — the backward runs in a background thread while inference prefill/decode continues.
- Backward is **paused** (`pause_backward`) during large decode steps (>1000 tokens, >25 decode steps) or small inference batches; **resumed** (`resume_backward`) when conditions are favorable.
- SLO gating: `check_will_starve` uses `PrefillExecutionEstimator` and `DecodeExecutionEstimator` to predict if queued requests will miss TTFT SLO before the FT tokens are admitted.

### Performance Estimation (`slora/server/router/tracker.py`)
- `BatchExecutionTracker` records actual prefill/decode durations with their token counts.
- `PrefillExecutionEstimator` and `DecodeExecutionEstimator` fit linear models to predict execution time. Re-fit every 256 batches.
- The router runs `estimate_finetuning_overhead()` on startup (runs synthetic batches) to warm up the estimators before serving real traffic.

### Memory Management (`slora/common/`)
- **Unified Memory Allocator** (`unified_mem_allocator.py`) — single paged pool shared between KV cache tensors and LoRA adapter weights. Enabled with `--enable_unified_mem_manager` (default in `launch_llama3.py`).
- Without it, separate allocators manage KV cache (`mem_manager.py`) and adapters.
- GQA models use `gqa_mem_manager.py` for KV cache (fewer KV heads than Q heads).

### Model Hierarchy (`slora/models/`)
```
llama/model.py (LlamaTpPartModel)               ← base implementation, MHA
  ├─ llama2/model.py (Llama2TpPartModel)        ← minor weight name differences
  └─ llama3/model.py (Llama3TpPartModel)        ← adds GQA, sets backward_service_class
       └─ mixtral/model.py (MixtralTpPartModel) ← MoE FFN, Sliding Window Attention
            └─ (MixtralEPTpPartModel)           ← Expert Parallelism variant

llama/SFT_service.py (LlamaSFTBackwardService)       ← base backward pass (MHA)
  └─ llama3/SFT_service.py (Llama3SFTBackwardService) ← overrides _backpop_attention for GQA
```
Each model class sets `transformer_weight_class`, `transformer_layer_infer_class`, and optionally `backward_service_class`. The `SFT_service` stores:
- `BaseModelWeights` — references to frozen base model weights
- `AdapterWeights` — lora_weights list (optimizer leaves, shape `[2, 4r, Hq, Hd]` per layer)
- `Activations` / `SharedActivations` — saved forward activations for backward

### Mixtral / MoE Support (`slora/models/mixtral/`)
Mixtral is dispatched in `model_rpc.py` when `model_type in ("mistral", "mixtral")`. Two parallelism strategies are available via `--ep`:

**TP mode (default)** — `MixtralTpPartModel` + `MixtralTransformerLayerInfer`
- Router gate weight replicated on all ranks; expert `w1`/`w3` are column-parallel (sharding the `intermediate` dim), `w2` is row-parallel. After MoE FFN, the inherited `all_reduce` sums partial results across TP ranks.
- Top-k routing: iterates over all `num_local_experts` experts per forward pass, dispatching only the tokens that selected each expert.

**EP mode (`--ep`)** — `MixtralEPTpPartModel` + `MixtralEPTransformerLayerInfer`
- Each rank owns `num_local_experts // world_size` experts at **full** intermediate size.
- Uses `dist.all_to_all_single` to route token embeddings to the rank owning each selected expert, runs local FFN, then scatters results back. Overrides `_context_ffn`/`_token_ffn` to **skip** the template's `all_reduce` (EP is already globally reduced).

**Sliding Window Attention (SWA)** — read from `sliding_window` in `config.json`. When set, uses custom triton kernels:
- `triton_kernel/context_flashattention_nopad_swa.py` — prefill with causal + SWA masking
- `triton_kernel/token_attention_softmax_and_reducev_swa.py` — decode with SWA masking
- Falls back to full attention kernels when `sliding_window` is absent.

**Limitations**: SFT co-serving backward is **not implemented** for Mixtral — `backward_service_class` is inherited from `Llama3TpPartModel` but the MoE FFN backward pass is missing. Mixtral can only be used for inference.

### LoRA Computation (`slora/models/peft/` and `slora/csrc/bgmv/`)
- Custom CUDA kernels in `csrc/bgmv/*.cu` implement `bgmv` (batched grouped matrix-vector) for heterogeneous LoRA ranks without padding.
- `LoraUnorderedBatchInfer` — inference-only batching using custom kernels.
- `LoraUnorderedBatchMixed` — mixed inference + backward batching for co-serving; handles the interaction between live inference activations and saved FT activations.
- Adapters are stored in CPU memory and swapped to GPU on demand (`--swap` flag).

### SFT Config Fields (`finetuning_config.json`)
Key fields that control co-serving behavior:
- `finetuning_type` — `"SFT"` (default) or `"SFT Profile"` (overhead measurement mode)
- `finetuning_data_path` — CSV training data
- `finetuning_lora_path` — adapter to train (must pre-exist as a copy of the inference adapter)
- `num_epochs`, `learning_rate`, `weight_decay`, `gamma`
- `max_saved_finetuning_tokens` — max pending activation tokens buffered before backward is blocked
- `max_finetuning_tokens_in_batch` — per-batch token cap for FT forward
- `optimizer_threading` — runs optimizer step on a background thread
- `ttft_slo`, `avg_tbt_slo`, `max_tbt_slo` — latency thresholds in seconds; backward passes are suppressed when inference SLOs are at risk
- `start_on_launch` — if `true`, finetuning starts immediately without needing `POST /start_finetuning`
- `ft_log_path` — path to write per-step backward loss CSV

### Capacity Planning (`slora/mprophet/`)
`ModelProphet` computes memory footprint and throughput estimates for a given model configuration without running the model. Used by `api_server.py` to validate memory budgets at startup. Key class: `ModelProphet(name, model_dir=...)`.
