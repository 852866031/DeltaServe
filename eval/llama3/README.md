# eval/llama3

End-to-end evaluation harness for DeltaServe with **Meta-Llama-3-8B**. Supports two modes:

- **Inference only** — standard LoRA serving benchmark
- **Co-serving** (`--co`) — simultaneous inference + online finetuning, where the finetuning adapter is updated in the background while requests are served

---

## Directory layout

```
eval/llama3/
├── init_adapters.py          # Train and initialize LoRA adapters (run once before benchmarking)
├── launch_llama3.py          # Start the DeltaServe API server
├── auto_benchmark.py         # Full benchmark orchestrator (launch → warmup → run timeline → record)
├── simple_test.py            # Single-request smoke test
├── auto_plot.py              # Plot benchmark results from CSV outputs
│
├── config/
│   ├── finetuning_config.json        # Co-serving config (12 epochs, lr=5e-4, SLO thresholds)
│   ├── no_finetuning_config.json     # Inference-only config (4 epochs, lr=1e-3, tighter SLOs)
│   ├── emotion.txt                   # Finetuning dataset (text samples)
│   └── load_emotion.py               # Download the HuggingFace emotion dataset
│
├── adapters/
│   ├── llama3-toy-lora/      # Inference adapter (served at request time)
│   └── llama3-toy-lora-ft/   # Finetuning target adapter (updated during co-serving)
│
├── timeline_live.csv         # Request timeline: timestamp_s, prompt_length, max_new_tokens
├── timeline_results.csv      # Benchmark output: latency, TTFT, TBT per request
├── bwd_log.csv               # Finetuning backward pass log: timestamp, batch_tokens, loss
└── auto_benchmark_summary.png  # Plot output from auto_plot.py
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -U transformers datasets accelerate peft aiohttp
huggingface-cli login   # required for Meta-Llama-3 weights
```

### 2. Initialize adapters

Trains a small LoRA adapter (Q/K/V/O projections, r=16) on a toy dataset and saves it to both adapter directories. Run this once before any benchmark.

```bash
python init_adapters.py
```

Both `adapters/llama3-toy-lora/` and `adapters/llama3-toy-lora-ft/` will be (re)created from scratch.

### 3. Run a smoke test

Launches the server, waits for it to be ready, sends one request, then shuts down.

```bash
python simple_test.py [--co] [--port 9000]
```

### 4. Run the full benchmark

Launches the server, runs warmup requests, then replays `timeline_live.csv` and writes results to `timeline_results.csv`.

```bash
# Inference only
python auto_benchmark.py

# Co-serving (finetuning starts after warmup)
python auto_benchmark.py --co

# With CUDA graph
python auto_benchmark.py --co --graph
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--timeline_csv` | `timeline_live.csv` | Request timeline to replay |
| `--base_model` | `meta-llama/Meta-Llama-3-8B` | HuggingFace model ID |
| `--lora_dir` | `adapters/llama3-toy-lora` | Adapter served at inference time |
| `--port` | `9000` | Server port |
| `--co` | off | Enable co-serving (finetuning + inference) |
| `--graph` | off | Enable CUDA graph for decode steps |
| `--warmup_count` | `15` | Number of warmup requests |
| `--out_csv` | `timeline_results.csv` | Output path for results |

### 5. Plot results

```bash
python auto_plot.py
```

Reads `timeline_live.csv`, `timeline_results.csv`, and `bwd_log.csv` (all relative to the script), and writes `auto_benchmark_summary.png` with four panels:

1. Prompt tokens/s over time
2. TTFT percentile curve
3. E2E latency scatter plot
4. Cumulative finetuning tokens (co-serving mode)

---

## Server configuration

`launch_llama3.py` assembles the `dserve.server.api_server` command from two sources:

- **`CONFIG["online"]`** — model ID and adapter paths
- **`CONFIG["defaults"]`** — memory manager settings, token budget, pool size

The finetuning config passed to the server (`--finetuning_config_path`) is selected automatically:
- `--enable-finetuning` → `config/finetuning_config.json`
- otherwise → `config/no_finetuning_config.json`

Both JSON files point at `emotion.txt` as the finetuning data source and `adapters/llama3-toy-lora-ft` as the adapter to update.

---

## SLO thresholds

| Config | TTFT SLO | Avg TBT SLO | Max TBT SLO |
|---|---|---|---|
| `finetuning_config.json` (co-serving) | 350 ms | 150 ms | 350 ms |
| `no_finetuning_config.json` | 100 ms | 150 ms | 350 ms |