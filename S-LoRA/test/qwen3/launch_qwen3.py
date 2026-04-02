"""
launch_qwen3.py — Start DeltaServe with Qwen3-30B-A3B (inference only, EP mode).

Purpose: observe how PrefillExecutionEstimator behaves under real MoE traffic.
- Uses the slora_plus scheduler (default), which activates the full predictor pipeline:
    1. estimate_finetuning_overhead() at startup → initial fit on uniform ~50-tok batches
    2. check_refit() every 256 batches → online refit on accumulated real traffic
- No co-serving (start_on_launch=false, no finetuning_lora_path set)
- On exit, batch_prediction_stats.csv is written automatically by RouterManager.clean_up()

Architecture notes (vs Mixtral-8x7B):
  - hidden_size: 2048 (vs 4096)     — smaller attention
  - num_experts: 128 (vs 8)         — many more experts
  - num_experts_per_tok: 8 (vs 2)   — more experts active per token
  - moe_intermediate_size: 768      — small per-expert FFN
  - head_dim: 128 (explicit)        — non-standard (hidden/heads = 64)
  - num_kv_heads: 4                 — aggressive GQA
  - num_layers: 48 (vs 32)
  - EP sharding: 64 experts/GPU on 2×A100

Usage:
    cd S-LoRA
    python test/qwen3/launch_qwen3.py [--port 8000]

Then drive traffic with:
    python test/qwen3/auto_benchmark.py

The predictor's fitted RMSE is printed to stdout on every 256-batch refit:
    [Router]: Error for prefill estimator: <rmse>
    [Router]: Error for decode estimator: <rmse>
"""

import argparse
import os
import sys

MODEL_DIR = "/mnt/nfs/home/ramya/slora-plus/S-LoRA"
QWEN3     = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
CONFIG    = os.path.join(MODEL_DIR, "test/qwen3/config/inference_config.json")

if __name__ == "__main__":
    if not os.path.isdir(QWEN3):
        print(f"Model not found at {QWEN3}")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=20000,
                        help="KV pool size. Qwen3's small KV heads (4) allow larger pools "
                             "than Mixtral on the same hardware.")
    args = parser.parse_args()

    cmd = (
        f"python -m slora.server.api_server"
        f" --model {QWEN3}"
        f" --tokenizer_mode auto"
        f" --tp 2"
        f" --ep"
        f" --nccl_port 28767"       # avoid clash with Mixtral (28766) and sweep scripts
        f" --max_total_token_num {args.max_tokens}"
        f" --max_req_total_len 4096"
        f" --max_req_input_len 4000"
        f" --eos_id 151645"         # <|im_end|> — primary eos for Qwen3
        f" --port {args.port}"
        f" --no-lora"               # base model only — no adapter needed
        f" --swap"
        f" --finetuning_config_path {CONFIG}"
        # scheduler defaults to slora_plus — predictor + refit pipeline active
    )

    print("=" * 70)
    print("Qwen3-30B-A3B EP Inference Server")
    print("Predictor pipeline: ACTIVE (slora_plus scheduler)")
    print("Co-serving:         DISABLED (start_on_launch=false)")
    print("=" * 70)
    print(cmd)
    print()
    print("Predictor refit fires every 256 batches — watch for:")
    print("  [Router]: Error for prefill estimator: <rmse>")
    print()
    os.system(cmd)
