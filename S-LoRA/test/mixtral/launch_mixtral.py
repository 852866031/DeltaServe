"""
launch_mixtral.py — Start DeltaServe with Mixtral-8x7B-v0.1 (inference only, EP mode).

Purpose: observe how PrefillExecutionEstimator behaves under real MoE traffic.
- Uses the slora_plus scheduler (default), which activates the full predictor pipeline:
    1. estimate_finetuning_overhead() at startup → initial fit on uniform ~50-tok batches
    2. check_refit() every 256 batches → online refit on accumulated real traffic
- No co-serving (start_on_launch=false, no finetuning_lora_path set)
- On exit, batch_prediction_stats.csv is written automatically by RouterManager.clean_up()

Usage:
    cd S-LoRA
    python test/mixtral/launch_mixtral.py [--port 8000]

Then drive traffic with:
    python test/mixtral/auto_benchmark.py

The predictor's fitted RMSE is printed to stdout on every 256-batch refit:
    [Router]: Error for prefill estimator: <rmse>
    [Router]: Error for decode estimator: <rmse>
"""

import argparse
import os
import sys

MODEL_DIR = "/mnt/nfs/home/ramya/slora-plus/S-LoRA"
MIXTRAL   = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
CONFIG    = os.path.join(MODEL_DIR, "test/mixtral/config/inference_config.json")

if __name__ == "__main__":
    if not os.path.isdir(MIXTRAL):
        print(f"Model not found at {MIXTRAL}")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=15000,
                        help="KV pool size. Max safe ~20000 for 2×80GB A100 with Mixtral-8x7B")
    args = parser.parse_args()

    cmd = (
        f"python -m slora.server.api_server"
        f" --model {MIXTRAL}"
        f" --tokenizer_mode auto"
        f" --tp 2"
        f" --ep"
        f" --nccl_port 28766"       # avoid clash with stress_test.py (29502) and sweeps (29500)
        f" --max_total_token_num {args.max_tokens}"
        f" --max_req_total_len 4096"
        f" --max_req_input_len 4000"
        f" --eos_id 2"
        f" --port {args.port}"
        f" --no-lora"               # base model only — no adapter needed
        f" --swap"
        f" --finetuning_config_path {CONFIG}"
        # scheduler defaults to slora_plus — predictor + refit pipeline active
        # do NOT pass --scheduler to keep the default
    )

    print("=" * 70)
    print("Mixtral-8x7B EP Inference Server")
    print("Predictor pipeline: ACTIVE (slora_plus scheduler)")
    print("Co-serving:         DISABLED (start_on_launch=false)")
    print("=" * 70)
    print(cmd)
    print()
    print("Predictor refit fires every 256 batches — watch for:")
    print("  [Router]: Error for prefill estimator: <rmse>")
    print()
    os.system(cmd)
