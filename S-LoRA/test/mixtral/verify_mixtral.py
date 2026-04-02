#!/usr/bin/env python3
"""
verify_mixtral.py — Load real Mixtral weights and run greedy generation.

This is a sanity-check script: if the output is coherent English (e.g.
"The capital of France is Paris"), the real checkpoint was loaded and the
full forward/decode pipeline is working correctly.

Requires a Mixtral 8x7B checkpoint — either a HuggingFace repo id (needs
internet + HF access) or a local directory that contains config.json and the
model shards.  Two GPUs are needed for Mixtral 8x7B (≈94 GB in fp16); with
--world_size 1 you can test on smaller toy checkpoints.

Usage — 2-GPU TP (default, suitable for 8x7B on 2×A100-80GB):
    cd S-LoRA
    python test/mixtral/verify_mixtral.py \\
        --model_dir mistralai/Mixtral-8x7B-v0.1 \\
        --world_size 2 \\
        --prompt "The capital of France is" \\
        --max_new_tokens 40

Usage — 2-GPU EP (matches the predictor experiments):
    python test/mixtral/verify_mixtral.py \\
        --model_dir mistralai/Mixtral-8x7B-v0.1 \\
        --world_size 2 --ep \\
        --prompt "The capital of France is"

Usage — single GPU (toy / quantized checkpoint):
    python test/mixtral/verify_mixtral.py \\
        --model_dir /path/to/small-mixtral \\
        --world_size 1 \\
        --prompt "Once upon a time"
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# KV cache pool: enough for prompt + generated tokens on a single short run.
# Increase if you use longer prompts or more new tokens.
MAX_KV_POOL = 2048


# ---------------------------------------------------------------------------
# Worker — runs on each GPU rank
# ---------------------------------------------------------------------------

def _worker(rank, world_size, use_ep, model_dir, prompt_ids, max_new_tokens, results):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29502",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    # ── load model ──────────────────────────────────────────────────────────
    if use_ep:
        from slora.models.mixtral.model import MixtralEPTpPartModel as Model
    else:
        from slora.models.mixtral.model import MixtralTpPartModel as Model

    if rank == 0:
        mode_str = "EP" if use_ep else "TP"
        print(f"[rank 0] Loading {model_dir} — {mode_str} mode, world_size={world_size}")

    model = Model(
        tp_rank=rank,
        world_size=world_size,
        weight_dir=model_dir,
        max_total_token_num=MAX_KV_POOL,
        mem_adapter_size=0,
        dummy=False,         # <── real weights
    )

    if rank == 0:
        # Print a quick weight-sanity check: the norm of the first layer's
        # q_weight should be a real non-zero value, not NaN.
        w = model.trans_layers_weight[0]
        qw = getattr(w, "q_weight_", None)
        if qw is not None:
            print(f"[rank 0] Layer-0 q_weight  shape={list(qw.shape)}  "
                  f"norm={qw.float().norm().item():.4f}  "
                  f"dtype={qw.dtype}")
        print(f"[rank 0] Model loaded — starting generation\n")

    # ── build prefill tensors ────────────────────────────────────────────────
    prompt_len  = len(prompt_ids)
    bs          = 1

    flat_ids    = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
    b_seq_len   = torch.tensor([prompt_len], dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(1, dtype=torch.long, device="cuda")
    # b_loc shape [bs, max_seq_len]; init_bloc (called inside _prefill) fills
    # it with KV-cache slot indices.
    b_loc       = torch.zeros(bs, prompt_len, dtype=torch.long, device="cuda")

    # ── prefill ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model.forward(
            batch_size=bs,
            total_token_num=prompt_len,
            max_len_in_batch=prompt_len,
            input_ids=flat_ids,
            b_loc=b_loc,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            is_prefill=True,
        )
    # logits: [bs, vocab_size] — pick the first new token greedily
    next_token = int(logits[0].argmax().item())
    generated  = [next_token]

    # ── decode loop ──────────────────────────────────────────────────────────
    for step in range(max_new_tokens - 1):
        # Grow b_loc by one column; _decode writes the new KV slot index there.
        b_loc     = torch.cat(
            [b_loc, torch.zeros(bs, 1, dtype=torch.long, device="cuda")], dim=1
        )
        b_seq_len = b_seq_len + 1

        cur_total  = int(b_seq_len.sum().item())
        cur_maxlen = int(b_seq_len.max().item())

        with torch.no_grad():
            logits = model.forward(
                batch_size=bs,
                total_token_num=cur_total,
                max_len_in_batch=cur_maxlen,
                input_ids=torch.tensor([next_token], dtype=torch.long, device="cuda"),
                b_loc=b_loc,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                is_prefill=False,
            )

        next_token = int(logits[0].argmax().item())
        generated.append(next_token)

    if rank == 0:
        results["generated_ids"] = generated

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Greedy generation with real Mixtral weights")
    parser.add_argument("--model_dir",      default="mistralai/Mixtral-8x7B-v0.1",
                        help="HuggingFace repo id or local path to a Mixtral checkpoint")
    parser.add_argument("--world_size",     type=int, default=2,
                        help="Number of GPUs (TP degree).  2 is required for 8x7B in fp16.")
    parser.add_argument("--ep",             action="store_true",
                        help="Use Expert Parallelism instead of TP for the FFN "
                             "(matches the predictor-experiment setup)")
    parser.add_argument("--prompt",         default="The capital of France is",
                        help="Prompt string to complete")
    parser.add_argument("--max_new_tokens", type=int, default=40,
                        help="Number of tokens to generate")
    args = parser.parse_args()

    # ── tokenise ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model_dir} ...")
    tok       = AutoTokenizer.from_pretrained(args.model_dir)
    enc       = tok(args.prompt, add_special_tokens=True)
    prompt_ids = enc["input_ids"]

    print(f"Prompt    : {args.prompt!r}")
    print(f"Token IDs : {prompt_ids}  (len={len(prompt_ids)})")
    print()

    if args.world_size < 2 and args.ep:
        parser.error("--ep requires --world_size >= 2")

    if args.ep and (8 % args.world_size != 0):
        print("WARNING: Mixtral 8x7B has 8 experts; world_size should divide 8 for EP.")

    # ── spawn workers ─────────────────────────────────────────────────────────
    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        _worker,
        args=(args.world_size, args.ep, args.model_dir,
              prompt_ids, args.max_new_tokens, results),
        nprocs=args.world_size,
        join=True,
    )

    # ── decode & display ──────────────────────────────────────────────────────
    gen_ids  = results["generated_ids"]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    print("=" * 64)
    print(f"Prompt         : {args.prompt!r}")
    print(f"Generated text : {gen_text!r}")
    print(f"Full output    : {(args.prompt + gen_text)!r}")
    print()
    print(f"Generated token ids : {gen_ids}")
    print("=" * 64)

    # ── verification hint ─────────────────────────────────────────────────────
    # A well-loaded Mixtral-8x7B should complete "The capital of France is"
    # with something starting with "Paris".  If you see random tokens or
    # repeated punctuation, the weights did not load correctly.
    if "paris" in gen_text.lower():
        print("✓  Output looks correct — 'Paris' appears in the completion.")
    else:
        print("?  'Paris' not found in completion; verify the checkpoint is correct.")


if __name__ == "__main__":
    main()
