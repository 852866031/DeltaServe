#!/usr/bin/env python3
"""
verify_qwen3.py — Load real Qwen3-30B-A3B weights and run greedy generation.

Sanity-check script: if the output is coherent (e.g. "Paris" follows
"The capital of France is"), the checkpoint loaded correctly and the full
forward/decode pipeline (including head_dim=128 override, q_norm/k_norm,
EP all_to_all routing) is working.

Usage:
    cd S-LoRA
    python test/qwen3/verify_qwen3.py
    python test/qwen3/verify_qwen3.py --prompt "The square root of 144 is" --max_new_tokens 10
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_DIR    = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
MAX_KV_POOL  = 2048
NCCL_PORT    = 29524    # avoid clash with Llama3 exp_a/exp3/exp_refit (29503)


def _worker(rank, world_size, model_dir, prompt_ids, max_new_tokens, results):
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{NCCL_PORT}",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    from slora.models.qwen3_moe.model import Qwen3MoeEPTpPartModel

    if rank == 0:
        print(f"[rank 0] Loading {model_dir} — EP mode, world_size={world_size}")

    model = Qwen3MoeEPTpPartModel(
        tp_rank=rank,
        world_size=world_size,
        weight_dir=model_dir,
        max_total_token_num=MAX_KV_POOL,
        mem_adapter_size=0,
        dummy=False,
    )

    if rank == 0:
        w = model.trans_layers_weight[0]
        qw = getattr(w, "q_weight_", None)
        if qw is not None:
            print(f"[rank 0] Layer-0 q_weight  shape={list(qw.shape)}  "
                  f"norm={qw.float().norm().item():.4f}  dtype={qw.dtype}")
        print(f"[rank 0] Model loaded — starting generation\n")

    prompt_len  = len(prompt_ids)
    bs          = 1
    flat_ids    = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
    b_seq_len   = torch.tensor([prompt_len], dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(1, dtype=torch.long, device="cuda")
    b_loc       = torch.zeros(bs, prompt_len, dtype=torch.long, device="cuda")

    # Prefill
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
    next_token = int(logits[0].argmax().item())
    generated  = [next_token]

    # Decode loop
    for step in range(max_new_tokens - 1):
        b_loc     = torch.cat([b_loc, torch.zeros(bs, 1, dtype=torch.long, device="cuda")], dim=1)
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


def run_prompt(prompt, max_new_tokens, tok):
    enc        = tok(prompt, add_special_tokens=False)
    prompt_ids = enc["input_ids"]
    print(f"Prompt    : {prompt!r}")
    print(f"Token IDs : {prompt_ids}  (len={len(prompt_ids)})")

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        _worker,
        args=(2, MODEL_DIR, prompt_ids, max_new_tokens, results),
        nprocs=2,
        join=True,
    )

    gen_ids  = results["generated_ids"]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    print(f"Generated : {gen_text!r}")
    print(f"Full      : {(prompt + gen_text)!r}")
    print(f"Token IDs : {gen_ids}")
    return gen_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=40)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {MODEL_DIR} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print()

    prompts = [
        "The capital of France is",
        "The square root of 144 is",
    ]

    for prompt in prompts:
        print("=" * 64)
        gen = run_prompt(prompt, args.max_new_tokens, tok)
        print("=" * 64)
        print()

    print("If completions are coherent (Paris / 12), the model is working correctly.")


if __name__ == "__main__":
    main()
