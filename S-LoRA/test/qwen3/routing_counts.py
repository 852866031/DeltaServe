#!/usr/bin/env python3
"""
For each prompt, count how many tokens are routed to each GPU at every MoE layer.

On each GPU, routing_imbalance_log stores recv_counts per layer:
  recv_counts[0] = token-expert pairs this GPU receives FROM GPU 0
  recv_counts[1] = token-expert pairs this GPU receives FROM GPU 1
  total          = total work landing on this GPU this layer

Output: results/routing_counts/routing_counts.csv
  prompt_id, seq_len, layer_idx,
  gpu0_from_gpu0, gpu0_from_gpu1, total_on_gpu0,
  gpu1_from_gpu0, gpu1_from_gpu1, total_on_gpu1

Usage:
    cd S-LoRA
    python test/qwen3/routing_counts.py
"""

import os, sys, csv
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
MAX_POOL  = 20_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results", "routing_counts")
NCCL_PORT = 29527

SEQ_LENS        = [64, 128, 256, 512, 1024]
PROMPTS_PER_LEN = 4   # 20 prompts total


def run_prefill(model, token_ids_list):
    bs       = len(token_ids_list)
    seq_lens = [len(x) for x in token_ids_list]
    total_p  = sum(seq_lens)
    max_len  = max(seq_lens)

    flat        = np.concatenate([np.asarray(x, np.int64) for x in token_ids_list])
    input_ids_p = torch.from_numpy(flat).cuda()
    b_seq_len   = torch.tensor(seq_lens, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(bs, dtype=torch.long, device="cuda")
    for i in range(1, bs):
        b_start_loc[i] = b_start_loc[i-1] + seq_lens[i-1]
    b_loc = torch.zeros(bs, max_len, dtype=torch.long, device="cuda")

    model.mem_manager.free_all()
    torch.cuda.synchronize(); dist.barrier()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_p, max_len_in_batch=max_len,
                      input_ids=input_ids_p, b_loc=b_loc, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len, is_prefill=True)
    torch.cuda.synchronize(); dist.barrier()


def worker(rank, world_size, prompts, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{NCCL_PORT}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.qwen3_moe.model import Qwen3MoeEPTpPartModel
    import slora.models.qwen3_moe.layer_infer.transformer_layer_infer_ep as ep_module

    if rank == 0:
        print(f"Loading model...")
    model = Qwen3MoeEPTpPartModel(tp_rank=rank, world_size=world_size,
                                   weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                   mem_adapter_size=0, dummy=False)
    if rank == 0:
        print(f"Model loaded. Running {len(prompts)} prompts...\n")

    rank_rows = []
    for prompt_id, (seq_len, token_ids) in enumerate(prompts):
        ep_module.routing_imbalance_log.clear()

        run_prefill(model, [token_ids])

        for layer_idx, recv_counts in enumerate(ep_module.routing_imbalance_log):
            rank_rows.append({
                "prompt_id":  prompt_id,
                "seq_len":    seq_len,
                "layer_idx":  layer_idx,
                "from_gpu0":  recv_counts[0],
                "from_gpu1":  recv_counts[1],
                "total":      recv_counts[0] + recv_counts[1],
            })

        if rank == 0:
            print(f"  prompt {prompt_id:02d}  seq_len={seq_len:4d}")

    mp_results[f"rank{rank}_rows"] = rank_rows

    dist.destroy_process_group()


def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    vocab_size = tok.vocab_size

    rng = np.random.default_rng(42)
    prompts = []
    for seq_len in SEQ_LENS:
        for _ in range(PROMPTS_PER_LEN):
            ids = rng.integers(0, vocab_size, size=seq_len).tolist()
            prompts.append((seq_len, ids))

    print(f"{len(prompts)} prompts, seq_lens={SEQ_LENS}")

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, prompts, mp_results), nprocs=2, join=True)

    r0_rows = mp_results["rank0_rows"]
    r1_rows = mp_results["rank1_rows"]

    # Merge: one row per (prompt, layer) with stats from both GPUs
    assert len(r0_rows) == len(r1_rows)
    merged = []
    for r0, r1 in zip(r0_rows, r1_rows):
        assert r0["prompt_id"] == r1["prompt_id"] and r0["layer_idx"] == r1["layer_idx"]
        merged.append({
            "prompt_id":    r0["prompt_id"],
            "seq_len":      r0["seq_len"],
            "layer_idx":    r0["layer_idx"],
            "gpu0_from_gpu0": r0["from_gpu0"],
            "gpu0_from_gpu1": r0["from_gpu1"],
            "total_on_gpu0":  r0["total"],
            "gpu1_from_gpu0": r1["from_gpu0"],
            "gpu1_from_gpu1": r1["from_gpu1"],
            "total_on_gpu1":  r1["total"],
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "routing_counts.csv")
    fields = ["prompt_id", "seq_len", "layer_idx",
              "gpu0_from_gpu0", "gpu0_from_gpu1", "total_on_gpu0",
              "gpu1_from_gpu0", "gpu1_from_gpu1", "total_on_gpu1"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)
    print(f"\nSaved {len(merged)} rows → {out_path}")

    # Quick summary per seq_len
    print(f"\n{'seq_len':>7}  {'gpu0_mean':>9}  {'gpu1_mean':>9}  {'diff_mean':>9}")
    for seq_len in SEQ_LENS:
        sub = [r for r in merged if r["seq_len"] == seq_len]
        g0 = np.mean([r["total_on_gpu0"] for r in sub])
        g1 = np.mean([r["total_on_gpu1"] for r in sub])
        print(f"{seq_len:>7}  {g0:>9.1f}  {g1:>9.1f}  {g0-g1:>+9.1f}")


if __name__ == "__main__":
    main()
