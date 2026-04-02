#!/usr/bin/env python3
"""
Seq-length routing sweep — Qwen3-30B-A3B EP, 2 GPUs
=====================================================
For each (n_seqs, seq_len) config, run N_TRIALS prefills and record per layer:
  - recv_rank0 / recv_rank1  : raw token-expert pair counts landing on each GPU
  - fwd_comm_ms              : forward all_to_all time (send embeddings to experts)
  - bwd_comm_ms              : backward all_to_all time (return results)
  - expert_ms                : local expert computation time

Timings are collected on BOTH GPUs independently.

Output: results/seq_len_routing/layer_routing.csv

Usage:
    cd S-LoRA
    python test/qwen3/seq_len_routing.py
"""

import os, sys, csv, time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
MAX_POOL  = 20_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results", "seq_len_routing")
NCCL_PORT = 29526
N_TRIALS  = 5
N_WARMUP  = 5
K_EXPERTS = 8  # num_experts_per_tok for Qwen3-30B-A3B

# (n_seqs, seq_len) pairs to sweep
CONFIGS = [
    (1,   64),
    (1,  128),
    (1,  256),
    (1,  512),
    (1, 1024),
    (1, 2048),
    (4,   64),
    (4,  128),
    (4,  256),
    (4,  512),
    (4, 1024),
]


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
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_p, max_len_in_batch=max_len,
                      input_ids=input_ids_p, b_loc=b_loc, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len, is_prefill=True)
    torch.cuda.synchronize(); dist.barrier()
    return time.perf_counter() - t0


def worker(rank, world_size, configs, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{NCCL_PORT}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.qwen3_moe.model import Qwen3MoeEPTpPartModel
    import slora.models.qwen3_moe.layer_infer.transformer_layer_infer_ep as ep_module

    if rank == 0:
        print(f"Loading model from {MODEL_DIR}...")
    model = Qwen3MoeEPTpPartModel(tp_rank=rank, world_size=world_size,
                                   weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                   mem_adapter_size=0, dummy=False)
    vocab_size = model.config["vocab_size"]
    if rank == 0:
        print(f"Model loaded. vocab_size={vocab_size}")

    rng = np.random.default_rng(42 + rank)

    # Warmup (timing off)
    if rank == 0:
        print(f"Warming up ({N_WARMUP} batches)...")
    ep_module.timing_enabled = False
    for _ in range(N_WARMUP):
        batch = [rng.integers(0, vocab_size, size=128).tolist() for _ in range(4)]
        run_prefill(model, batch)
    if rank == 0:
        print("Warmup done.\n")

    # Enable timing for the sweep
    ep_module.timing_enabled = True

    rank_rows = []
    for (n_seqs, seq_len) in configs:
        if rank == 0:
            print(f"  n_seqs={n_seqs}  seq_len={seq_len}  ({N_TRIALS} trials)...")
        batch_template = [rng.integers(0, vocab_size, size=seq_len).tolist()
                          for _ in range(n_seqs)]
        expected = n_seqs * seq_len * K_EXPERTS / 2

        for trial in range(N_TRIALS):
            ep_module.routing_imbalance_log.clear()
            ep_module.comm_time_log.clear()
            ep_module.expert_time_log.clear()

            elapsed = run_prefill(model, batch_template)

            routing_snap = list(ep_module.routing_imbalance_log)
            comm_snap    = list(ep_module.comm_time_log)
            expert_snap  = list(ep_module.expert_time_log)

            n_layers = max(len(routing_snap), len(comm_snap), len(expert_snap))
            for layer_idx in range(n_layers):
                recv0, recv1 = (routing_snap[layer_idx] if layer_idx < len(routing_snap)
                                else (None, None))
                fwd_ms, bwd_ms = (comm_snap[layer_idx] if layer_idx < len(comm_snap)
                                  else (None, None))
                exp_ms = (expert_snap[layer_idx] if layer_idx < len(expert_snap) else None)

                total_recv = (recv0 + recv1) if recv0 is not None else None
                # Only rank 0 has routing data; both ranks have timing data
                rank_rows.append({
                    "rank":         rank,
                    "seq_len":      seq_len,
                    "n_seqs":       n_seqs,
                    "total_tokens": n_seqs * seq_len,
                    "trial":        trial,
                    "layer_idx":    layer_idx,
                    "recv_rank0":   recv0,   # tokens rank 0 receives from rank 0 (rank 0 only)
                    "recv_rank1":   recv1,   # tokens rank 0 receives from rank 1 (rank 0 only)
                    "total_recv":   total_recv,
                    "expected":     expected,
                    "ratio":        (total_recv / expected if total_recv and expected else None),
                    "fwd_comm_ms":  fwd_ms,
                    "bwd_comm_ms":  bwd_ms,
                    "expert_ms":    exp_ms,
                    "elapsed_ms":   elapsed * 1000,
                })

    mp_results[f"rank{rank}_rows"] = rank_rows

    dist.destroy_process_group()


def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, CONFIGS, mp_results), nprocs=2, join=True)

    # Merge rows from both ranks
    all_rows = list(mp_results["rank0_rows"]) + list(mp_results["rank1_rows"])
    all_rows.sort(key=lambda r: (r["n_seqs"], r["seq_len"], r["trial"],
                                  r["layer_idx"], r["rank"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "layer_routing.csv")
    fields = ["rank", "seq_len", "n_seqs", "total_tokens", "trial", "layer_idx",
              "recv_rank0", "recv_rank1", "total_recv", "expected", "ratio",
              "fwd_comm_ms", "bwd_comm_ms", "expert_ms", "elapsed_ms"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows → {out_path}")

    # Summary per (n_seqs, seq_len) — rank 0 routing + mean timings per rank
    print(f"\n{'n_seqs':>6}  {'seq_len':>7}  {'T':>5}  "
          f"{'recv_r0':>8}  {'recv_r1':>8}  {'ratio':>6}  "
          f"{'r0_fwd_ms':>9}  {'r0_exp_ms':>9}  {'r0_bwd_ms':>9}  "
          f"{'r1_fwd_ms':>9}  {'r1_exp_ms':>9}  {'r1_bwd_ms':>9}")
    for (n_seqs, seq_len) in CONFIGS:
        r0 = [r for r in all_rows if r["rank"] == 0
              and r["n_seqs"] == n_seqs and r["seq_len"] == seq_len]
        r1 = [r for r in all_rows if r["rank"] == 1
              and r["n_seqs"] == n_seqs and r["seq_len"] == seq_len]
        if not r0:
            continue

        recv0_mean = np.mean([r["recv_rank0"] for r in r0 if r["recv_rank0"] is not None])
        recv1_mean = np.mean([r["recv_rank1"] for r in r0 if r["recv_rank1"] is not None])
        ratio_mean = np.mean([r["ratio"] for r in r0 if r["ratio"] is not None])

        def layer_mean(rows, key):
            vals = [r[key] for r in rows if r[key] is not None]
            return np.mean(vals) if vals else float("nan")

        print(f"{n_seqs:>6}  {seq_len:>7}  {n_seqs*seq_len:>5}  "
              f"{recv0_mean:>8.1f}  {recv1_mean:>8.1f}  {ratio_mean:>6.3f}  "
              f"{layer_mean(r0,'fwd_comm_ms'):>9.3f}  {layer_mean(r0,'expert_ms'):>9.3f}  "
              f"{layer_mean(r0,'bwd_comm_ms'):>9.3f}  "
              f"{layer_mean(r1,'fwd_comm_ms'):>9.3f}  {layer_mean(r1,'expert_ms'):>9.3f}  "
              f"{layer_mean(r1,'bwd_comm_ms'):>9.3f}")


if __name__ == "__main__":
    main()
