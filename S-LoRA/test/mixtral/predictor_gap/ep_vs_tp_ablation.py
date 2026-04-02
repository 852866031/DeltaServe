#!/usr/bin/env python3
"""
EP vs TP Ablation — Fixed Composition Variance
===============================================
Runs the identical fixed-composition experiment under two parallelism modes:
  - EP (MixtralEPTpPartModel): uses dist.all_to_all_single to route tokens
  - TP (MixtralTpPartModel):   uses all_reduce after sharded FFN, no all_to_all

Same model weights, same 2 GPUs, same batch (identical token IDs), same T_in values.
The only variable is the collective communication primitive.

If EP std >> TP std at matched T_in, all_to_all is the variance source.
If they are equal, the variance comes from somewhere else (attention, FFN compute).

Usage:
    cd S-LoRA
    python test/mixtral/predictor_gap/ep_vs_tp_ablation.py
"""

import os, sys, csv, time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR  = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
MAX_POOL   = 15_000
OUT_DIR    = os.path.join(os.path.dirname(__file__), "results")
PORT_EP    = "29514"
PORT_TP    = "29515"

BS           = 4
T_IN_VALUES  = [256, 512, 1024, 2048]
N_WARMUP     = 20
N_TRIALS     = 100


# ---------------------------------------------------------------------------
# Forward-pass helper (same for both modes)
# ---------------------------------------------------------------------------

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


def warmup(model, vocab_size, rank, n=20):
    if rank == 0:
        print(f"    Warming up ({n} batches)...")
    rng = np.random.default_rng(0)
    for _ in range(n):
        batch = [rng.integers(0, vocab_size, size=64) for _ in range(BS)]
        run_prefill(model, batch)


def run_experiment(model, vocab_size, rank):
    results = {}
    for T_in in T_IN_VALUES:
        seq_len     = T_in // BS
        rng         = np.random.default_rng(T_in)
        fixed_batch = [rng.integers(0, vocab_size, size=seq_len).tolist() for _ in range(BS)]
        warmup(model, vocab_size, rank, N_WARMUP)

        if rank == 0:
            print(f"    T_in={T_in}: {N_TRIALS} trials...")

        times = []
        for _ in range(N_TRIALS):
            t = run_prefill(model, fixed_batch)
            if rank == 0:
                times.append(t * 1000)

        if rank == 0:
            results[T_in] = times
    return results


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def worker_ep(rank, world_size, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{PORT_EP}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    from slora.models.mixtral.model import MixtralEPTpPartModel
    if rank == 0:
        print("  [EP] Loading model...")
    model = MixtralEPTpPartModel(tp_rank=rank, world_size=world_size,
                                  weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                  mem_adapter_size=0, dummy=False)
    vocab_size = model.config["vocab_size"]
    if rank == 0:
        print(f"  [EP] Model loaded. Running experiment...")
    results = run_experiment(model, vocab_size, rank)
    if rank == 0:
        mp_results["ep"] = results
    dist.destroy_process_group()


def worker_tp(rank, world_size, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{PORT_TP}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    from slora.models.mixtral.model import MixtralTpPartModel
    if rank == 0:
        print("  [TP] Loading model...")
    model = MixtralTpPartModel(tp_rank=rank, world_size=world_size,
                                weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                mem_adapter_size=0, dummy=False)
    vocab_size = model.config["vocab_size"]
    if rank == 0:
        print(f"  [TP] Model loaded. Running experiment...")
    results = run_experiment(model, vocab_size, rank)
    if rank == 0:
        mp_results["tp"] = results
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(ep_results, tp_results):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "ep_vs_tp.csv")
    fields = ["mode", "T_in", "trial_id", "actual_ms"]
    rows = []
    for T_in, times in ep_results.items():
        for i, t in enumerate(times):
            rows.append({"mode": "EP", "T_in": T_in, "trial_id": i, "actual_ms": t})
    for T_in, times in tp_results.items():
        for i, t in enumerate(times):
            rows.append({"mode": "TP", "T_in": T_in, "trial_id": i, "actual_ms": t})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {path}")


def print_summary(ep_results, tp_results):
    print("\n" + "=" * 65)
    print("EP vs TP ABLATION — Mixtral-8x7B (2 GPUs, fixed composition)")
    print("=" * 65)
    print(f"  {'T_in':>6}  {'EP mean':>9}  {'EP std':>8}  {'TP mean':>9}  {'TP std':>8}  {'std ratio':>10}")
    for T_in in T_IN_VALUES:
        ep = np.array(ep_results[T_in])
        tp = np.array(tp_results[T_in])
        ratio = np.std(ep) / np.std(tp) if np.std(tp) > 0 else float("inf")
        print(f"  {T_in:>6}  {np.mean(ep):>9.2f}  {np.std(ep):>8.3f}  "
              f"{np.mean(tp):>9.2f}  {np.std(tp):>8.3f}  {ratio:>10.2f}×")
    print()
    print("  If EP std >> TP std: all_to_all is the variance source.")
    print("  If EP std ≈ TP std:  variance comes from FFN compute or attention.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    manager    = mp.Manager()
    mp_results = manager.dict()

    print("Phase 1: EP mode (all_to_all)")
    mp.spawn(worker_ep, args=(2, mp_results), nprocs=2, join=True)

    print("\nPhase 2: TP mode (all_reduce)")
    mp.spawn(worker_tp, args=(2, mp_results), nprocs=2, join=True)

    ep_results = dict(mp_results["ep"])
    tp_results = dict(mp_results["tp"])
    save_results(ep_results, tp_results)
    print_summary(ep_results, tp_results)


if __name__ == "__main__":
    main()
