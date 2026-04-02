#!/usr/bin/env python3
"""
Does GPU 0's higher token load actually make it the bottleneck per layer?

Sweeps batch_size × seq_len. Per layer per config, collects:
  - token-expert pairs routed to each GPU  (routing_imbalance_log)
  - expert compute time on each GPU        (expert_time_log)

Skips configs where batch_size × seq_len > MAX_TOTAL_TOKENS to avoid OOM
on EP routing tensors (shape [T*K, H] in fp16).

Output: results/routing_bottleneck/bottleneck.csv
  batch_size, seq_len, total_tokens, trial, layer_idx,
  total_on_gpu0, total_on_gpu1, load_ratio,
  gpu0_expert_ms, gpu1_expert_ms, expert_time_ratio

Usage:
    cd S-LoRA
    python test/qwen3/routing_bottleneck.py
"""

import os, sys, csv
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_DIR        = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
MAX_POOL         = 100_000
MAX_TOTAL_TOKENS = 100_000   # skip configs above this to avoid EP tensor OOM
OUT_DIR          = os.path.join(os.path.dirname(__file__), "results", "routing_bottleneck")
NCCL_PORT        = 29528
N_TRIALS         = 3

BATCH_SIZES = [2, 4, 8, 32, 64]
SEQ_LENS    = [256, 512, 1024, 2048, 3072, 4096, 8192]


def build_configs(vocab_size, rng):
    """Return list of (batch_size, seq_len, list_of_trials) where each trial
    is a list of token_id lists."""
    configs = []
    skipped = []
    for bs in BATCH_SIZES:
        for sl in SEQ_LENS:
            total = bs * sl
            if total > MAX_TOTAL_TOKENS:
                skipped.append((bs, sl, total))
                continue
            trials = []
            for _ in range(N_TRIALS):
                batch = [rng.integers(0, vocab_size, size=sl).tolist()
                         for _ in range(bs)]
                trials.append(batch)
            configs.append((bs, sl, trials))
    if skipped:
        print(f"Skipping {len(skipped)} configs (total_tokens > {MAX_TOTAL_TOKENS}):")
        for bs, sl, tot in skipped:
            print(f"  bs={bs} sl={sl} total={tot}")
    return configs


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


def worker(rank, world_size, configs, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{NCCL_PORT}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.qwen3_moe.model import Qwen3MoeEPTpPartModel
    import slora.models.qwen3_moe.layer_infer.transformer_layer_infer_ep as ep_module

    # Patch MemoryAllocator.reset_all_pool before model init to skip SFT activation
    # buffers (finetune_activation_buffer, input_layer_output, ffn_input_buffer).
    # These are tot_size*10 entries per layer — enormous at large MAX_POOL but unused
    # during inference. The forward pass only needs key_buffer and value_buffer.
    from slora.common.mem_allocator import MemoryAllocator

    def _reset_no_sft(self):
        self.mem_state = torch.ones((self.tot_size,), dtype=torch.bool, device="cuda")
        self._mem_cum_sum = torch.empty((self.tot_size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, self.tot_size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = self.tot_size
        self.key_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                       dtype=self.dtype, device="cuda")
                           for _ in range(self.layer_num)]
        self.value_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                         dtype=self.dtype, device="cuda")
                             for _ in range(self.layer_num)]
        self.finetune_activation_buffer = None
        self.input_layer_output = None
        self.ffn_input_buffer = None
        self.request_token_info = []
        self.finetune_input_ids = []
        self.alignment_completion_masks = []
        self.finetune_logits_per_request = []
        self.reference_logits_per_request = []
        self.alignment_labels = []
        self.request_token_info_checkpoint = None
        self.saved_q = self.saved_k = self.saved_v = self.saved_o = None

    MemoryAllocator.reset_all_pool = _reset_no_sft

    if rank == 0:
        print("Loading model...")
    model = Qwen3MoeEPTpPartModel(tp_rank=rank, world_size=world_size,
                                   weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                   mem_adapter_size=0, dummy=False)

    if rank == 0:
        n_configs = len(configs)
        print(f"Model loaded. Running {n_configs} configs × {N_TRIALS} trials...\n")

    ep_module.timing_enabled = True

    rank_rows = []
    for cfg_idx, (bs, sl, trials) in enumerate(configs):
        for trial_idx, batch in enumerate(trials):
            ep_module.routing_imbalance_log.clear()
            ep_module.expert_time_log.clear()

            run_prefill(model, batch)

            routing_snap = list(ep_module.routing_imbalance_log)
            expert_snap  = list(ep_module.expert_time_log)

            for layer_idx in range(len(routing_snap)):
                recv    = routing_snap[layer_idx]
                exp_ms  = expert_snap[layer_idx] if layer_idx < len(expert_snap) else None
                rank_rows.append({
                    "batch_size":   bs,
                    "seq_len":      sl,
                    "total_tokens": bs * sl,
                    "trial":        trial_idx,
                    "layer_idx":    layer_idx,
                    "total_recv":   recv[0] + recv[1],
                    "expert_ms":    exp_ms,
                })

        if rank == 0:
            print(f"  [{cfg_idx+1}/{len(configs)}] bs={bs} sl={sl}")

    mp_results[f"rank{rank}_rows"] = rank_rows
    dist.destroy_process_group()


def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    vocab_size = tok.vocab_size

    rng = np.random.default_rng(42)
    configs = build_configs(vocab_size, rng)
    print(f"\n{len(configs)} configs will run.\n")

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, configs, mp_results), nprocs=2, join=True)

    r0 = mp_results["rank0_rows"]
    r1 = mp_results["rank1_rows"]
    assert len(r0) == len(r1)

    merged = []
    for a, b in zip(r0, r1):
        assert a["batch_size"] == b["batch_size"]
        assert a["layer_idx"]  == b["layer_idx"]
        load0 = a["total_recv"]
        load1 = b["total_recv"]
        exp0  = a["expert_ms"]
        exp1  = b["expert_ms"]
        merged.append({
            "batch_size":        a["batch_size"],
            "seq_len":           a["seq_len"],
            "total_tokens":      a["total_tokens"],
            "trial":             a["trial"],
            "layer_idx":         a["layer_idx"],
            "total_on_gpu0":     load0,
            "total_on_gpu1":     load1,
            "load_ratio":        round(load0 / load1, 4) if load1 else None,
            "gpu0_expert_ms":    round(exp0, 4) if exp0 is not None else None,
            "gpu1_expert_ms":    round(exp1, 4) if exp1 is not None else None,
            "expert_time_ratio": round(exp0 / exp1, 4) if (exp0 and exp1) else None,
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "bottleneck.csv")
    fields = ["batch_size", "seq_len", "total_tokens", "trial", "layer_idx",
              "total_on_gpu0", "total_on_gpu1", "load_ratio",
              "gpu0_expert_ms", "gpu1_expert_ms", "expert_time_ratio"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)
    print(f"\nSaved {len(merged)} rows → {out_path}")

    # Summary table: mean load_ratio and expert_time_ratio per (bs, sl)
    configs_seen = sorted(set((r["batch_size"], r["seq_len"]) for r in merged))
    print(f"\n{'bs':>4}  {'sl':>5}  {'total':>7}  {'load_ratio':>10}  "
          f"{'exp_time_ratio':>14}  {'gpu0_ms':>8}  {'gpu1_ms':>8}  {'bottleneck':>10}")
    for (bs, sl) in configs_seen:
        sub = [r for r in merged if r["batch_size"] == bs and r["seq_len"] == sl]
        lr  = np.mean([r["load_ratio"]        for r in sub if r["load_ratio"]])
        tr  = np.mean([r["expert_time_ratio"]  for r in sub if r["expert_time_ratio"]])
        e0  = np.mean([r["gpu0_expert_ms"]     for r in sub if r["gpu0_expert_ms"] is not None])
        e1  = np.mean([r["gpu1_expert_ms"]     for r in sub if r["gpu1_expert_ms"] is not None])
        bottleneck = "GPU0" if tr > 1.05 else ("GPU1" if tr < 0.95 else "balanced")
        print(f"{bs:>4}  {sl:>5}  {bs*sl:>7}  {lr:>10.3f}  {tr:>14.3f}  "
              f"{e0:>8.3f}  {e1:>8.3f}  {bottleneck:>10}")

    # Overall correlation
    lr_all = [r["load_ratio"]       for r in merged if r["load_ratio"] and r["expert_time_ratio"]]
    tr_all = [r["expert_time_ratio"] for r in merged if r["load_ratio"] and r["expert_time_ratio"]]
    corr = np.corrcoef(lr_all, tr_all)[0, 1]
    print(f"\ncorr(load_ratio, expert_time_ratio) across all configs = {corr:+.4f}")


if __name__ == "__main__":
    main()
