# Piecewise CUDA Graph for FT Prefill (sglang-style)

Adds an alternative prefill CUDA-graph path that **supports fine-tuning
batches** by splitting each transformer layer into static-shape graph
pieces with eager hooks between them.

Branch: `feature/cuda-graph-prefill-finetuning`

## Why piecewise

The existing full-graph prefill (`PrefillCudaGraphRunner` in `cuda_graph_runner.py`)
captures the entire `_context_forward` into a single CUDA graph. That works
for inference but **cannot accommodate fine-tuning saves**:

- `mem_manager.save_embedding_output(input_embs, infer_state)`
- `mem_manager.save_activations_by_layer(i, input_embs, infer_state, FFN_INPUT_ACTIVATION)`
- `mem_manager.save_activations_by_layer(i, input_embs, infer_state, ATTENTION_INPUT_ACTIVATION)`
- `mem_manager.write_to_logit_tensor(...)`

All four use **boolean indexing** `input_embs[finetune_mask]` whose output
shape depends on how many tokens are flagged as fine-tune in the batch
(data-dependent dynamic shape, not graph-friendly), and they call back into
the page-allocator (CPU work).

The piecewise runner takes the same approach as sglang: keep the static
matmul / RMSNorm / rotary / attention work in graphs, run dynamic-shape
work eagerly in between.

## Layer split

```
input_embs (static buffer, T_bucket × hidden)
   │
   ├─ pre_attn_graph[i]:           RMSNorm + Q/K/V + LoRA + rotary + destindex_copy_kv
   ├─ eager attention:             context_attention_fwd(q, k, v, o_static, ...)
   ├─ post_attn_resid_graph[i]:    O proj + LoRA O + residual add
   ├─ EAGER save_FFN_INPUT_ACTIVATION  (FT only)
   ├─ ffn_graph[i]:                FFN_norm + gate/up/silu/down + residual add
   └─ EAGER save_ATTENTION_INPUT_ACTIVATION  (FT only)
```

Plus `pre_infer_graph` (embeddings) at the start, an eager
`save_embedding_output` after pre_infer, and an eager post-infer
(`token_forward_with_finetune_outputs` + `write_to_logit_tensor`) at the
end.

Per bucket, **3 × num_layers + 1** sub-graphs are captured. With
LLaMA-3-8B (32 layers) that's 97 sub-graphs per `(bs_bucket, T_bucket)`
combination.

## Cache key

```
(bs_bucket, T_bucket)
bs_bucket  ∈ {1, 2, 4, 8, 16, 32, 64}
T_bucket   = ceil(total_token_num, 128)
```

Padding semantics:
- Padded request rows (`b_seq_len = 0`) are no-ops in attention.
- Padded tokens at positions `[total_token_num, T_bucket)` write to
  dedicated KV scratch slots that are never read.
- `finetune_mask[total_token_num:]` is forced to zero so eager save
  hooks ignore padding tokens.

## Files

| File | Change |
|------|--------|
| `dserve/common/cuda_graph_runner.py` | New `PiecewiseCudaGraphRunner` class (~270 lines). |
| `dserve/models/peft/lora_unordered_batch_mixed.py` | New `_pre_attn_piece`, `_post_attn_residual_piece`, `_ffn_piece` helpers; new `_prefill_with_piecewise_cuda_graph` orchestration; routing in `_prefill`. |

## Usage

```bash
DSERVE_PIECEWISE_PREFILL=1 \
LD_LIBRARY_PATH=$CONDA_ENV/lib/python3.9/site-packages/torch/lib \
python -m dserve.server.api_server \
  --config eval/llama3/config/serving_config_finetuning.yaml \
  --override cuda_graph.enable_prefill_cuda_graph=true \
  --override serving.max_total_token_num=80000 \
  --override memory.unified_mem_manager_max_size_gb=12 \
  ...
```

`--enable-prefill-cuda-graph` flag controls the gate. The piecewise path is
selected when `DSERVE_PIECEWISE_PREFILL=1` is also set; otherwise the
existing full-graph path runs (inference-only batches only).

## Caveats

- **Higher KV pool requirement.** Padding to `T_bucket` means each prefill
  reserves up to `T_bucket - total_token_num` extra KV slots. For short
  prompts this is a lot of waste; bump `serving.max_total_token_num` to
  ≥ `2 × max_concurrent_prefill_tokens`.
- **First capture per bucket is expensive.** 97 sub-graphs per bucket
  ≈ 4–6 s of capture time. Subsequent requests hitting the same bucket
  replay in tens of milliseconds.
- **Padding works only for inference-style FT batches.** When the FT mask
  is non-trivial (real FT tokens present), eager save hooks run and the
  page allocator allocates real backing pages for those tokens — the same
  behavior as the non-graph path.
- **Still requires bucketing.** Even with attention semi-eager, the matmul
  and LoRA dispatches inside the graph segments have shape-static kernel
  launches. Hence the `(bs_bucket, T_bucket)` cache.

## Validation

Server starts up cleanly under `DSERVE_PIECEWISE_PREFILL=1` with FT mode
enabled. The 131-batch profile sweep that mixes inference / FT-only /
coserve batches (the dserve scheduler's startup bench) completes in ~3 min
without errors. Inference requests against the running server return the
expected outputs.

End-to-end FT data-driven training was not exercised in this branch's
validation; reviewing FT correctness end-to-end would need a longer run
that processes a real FT data file and checks the gradients/loss curve
against a baseline.
