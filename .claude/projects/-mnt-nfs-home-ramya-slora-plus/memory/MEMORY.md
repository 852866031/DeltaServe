# DeltaServe / S-LoRA Project Memory

## Active Work Item
**MoE Predictor Failure Experiment** — proving the prefill/decode time
predictors are wrong for Mixtral. Detailed plan is in
`moe_predictor_failure_experiment.md`. Detailed implementation notes are in
`memory/moe_predictor_impl.md`.

**Status**: Plan written, implementation NOT started yet. Resume from Step 1.

## Project Quick Facts
- Package: `slora` inside `S-LoRA/`
- Python env: `conda activate dserve`
- Base model used in experiments: `meta-llama/Meta-Llama-3-8B` (Llama3)
- Mixtral weights: NOT on NFS. Must `huggingface-cli login` then download.
  Suggested small checkpoint: `mistralai/Mixtral-8x7B-v0.1` (full, ~90GB) or
  use `--dummy` flag (see impl notes) for a mock run without real weights.
- MPS required for all server launches: `sudo nvidia-cuda-mps-control -d`
- Server port convention: 9000 (Llama3), 9001 (Mixtral)

## Key File Paths
| Role | Path |
|------|------|
| Predictor (read this first) | `S-LoRA/slora/server/router/tracker.py` |
| Mixtral MoE FFN | `S-LoRA/slora/models/mixtral/layer_infer/transformer_layer_infer.py` |
| ModelRpcServer (forward) | `S-LoRA/slora/server/router/model_infer/model_rpc.py` |
| Router/manager (tracker usage) | `S-LoRA/slora/server/router/manager.py` |
| Llama3 launcher (template) | `S-LoRA/test/llama3/launch_llama3.py` |
| Predictor benchmark (template) | `S-LoRA/test/llama3/predictor_benchmark.py` |

## See Also
- `memory/moe_predictor_impl.md` — full step-by-step implementation guide
