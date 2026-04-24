"""
CUDA Graph runner for decode (token generation) steps.

Key design decisions:
  - Cache key is (batch_size_bucket, max_len_bucket).
  - All batch-dimensional tensors are padded to bs_bucket during capture.
  - b_loc is RIGHT-ALIGNED to ml_bucket before capture/replay.
  - Force non-contiguous decode path for stable buffer addresses.
  - Pre-allocate att_m_buffers (one per layer) with max_total_tokens size.
"""

import torch
from typing import Callable, Dict, Tuple


def _pad_to(tensor: torch.Tensor, target_dim0: int) -> torch.Tensor:
    """Pad tensor's first dimension to target_dim0 with zeros."""
    if tensor.shape[0] >= target_dim0:
        return tensor[:target_dim0].clone()
    pad_shape = list(tensor.shape)
    pad_shape[0] = target_dim0
    out = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    out[:tensor.shape[0]] = tensor
    return out


class CudaGraphRunner:
    """Caches and replays CUDA graphs keyed by (batch_size_bucket, max_len_bucket)."""

    BATCH_BUCKETS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    MAX_LEN_BUCKET_SIZE = 128

    # Prefill bucket size: total_token_num is padded to nearest multiple of this.
    PREFILL_TOKEN_BUCKET_SIZE = 128

    def __init__(self, max_total_tokens: int = 25000, num_layers: int = 32,
                 tp_q_head_num: int = 32, tp_k_head_num: int = 8,
                 head_dim: int = 128,
                 embed_dim: int = 4096,
                 b_loc_width: int = None):
        self.max_total_tokens = max_total_tokens
        self.num_layers = num_layers
        self.tp_q_head_num = tp_q_head_num
        self.tp_k_head_num = tp_k_head_num
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.b_loc_width = b_loc_width  # second dim of b_loc_key/value tensor

        # Decode cache: (bs, ml_bucket) -> (graph, bufs, output, ml_bucket)
        self._cache: Dict[Tuple[int, int], Tuple[torch.cuda.CUDAGraph, dict, torch.Tensor, int]] = {}
        # Prefill cache: (bs, T_bucket) -> (graph, bufs, output, T_bucket)
        self._prefill_cache: Dict[Tuple[int, int], Tuple[torch.cuda.CUDAGraph, dict, torch.Tensor, int]] = {}
        self._warmup_iters = 3

    @staticmethod
    def get_batch_bucket(batch_size: int) -> int:
        for b in CudaGraphRunner.BATCH_BUCKETS:
            if batch_size <= b:
                return b
        return batch_size

    @staticmethod
    def get_max_len_bucket(max_len: int) -> int:
        bs = CudaGraphRunner.MAX_LEN_BUCKET_SIZE
        return ((max_len + bs - 1) // bs) * bs

    def get_cache_key(self, batch_size: int, max_len_in_batch: int) -> Tuple[int, int]:
        # Use exact batch_size (not bucketed) because LoRA dispatch requires
        # exact adapter mappings that can't be padded
        return (batch_size, self.get_max_len_bucket(max_len_in_batch))

    def has_graph(self, batch_size: int, max_len_in_batch: int) -> bool:
        return self.get_cache_key(batch_size, max_len_in_batch) in self._cache

    # ─── Prefill bucketing helpers ──────────────────────────────────────
    @staticmethod
    def get_prefill_token_bucket(total_token_num: int) -> int:
        bs = CudaGraphRunner.PREFILL_TOKEN_BUCKET_SIZE
        return ((total_token_num + bs - 1) // bs) * bs

    def get_prefill_cache_key(self, batch_size: int, total_token_num: int) -> Tuple[int, int]:
        return (batch_size, self.get_prefill_token_bucket(total_token_num))

    def has_prefill_graph(self, batch_size: int, total_token_num: int) -> bool:
        return self.get_prefill_cache_key(batch_size, total_token_num) in self._prefill_cache

    def _realign_b_loc(self, static_buf, real_buf, batch_size, actual_max_len, bucket_max_len):
        shift = bucket_max_len - actual_max_len
        static_buf[:batch_size].zero_()
        if actual_max_len > 0 and shift >= 0:
            src_cols = min(actual_max_len, real_buf.shape[1])
            static_buf[:batch_size, shift:shift + src_cols].copy_(
                real_buf[:batch_size, :src_cols])

    def _create_att_m_buffers(self, dtype):
        buffers = {}
        for layer_id in range(self.num_layers):
            buffers[layer_id] = torch.empty(
                (self.tp_q_head_num, self.max_total_tokens),
                dtype=dtype, device="cuda")
        return buffers

    def capture(self, batch_size, max_len_in_batch, token_forward_fn,
                input_ids, infer_state, forward_kwargs) -> torch.Tensor:
        infer_state.mem_manager.page_table_lock.acquire()
        ml_bucket = self.get_max_len_bucket(max_len_in_batch)
        cache_key = (batch_size, ml_bucket)

        # Pre-allocate att_m_buffers (must be float16, NOT input_ids.dtype which is int)
        att_m_buffers = self._create_att_m_buffers(torch.float16)
        infer_state._att_m_buffers = att_m_buffers

        # Clone all tensors (exact batch_size, no padding)
        static_input_ids = input_ids.clone()
        static_b_seq_len = infer_state.b_seq_len.clone()
        static_b_start_loc = infer_state.b_start_loc.clone()
        static_position_cos = infer_state.position_cos.clone()
        static_position_sin = infer_state.position_sin.clone()

        # b_loc (unified mem path) - re-align to ml_bucket
        static_b_loc_key = None
        static_b_loc_value = None
        if hasattr(infer_state, 'b_loc_key') and infer_state.b_loc_key is not None:
            static_b_loc_key = torch.zeros_like(infer_state.b_loc_key)
            self._realign_b_loc(static_b_loc_key, infer_state.b_loc_key,
                                batch_size, max_len_in_batch, ml_bucket)
            static_b_loc_value = torch.zeros_like(infer_state.b_loc_value)
            self._realign_b_loc(static_b_loc_value, infer_state.b_loc_value,
                                batch_size, max_len_in_batch, ml_bucket)

        # b_loc (regular mem path)
        static_b_loc = None
        if hasattr(infer_state, 'b_loc') and infer_state.b_loc is not None:
            static_b_loc = torch.zeros_like(infer_state.b_loc)
            self._realign_b_loc(static_b_loc, infer_state.b_loc,
                                batch_size, max_len_in_batch, ml_bucket)

        # Decode mem indices
        static_decode_mem_index_key = None
        static_decode_mem_index_value = None
        static_decode_mem_index = None
        if hasattr(infer_state, 'decode_mem_index_key') and infer_state.decode_mem_index_key is not None:
            static_decode_mem_index_key = infer_state.decode_mem_index_key.clone()
            static_decode_mem_index_value = infer_state.decode_mem_index_value.clone()
        if hasattr(infer_state, 'decode_mem_index') and infer_state.decode_mem_index is not None:
            static_decode_mem_index = infer_state.decode_mem_index.clone()

        # Decode KV scratch buffers
        static_decode_key_buffer = None
        static_decode_value_buffer = None
        if hasattr(infer_state, 'decode_key_buffer') and infer_state.decode_key_buffer is not None:
            static_decode_key_buffer = infer_state.decode_key_buffer.clone()
            static_decode_value_buffer = infer_state.decode_value_buffer.clone()

        # Wire static buffers into infer_state
        infer_state.b_seq_len = static_b_seq_len
        infer_state.b_start_loc = static_b_start_loc
        infer_state.position_cos = static_position_cos
        infer_state.position_sin = static_position_sin
        infer_state.max_len_in_batch = ml_bucket

        if static_b_loc_key is not None:
            infer_state.b_loc_key = static_b_loc_key
            infer_state.b_loc_value = static_b_loc_value
        if static_b_loc is not None:
            infer_state.b_loc = static_b_loc
        if static_decode_mem_index_key is not None:
            infer_state.decode_mem_index_key = static_decode_mem_index_key
            infer_state.decode_mem_index_value = static_decode_mem_index_value
        if static_decode_mem_index is not None:
            infer_state.decode_mem_index = static_decode_mem_index
        if static_decode_key_buffer is not None:
            infer_state.decode_key_buffer = static_decode_key_buffer
            infer_state.decode_value_buffer = static_decode_value_buffer

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._warmup_iters):
                _ = token_forward_fn(static_input_ids, infer_state, **forward_kwargs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = token_forward_fn(static_input_ids, infer_state, **forward_kwargs)

        input_buffers = {
            'input_ids': static_input_ids,
            'b_seq_len': static_b_seq_len,
            'b_start_loc': static_b_start_loc,
            'position_cos': static_position_cos,
            'position_sin': static_position_sin,
            'b_loc_key': static_b_loc_key,
            'b_loc_value': static_b_loc_value,
            'b_loc': static_b_loc,
            'decode_mem_index_key': static_decode_mem_index_key,
            'decode_mem_index_value': static_decode_mem_index_value,
            'decode_mem_index': static_decode_mem_index,
            'decode_key_buffer': static_decode_key_buffer,
            'decode_value_buffer': static_decode_value_buffer,
            'att_m_buffers': att_m_buffers,
        }

        self._cache[cache_key] = (graph, input_buffers, static_output, ml_bucket)
        infer_state.mem_manager.page_table_lock.release()
        return static_output[:batch_size]

    def replay(self, batch_size, max_len_in_batch, input_ids, infer_state) -> torch.Tensor:
        cache_key = self.get_cache_key(batch_size, max_len_in_batch)
        graph, bufs, static_output, ml_bucket = self._cache[cache_key]

        # Copy current values into static buffers (exact size, no padding)
        bufs['input_ids'].copy_(input_ids)
        bufs['b_seq_len'].copy_(infer_state.b_seq_len)
        bufs['b_start_loc'].copy_(infer_state.b_start_loc)
        bufs['position_cos'].copy_(infer_state.position_cos)
        bufs['position_sin'].copy_(infer_state.position_sin)

        # Re-align b_loc to ml_bucket
        if bufs['b_loc_key'] is not None and hasattr(infer_state, 'b_loc_key') and infer_state.b_loc_key is not None:
            self._realign_b_loc(bufs['b_loc_key'], infer_state.b_loc_key,
                                batch_size, max_len_in_batch, ml_bucket)
            self._realign_b_loc(bufs['b_loc_value'], infer_state.b_loc_value,
                                batch_size, max_len_in_batch, ml_bucket)
        if bufs['b_loc'] is not None and hasattr(infer_state, 'b_loc') and infer_state.b_loc is not None:
            self._realign_b_loc(bufs['b_loc'], infer_state.b_loc,
                                batch_size, max_len_in_batch, ml_bucket)

        # Update decode mem indices
        if bufs['decode_mem_index_key'] is not None and hasattr(infer_state, 'decode_mem_index_key') and infer_state.decode_mem_index_key is not None:
            bufs['decode_mem_index_key'].copy_(infer_state.decode_mem_index_key)
            bufs['decode_mem_index_value'].copy_(infer_state.decode_mem_index_value)
        if bufs['decode_mem_index'] is not None and hasattr(infer_state, 'decode_mem_index') and infer_state.decode_mem_index is not None:
            bufs['decode_mem_index'].copy_(infer_state.decode_mem_index)

        # Point infer_state to static buffers
        infer_state.b_seq_len = bufs['b_seq_len']
        infer_state.b_start_loc = bufs['b_start_loc']
        infer_state.position_cos = bufs['position_cos']
        infer_state.position_sin = bufs['position_sin']
        infer_state.max_len_in_batch = ml_bucket
        infer_state._att_m_buffers = bufs['att_m_buffers']

        if bufs['b_loc_key'] is not None:
            infer_state.b_loc_key = bufs['b_loc_key']
            infer_state.b_loc_value = bufs['b_loc_value']
        if bufs['b_loc'] is not None:
            infer_state.b_loc = bufs['b_loc']
        if bufs['decode_mem_index_key'] is not None:
            infer_state.decode_mem_index_key = bufs['decode_mem_index_key']
            infer_state.decode_mem_index_value = bufs['decode_mem_index_value']
        if bufs['decode_mem_index'] is not None:
            infer_state.decode_mem_index = bufs['decode_mem_index']
        if bufs['decode_key_buffer'] is not None:
            infer_state.decode_key_buffer = bufs['decode_key_buffer']
            infer_state.decode_value_buffer = bufs['decode_value_buffer']

        graph.replay()
        return static_output

    # ─── Prefill capture / replay ───────────────────────────────────────
    #
    # MVP scope: batch_size == 1, inference-only (no finetune), non-int8 KV.
    # Cache key: (batch_size, total_token_num_bucket).
    #
    # The prefill graph captures the full _context_forward path including
    # pre_infer.context_forward and a graph-friendly post_infer output path.
    # Dynamic-length tensors are padded to T_bucket; attention reads only
    # the first L_real tokens via b_start_loc/b_seq_len, so padding is inert
    # for attention output. Padding K/V slots get written to but never read.
    def prefill_capture(self, engine, batch_size, total_token_num, context_forward_fn,
                        input_ids, infer_state, batch_req_bins, forward_kwargs) -> torch.Tensor:
        """Capture a CUDA graph for prefill.

        `engine` is the LoraUnorderedBatchMixed instance — we update engine.batch_req_bins
        and engine.delta to point at static padded buffers so the captured kernels
        reference stable addresses.
        `batch_req_bins` is the real (unpadded) [total_token_num] index tensor.
        """
        assert batch_size == 1, "prefill CUDA graph MVP supports batch_size=1 only"
        infer_state.mem_manager.page_table_lock.acquire()
        try:
            T_bucket = self.get_prefill_token_bucket(total_token_num)
            cache_key = (batch_size, T_bucket)

            # Swap in T_bucket-sized delta scratch buffers for LoRA (persistent across
            # capture/replay of the same bucket).
            max_lora_dim = engine.max_lora_dim
            static_delta = [
                torch.zeros((T_bucket, max_lora_dim), dtype=torch.float16, device="cuda")
                for _ in range(3)
            ]
            engine.delta = static_delta

            infer_state._att_m_buffers = self._create_att_m_buffers(torch.float16)

            # Pad input_ids
            static_input_ids = torch.zeros(T_bucket, dtype=input_ids.dtype, device="cuda")
            static_input_ids[:total_token_num].copy_(input_ids)

            # Pad position_cos / position_sin
            pc = infer_state.position_cos
            ps = infer_state.position_sin
            static_position_cos = torch.zeros((T_bucket, pc.shape[1]), dtype=pc.dtype, device="cuda")
            static_position_sin = torch.zeros((T_bucket, ps.shape[1]), dtype=ps.dtype, device="cuda")
            static_position_cos[:total_token_num].copy_(pc)
            static_position_sin[:total_token_num].copy_(ps)

            # b_start_loc / b_seq_len are small ([batch_size]) — no padding needed
            static_b_start_loc = infer_state.b_start_loc.clone()
            static_b_seq_len = infer_state.b_seq_len.clone()

            # Pad batch_req_bins
            static_batch_req_bins = torch.zeros(T_bucket, dtype=batch_req_bins.dtype, device="cuda")
            static_batch_req_bins[:total_token_num].copy_(batch_req_bins)

            # prefill_mem_index_{key,value} are already T_bucket-sized (allocated by
            # _prefill when CUDA graph is enabled). Clone for stable addresses.
            mk = infer_state.prefill_mem_index_key
            mv = infer_state.prefill_mem_index_value
            assert mk.shape[0] == T_bucket, (
                f"prefill_mem_index_key size {mk.shape[0]} != T_bucket {T_bucket}; "
                f"_prefill must allocate T_bucket KV slots when use_prefill_cg=True")
            static_mem_index_key = mk.clone()
            static_mem_index_value = mv.clone()
            static_mem_index_cat = torch.cat([static_mem_index_key, static_mem_index_value], dim=0)

            # Prefill K/V scratch buffers sized to T_bucket
            static_prefill_key_buffer = torch.empty(
                (T_bucket, self.tp_k_head_num, self.head_dim),
                dtype=torch.float16, device="cuda")
            static_prefill_value_buffer = torch.empty(
                (T_bucket, self.tp_k_head_num, self.head_dim),
                dtype=torch.float16, device="cuda")

            # b_loc_key/value — clone so address is stable across capture/replay
            static_b_loc_key = infer_state.b_loc_key.clone()
            static_b_loc_value = infer_state.b_loc_value.clone()

            # finetune_mask: all zeros (inference-only path)
            static_finetune_mask = torch.zeros(T_bucket, dtype=torch.bool, device="cuda")

            # Wire statics into infer_state + engine
            infer_state.b_start_loc = static_b_start_loc
            infer_state.b_seq_len = static_b_seq_len
            infer_state.position_cos = static_position_cos
            infer_state.position_sin = static_position_sin
            infer_state.prefill_mem_index_key = static_mem_index_key
            infer_state.prefill_mem_index_value = static_mem_index_value
            infer_state.prefill_mem_index_cat = static_mem_index_cat
            infer_state.prefill_key_buffer = static_prefill_key_buffer
            infer_state.prefill_value_buffer = static_prefill_value_buffer
            infer_state.b_loc_key = static_b_loc_key
            infer_state.b_loc_value = static_b_loc_value
            infer_state.finetune_mask = static_finetune_mask
            infer_state.total_token_num = T_bucket
            infer_state.max_len_in_batch = int(total_token_num)
            engine.batch_req_bins = static_batch_req_bins

            # Prefill warmup: 1 iter (not 3 like decode). Empirically, more warmup
            # iters on the side stream leave state that makes the captured forward
            # produce wrong logits on the first request. The post-capture replay()
            # below ensures static_output reflects the current input.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                _ = context_forward_fn(static_input_ids, infer_state, **forward_kwargs)
            torch.cuda.current_stream().wait_stream(s)

            # Capture, then replay once so static_output is populated for the CAPTURE
            # caller (otherwise the first request returns stale/uninitialized values).
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = context_forward_fn(static_input_ids, infer_state, **forward_kwargs)
            graph.replay()
            torch.cuda.current_stream().synchronize()

            bufs = {
                'input_ids': static_input_ids,
                'b_start_loc': static_b_start_loc,
                'b_seq_len': static_b_seq_len,
                'position_cos': static_position_cos,
                'position_sin': static_position_sin,
                'batch_req_bins': static_batch_req_bins,
                'prefill_mem_index_key': static_mem_index_key,
                'prefill_mem_index_value': static_mem_index_value,
                'prefill_mem_index_cat': static_mem_index_cat,
                'prefill_key_buffer': static_prefill_key_buffer,
                'prefill_value_buffer': static_prefill_value_buffer,
                'b_loc_key': static_b_loc_key,
                'b_loc_value': static_b_loc_value,
                'finetune_mask': static_finetune_mask,
                'att_m_buffers': infer_state._att_m_buffers,
                'delta': static_delta,
            }
            self._prefill_cache[cache_key] = (graph, bufs, static_output, T_bucket)
            return static_output
        finally:
            infer_state.mem_manager.page_table_lock.release()

    def prefill_replay(self, engine, batch_size, total_token_num, input_ids, infer_state,
                       batch_req_bins) -> torch.Tensor:
        cache_key = self.get_prefill_cache_key(batch_size, total_token_num)
        graph, bufs, static_output, T_bucket = self._prefill_cache[cache_key]

        # Swap engine's delta to the persistent static buffers captured with this graph.
        engine.delta = bufs['delta']

        bufs['input_ids'][total_token_num:].zero_()
        bufs['input_ids'][:total_token_num].copy_(input_ids)

        bufs['position_cos'][total_token_num:].zero_()
        bufs['position_cos'][:total_token_num].copy_(infer_state.position_cos)
        bufs['position_sin'][total_token_num:].zero_()
        bufs['position_sin'][:total_token_num].copy_(infer_state.position_sin)

        bufs['b_start_loc'].copy_(infer_state.b_start_loc)
        bufs['b_seq_len'].copy_(infer_state.b_seq_len)

        bufs['batch_req_bins'][total_token_num:].zero_()
        bufs['batch_req_bins'][:total_token_num].copy_(batch_req_bins)

        # prefill_mem_index_{key,value} are already T_bucket-sized (dedicated padding slots)
        assert infer_state.prefill_mem_index_key.shape[0] == T_bucket
        bufs['prefill_mem_index_key'].copy_(infer_state.prefill_mem_index_key)
        bufs['prefill_mem_index_value'].copy_(infer_state.prefill_mem_index_value)
        bufs['prefill_mem_index_cat'][:T_bucket].copy_(bufs['prefill_mem_index_key'])
        bufs['prefill_mem_index_cat'][T_bucket:].copy_(bufs['prefill_mem_index_value'])

        bufs['b_loc_key'].copy_(infer_state.b_loc_key)
        bufs['b_loc_value'].copy_(infer_state.b_loc_value)

        bufs['finetune_mask'].zero_()

        # Rewire
        infer_state.b_start_loc = bufs['b_start_loc']
        infer_state.b_seq_len = bufs['b_seq_len']
        infer_state.position_cos = bufs['position_cos']
        infer_state.position_sin = bufs['position_sin']
        infer_state.prefill_mem_index_key = bufs['prefill_mem_index_key']
        infer_state.prefill_mem_index_value = bufs['prefill_mem_index_value']
        infer_state.prefill_mem_index_cat = bufs['prefill_mem_index_cat']
        infer_state.prefill_key_buffer = bufs['prefill_key_buffer']
        infer_state.prefill_value_buffer = bufs['prefill_value_buffer']
        infer_state.b_loc_key = bufs['b_loc_key']
        infer_state.b_loc_value = bufs['b_loc_value']
        infer_state.finetune_mask = bufs['finetune_mask']
        infer_state._att_m_buffers = bufs['att_m_buffers']
        infer_state.total_token_num = T_bucket
        infer_state.max_len_in_batch = int(total_token_num)
        engine.batch_req_bins = bufs['batch_req_bins']

        graph.replay()
        return static_output

    def reset(self):
        self._cache.clear()
        self._prefill_cache.clear()


# ────────────────────────────────────────────────────────────────────────────
# Piecewise CUDA Graph Runner (sglang-style)
# ────────────────────────────────────────────────────────────────────────────
#
# Instead of one big graph covering the whole forward (including attention),
# split each transformer layer into two sub-graphs around the attention call:
#
#     input_embs
#       └─ pre_attn_graph[i]:  RMSNorm → Q/K/V + LoRA → rotary → destindex_copy_kv
#       └─ eager attention:    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_len)
#       └─ post_attn_graph[i]: O + LoRA → residual → FFN norm+matmul → residual
#
# Plus one pre_infer graph (embedding lookup) and one post_infer graph
# (last-token logit projection).
#
# Advantages vs the full-graph prefill path:
#   - attention runs eagerly → max_input_len is a real scalar, not frozen
#   - variable seq lengths within a bucket attend over the correct token range
#   - batch_size > 1 becomes feasible (attention's per-request loop is dynamic)
#
# Still requires bucketing on total_token_num because the static pieces have
# shape [T_bucket, ...].
#

class PiecewiseCudaGraphRunner:
    """Captures prefill as pre_infer + {pre_attn, post_attn}*L + post_infer sub-graphs.

    Cache key: (batch_size, total_token_num_bucket).
    Eager attention runs between each pre_attn and post_attn.
    """

    TOKEN_BUCKET_SIZE = 128
    BATCH_BUCKETS = [1, 2, 4, 8]

    def __init__(self, max_total_tokens: int = 25000, num_layers: int = 32,
                 tp_q_head_num: int = 32, tp_k_head_num: int = 8,
                 head_dim: int = 128, embed_dim: int = 4096):
        self.max_total_tokens = max_total_tokens
        self.num_layers = num_layers
        self.tp_q_head_num = tp_q_head_num
        self.tp_k_head_num = tp_k_head_num
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        # cache: (batch_size, T_bucket) -> bucket state
        self._cache: Dict[Tuple[int, int], dict] = {}

    @staticmethod
    def get_token_bucket(total_token_num: int) -> int:
        bs = PiecewiseCudaGraphRunner.TOKEN_BUCKET_SIZE
        return ((total_token_num + bs - 1) // bs) * bs

    @staticmethod
    def get_batch_bucket(batch_size: int) -> int:
        for b in PiecewiseCudaGraphRunner.BATCH_BUCKETS:
            if batch_size <= b:
                return b
        return batch_size

    def get_cache_key(self, batch_size: int, total_token_num: int) -> Tuple[int, int]:
        return (self.get_batch_bucket(batch_size),
                self.get_token_bucket(total_token_num))

    def has_graph(self, batch_size: int, total_token_num: int) -> bool:
        return self.get_cache_key(batch_size, total_token_num) in self._cache

    def capture(self, engine, batch_size: int, total_token_num: int,
                input_ids, infer_state, batch_req_bins,
                pre_infer_fn, pre_attn_fn, attention_fn, post_attn_fn,
                post_infer_fn, no_lora_compute: bool = False) -> torch.Tensor:
        """Capture pre_infer + per-layer {pre_attn, post_attn} + post_infer graphs.

        pre_infer_fn(input_ids_static, infer_state) -> input_embs tensor
        pre_attn_fn(layer_id, input_embs, infer_state, no_lora_compute) -> (q, k, v)
        attention_fn(q, cache_k, cache_v, o_static, infer_state) — eager, NOT captured
        post_attn_fn(layer_id, o, input_embs, infer_state, no_lora_compute) -> None
        post_infer_fn(input_embs, infer_state) -> logits
        """
        bs_bucket = self.get_batch_bucket(batch_size)
        T_bucket = self.get_token_bucket(total_token_num)
        key = (bs_bucket, T_bucket)

        infer_state.mem_manager.page_table_lock.acquire()
        try:
            # ── Pre-allocate static buffers (persist across all subgraphs) ──
            static_input_ids = torch.zeros(T_bucket, dtype=input_ids.dtype, device="cuda")
            static_input_ids[:total_token_num].copy_(input_ids)

            pc = infer_state.position_cos
            ps = infer_state.position_sin
            static_position_cos = torch.zeros((T_bucket, pc.shape[1]), dtype=pc.dtype, device="cuda")
            static_position_sin = torch.zeros((T_bucket, ps.shape[1]), dtype=ps.dtype, device="cuda")
            static_position_cos[:total_token_num].copy_(pc)
            static_position_sin[:total_token_num].copy_(ps)

            static_batch_req_bins = torch.zeros(T_bucket, dtype=batch_req_bins.dtype, device="cuda")
            static_batch_req_bins[:total_token_num].copy_(batch_req_bins)

            # b_start_loc / b_seq_len padded to bs_bucket
            static_b_start_loc = torch.zeros(bs_bucket, dtype=infer_state.b_start_loc.dtype, device="cuda")
            static_b_seq_len   = torch.zeros(bs_bucket, dtype=infer_state.b_seq_len.dtype, device="cuda")
            static_b_start_loc[:batch_size].copy_(infer_state.b_start_loc)
            static_b_seq_len[:batch_size].copy_(infer_state.b_seq_len)

            # prefill KV indices (T_bucket-sized, with dedicated padding slots)
            mk = infer_state.prefill_mem_index_key
            mv = infer_state.prefill_mem_index_value
            assert mk.shape[0] == T_bucket, f"_prefill must allocate T_bucket slots"
            static_mem_index_key = mk.clone()
            static_mem_index_value = mv.clone()
            static_mem_index_cat = torch.cat([static_mem_index_key, static_mem_index_value], dim=0)

            # Per-layer scratch: cache_k/v alias into these so destindex_copy_kv and
            # attention see stable addresses.
            static_prefill_key_buffer = torch.empty(
                (T_bucket, self.tp_k_head_num, self.head_dim),
                dtype=torch.float16, device="cuda")
            static_prefill_value_buffer = torch.empty(
                (T_bucket, self.tp_k_head_num, self.head_dim),
                dtype=torch.float16, device="cuda")

            # O-buffer: eager attention writes here, post_attn reads.
            # Reused across all layers (each layer's attention overwrites).
            static_o = torch.empty(
                (T_bucket, self.tp_q_head_num, self.head_dim),
                dtype=torch.float16, device="cuda")

            # LoRA delta scratch
            max_lora_dim = engine.max_lora_dim
            static_delta = [
                torch.zeros((T_bucket, max_lora_dim), dtype=torch.float16, device="cuda")
                for _ in range(3)
            ]

            # finetune_mask (all zeros in inference-only)
            static_finetune_mask = torch.zeros(T_bucket, dtype=torch.bool, device="cuda")

            # Wire statics into infer_state + engine
            infer_state.b_start_loc = static_b_start_loc
            infer_state.b_seq_len   = static_b_seq_len
            infer_state.position_cos = static_position_cos
            infer_state.position_sin = static_position_sin
            infer_state.prefill_mem_index_key = static_mem_index_key
            infer_state.prefill_mem_index_value = static_mem_index_value
            infer_state.prefill_mem_index_cat = static_mem_index_cat
            infer_state.prefill_key_buffer = static_prefill_key_buffer
            infer_state.prefill_value_buffer = static_prefill_value_buffer
            infer_state.finetune_mask = static_finetune_mask
            infer_state.total_token_num = T_bucket
            # Keep max_len_in_batch as real value — but attention is eager, so it's
            # set per-call anyway; here it just influences anything that reads it
            # inside the captured pre_attn (nothing does, it's only used by attention).
            infer_state.max_len_in_batch = int(total_token_num)
            engine.delta = static_delta
            engine.batch_req_bins = static_batch_req_bins

            # Warmup once on side stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                input_embs_tmp = pre_infer_fn(static_input_ids, infer_state)
                for i in range(self.num_layers):
                    q, ck, cv = pre_attn_fn(i, input_embs_tmp, infer_state, no_lora_compute)
                    attention_fn(q, ck, cv, static_o, infer_state)
                    post_attn_fn(i, static_o, input_embs_tmp, infer_state, no_lora_compute)
                _ = post_infer_fn(input_embs_tmp, infer_state)
            torch.cuda.current_stream().wait_stream(s)

            # ── Capture pre_infer graph ──
            pre_infer_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(pre_infer_graph):
                input_embs_static = pre_infer_fn(static_input_ids, infer_state)
            pre_infer_graph.replay()
            torch.cuda.current_stream().synchronize()

            # ── Per-layer: pre_attn, post_attn ──
            pre_attn_graphs = []
            pre_attn_outputs = []   # list of (q_tensor, ck_tensor, cv_tensor)
            post_attn_graphs = []
            for i in range(self.num_layers):
                g_pre = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_pre):
                    q_i, ck_i, cv_i = pre_attn_fn(i, input_embs_static, infer_state, no_lora_compute)
                g_pre.replay()
                torch.cuda.current_stream().synchronize()
                pre_attn_graphs.append(g_pre)
                pre_attn_outputs.append((q_i, ck_i, cv_i))

                # Eager attention so the subsequent post_attn graph captures with
                # a consistent static_o value (not strictly required but matches replay).
                attention_fn(q_i, ck_i, cv_i, static_o, infer_state)

                g_post = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_post):
                    post_attn_fn(i, static_o, input_embs_static, infer_state, no_lora_compute)
                g_post.replay()
                torch.cuda.current_stream().synchronize()
                post_attn_graphs.append(g_post)

            # ── Post-infer graph ──
            post_infer_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(post_infer_graph):
                logits_static = post_infer_fn(input_embs_static, infer_state)
            post_infer_graph.replay()
            torch.cuda.current_stream().synchronize()

            # After all captures, logits_static holds the CAPTURE-time output.
            # Return it directly — don't call replay() immediately after capture
            # because infer_state is still wired to static buffers and replay's
            # copy_(infer_state.x -> state['x']) would be a self-copy.
            self._cache[key] = {
                # Input statics
                'input_ids': static_input_ids,
                'position_cos': static_position_cos,
                'position_sin': static_position_sin,
                'batch_req_bins': static_batch_req_bins,
                'b_start_loc': static_b_start_loc,
                'b_seq_len': static_b_seq_len,
                'prefill_mem_index_key': static_mem_index_key,
                'prefill_mem_index_value': static_mem_index_value,
                'prefill_mem_index_cat': static_mem_index_cat,
                'prefill_key_buffer': static_prefill_key_buffer,
                'prefill_value_buffer': static_prefill_value_buffer,
                'finetune_mask': static_finetune_mask,
                'delta': static_delta,
                # Pipeline statics
                'input_embs': input_embs_static,
                'o_static': static_o,
                'logits_static': logits_static,
                # Sub-graphs
                'pre_infer_graph': pre_infer_graph,
                'pre_attn_graphs': pre_attn_graphs,
                'pre_attn_outputs': pre_attn_outputs,
                'post_attn_graphs': post_attn_graphs,
                'post_infer_graph': post_infer_graph,
                # Metadata
                'bs_bucket': bs_bucket,
                'T_bucket': T_bucket,
            }
            return logits_static
        finally:
            infer_state.mem_manager.page_table_lock.release()

    def replay(self, engine, batch_size: int, total_token_num: int,
               input_ids, infer_state, batch_req_bins,
               attention_fn) -> torch.Tensor:
        """Replay the captured sub-graphs with fresh inputs. Attention runs eagerly."""
        key = self.get_cache_key(batch_size, total_token_num)
        state = self._cache[key]
        bs_bucket = state['bs_bucket']
        T_bucket = state['T_bucket']

        # ── Copy fresh inputs into static buffers ──
        state['input_ids'].zero_()
        state['input_ids'][:total_token_num].copy_(input_ids)

        state['position_cos'].zero_()
        state['position_cos'][:total_token_num].copy_(infer_state.position_cos)
        state['position_sin'].zero_()
        state['position_sin'][:total_token_num].copy_(infer_state.position_sin)

        state['batch_req_bins'].zero_()
        state['batch_req_bins'][:total_token_num].copy_(batch_req_bins)

        state['b_start_loc'].zero_()
        state['b_seq_len'].zero_()
        state['b_start_loc'][:batch_size].copy_(infer_state.b_start_loc)
        state['b_seq_len'][:batch_size].copy_(infer_state.b_seq_len)

        assert infer_state.prefill_mem_index_key.shape[0] == T_bucket
        state['prefill_mem_index_key'].copy_(infer_state.prefill_mem_index_key)
        state['prefill_mem_index_value'].copy_(infer_state.prefill_mem_index_value)
        state['prefill_mem_index_cat'][:T_bucket].copy_(state['prefill_mem_index_key'])
        state['prefill_mem_index_cat'][T_bucket:].copy_(state['prefill_mem_index_value'])

        state['finetune_mask'].zero_()

        # ── Wire infer_state + engine to the static buffers ──
        infer_state.b_start_loc = state['b_start_loc']
        infer_state.b_seq_len = state['b_seq_len']
        infer_state.position_cos = state['position_cos']
        infer_state.position_sin = state['position_sin']
        infer_state.prefill_mem_index_key = state['prefill_mem_index_key']
        infer_state.prefill_mem_index_value = state['prefill_mem_index_value']
        infer_state.prefill_mem_index_cat = state['prefill_mem_index_cat']
        infer_state.prefill_key_buffer = state['prefill_key_buffer']
        infer_state.prefill_value_buffer = state['prefill_value_buffer']
        infer_state.finetune_mask = state['finetune_mask']
        infer_state.total_token_num = T_bucket
        infer_state.max_len_in_batch = int(total_token_num)
        engine.delta = state['delta']
        engine.batch_req_bins = state['batch_req_bins']

        # ── Orchestrate pieces ──
        state['pre_infer_graph'].replay()
        for i in range(self.num_layers):
            state['pre_attn_graphs'][i].replay()
            q_i, ck_i, cv_i = state['pre_attn_outputs'][i]
            # Eager attention — uses REAL max_input_len, supports variable shapes
            attention_fn(q_i, ck_i, cv_i, state['o_static'], infer_state)
            state['post_attn_graphs'][i].replay()
        state['post_infer_graph'].replay()

        return state['logits_static']

    def reset(self):
        self._cache.clear()
