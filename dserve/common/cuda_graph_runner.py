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

    def __init__(self, max_total_tokens: int = 25000, num_layers: int = 32,
                 tp_q_head_num: int = 32, tp_k_head_num: int = 8,
                 head_dim: int = 128):
        self.max_total_tokens = max_total_tokens
        self.num_layers = num_layers
        self.tp_q_head_num = tp_q_head_num
        self.tp_k_head_num = tp_k_head_num
        self.head_dim = head_dim

        # cache: (bs_bucket, ml_bucket) -> (graph, bufs, output, ml_bucket)
        self._cache: Dict[Tuple[int, int], Tuple[torch.cuda.CUDAGraph, dict, torch.Tensor, int]] = {}
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
        infer_state.alt_mem_manager.page_table_lock.acquire()
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
        infer_state.alt_mem_manager.page_table_lock.release()
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

    def reset(self):
        self._cache.clear()
