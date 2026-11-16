from __future__ import annotations
import torch
import torch.nn.functional as F
from . import attention as attention_backend

TEMPORAL_ATTENTION_MODES = ("dense", "frame_causal")
TEMPORAL_ATTENTION_BACKENDS = ("auto", "flash", "sdpa")


def validate_temporal_attention(mode: str, backend: str) -> None:
    if mode not in TEMPORAL_ATTENTION_MODES:
        raise ValueError(f"unknown temporal_attention_mode {mode!r}")
    if backend not in TEMPORAL_ATTENTION_BACKENDS:
        raise ValueError(f"unknown temporal_attention_backend {backend!r}")


def check_causal_self_attention(module, *, positions=None):
    if positions is not None or tuple(module.window_size) != (-1, -1):
        raise ValueError("Block-causal inference requires full spatial attention")


def _rectangular_attention(q, k, v, backend):
    if backend == "flash":
        return attention_backend.flash_attention(q, k, v, causal=False, dtype=v.dtype)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if q.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return output.transpose(1, 2)


def frame_causal_attention(q, k, v, grid_sizes, seq_lens=None, *, backend="auto"):
    validate_temporal_attention("frame_causal", backend)
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("frame_causal needs matching [B,S,H,D] Q/K/V shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if not all((x.is_floating_point() for x in (q, k, v))):
        raise ValueError("Q/K/V must be floating point")
    batch, sequence = q.shape[:2]
    grids = torch.as_tensor(grid_sizes).tolist()
    if len(grids) != batch or any((len(g) != 3 for g in grids)):
        raise ValueError("grid_sizes must have shape [B,3]")
    lengths = (
        [int(t * h * w) for t, h, w in grids]
        if seq_lens is None
        else torch.as_tensor(seq_lens).tolist()
    )
    if len(lengths) != batch:
        raise ValueError("seq_lens must have shape [B]")
    for grid, length in zip(grids, lengths):
        if any((int(n) != n or n <= 0 for n in grid)):
            raise ValueError(f"invalid frame grid {grid}")
        if length != grid[0] * grid[1] * grid[2] or length > sequence:
            raise ValueError(
                "seq_lens must equal complete frame grids within S; packed/distributed sequence shards are unsupported"
            )
    has_flash = attention_backend.FLASH_ATTN_2_AVAILABLE or attention_backend.FLASH_ATTN_3_AVAILABLE
    if backend == "auto":
        backend = "flash" if q.is_cuda and has_flash else "sdpa"
    if backend == "flash" and (not q.is_cuda or not has_flash):
        raise RuntimeError("backend=flash requires CUDA and an installed FlashAttention")
    if q.is_cuda and v.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CUDA frame_causal requires float16/bfloat16 V for fused attention")
    q, k = (q.to(v.dtype), k.to(v.dtype))
    results = []
    for b, ((frames, height, width), length) in enumerate(zip(grids, lengths)):
        spatial = int(height * width)
        pieces = []
        for frame in range(int(frames)):
            start, end = (frame * spatial, (frame + 1) * spatial)
            pieces.append(
                _rectangular_attention(
                    q[b : b + 1, start:end], k[b : b + 1, :end], v[b : b + 1, :end], backend
                )
            )
        if length < sequence:
            pieces.append(v.new_zeros(1, sequence - int(length), *v.shape[2:]))
        results.append(torch.cat(pieces, dim=1))
    return torch.cat(results, dim=0)
