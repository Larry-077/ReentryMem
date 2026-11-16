# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for ReentryMem inference.
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .attention import attention as flash_attention
from .causal_attention import (
    check_causal_self_attention,
    frame_causal_attention,
    validate_temporal_attention,
)

__all__ = ["ReentryMemModel"]
TRANS_TEXT = [
    "Camera does not move.",
    "Camera moves forward.",
    "Camera moves backward.",
    "Camera moves right.",
    "Camera moves left.",
    "Camera moves forward-right.",
    "Camera moves forward-left.",
    "Camera moves backward-right.",
    "Camera moves backward-left.",
]
ROT_TEXT = [
    "Camera does not rotate.",
    "Camera pitches up.",
    "Camera pitches down.",
    "Camera yaws right.",
    "Camera yaws left.",
    "Camera pitches up and yaws right.",
    "Camera pitches up and yaws left.",
    "Camera pitches down and yaws right.",
    "Camera pitches down and yaws left.",
]


def build_action_text_table() -> List[str]:
    table = []
    for t_idx in range(9):
        for r_idx in range(9):
            table.append(f"{TRANS_TEXT[t_idx]} {ROT_TEXT[r_idx]}")
    return table


ACTION_TEXT_TABLE = build_action_text_table()


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, positions=None):
    n, c = (x.size(2), x.size(3) // 2)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i in range(x.size(0)):
        if positions is not None:
            pos_i = positions[i]
            seq_len = pos_i.size(0)
            t_idx = pos_i[:, 0].long().clamp_(0, freqs[0].size(0) - 1)
            h_idx = pos_i[:, 1].long().clamp_(0, freqs[1].size(0) - 1)
            w_idx = pos_i[:, 2].long().clamp_(0, freqs[2].size(0) - 1)
            freqs_i = torch.cat(
                [freqs[0][t_idx], freqs[1][h_idx], freqs[2][w_idx]], dim=-1
            ).reshape(seq_len, 1, -1)
        else:
            f, h, w = grid_sizes[i].tolist()
            seq_len = f * h * w
            freqs_i = torch.cat(
                [
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-06, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-06):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.temporal_attention_mode = "dense"
        self.temporal_attention_backend = "auto"

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = (*x.shape[:2], self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        if self.temporal_attention_mode == "frame_causal":
            value = frame_causal_attention(
                q, k, v, grid_sizes, seq_lens, backend=self.temporal_attention_backend
            )
        else:
            value = flash_attention(q=q, k=k, v=v, k_lens=seq_lens, window_size=self.window_size)
        return self.o(value.flatten(2))


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        b, n, d = (x.size(0), self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        value = flash_attention(q, k, v, k_lens=context_lens)
        return self.o(value.flatten(2))


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-06,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        per_frame_context=None,
        per_frame_context_lens=None,
        tokens_per_frame=None,
        num_latent_frames=None,
    ):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        value = self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        x = x + self.self_attn(value, seq_lens, grid_sizes, freqs) * e[2].squeeze(2)
        normalized = self.norm3(x)
        if per_frame_context is not None:
            batch, frames, spatial = (x.shape[0], num_latent_frames, tokens_per_frame)
            queries = normalized[:, : frames * spatial].reshape(batch * frames, spatial, -1)
            value = self.cross_attn(queries, per_frame_context, per_frame_context_lens)
            correction = torch.zeros_like(normalized)
            correction[:, : frames * spatial] = value.reshape(batch, frames * spatial, -1)
            x = x + correction
        else:
            x = x + self.cross_attn(normalized, context, context_lens)
        value = self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        return x + self.ffn(value) * e[5].squeeze(2)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-06):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with torch.autocast(x.device.type, enabled=False):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class ReentryMemModel(ModelMixin, ConfigMixin):
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-06,
        temporal_attention_mode="dense",
        temporal_attention_backend="auto",
    ):
        super().__init__()
        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        assert dim % num_heads == 0 and dim // num_heads % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
        self.configure_temporal_attention(temporal_attention_mode, temporal_attention_backend)
        self.init_weights()

    def configure_temporal_attention(self, mode="dense", backend="auto"):
        validate_temporal_attention(mode, backend)
        if mode == "frame_causal":
            if int(self.patch_size[0]) != 1:
                raise ValueError("frame_causal requires temporal patch_size=1")
            for block in self.blocks:
                check_causal_self_attention(block.self_attn)
        self.temporal_attention_mode = mode
        self.temporal_attention_backend = backend
        for block in self.blocks:
            block.self_attn.temporal_attention_mode = mode
            block.self_attn.temporal_attention_backend = backend
        self.register_to_config(temporal_attention_mode=mode, temporal_attention_backend=backend)
        return {"temporal_attention_mode": mode, "temporal_attention_backend": backend}

    def add_camera_control_parameters(self, num_actions: int = 81):
        from .action_embedder import CameraActionEmbedder

        self.hycam_action_embedder = CameraActionEmbedder(
            hidden_size=self.dim, freq_dim=self.freq_dim, num_actions=num_actions
        )
        dp = sum((p.numel() for p in self.hycam_action_embedder.parameters()))
        print(f"[ReentryMem] camera conditioning: {dp:,} params")

    def precompute_cam_text_embeddings(self, encode_fn, device, dtype=torch.bfloat16):
        action_texts = ACTION_TEXT_TABLE
        raw_embs = encode_fn(action_texts)
        self._cam_text_raw = [e.to(device=device, dtype=dtype) for e in raw_embs]
        self._use_cam_text_cross_attn = True
        _S = max((int(e.shape[0]) for e in self._cam_text_raw))
        _td = int(self._cam_text_raw[0].shape[1])
        _pad = torch.zeros(len(self._cam_text_raw), _S, _td, device=device, dtype=dtype)
        for _i, _e in enumerate(self._cam_text_raw):
            _pad[_i, : _e.shape[0]] = _e
        self._cam_text_padded = _pad
        print(
            f"[ReentryMem] cached {len(self._cam_text_raw)} raw embeddings, text_dim={self._cam_text_raw[0].shape[1]}"
        )

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        action_labels=None,
        cond_latent_frames=0,
        cond_timestep="zero",
    ):
        if self.model_type == "i2v":
            assert y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, "Token sequence exceeds seq_len"
        x = torch.cat(
            [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
        )
        if cond_timestep not in ("zero", "shared"):
            raise ValueError(f"cond_timestep must be 'zero' or 'shared', got {cond_timestep!r}")
        if cond_latent_frames > 0 and cond_timestep == "zero":
            F_lat = grid_sizes[0, 0].item()
            H_p = grid_sizes[0, 1].item()
            W_p = grid_sizes[0, 2].item()
            tokens_per_frame = H_p * W_p
            cond_latent_frames = min(cond_latent_frames, F_lat - 1)
            cond_tokens = cond_latent_frames * tokens_per_frame
            gen_tokens = (F_lat - cond_latent_frames) * tokens_per_frame
            per_token_t = torch.cat(
                [torch.zeros(cond_tokens, device=t.device, dtype=t.dtype), t[0].expand(gen_tokens)]
            )
            with torch.autocast(x.device.type, enabled=False):
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, per_token_t).float())
                e = e.unsqueeze(0)
                if e.shape[1] < seq_len:
                    e = torch.cat([e, e.new_zeros(1, seq_len - e.shape[1], e.shape[2])], dim=1)
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.autocast(x.device.type, enabled=False):
                bt = t.size(0)
                t_flat = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t_flat)
                    .unflatten(0, (bt, seq_len))
                    .float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        per_frame_context = None
        per_frame_context_lens = None
        _tokens_per_frame = None
        _num_latent_frames = None
        _use_cam_text = (
            getattr(self, "_use_cam_text_cross_attn", False)
            and action_labels is not None
            and hasattr(self, "_cam_text_raw")
        )
        if _use_cam_text:
            B = len(context) if isinstance(context, list) else context.shape[0]
            T = grid_sizes[0, 0].item()
            H_p = grid_sizes[0, 1].item()
            W_p = grid_sizes[0, 2].item()
            _tokens_per_frame = H_p * W_p
            _num_latent_frames = T
            al = action_labels
            if al.shape[1] != T:
                al = (
                    F.interpolate(al.float().unsqueeze(1), size=T, mode="nearest").squeeze(1).long()
                )
            assert al.min() >= 0 and al.max() < len(self._cam_text_raw), (
                f"action_labels out of range: min={al.min().item()}, max={al.max().item()}, num_actions={len(self._cam_text_raw)}"
            )
            text_dim_raw = self._cam_text_raw[0].shape[1]
            S_target = self.text_len
            _cam = self._cam_text_padded[al]
            _S_cam = _cam.shape[2]
            _S_capmax = max((int(u.shape[0]) for u in context))
            _cap = torch.stack(
                [torch.cat([u, u.new_zeros(_S_capmax - u.shape[0], text_dim_raw)]) for u in context]
            )
            _cap = _cap[:, None].expand(B, T, _S_capmax, text_dim_raw)
            _comb = torch.cat([_cam, _cap], dim=2).reshape(B * T, _S_cam + _S_capmax, text_dim_raw)
            if _comb.shape[1] >= S_target:
                per_frame_raw = _comb[:, :S_target].contiguous()
            else:
                per_frame_raw = torch.cat(
                    [_comb, _comb.new_zeros(B * T, S_target - _comb.shape[1], text_dim_raw)], dim=1
                )
            per_frame_context = self.text_embedding(per_frame_raw)
        else:
            context = self.text_embedding(
                torch.stack(
                    [
                        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                        for u in context
                    ]
                )
            )
        context_lens = None
        per_frame_context_lens = None
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            per_frame_context=per_frame_context,
            per_frame_context_lens=per_frame_context_lens,
            tokens_per_frame=_tokens_per_frame,
            num_latent_frames=_num_latent_frames,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, **kwargs)
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)
