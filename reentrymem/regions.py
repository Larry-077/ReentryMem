from __future__ import annotations
import json
import pathlib
from typing import Any, Mapping, Sequence
import numpy as np
import torch


def subject_attention_heads(
    q: torch.Tensor, k: torch.Tensor, subject_indices: torch.Tensor, *, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        q.ndim != 4
        or k.ndim != 4
        or q.shape[2:] != k.shape[2:]
        or (int(k.shape[0]) not in (1, int(q.shape[0])))
    ):
        raise ValueError(
            f"expected q [N,M,H,D] and k [N|1,S,H,D], got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if subject_indices.numel() == 0:
        raise ValueError("no subject tokens to read attention towards")
    index = subject_indices.to(device=q.device, dtype=torch.long)
    if int(index.max()) >= int(k.shape[1]):
        raise ValueError(f"subject token {int(index.max())} outside {int(k.shape[1])} positions")
    count, length, heads, head_dim = (int(v) for v in q.shape)
    scale = 1.0 / float(head_dim) ** 0.5
    share = torch.empty(count, length, heads, dtype=torch.float32, device=q.device)
    logit = torch.empty_like(share)
    along_frames = count > 1
    total = count if along_frames else length
    for start in range(0, total, int(chunk)):
        stop = min(start + int(chunk), total)
        if along_frames:
            queries = q[start:stop]
            keys = k[start:stop] if int(k.shape[0]) == count else k
        else:
            queries, keys = (q[:, start:stop], k)
        scores = (
            torch.matmul(
                queries.to(torch.float32).permute(0, 2, 1, 3),
                keys.to(torch.float32).permute(0, 2, 3, 1),
            )
            * scale
        )
        part_share = torch.softmax(scores, dim=-1).index_select(-1, index).sum(-1)
        part_logit = scores.index_select(-1, index).mean(-1)
        if along_frames:
            share[start:stop] = part_share.permute(0, 2, 1)
            logit[start:stop] = part_logit.permute(0, 2, 1)
        else:
            share[:, start:stop] = part_share.permute(0, 2, 1)
            logit[:, start:stop] = part_logit.permute(0, 2, 1)
        del scores, part_share, part_logit
    return (share, logit)


class EntityAttentionReader:
    def __init__(
        self,
        model: Any,
        subject_indices: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        frames: int,
        tokens_per_frame: int,
        frame_chunk: int = 8,
        keep_layers: bool = False,
        keep_heads: bool = False,
        head_sets: Mapping[str, Sequence[Sequence[int]]] | None = None,
    ) -> None:
        indices = np.asarray(subject_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("no subject tokens: the scene has no mention to read attention to")
        self.per_frame = bool(getattr(model, "_use_cam_text_cross_attn", False))
        self.cam_offset = 0
        if self.per_frame:
            padded = getattr(model, "_cam_text_padded", None)
            if padded is None:
                raise RuntimeError(
                    "the cam-text cross-attention is on but no camera text was precomputed, so the caption's offset is unknown"
                )
            self.cam_offset = int(padded.shape[1])
        text_len = int(getattr(model, "text_len", 512))
        shifted = indices + self.cam_offset
        kept = shifted[shifted < text_len]
        self.dropped_tokens = int(shifted.size - kept.size)
        if kept.size == 0:
            raise ValueError(
                f"every subject token falls past the {text_len}-position context behind {self.cam_offset} camera-text positions"
            )
        self.model = model
        self.indices = torch.as_tensor(kept)
        self.frames = int(frames)
        self.tokens_per_frame = int(tokens_per_frame)
        self.frame_chunk = int(frame_chunk)
        self.keep_layers = bool(keep_layers)
        self.keep_heads = bool(keep_heads)
        self._sum: torch.Tensor | None = None
        self._layer_share: list[torch.Tensor] = []
        self._layer_logit: list[torch.Tensor] = []
        self._head_share: list[torch.Tensor] = []
        self._sets: dict[str, dict[int, torch.Tensor]] = {}
        self._set_size: dict[str, int] = {}
        for name, pairs in (head_sets or {}).items():
            by_block: dict[int, set[int]] = {}
            for block, head in pairs:
                if not 0 <= int(block) < len(model.blocks):
                    raise ValueError(
                        f"head set {name!r} names block {block} of {len(model.blocks)}"
                    )
                by_block.setdefault(int(block), set()).add(int(head))
            if not by_block:
                raise ValueError(f"head set {name!r} is empty")
            self._sets[name] = {
                block: torch.as_tensor(sorted(heads), dtype=torch.long)
                for block, heads in by_block.items()
            }
            self._set_size[name] = sum((len(heads) for heads in by_block.values()))
        self._set_sum: dict[str, torch.Tensor] = {}
        self._set_read: dict[str, int] = {}
        self.layers_read = 0
        self._handles: list[Any] = []

    def _hook(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], block: int = -1
    ) -> None:
        if len(args) < 2:
            raise RuntimeError("cross-attention hook expected (x, context, context_lens)")
        x, context = (args[0], args[1])
        if self.per_frame:
            if x.ndim != 3 or tuple((int(v) for v in x.shape[:2])) != (
                self.frames,
                self.tokens_per_frame,
            ):
                raise RuntimeError(
                    f"cross-attention was called on {tuple(x.shape)}, expected {self.frames} frames of {self.tokens_per_frame} tokens"
                )
            chunk = self.frame_chunk
        else:
            expected = self.frames * self.tokens_per_frame
            if x.ndim != 3 or int(x.shape[0]) != 1 or int(x.shape[1]) != expected:
                raise RuntimeError(
                    f"cross-attention was called on {tuple(x.shape)}, expected the whole sequence of {expected} tokens"
                )
            chunk = self.frame_chunk * self.tokens_per_frame
        with torch.no_grad():
            heads, head_dim = (int(module.num_heads), int(module.head_dim))
            q = module.norm_q(module.q(x)).view(x.shape[0], -1, heads, head_dim)
            k = module.norm_k(module.k(context)).view(context.shape[0], -1, heads, head_dim)
            per_head, logit = subject_attention_heads(q, k, self.indices, chunk=chunk)
            per_head = per_head.reshape(1, -1, heads)
            share = per_head.mean(-1)
            self._sum = share if self._sum is None else self._sum + share
            if self.keep_layers:
                self._layer_share.append(share[0].to(torch.float16).cpu())
                self._layer_logit.append(
                    logit.reshape(1, -1, heads).mean(-1)[0].to(torch.float16).cpu()
                )
            if self.keep_heads:
                self._head_share.append(per_head[0].T.to(torch.float16).cpu())
            for name, blocks in self._sets.items():
                picked = blocks.get(block)
                if picked is None:
                    continue
                if int(picked.max()) >= heads:
                    raise ValueError(f"head set {name!r} names head {int(picked.max())} of {heads}")
                part = per_head.index_select(-1, picked.to(per_head.device)).sum(-1)
                self._set_sum[name] = (
                    part if name not in self._set_sum else self._set_sum[name] + part
                )
                self._set_read[name] = self._set_read.get(name, 0) + int(picked.numel())
            self.layers_read += 1

    def __enter__(self) -> "EntityAttentionReader":
        if self._handles:
            raise RuntimeError("reader cannot be entered twice")
        self._sum, self.layers_read = (None, 0)
        self._layer_share, self._layer_logit, self._head_share = ([], [], [])
        self._set_sum, self._set_read = ({}, {})
        for index, block in enumerate(self.model.blocks):
            self._handles.append(
                block.cross_attn.register_forward_pre_hook(
                    lambda module, args, kwargs, index=index: self._hook(
                        module, args, kwargs, index
                    ),
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def field(self) -> torch.Tensor:
        if self._sum is None or self.layers_read == 0:
            raise RuntimeError(
                "no cross-attention call was read; the forward did not run inside the reader, or cross-attention was not called"
            )
        if self.layers_read != len(self.model.blocks):
            raise RuntimeError(f"read {self.layers_read} blocks of {len(self.model.blocks)}")
        if self._sum.shape[0] != 1:
            raise ValueError("field() reads batch 1")
        return self._sum[0] / float(self.layers_read)

    def layer_fields(self) -> dict[str, torch.Tensor]:
        if not self.keep_layers:
            raise RuntimeError("the reader was not asked to keep layers")
        if len(self._layer_share) != len(self.model.blocks):
            raise RuntimeError(f"kept {len(self._layer_share)} blocks of {len(self.model.blocks)}")
        return {"share": torch.stack(self._layer_share), "logit": torch.stack(self._layer_logit)}

    def head_set_field(self, name: str) -> torch.Tensor:
        if name not in self._sets:
            raise KeyError(f"no head set {name!r}; the reader has {sorted(self._sets)}")
        if self._set_read.get(name, 0) != self._set_size[name]:
            raise RuntimeError(
                f"head set {name!r} read {self._set_read.get(name, 0)} of {self._set_size[name]} heads; the forward did not run every block"
            )
        return self._set_sum[name][0] / float(self._set_size[name])

    def head_fields(self) -> torch.Tensor:
        if not self.keep_heads:
            raise RuntimeError("the reader was not asked to keep heads")
        if len(self._head_share) != len(self.model.blocks):
            raise RuntimeError(f"kept {len(self._head_share)} blocks of {len(self.model.blocks)}")
        return torch.stack(self._head_share)


def load_region_config(
    path: str | pathlib.Path, *, blocks: int | None = None, heads: int | None = None
) -> dict[str, Any]:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} is not a self-mask spec (schema_version 1)")
    for key in ("shape_heads", "presence_heads"):
        pairs = payload.get(key) or []
        if not pairs:
            raise ValueError(f"{path}: {key} is empty")
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"{path}: {key} entry {pair} is not [block, head]")
            if blocks is not None and (not 0 <= int(pair[0]) < blocks):
                raise ValueError(f"{path}: {key} names block {pair[0]} of {blocks}")
            if heads is not None and (not 0 <= int(pair[1]) < heads):
                raise ValueError(f"{path}: {key} names head {pair[1]} of {heads}")
    presence = payload.get("presence") or {}
    if presence.get("normalise") != "segment_zscore_peak":
        raise ValueError(f"{path}: presence.normalise must be segment_zscore_peak")
    if "threshold" not in presence:
        raise ValueError(f"{path}: presence.threshold is missing")
    if int(presence.get("median_frames", 1)) < 1 or int(presence.get("median_frames", 1)) % 2 == 0:
        raise ValueError(f"{path}: presence.median_frames must be a positive odd number")
    relative = presence.get("relative")
    if relative is not None and (not 0.0 < float(relative) <= 1.0):
        raise ValueError(f"{path}: presence.relative must be in (0, 1]")
    ladder = presence.get("threshold_by_sigma")
    if ladder is not None:
        points = [(float(s), float(t)) for s, t in ladder]
        if not points or points != sorted(points):
            raise ValueError(
                f"{path}: presence.threshold_by_sigma must be [[sigma, threshold], ...] in ascending sigma"
            )
    if not 0.0 < float(payload.get("grow_ratio", -1)) < 1.0:
        raise ValueError(f"{path}: grow_ratio must be in (0, 1)")
    dilate = payload.get("dilate") or {}
    if set(dilate) != {"bias", "loss"} or any((int(v) < 0 for v in dilate.values())):
        raise ValueError(f"{path}: dilate needs non-negative bias and loss")
    return payload


def presence_threshold(spec: Mapping[str, Any], sigma: float | None = None) -> float:
    presence = spec["presence"]
    ladder = presence.get("threshold_by_sigma")
    if sigma is None or not ladder:
        return float(presence["threshold"])
    points = [(float(s), float(t)) for s, t in ladder]
    sigma = float(sigma)
    if sigma <= points[0][0]:
        return points[0][1]
    for (low, low_tau), (high, high_tau) in zip(points, points[1:]):
        if sigma <= high:
            return low_tau + (high_tau - low_tau) * (sigma - low) / (high - low)
    return points[-1][1]


def region_head_sets(spec: Mapping[str, Any]) -> dict[str, list[list[int]]]:
    return {
        "shape": [list(p) for p in spec["shape_heads"]],
        "presence": [list(p) for p in spec["presence_heads"]],
    }


def extract_head_mask(
    shape_field: torch.Tensor | np.ndarray,
    presence_field: torch.Tensor | np.ndarray,
    *,
    frames: int,
    height: int,
    width: int,
    cond_frames: int,
    threshold: float,
    median_frames: int = 3,
    relative: float | None = None,
    grow_ratio: float = 0.5,
    dilate: int = 1,
) -> dict[str, Any]:
    from scipy import ndimage

    def grid(field):
        values = (
            field.detach().float().cpu().numpy()
            if isinstance(field, torch.Tensor)
            else np.asarray(field, dtype=np.float64)
        )
        if values.size != frames * height * width:
            raise ValueError(f"field has {values.size} values for {frames}x{height}x{width}")
        return values.reshape(frames, height, width)[cond_frames:].astype(np.float64)

    shape, presence = (grid(shape_field), grid(presence_field))
    generated = frames - cond_frames
    flat = presence.reshape(generated, -1)
    score = ((flat - flat.mean()) / (flat.std() + 1e-12)).max(axis=1)
    if int(median_frames) > 1:
        score = ndimage.median_filter(score, size=int(median_frames), mode="nearest")
    present = score > float(threshold)
    if relative is not None:
        present &= score >= float(relative) * float(score.max())
    target = np.zeros((generated, height, width), dtype=bool)
    for frame in np.nonzero(present)[0]:
        values = shape[frame]
        peak = np.unravel_index(int(np.argmax(values)), values.shape)
        floor = float(values.min())
        labels, _ = ndimage.label(values >= floor + float(grow_ratio) * (values[peak] - floor))
        blob = labels == labels[peak]
        target[frame] = ndimage.binary_dilation(blob, iterations=int(dilate)) if dilate else blob
    token_mask = np.zeros((frames, height, width), dtype=bool)
    token_mask[cond_frames:] = target
    tokens = torch.from_numpy(token_mask)
    latent = tokens.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).unsqueeze(0)
    return {
        "token_mask": tokens,
        "latent_mask": latent,
        "target_cells": int(target.sum()),
        "target_frames_with_mask": int(target.any(axis=(1, 2)).sum()),
        "present_frames": int(present.sum()),
        "presence_score": score,
    }
