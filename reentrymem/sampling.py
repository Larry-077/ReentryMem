from __future__ import annotations
from typing import Protocol
import torch


def overshoot_update(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
    sigma_next: float,
    *,
    gain: torch.Tensor,
    c: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    step = sigma - sigma_next
    if c <= 0.0 or step <= 0.0 or sigma_next <= 0.0:
        return sample + velocity * (sigma_next - sigma)
    s_o = torch.clamp(sigma_next - c * gain * step, min=0.0)
    x_o = sample + velocity * (s_o - sigma)
    a = (1.0 - sigma_next) / (1.0 - s_o)
    b = torch.sqrt(torch.clamp(sigma_next**2 - (a * s_o) ** 2, min=0.0))
    return torch.where(gain > 0, a * x_o + b * noise, sample + velocity * (sigma_next - sigma))


class GainSource(Protocol):
    def gain(
        self, step: int, sigma: float, sample: torch.Tensor, x0_hat: torch.Tensor
    ) -> torch.Tensor: ...


class ReentryMemSampler:
    def __init__(
        self,
        c: float,
        source: GainSource,
        *,
        cond_latent_frames: int,
        seed: int,
        device: torch.device | str = "cuda",
        keep_fields: bool = False,
    ) -> None:
        self.c = float(c)
        self.source = source
        self.cond_latent_frames = int(cond_latent_frames)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.device = torch.device(device)
        self.trace: list[dict[str, float]] = []
        self.fields: list[torch.Tensor] | None = [] if keep_fields else None

    def gain(
        self, step: int, sigma: float, sample: torch.Tensor, x0_hat: torch.Tensor
    ) -> torch.Tensor:
        m = self.source.gain(step, sigma, sample, x0_hat).to(
            device=sample.device, dtype=sample.dtype
        )
        while m.dim() < sample.dim():
            m = m.unsqueeze(0)
        m = torch.clamp(m, 0.0, 1.0)
        m[..., : self.cond_latent_frames, :, :] = 0.0
        return m

    def step(
        self,
        step: int,
        sigma: float,
        sigma_next: float,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        x0_hat: torch.Tensor,
    ) -> torch.Tensor:
        m = self.gain(step, sigma, sample, x0_hat)
        noise = torch.randn(sample.shape, generator=self.generator, dtype=torch.float32).to(
            sample.device
        )
        out = overshoot_update(sample, velocity, sigma, sigma_next, gain=m, c=self.c, noise=noise)
        target = m[..., self.cond_latent_frames :, :, :]
        self.trace.append(
            {
                "step": step,
                "sigma": sigma,
                "gain_mean": float(target.mean()) if target.numel() else 0.0,
                "gain_max": float(target.max()) if target.numel() else 0.0,
            }
        )
        if self.fields is not None:
            self.fields.append(m.detach().reshape(m.shape[-3:]).to("cpu", torch.float16))
        return out

    def report(self) -> dict[str, float]:
        if not self.trace:
            return {}
        return {
            "steps": len(self.trace),
            "gain_mean": sum((row["gain_mean"] for row in self.trace)) / len(self.trace),
            "gain_max": max((row["gain_max"] for row in self.trace)),
        }


class EntityRegions:
    def __init__(
        self,
        spec: dict,
        *,
        frames: int,
        height: int,
        width: int,
        cond_frames: int,
        trusted_sigma: float = 0.8,
        dilate_key: str = "loss",
    ) -> None:
        self.spec = spec
        self.frames, self.height, self.width = (int(frames), int(height), int(width))
        self.cond_frames = int(cond_frames)
        self.trusted_sigma = float(trusted_sigma)
        self.dilate_key = dilate_key
        self._shape: torch.Tensor | None = None
        self._presence: torch.Tensor | None = None
        self.reads: list[dict[str, float]] = []

    def capture(self, shape_field, presence_field) -> None:
        self._shape, self._presence = (shape_field, presence_field)

    def capture_from(self, reader) -> None:
        self.capture(reader.head_set_field("shape"), reader.head_set_field("presence"))

    def gain(self, step, sigma, sample, x0_hat):
        empty = torch.zeros_like(sample[:, :1])
        if self._shape is None or self._presence is None or sigma > self.trusted_sigma:
            return empty
        from .regions import extract_head_mask, presence_threshold

        presence = self.spec["presence"]
        extracted = extract_head_mask(
            self._shape,
            self._presence,
            frames=self.frames,
            height=self.height,
            width=self.width,
            cond_frames=self.cond_frames,
            threshold=presence_threshold(self.spec, sigma),
            median_frames=int(presence.get("median_frames", 1)),
            relative=presence.get("relative"),
            grow_ratio=float(self.spec["grow_ratio"]),
            dilate=int(self.spec["dilate"][self.dilate_key]),
        )
        self.reads.append(
            {
                "step": step,
                "sigma": sigma,
                "present_frames": extracted["present_frames"],
                "cells": extracted["target_cells"],
            }
        )
        return extracted["latent_mask"].unsqueeze(0).to(device=sample.device, dtype=sample.dtype)

    def report(self) -> dict[str, float]:
        if not self.reads:
            return {"mask_steps": 0}
        return {
            "mask_steps": len(self.reads),
            "present_frames": sum((r["present_frames"] for r in self.reads)) / len(self.reads),
            "cells": sum((r["cells"] for r in self.reads)) / len(self.reads),
        }
