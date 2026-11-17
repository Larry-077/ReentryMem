#!/usr/bin/env python3
"""Export a portable model folder; no optimizer, scheduler or training data is copied."""

import argparse
import json
import os
from pathlib import Path
import shutil
from reentrymem.checkpoints import (
    CONFIG_KEYS,
    MODEL_DEFAULTS,
    MODEL_NAME,
    sha256,
    safetensors_header,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", default="ReentryMem")
    p.add_argument("--variant", default="single-entity")
    p.add_argument("--step", type=int, default=1496)
    p.add_argument(
        "--hardlink",
        action="store_true",
        help="Local staging without a second 10GB copy; do not edit linked weights",
    )
    a = p.parse_args()
    source, dest = Path(a.checkpoint).resolve(), Path(a.output_dir).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Export destination must be new or empty")
    header = safetensors_header(source / MODEL_NAME)
    if "hycam_action_embedder.mlp.0.weight" not in header:
        raise ValueError("Expected the merged camera-conditioned Wan2.2 checkpoint")
    original = json.loads((source / "config.json").read_text())
    unknown = [k for k in original if not k.startswith("_") and k not in CONFIG_KEYS]
    if unknown:
        raise ValueError(f"Unsupported architecture options: {unknown}")
    config = {**MODEL_DEFAULTS, **{k: v for k, v in original.items() if k in CONFIG_KEYS}}
    dest.mkdir(parents=True, exist_ok=True)
    if a.hardlink:
        os.link(source / MODEL_NAME, dest / MODEL_NAME)
    else:
        shutil.copyfile(source / MODEL_NAME, dest / MODEL_NAME)
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    release = Path(__file__).resolve().parents[1]
    for f in ["LICENSE", "NOTICE"]:
        shutil.copy2(release / f, dest / f)
    shutil.copy2(release / "reentrymem/configs/entity_regions.json", dest / "entity_regions.json")
    runtime = dict(
        name=a.name,
        variant=a.variant,
        checkpoint_step=a.step,
        base_model="Wan-AI/Wan2.2-TI2V-5B",
        dtype="bfloat16",
        height=480,
        width=832,
        fps=15,
        observed_frames=77,
        generated_frames=76,
        steps=50,
        cfg=5.0,
        sigma_shift=5.0,
        correction_strength=8.0,
        sigma_max=0.92,
        seed=42,
        correction_seed=0,
        weights="merged DiT with camera-conditioning parameters",
    )
    (dest / "inference_config.json").write_text(json.dumps(runtime, indent=2) + "\n")
    card = f"""---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B
library_name: pytorch
tags:
- video-generation
- video-to-video
- world-model
- reentrymem
---

# {a.name}

Inference checkpoint for **Out of View, Still in Motion: Dynamic Entity Re-entry in Video World Models**.

Inference code: [Larry-077/ReentryMem](https://github.com/Larry-077/ReentryMem).

This bundle contains the merged BF16 DiT and camera-conditioning parameters at step {a.step}.
Variant: `{a.variant}`. It contains no optimizer state or training dataset.
The Wan2.2 text encoder, tokenizer and VAE are downloaded separately from the base model.

Use the accompanying ReentryMem inference repository. This is a custom camera-conditioned
model, not an unmodified Diffusers video pipeline. `localized` and `euler` use the same weights;
`localized` enables calibrated entity-region sampling. Default generation uses 77 observed
and 76 future RGB frames, 832x480, 15fps, 50 steps and CFG5. It requires no future RGB or masks.

This bundle uses the `{a.variant}` checkpoint for the paper's virtual-scene re-entry task.
This is the complete trained transformer with camera-conditioning parameters.
Use one entity phrase from the caption with the single-entity model.

Files are verified by `checksums.json`. See `LICENSE` and `NOTICE` for upstream attribution.
"""
    (dest / "README.md").write_text(card)
    manifest = {"schema_version": 1, "files": {}}
    for f in sorted(dest.iterdir()):
        if f.is_file() and f.name != "checksums.json":
            print(f"Hashing {f.name}", flush=True)
            manifest["files"][f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    (dest / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "bundle": str(dest),
                "weight_bytes": (dest / MODEL_NAME).stat().st_size,
                "tensor_count": len(header),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
