#!/usr/bin/env python3
"""Export a portable model folder; no optimizer, scheduler or training data is copied."""

import argparse
import json
import os
from pathlib import Path
import shutil
from string import Template
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
    card = Template((release / "docs/MODEL_CARD.md").read_text()).substitute(
        model_name=a.name, variant=a.variant, checkpoint_step=a.step
    )
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
