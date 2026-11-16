#!/usr/bin/env python3
"""Download only the required upstream assets and a separate ReentryMem bundle."""

import argparse
from huggingface_hub import snapshot_download
from reentrymem.checkpoints import verify_bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="pretrained/Wan2.2-TI2V-5B")
    p.add_argument("--base-revision", default="main")
    p.add_argument(
        "--repo-id", help="Your published ReentryMem model repo; omit for base assets only"
    )
    p.add_argument("--revision", default="main", help="Model revision or pinned commit")
    p.add_argument("--checkpoint-dir", default="checkpoints/ReentryMem")
    p.add_argument("--skip-base", action="store_true")
    a = p.parse_args()
    if not a.skip_base:
        snapshot_download(
            "Wan-AI/Wan2.2-TI2V-5B",
            revision=a.base_revision,
            local_dir=a.base_dir,
            allow_patterns=[
                "Wan2.2_VAE.pth",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "google/umt5-xxl/*",
                "config.json",
            ],
        )
        print(f"Base assets ready: {a.base_dir}")
    if a.repo_id:
        snapshot_download(
            a.repo_id,
            revision=a.revision,
            local_dir=a.checkpoint_dir,
            allow_patterns=[
                "*.safetensors",
                "config.json",
                "inference_config.json",
                "entity_regions.json",
                "checksums.json",
                "README.md",
                "LICENSE",
                "NOTICE",
            ],
        )
        verify_bundle(a.checkpoint_dir)
        print(f"Checkpoint downloaded and verified: {a.checkpoint_dir}")


if __name__ == "__main__":
    main()
