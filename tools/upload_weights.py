#!/usr/bin/env python3
"""Upload a verified bundle to Hugging Face; credentials use normal HF login/cache."""

import argparse
from pathlib import Path
from huggingface_hub import HfApi
from reentrymem.checkpoints import verify_bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--repo-id", required=True, help="owner/ReentryMem-Wan2.2-5B")
    p.add_argument(
        "--public",
        action="store_true",
        help="Make the model public after the verified files have uploaded",
    )
    a = p.parse_args()
    folder = Path(a.bundle).resolve()
    manifest = verify_bundle(folder)
    api = HfApi()
    api.whoami()
    api.create_repo(a.repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(a.repo_id, repo_type="model").private:
        raise RuntimeError("Staged upload requires a new or private model repository")
    kwargs = dict(
        repo_id=a.repo_id,
        repo_type="model",
        folder_path=folder,
        allow_patterns=list(manifest["files"]) + ["checksums.json"],
    )
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(**kwargs)
    else:
        api.upload_folder(**kwargs)
    remote = set(api.list_repo_files(a.repo_id, repo_type="model"))
    needed = set(manifest["files"]) | {"checksums.json"}
    if not needed <= remote:
        raise RuntimeError("Upload is incomplete; rerun the same command to resume")
    if a.public:
        if hasattr(api, "update_repo_settings"):
            api.update_repo_settings(repo_id=a.repo_id, repo_type="model", private=False)
        else:
            api.update_repo_visibility(repo_id=a.repo_id, repo_type="model", private=False)
    print(f"https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
