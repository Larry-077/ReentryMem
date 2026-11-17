#!/usr/bin/env python3
"""Upload a verified bundle to Hugging Face; credentials use normal HF login/cache."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from requests.exceptions import ConnectionError, Timeout
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from reentrymem.checkpoints import verify_bundle

NETWORK_ERRORS = (ConnectionError, Timeout)
try:
    import httpx
except ImportError:
    pass
else:
    NETWORK_ERRORS += (httpx.TransportError,)


def retry_call(label, operation, *, attempts=5):
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            transient = isinstance(error, NETWORK_ERRORS) or (
                isinstance(error, HfHubHTTPError) and status in {408, 429, 500, 502, 503, 504}
            )
            if not transient or attempt == attempts:
                raise
            delay = min(2**attempt, 30)
            print(
                f"{label}: {type(error).__name__}; retrying in {delay}s "
                f"({attempt}/{attempts - 1}).",
                flush=True,
            )
            time.sleep(delay)


def remote_mismatches(info, folder, manifest):
    remote = {item.rfilename: item for item in info.siblings}
    mismatches = []
    for name in list(manifest["files"]) + ["checksums.json"]:
        path = folder / name
        item = remote.get(name)
        if item is None or item.size != path.stat().st_size:
            mismatches.append(name)
            continue
        if item.lfs is not None:
            expected = manifest["files"].get(name, {}).get("sha256")
            actual = (
                item.lfs.get("sha256")
                if isinstance(item.lfs, dict)
                else getattr(item.lfs, "sha256", None)
            )
            if expected is None or actual != expected:
                mismatches.append(name)
        elif path.stat().st_size > 10 * 1024**2:
            mismatches.append(name)
        else:
            content = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if item.blob_id != blob:
                mismatches.append(name)
    return mismatches


def upload_bundle(api, repo_id, folder, manifest, *, public=False):
    account = retry_call("Login check", api.whoami)
    print(f"Authenticated as {account['name']}. Preparing {repo_id}...", flush=True)
    retry_call(
        "Repository setup",
        lambda: api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True),
    )

    def info(revision=None):
        return api.repo_info(
            repo_id, repo_type="model", revision=revision, files_metadata=True, timeout=30
        )

    def upload_if_needed():
        current = info()
        mismatches = remote_mismatches(current, folder, manifest)
        if not mismatches:
            print("Remote bundle already matches; no upload is needed.", flush=True)
            return current
        if not current.private:
            raise RuntimeError("Existing public repository differs from this bundle")
        print(
            "Uploading missing or changed files. Previously transferred chunks can be reused...",
            flush=True,
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder,
            allow_patterns=list(manifest["files"]) + ["checksums.json"],
        )
        print("Upload returned. Verifying the committed files...", flush=True)
        current = info(commit.oid)
        mismatches = remote_mismatches(current, folder, manifest)
        if mismatches:
            raise RuntimeError(f"Remote file verification failed: {', '.join(mismatches)}")
        return current

    current = retry_call("Upload/commit", upload_if_needed)
    if public and current.private:
        print("Remote hashes match. Making the model public...", flush=True)

        def publish():
            if hasattr(api, "update_repo_settings"):
                return api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)
            return api.update_repo_visibility(repo_id=repo_id, repo_type="model", private=False)

        retry_call("Publish", publish)
    final = retry_call("Final verification", info)
    mismatches = remote_mismatches(final, folder, manifest)
    if mismatches or (public and final.private):
        raise RuntimeError("The remote bundle or repository visibility is not ready")
    print(f"Verified commit: {final.sha}", flush=True)
    print(f"https://huggingface.co/{repo_id}", flush=True)
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--repo-id", required=True, help="owner/ReentryMem")
    p.add_argument("--dry-run", action="store_true", help="Verify the bundle without uploading")
    p.add_argument(
        "--public",
        action="store_true",
        help="Make the model public after the verified files have uploaded",
    )
    a = p.parse_args()
    folder = Path(a.bundle).resolve()
    if not a.dry_run:
        print(f"Verifying local model bundle: {folder}", flush=True)
        print("Reading the complete checkpoint for SHA256 verification before upload.", flush=True)
    manifest = verify_bundle(folder)
    if a.dry_run:
        print(
            json.dumps(
                dict(
                    bundle=str(folder),
                    repo_id=a.repo_id,
                    files=sorted(manifest["files"]),
                    total_bytes=sum(item["bytes"] for item in manifest["files"].values()),
                    uploaded=False,
                ),
                indent=2,
            )
        )
        return
    print("Local checksums passed. Checking Hugging Face login...", flush=True)
    upload_bundle(HfApi(), a.repo_id, folder, manifest, public=a.public)


if __name__ == "__main__":
    main()
