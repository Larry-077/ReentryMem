import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from requests.exceptions import ConnectionError


spec = importlib.util.spec_from_file_location(
    "upload_weights", Path(__file__).resolve().parents[1] / "tools/upload_weights.py"
)
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class FakeHub:
    def __init__(self, folder, *, complete=False, private=True, reset=None, corrupt=False):
        self.folder = folder
        self.complete = complete
        self.private = private
        self.reset = reset
        self.corrupt = corrupt
        self.uploads = 0
        self.publishes = 0

    def whoami(self):
        return {"name": "release-test"}

    def create_repo(self, *args, **kwargs):
        pass

    def repo_info(self, *args, **kwargs):
        siblings = []
        if self.complete:
            for path in self.folder.iterdir():
                content = path.read_bytes()
                blob = hashlib.sha1(
                    b"blob " + str(len(content)).encode() + b"\0" + content
                ).hexdigest()
                large = path.suffix == ".safetensors"
                digest = "wrong" if self.corrupt else hashlib.sha256(content).hexdigest()
                siblings.append(
                    SimpleNamespace(
                        rfilename=path.name,
                        size=len(content),
                        blob_id=blob,
                        lfs=SimpleNamespace(sha256=digest) if large else None,
                    )
                )
        return SimpleNamespace(siblings=siblings, private=self.private, sha="test-commit")

    def upload_folder(self, **kwargs):
        self.uploads += 1
        if self.reset == "before_commit" and self.uploads == 1:
            raise ConnectionError("reset before commit")
        self.complete = True
        if self.reset == "after_commit" and self.uploads == 1:
            raise ConnectionError("commit accepted but response lost")
        return SimpleNamespace(oid="test-commit")

    def update_repo_settings(self, **kwargs):
        self.publishes += 1
        self.private = False


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.manifest = {"files": {}}
        for name, content in [
            ("diffusion_pytorch_model.safetensors", b"weights"),
            ("config.json", b"{}"),
        ]:
            (self.folder / name).write_bytes(content)
            self.manifest["files"][name] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        (self.folder / "checksums.json").write_text(json.dumps(self.manifest))
        self.sleep = patch.object(upload.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def run_upload(self, hub):
        return upload.upload_bundle(
            hub, "release-test/ReentryMem", self.folder, self.manifest, public=True
        )

    def test_reset_before_commit_retries_upload(self):
        hub = FakeHub(self.folder, reset="before_commit")
        self.run_upload(hub)
        self.assertEqual(hub.uploads, 2)
        self.assertEqual(hub.publishes, 1)

    def test_lost_commit_response_does_not_repeat_upload(self):
        hub = FakeHub(self.folder, reset="after_commit")
        self.run_upload(hub)
        self.assertEqual(hub.uploads, 1)
        self.assertEqual(hub.publishes, 1)

    def test_matching_public_repository_is_successful(self):
        hub = FakeHub(self.folder, complete=True, private=False)
        self.run_upload(hub)
        self.assertEqual(hub.uploads, 0)
        self.assertEqual(hub.publishes, 0)

    def test_wrong_remote_hash_prevents_publication(self):
        hub = FakeHub(self.folder, corrupt=True)
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            self.run_upload(hub)
        self.assertTrue(hub.private)
        self.assertEqual(hub.publishes, 0)


if __name__ == "__main__":
    unittest.main()
