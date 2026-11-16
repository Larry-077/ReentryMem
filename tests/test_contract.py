import json
from pathlib import Path
import tempfile
import unittest
from reentrymem.case import load_case, stable_seed, subject_indices


class ContractTests(unittest.TestCase):
    def fixture(self, root, **updates):
        (root / "prefix.mp4").write_bytes(b"video-existence-fixture")
        data = dict(
            sample_id="example",
            condition_video="prefix.mp4",
            prompt="A swan returns.",
            subjects=["swan"],
            actions=[0] * 39,
        )
        data.update(updates)
        p = root / "case.json"
        p.write_text(json.dumps(data))
        return p

    def test_relative_paths_and_control_length(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = load_case(self.fixture(root))
            self.assertEqual(c.condition_video, root / "prefix.mp4")
            self.assertEqual(len(c.actions), 39)

    def test_reject_short_actions(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_case(self.fixture(Path(d), actions=[0] * 38))

    def test_reject_action_outside_label_space(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_case(self.fixture(Path(d), actions=[81] * 39))

    def test_seed_matches_experiment_rule(self):
        self.assertEqual(
            stable_seed("example", 42),
            int(__import__("hashlib").sha256(b"42:example").hexdigest()[:8], 16),
        )

    def test_phrase_offsets_and_all_mentions(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                import re

                return {
                    "offset_mapping": [(0, 0)]
                    + [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
                    + [(0, 0)]
                }

        self.assertEqual(subject_indices("A girl sees another girl", "girl", Tokenizer()), [2, 5])

    def test_absent_phrase_fails(self):
        with self.assertRaises(ValueError):
            subject_indices("A swan returns.", "girl", None)


if __name__ == "__main__":
    unittest.main()
