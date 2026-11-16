from pathlib import Path
import tempfile
import unittest

import numpy as np

from reentrymem.video import read_condition, write_video


class VideoTests(unittest.TestCase):
    def test_observed_frame_count_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observed.mp4"
            frames = np.zeros((77, 32, 48, 3), dtype=np.uint8)
            frames[..., 1] = np.arange(77, dtype=np.uint8)[:, None, None]
            write_video(path, frames)
            decoded = read_condition(path, width=48, height=32)
            self.assertEqual(decoded.shape, frames.shape)
            self.assertLess(float(np.abs(decoded.astype(float) - frames).mean()), 3)

    def test_short_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "short.mp4"
            write_video(path, np.zeros((3, 32, 48, 3), dtype=np.uint8))
            with self.assertRaises(ValueError):
                read_condition(path, width=48, height=32)


if __name__ == "__main__":
    unittest.main()
