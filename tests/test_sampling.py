import unittest
import torch
from reentrymem.sampling import overshoot_update


class SamplingTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.x = torch.randn(1, 4, 5, 2, 2, generator=generator)
        self.v = torch.randn(1, 4, 5, 2, 2, generator=generator)
        self.noise = torch.randn(1, 4, 5, 2, 2, generator=generator)

    def test_zero_support_is_exact_euler(self):
        out = overshoot_update(
            self.x, self.v, 0.7, 0.6, gain=torch.zeros(1, 1, 5, 2, 2), c=8, noise=self.noise
        )
        torch.testing.assert_close(out, self.x + self.v * (0.6 - 0.7), rtol=0, atol=0)

    def test_only_supported_elements_change(self):
        mask = torch.zeros(1, 1, 5, 2, 2)
        mask[:, :, 3:] = 1
        out = overshoot_update(self.x, self.v, 0.7, 0.6, gain=mask, c=8, noise=self.noise)
        torch.testing.assert_close(
            out[:, :, :3], (self.x + self.v * (0.6 - 0.7))[:, :, :3], rtol=0, atol=0
        )
        self.assertFalse(torch.equal(out[:, :, 3:], (self.x + self.v * (0.6 - 0.7))[:, :, 3:]))

    def test_last_step_is_euler(self):
        out = overshoot_update(
            self.x, self.v, 0.1, 0, gain=torch.ones(1, 1, 5, 2, 2), c=8, noise=self.noise
        )
        torch.testing.assert_close(out, self.x - 0.1 * self.v, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
