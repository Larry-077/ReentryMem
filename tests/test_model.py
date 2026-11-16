import unittest
import warnings

import torch

from reentrymem.modules.model import ReentryMemModel
from reentrymem.regions import EntityAttentionReader


class ModelTests(unittest.TestCase):
    def model(self, mode="dense"):
        torch.manual_seed(17)
        model = ReentryMemModel(
            model_type="ti2v",
            dim=24,
            ffn_dim=48,
            freq_dim=8,
            text_dim=16,
            in_dim=4,
            out_dim=4,
            num_heads=2,
            num_layers=2,
            text_len=12,
            temporal_attention_mode=mode,
        ).eval()
        torch.nn.init.normal_(model.head.head.weight, std=0.1)
        model.add_camera_control_parameters(81)
        contexts = [torch.randn(3, 16) for _ in range(81)]
        model.precompute_cam_text_embeddings(lambda _: contexts, "cpu", torch.float32)
        return model

    def test_dense_and_causal_forward_with_region_hooks(self):
        for mode in ["dense", "frame_causal"]:
            with self.subTest(mode=mode):
                model = self.model(mode)
                video = [torch.randn(4, 3, 4, 4)]
                reader = EntityAttentionReader(
                    model,
                    [1],
                    frames=3,
                    tokens_per_frame=4,
                    head_sets={"shape": [[0, 0]], "presence": [[1, 1]]},
                )
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with reader:
                        result = model(
                            video,
                            torch.tensor([600.0]),
                            [torch.randn(4, 16)],
                            seq_len=12,
                            action_labels=torch.zeros(1, 3, dtype=torch.long),
                            cond_latent_frames=1,
                        )[0]
                self.assertEqual(result.shape, video[0].shape)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(reader.head_set_field("shape").shape, (12,))
                self.assertTrue(
                    all(not block.cross_attn._forward_pre_hooks for block in model.blocks)
                )

    def test_mixed_precision_preserves_float_time_and_output_heads(self):
        model = self.model().to(torch.bfloat16)
        model.time_embedding.float()
        model.time_projection.float()
        model.head.float()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = model(
                    [torch.randn(4, 3, 4, 4).bfloat16()],
                    torch.tensor([500.0]),
                    [torch.randn(4, 16).bfloat16()],
                    seq_len=12,
                    action_labels=torch.zeros(1, 3, dtype=torch.long),
                    cond_latent_frames=1,
                )[0]
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
