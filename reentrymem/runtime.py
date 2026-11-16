from pathlib import Path
import torch
from safetensors.torch import load_file
from .checkpoints import MODEL_NAME, preflight
from .constants import WAN_LATENT_AVG, WAN_LATENT_STDDEV


def load_dit(checkpoint, base_model, device):
    from .modules.model import ReentryMemModel, rope_params

    report = preflight(checkpoint, base_model)
    config = report["config"]
    with torch.device("meta"):
        model = ReentryMemModel(**config)
        model.add_camera_control_parameters(81)
    state = load_file(str(Path(checkpoint) / MODEL_NAME), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    d = config["dim"] // config["num_heads"]
    with torch.device("cpu"):
        model.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
    model = model.to(device=device, dtype=torch.bfloat16)
    model.time_embedding.float()
    model.time_projection.float()
    model.head.float()
    return model.eval().requires_grad_(False), report


def load_text_encoder(base_model, device):
    from .modules.t5 import T5EncoderModel

    return T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(Path(base_model) / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(Path(base_model) / "google/umt5-xxl"),
    )


class VAE:
    def __init__(self, base_model, device):
        from .modules.vae2_2 import _video_vae

        self.model = _video_vae(
            pretrained_path=str(Path(base_model) / "Wan2.2_VAE.pth"),
            z_dim=48,
            dim=160,
            temperal_downsample=[False, True, True],
        )
        self.model = self.model.eval().requires_grad_(False).to(device)
        self.device = device
        mean = torch.tensor(WAN_LATENT_AVG, dtype=torch.bfloat16, device=device)
        std = torch.tensor(WAN_LATENT_STDDEV, dtype=torch.bfloat16, device=device)
        self.scale = [mean, 1 / std]

    @torch.no_grad()
    def encode(self, pixels):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.encode(pixels.unsqueeze(0), self.scale).float().squeeze(0)

    @torch.no_grad()
    def decode(self, latent):
        self.model.clear_cache()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rgb = self.model.decode(latent.unsqueeze(0), self.scale).float().clamp(-1, 1).squeeze(0)
        return ((rgb + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
