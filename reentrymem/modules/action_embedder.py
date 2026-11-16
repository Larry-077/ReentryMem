import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    orig_shape = position.shape
    position_flat = position.reshape(-1).type(torch.float64)
    freqs = torch.pow(
        10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)
    )
    sinusoid = torch.outer(position_flat, freqs)
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)
    x = x.reshape(*orig_shape, dim)
    return x.to(position.dtype)


class CameraActionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256, num_actions: int = 81):
        super().__init__()
        self.freq_dim = freq_dim
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        nn.init.zeros_(self.mlp[2].weight)
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, action_labels: torch.Tensor) -> torch.Tensor:
        action_emb = sinusoidal_embedding_1d(self.freq_dim, action_labels.float())
        return self.mlp(action_emb.to(self.mlp[0].weight.dtype))
