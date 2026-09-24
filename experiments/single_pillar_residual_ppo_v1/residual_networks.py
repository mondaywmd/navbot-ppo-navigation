"""Small symmetric policy head for bounded residual PPO."""

import numpy as np
import torch
from torch import nn


class ResidualActor(nn.Module):
    """Map the 16-value observation to two residual means in [-1, 1]."""

    def __init__(self, in_dim, out_dim, **_kwargs):
        super().__init__()
        if out_dim != 2:
            raise ValueError("ResidualActor requires exactly two outputs")
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
            nn.Tanh(),
        )

    def forward(self, observation, vision_feat=None):
        del vision_feat
        device = next(self.parameters()).device
        if isinstance(observation, np.ndarray):
            observation = torch.as_tensor(observation, dtype=torch.float32, device=device)
        else:
            observation = observation.to(device)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        return self.network(observation)
