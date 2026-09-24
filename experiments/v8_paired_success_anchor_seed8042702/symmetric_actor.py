"""Actor with exact left/right equivariance enforced by construction."""
import torch

from mirror import mirror_action, mirror_observation
from residual_networks import ResidualActor


class MirrorEquivariantActor(ResidualActor):
    def forward(self, observation, vision_feat=None):
        del vision_feat
        device = next(self.parameters()).device
        if not isinstance(observation, torch.Tensor):
            observation = torch.as_tensor(observation, dtype=torch.float32,
                                          device=device)
        else:
            observation = observation.to(device)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        direct = self.network(observation)
        reflected = mirror_action(self.network(mirror_observation(observation)))
        return 0.5 * (direct + reflected)
