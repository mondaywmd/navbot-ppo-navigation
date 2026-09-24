"""Differentiable Actor mirror-equivariance objective; no optimizer included."""
import torch
from mirror import mirror_action, mirror_observation


def actor_mirror_consistency_loss(actor, observations):
    if observations.ndim != 2 or observations.shape[1] != 16:
        raise ValueError("observations must have shape (batch, 16)")
    original = actor(observations)
    mirrored_prediction = actor(mirror_observation(observations))
    target_relation = mirror_action(original)
    component_mse = (mirrored_prediction - target_relation).pow(2).mean(dim=0)
    return component_mse.sum(), {
        "linear_mse": component_mse[0].detach(),
        "angular_mse": component_mse[1].detach(),
    }

