"""Clean-start 78-D temporal observation and exact mirror transform."""
import math
import numpy as np
import torch

DIM = 78

def build_observation(current_sectors, signed_rates, previous_action,
                      goal_distance, goal_heading, max_range=3.5,
                      distance_scale=10.75, rate_scale=1.0):
    current = np.asarray(current_sectors, dtype=np.float32)
    rates = np.asarray(signed_rates, dtype=np.float32)
    action = np.asarray(previous_action, dtype=np.float32)
    if current.shape != (36,) or rates.shape != (36,) or action.shape != (2,):
        raise ValueError("expected 36 ranges, 36 rates, and 2 previous actions")
    heading = (float(goal_heading) + math.pi) % (2 * math.pi) - math.pi
    result = np.concatenate((np.clip(current / max_range, 0, 1),
                             np.clip(rates / rate_scale, -1, 1), action,
                             np.asarray([goal_distance / distance_scale,
                                         math.sin(heading), math.cos(heading),
                                         heading / math.pi], dtype=np.float32)))
    if result.shape != (DIM,) or not np.isfinite(result).all():
        raise AssertionError("invalid temporal Actor observation")
    return result.astype(np.float32)

def mirror_observation(observation):
    if observation.shape[-1] != DIM:
        raise ValueError("observation last dimension must be 78")
    index = list(range(35, -1, -1)) + list(range(71, 35, -1)) + list(range(72, 78))
    signs = [1.0] * DIM
    for position in (73, 75, 77): signs[position] = -1.0
    if isinstance(observation, torch.Tensor):
        return observation.index_select(-1, torch.tensor(index, device=observation.device)) * torch.tensor(signs, dtype=observation.dtype, device=observation.device)
    return np.asarray(observation)[..., index] * np.asarray(signs, dtype=np.float32)

def mirror_action(action):
    signs = torch.tensor([1.0, -1.0], dtype=action.dtype, device=action.device) if isinstance(action, torch.Tensor) else np.asarray([1.0, -1.0], dtype=np.float32)
    return action * signs
