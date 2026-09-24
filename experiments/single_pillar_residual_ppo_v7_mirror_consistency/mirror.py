"""Exact left/right transforms derived from the established 16-D contract."""
import torch


OBS_INDEX = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                          10, 11, 12, 13, 14, 15])
OBS_SIGN = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, -1, 1, -1, 1, -1], dtype=torch.float32)
ACTION_SIGN = torch.tensor([1, -1], dtype=torch.float32)


def mirror_observation(observation):
    if observation.shape[-1] != 16:
        raise ValueError("observation last dimension must be 16")
    index = OBS_INDEX.to(observation.device)
    sign = OBS_SIGN.to(device=observation.device, dtype=observation.dtype)
    return observation.index_select(-1, index) * sign


def mirror_action(action):
    if action.shape[-1] != 2:
        raise ValueError("action last dimension must be 2")
    return action * ACTION_SIGN.to(device=action.device, dtype=action.dtype)

