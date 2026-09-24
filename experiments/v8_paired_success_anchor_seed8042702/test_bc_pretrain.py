import numpy as np
import torch

from bc_pretrain import losses, pair_split, state_digest
from residual_networks import ResidualActor


def main():
    levels = np.repeat([1, 2, 3], 16)
    pair_ids = np.tile(np.repeat(np.arange(8), 2), 3)
    train, val, keys = pair_split(levels, pair_ids)
    assert train.sum() == 36 and val.sum() == 12 and len(keys) == 6
    assert not set(zip(levels[train], pair_ids[train])) & set(zip(levels[val], pair_ids[val]))
    actor = ResidualActor(16, 2); before = state_digest(actor)
    obs = torch.rand(32, 16); action = torch.rand(32, 2) * 2 - 1
    teacher = torch.rand(32, 2) * 2 - 1; keep = torch.arange(32) % 3 == 0
    values = losses(actor, obs, action, teacher, keep)
    assert all(torch.isfinite(value).all() for value in values[:4])
    values[0].backward(); assert state_digest(actor) == before
    print("BC_PRETRAIN_TEST_PASS")


if __name__ == "__main__":
    main()
