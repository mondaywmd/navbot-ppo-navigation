import torch

from mirror import mirror_action, mirror_observation
from symmetric_actor import MirrorEquivariantActor


def main():
    torch.manual_seed(8042711)
    actor = MirrorEquivariantActor(16, 2)
    observations = torch.rand(1000, 16) * 2 - 1
    outputs = actor(observations)
    mirrored = actor(mirror_observation(observations))
    error = (mirrored - mirror_action(outputs)).abs().max().item()
    assert outputs.shape == (1000, 2)
    assert error < 1e-6
    assert outputs.abs().max().item() <= 1.0
    print("SYMMETRIC_ACTOR_TEST_PASS max_error=%.9f" % error)


if __name__ == "__main__":
    main()
