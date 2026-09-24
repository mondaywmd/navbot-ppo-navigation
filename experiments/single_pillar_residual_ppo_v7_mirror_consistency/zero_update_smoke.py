"""Forward/backward smoke that proves the loaded Actor receives no update."""
import argparse
import hashlib
import torch
from consistency import actor_mirror_consistency_loss
from residual_networks import ResidualActor


def digest(actor):
    value = hashlib.sha256()
    for name, tensor in actor.state_dict().items():
        value.update(name.encode("utf-8"))
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.manual_seed(7042604)
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    before_digest = digest(actor)
    observations = torch.rand(64, 16) * 2 - 1
    loss, parts = actor_mirror_consistency_loss(actor, observations)
    loss.backward()  # Gradient plumbing only: no optimizer exists and no step is called.
    unchanged = all(torch.equal(before[name], value)
                    for name, value in actor.state_dict().items())
    after_digest = digest(actor)
    if not unchanged or before_digest != after_digest:
        raise AssertionError("Actor parameters changed during zero-update smoke")
    print("ZERO_UPDATE_SMOKE passed loss=%.6f linear=%.6f angular=%.6f sha256=%s" %
          (float(loss.detach()), float(parts["linear_mse"]),
           float(parts["angular_mse"]), after_digest))


if __name__ == "__main__":
    main()
