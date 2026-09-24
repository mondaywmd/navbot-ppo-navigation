"""Scratch BC for the exactly mirror-equivariant V8 Actor."""
import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from bc_pretrain import pair_split, state_digest
from symmetric_actor import MirrorEquivariantActor


SEED = 8042711


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise RuntimeError("refusing overwrite: " + args.output_dir)
    torch.manual_seed(SEED); np.random.seed(SEED)
    data = np.load(args.dataset)
    train_mask, val_mask, validation = pair_split(data["levels"], data["pair_ids"])
    observations = torch.from_numpy(data["observations"])
    actions = torch.from_numpy(data["actions"])
    train = torch.from_numpy(train_mask); val = torch.from_numpy(val_mask)
    actor = MirrorEquivariantActor(16, 2); initial_digest = state_digest(actor)
    initial_val = float(nn.functional.mse_loss(actor(observations[val]), actions[val]))
    loss = nn.functional.mse_loss(actor(observations[train]), actions[train])
    loss.backward()
    if state_digest(actor) != initial_digest:
        raise AssertionError("backward without optimizer changed Actor")
    if args.smoke_only:
        print("SCRATCH_BC_ZERO_UPDATE_PASS train=%d val=%d loss=%.6f digest=%s" %
              (int(train.sum()), int(val.sum()), float(loss), initial_digest), flush=True)
        return
    os.makedirs(args.output_dir)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    best_loss = float("inf"); best_epoch = 0; best_state = None; history = []
    for epoch in range(1, args.epochs + 1):
        actor.train(); prediction = actor(observations[train])
        train_loss = nn.functional.mse_loss(prediction, actions[train])
        optimizer.zero_grad(); train_loss.backward(); optimizer.step()
        actor.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(actor(observations[val]), actions[val])
        if float(val_loss) < best_loss:
            best_loss = float(val_loss); best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        if epoch == 1 or epoch % 40 == 0:
            row = {"epoch": epoch, "train_mse": float(train_loss),
                   "validation_mse": float(val_loss)}
            history.append(row); print("V8_SCRATCH_BC " + json.dumps(row), flush=True)
    actor.load_state_dict(best_state)
    checkpoint = os.path.join(args.output_dir, "actor_bc_best.pth")
    torch.save(actor.state_dict(), checkpoint)
    summary = {"seed": SEED, "initialization": "random", "epochs": args.epochs,
               "best_epoch": best_epoch, "initial_validation_mse": initial_val,
               "best_validation_mse": best_loss, "train_samples": int(train.sum()),
               "validation_samples": int(val.sum()),
               "validation_pairs": sorted([list(x) for x in validation]),
               "exact_mirror_architecture": True, "v5_used": False,
               "ppo_started": False, "permanent_ood_accessed": False,
               "checkpoint": checkpoint, "actor_digest": state_digest(actor),
               "history": history}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    print("V8_SCRATCH_BC_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
