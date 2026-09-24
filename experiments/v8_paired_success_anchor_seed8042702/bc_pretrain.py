"""Short V5-initialized BC warm-up on complete paired successful routes."""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from torch import nn

from consistency import actor_mirror_consistency_loss
from residual_networks import ResidualActor


SEED = 8042709
EXPERT_WEIGHT = 1.0
ANCHOR_WEIGHT = 0.5
MIRROR_WEIGHT = 0.01


def state_digest(actor):
    digest = hashlib.sha256()
    for key, value in sorted(actor.state_dict().items()):
        digest.update(key.encode("utf-8")); digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def pair_split(levels, pair_ids):
    keys = sorted(set(zip(levels.tolist(), pair_ids.tolist())))
    validation = {(level, pair_id) for level in (1, 2, 3)
                  for pair_id in (6, 7)}
    if len(keys) != 24 or len(validation) != 6 or not validation.issubset(keys):
        raise AssertionError("unexpected composite pair inventory")
    val = np.asarray([(int(level), int(pair_id)) in validation
                      for level, pair_id in zip(levels, pair_ids)], dtype=bool)
    return ~val, val, validation


def losses(actor, observations, actions, teacher, keep):
    prediction = actor(observations)
    expert = nn.functional.mse_loss(prediction, actions)
    anchor = (nn.functional.mse_loss(prediction[keep], teacher[keep])
              if bool(keep.any()) else prediction.sum() * 0.0)
    mirror, parts = actor_mirror_consistency_loss(actor, observations)
    total = EXPERT_WEIGHT * expert + ANCHOR_WEIGHT * anchor + MIRROR_WEIGHT * mirror
    return total, expert, anchor, mirror, parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--v5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise RuntimeError("refusing overwrite: " + args.output_dir)
    torch.manual_seed(SEED); np.random.seed(SEED)
    data = np.load(args.dataset)
    train_mask, val_mask, validation = pair_split(data["levels"], data["pair_ids"])
    tensors = {name: torch.from_numpy(data[name]) for name in
               ("observations", "actions", "v5_teacher_actions", "keep_v5_mask")}
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(args.v5, map_location="cpu"))
    initial_digest = state_digest(actor)

    def subset(mask):
        m = torch.from_numpy(mask)
        return (tensors["observations"][m], tensors["actions"][m],
                tensors["v5_teacher_actions"][m], tensors["keep_v5_mask"][m].bool())
    train = subset(train_mask); val = subset(val_mask)
    with torch.no_grad():
        initial = losses(actor, *val)
    if args.smoke_only:
        trial = losses(actor, *train)
        trial[0].backward()
        if state_digest(actor) != initial_digest:
            raise AssertionError("zero-update smoke changed Actor state")
        print("BC_ZERO_UPDATE_PASS train=%d val=%d digest=%s loss=%.6f" %
              (len(train[0]), len(val[0]), initial_digest, trial[0].item()), flush=True)
        return

    os.makedirs(args.output_dir)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    best_score = float("inf"); best_epoch = 0; best_state = None; history = []
    for epoch in range(1, args.epochs + 1):
        actor.train(); total, expert, anchor, mirror, _ = losses(actor, *train)
        optimizer.zero_grad(); total.backward(); optimizer.step()
        actor.eval()
        with torch.no_grad():
            vt, ve, va, vm, vp = losses(actor, *val)
        score = float(vt)
        if score < best_score:
            best_score = score; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            row = {"epoch": epoch, "train_total": float(total),
                   "train_expert": float(expert), "train_anchor": float(anchor),
                   "train_mirror": float(mirror), "val_total": float(vt),
                   "val_expert": float(ve), "val_anchor": float(va),
                   "val_mirror": float(vm)}
            history.append(row); print("V8_BC " + json.dumps(row, sort_keys=True), flush=True)
    actor.load_state_dict(best_state)
    checkpoint = os.path.join(args.output_dir, "actor_bc_best.pth")
    torch.save(actor.state_dict(), checkpoint)
    summary = {"seed": SEED, "epochs": args.epochs, "best_epoch": best_epoch,
               "best_validation_total": best_score, "train_samples": int(train_mask.sum()),
               "validation_samples": int(val_mask.sum()),
               "validation_pairs": sorted([list(x) for x in validation]),
               "weights": {"expert": EXPERT_WEIGHT, "anchor": ANCHOR_WEIGHT,
                           "mirror": MIRROR_WEIGHT}, "v5_digest": initial_digest,
               "bc_digest": state_digest(actor), "checkpoint": checkpoint,
               "ppo_started": False, "permanent_ood_accessed": False,
               "history": history}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    print("V8_BC_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
