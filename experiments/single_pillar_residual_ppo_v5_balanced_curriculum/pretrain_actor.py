"""Direction-balanced behavior cloning for the independent V5 Actor."""

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from residual_networks import ResidualActor


def validate_training_dataset(data):
    required = {'observations', 'actions', 'direction_ids', 'episode_ids',
                'pillar_y', 'dataset_role'}
    missing = required.difference(data.files)
    if missing:
        raise ValueError('training dataset missing fields: %s' % sorted(missing))
    role = str(data['dataset_role'].item())
    if role != 'v5_training_demonstration':
        raise ValueError('dataset is not approved for V5 training: %s' % role)
    pillar_y = np.asarray(data['pillar_y'])
    if np.any(np.abs(pillar_y) >= 0.10 - 1e-7):
        raise ValueError('held-out pillar offset found in V5 training dataset')


def balanced_loss(prediction, targets, direction_ids, loss_fn):
    direction_losses = [
        loss_fn(prediction[direction_ids == direction], targets[direction_ids == direction])
        for direction in torch.unique(direction_ids)
    ]
    return torch.stack(direction_losses).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--epochs', type=int, default=1200)
    args = parser.parse_args()
    for path in (args.output, args.metrics):
        if os.path.exists(path):
            raise RuntimeError('refusing to overwrite: %s' % path)
    torch.manual_seed(17)
    data = np.load(args.dataset)
    validate_training_dataset(data)
    observations = torch.from_numpy(data['observations'])
    actions = torch.from_numpy(data['actions'])
    direction_ids = torch.from_numpy(data['direction_ids'])
    actor = ResidualActor(16, 2)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        initial_loss = balanced_loss(actor(observations), actions, direction_ids, loss_fn).item()
    history = []
    for epoch in range(1, args.epochs + 1):
        prediction = actor(observations)
        loss = balanced_loss(prediction, actions, direction_ids, loss_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0:
            history.append({'epoch': epoch, 'loss': float(loss.item())})
            print('V5_BC epoch=%d balanced_loss=%.8f' % (epoch, loss.item()), flush=True)
    actor.eval()
    with torch.no_grad():
        prediction = actor(observations)
        final_loss = balanced_loss(prediction, actions, direction_ids, loss_fn).item()
        per_direction = {}
        for direction in torch.unique(direction_ids).tolist():
            mask = direction_ids == direction
            values = prediction[mask]
            per_direction[str(direction)] = {
                'samples': int(mask.sum()),
                'loss': float(loss_fn(values, actions[mask]).item()),
                'prediction_min': values.min(dim=0).values.tolist(),
                'prediction_max': values.max(dim=0).values.tolist(),
                'prediction_mean': values.mean(dim=0).tolist(),
            }
    if not final_loss < initial_loss * 0.02:
        raise AssertionError('balanced BC loss did not decrease enough')
    torch.save(actor.state_dict(), args.output)
    metrics = {'seed': 17, 'epochs': args.epochs, 'initial_loss': initial_loss,
               'final_loss': final_loss, 'history': history,
               'per_direction': per_direction}
    with open(args.metrics, 'w') as output:
        json.dump(metrics, output, indent=2, sort_keys=True)
    print('V5_BC_SAVED initial=%.8f final=%.8f checkpoint=%s' % (
        initial_loss, final_loss, args.output
    ), flush=True)


if __name__ == '__main__':
    main()
