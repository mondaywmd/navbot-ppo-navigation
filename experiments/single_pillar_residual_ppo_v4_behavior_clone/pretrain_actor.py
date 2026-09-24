"""Offline behavior cloning of the successful residual expert."""

import argparse

import numpy as np
import torch
from torch import nn

from residual_networks import ResidualActor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=1200)
    args = parser.parse_args()
    torch.manual_seed(7)
    data = np.load(args.dataset)
    observations = torch.from_numpy(data['observations'])
    actions = torch.from_numpy(data['actions'])
    actor = ResidualActor(16, 2)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        initial_loss = loss_fn(actor(observations), actions).item()
    for epoch in range(1, args.epochs + 1):
        prediction = actor(observations)
        loss = loss_fn(prediction, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 200 == 0:
            print('V4_BC epoch=%d loss=%.8f' % (epoch, loss.item()), flush=True)

    actor.eval()
    with torch.no_grad():
        prediction = actor(observations)
        final_loss = loss_fn(prediction, actions).item()
        mean_action = prediction.mean(dim=0).numpy()
        minimum = prediction.min(dim=0).values.numpy()
        maximum = prediction.max(dim=0).values.numpy()
    if not final_loss < initial_loss * 0.01:
        raise AssertionError('behavior-cloning loss did not decrease enough')
    torch.save(actor.state_dict(), args.output)
    print(
        'V4_BC_SAVED initial_loss=%.8f final_loss=%.8f mean=(%.5f,%.5f) '
        'range0=[%.5f,%.5f] range1=[%.5f,%.5f] path=%s'
        % (initial_loss, final_loss, mean_action[0], mean_action[1],
           minimum[0], maximum[0], minimum[1], maximum[1], args.output),
        flush=True,
    )


if __name__ == '__main__':
    main()
