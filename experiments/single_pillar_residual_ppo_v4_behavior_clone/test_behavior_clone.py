import unittest

import torch
from torch import nn

from residual_networks import ResidualActor


class BehaviorCloneTest(unittest.TestCase):
    def test_short_fit_reduces_expert_loss(self):
        torch.manual_seed(7)
        actor = ResidualActor(16, 2)
        observations = torch.randn(32, 16)
        targets = torch.tensor([[1.0, -1.0]]).repeat(32, 1)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-3)
        initial = loss_fn(actor(observations), targets).item()
        for _ in range(100):
            loss = loss_fn(actor(observations), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final = loss_fn(actor(observations), targets).item()
        self.assertLess(final, initial * 0.05)


if __name__ == '__main__':
    unittest.main()
