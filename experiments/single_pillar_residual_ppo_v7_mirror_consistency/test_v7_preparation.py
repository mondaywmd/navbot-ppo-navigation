import unittest
import torch
from torch import nn
from consistency import actor_mirror_consistency_loss
from mirror import mirror_action, mirror_observation
from paired_scenes import generate


class SymmetricActor(nn.Module):
    def forward(self, observation):
        return torch.stack((observation[:, 12], observation[:, 13]), dim=1)


class PreparationTests(unittest.TestCase):
    def test_observation_involution_and_shape(self):
        obs = torch.rand(31, 16) * 2 - 1
        self.assertEqual(tuple(mirror_observation(obs).shape), (31, 16))
        self.assertTrue(torch.equal(mirror_observation(mirror_observation(obs)), obs))

    def test_action_involution(self):
        action = torch.rand(31, 2) * 2 - 1
        self.assertTrue(torch.equal(mirror_action(mirror_action(action)), action))

    def test_exact_scene_pairs(self):
        scenes = generate()
        self.assertEqual(len(scenes), 48)
        for positive, negative in zip(scenes[0::2], scenes[1::2]):
            self.assertEqual(positive["pair"], negative["pair"])
            self.assertEqual(positive["pillar_x"], negative["pillar_x"])
            self.assertEqual(positive["target_x"], negative["target_x"])
            self.assertEqual(positive["pillar_y"], -negative["pillar_y"])
            self.assertEqual(positive["target_y"], -negative["target_y"])
            self.assertEqual(positive["robot_yaw_deg"], -negative["robot_yaw_deg"])

    def test_zero_loss_for_equivariant_actor(self):
        obs = torch.rand(32, 16) * 2 - 1
        loss, parts = actor_mirror_consistency_loss(SymmetricActor(), obs)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(parts["linear_mse"]), 0.0)
        self.assertEqual(float(parts["angular_mse"]), 0.0)

    def test_finite_gradient(self):
        actor = nn.Sequential(nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 2), nn.Tanh())
        loss, _ = actor_mirror_consistency_loss(actor, torch.rand(32, 16))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in actor.parameters()))


if __name__ == "__main__":
    unittest.main()

