import unittest

import numpy as np
import torch
from torch import nn

from curriculum import HELD_OUT_LIMITS, LEVELS, seeded_scenarios
from pretrain_actor import balanced_loss, validate_training_dataset
from train_from_v4 import DEFAULT_V4_ACTOR


class CurriculumTest(unittest.TestCase):
    def test_training_limits_are_strictly_inside_heldout(self):
        for level in LEVELS:
            self.assertLess(level.pillar_y_limit, HELD_OUT_LIMITS['pillar_y'])
            self.assertLess(level.robot_yaw_deg_limit, HELD_OUT_LIMITS['robot_yaw_deg'])
            self.assertLess(level.target_y_limit, HELD_OUT_LIMITS['target_y'])

    def test_seeded_sampling_is_reproducible_and_bounded(self):
        for level in LEVELS:
            first = seeded_scenarios(level, 100, 23)
            second = seeded_scenarios(level, 100, 23)
            self.assertEqual(first, second)
            for sample in first:
                self.assertLessEqual(abs(sample['pillar_y']), level.pillar_y_limit)
                self.assertLessEqual(abs(sample['robot_yaw_deg']), level.robot_yaw_deg_limit)
                self.assertLessEqual(abs(sample['target_y']), level.target_y_limit)

    def test_actor_observation_contract_excludes_scene_coordinates(self):
        # The established Actor contract remains exactly the existing 16 values.
        actor_observation_keys = ('lidar_0', 'lidar_1', 'lidar_2', 'lidar_3',
                                  'lidar_4', 'lidar_5', 'lidar_6', 'lidar_7',
                                  'goal_distance', 'goal_heading',
                                  'past_linear_residual', 'past_angular_residual',
                                  'state_12', 'state_13', 'state_14', 'state_15')
        self.assertEqual(len(actor_observation_keys), 16)
        self.assertNotIn('pillar_y', actor_observation_keys)

    def test_v5_randomizes_only_pillar_inside_heldout(self):
        for level in LEVELS:
            self.assertEqual(level.robot_yaw_deg_limit, 0.0)
            self.assertEqual(level.target_y_limit, 0.0)
            self.assertLess(level.pillar_y_limit, 0.10)
        self.assertEqual(LEVELS[0].pillar_y_limit, 0.03)
        self.assertEqual(LEVELS[1].pillar_y_limit, 0.06)

    def test_default_initialization_is_final_v4_actor(self):
        self.assertTrue(DEFAULT_V4_ACTOR.endswith(
            'v4_bc_initialized_ppo_finetune_5k_seed7/checkpoints/'
            'actor_iter0008_step00005299.pth'
        ))


class BalancedLossTest(unittest.TestCase):
    def test_each_direction_has_equal_loss_weight(self):
        prediction = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        target = torch.tensor([[1.0], [1.0], [1.0], [3.0]])
        directions = torch.tensor([0, 0, 0, 1])
        loss = balanced_loss(prediction, target, directions, nn.MSELoss())
        self.assertAlmostEqual(float(loss), 5.0)

    def test_bc_loader_rejects_heldout_and_excluded_data(self):
        class FakeData(dict):
            @property
            def files(self):
                return list(self.keys())

        base = FakeData(
            observations=np.zeros((2, 16), dtype=np.float32),
            actions=np.zeros((2, 2), dtype=np.float32),
            direction_ids=np.asarray([0, 1]),
            episode_ids=np.asarray([0, 1]),
            pillar_y=np.asarray([0.06, -0.06], dtype=np.float32),
            dataset_role=np.asarray('v5_training_demonstration'),
        )
        validate_training_dataset(base)
        heldout = FakeData(base)
        heldout['pillar_y'] = np.asarray([0.10, -0.10], dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_training_dataset(heldout)
        excluded = FakeData(base)
        excluded['dataset_role'] = np.asarray('excluded_from_training')
        with self.assertRaises(ValueError):
            validate_training_dataset(excluded)


if __name__ == '__main__':
    unittest.main()
