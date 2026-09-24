import os
import unittest

import numpy as np
import torch

from control_gate import compose_gated_command
from residual_networks import ResidualActor
from scenario_spec import (EXCLUSION_RADII, MODE_RANGES, TRAIN_SEED,
                           V6ScenarioSampler, leaking_task_name,
                           load_permanent_tasks)
from train_v6 import V5_ACTOR
from training_env import PERMANENT_TASKS_PATH


class V6PreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_permanent_tasks(PERMANENT_TASKS_PATH)

    def test_all_permanent_tasks_are_rejected(self):
        self.assertEqual(len(self.tasks), 12)
        for task in self.tasks:
            self.assertEqual(leaking_task_name(task, self.tasks), task['task_name'])

    def test_representative_new_scenes_pass(self):
        examples = (
            {'pillar_x': 0.79, 'pillar_y': -0.15, 'target_y': 0.145,
             'robot_yaw_deg': 5.0},
            {'pillar_x': 1.24, 'pillar_y': 0.15, 'target_y': -0.145,
             'robot_yaw_deg': 10.0},
        )
        for scene in examples:
            self.assertIsNone(leaking_task_name(scene, self.tasks))

    def test_seeded_generation_is_reproducible_bounded_and_nonleaking(self):
        a = V6ScenarioSampler(self.tasks, TRAIN_SEED)
        b = V6ScenarioSampler(self.tasks, TRAIN_SEED)
        scenes_a = [a.sample_balanced(i) for i in range(100)]
        scenes_b = [b.sample_balanced(i) for i in range(100)]
        self.assertEqual(scenes_a, scenes_b)
        for index, scene in enumerate(scenes_a):
            bounds = MODE_RANGES[tuple(MODE_RANGES)[index % 2]]
            self.assertIsNone(leaking_task_name(scene, self.tasks))
            for key, (low, high) in bounds.items():
                self.assertLessEqual(low, scene[key])
                self.assertLessEqual(scene[key], high)

    def test_actor_checkpoint_dimension_and_action_safety(self):
        self.assertTrue(os.path.isfile(V5_ACTOR))
        actor = ResidualActor(16, 2)
        state = torch.load(V5_ACTOR, map_location='cpu')
        actor.load_state_dict(state)
        self.assertEqual(actor.network[0].in_features, 16)
        with torch.inference_mode():
            action = actor(np.zeros(16, dtype=np.float32)).squeeze(0).numpy()
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))
        for residual in (np.asarray([-1.0, -1.0]), np.asarray([1.0, 1.0])):
            command = compose_gated_command(2.0, 0.0, residual, True)
            self.assertGreaterEqual(command[0], 0.0)
            self.assertLessEqual(command[0], 0.25)
            self.assertGreaterEqual(command[1], -0.60)
            self.assertLessEqual(command[1], 0.60)

    def test_exclusion_radii_are_explicit(self):
        self.assertEqual(EXCLUSION_RADII, {
            'pillar_x': 0.08, 'pillar_y': 0.06,
            'target_y': 0.06, 'robot_yaw_deg': 4.0,
        })


if __name__ == '__main__':
    unittest.main()
