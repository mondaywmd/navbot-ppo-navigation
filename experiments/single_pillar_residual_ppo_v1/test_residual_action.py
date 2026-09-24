import math
import unittest

import numpy as np
import torch

from residual_action import (
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    compose_residual_command,
    goal_seeking_command,
)
from goal_seeking_baseline import speed_for_distance
from residual_networks import ResidualActor


class ResidualActionTest(unittest.TestCase):
    def test_residual_actor_is_symmetric_and_bounded(self):
        actor = ResidualActor(16, 2)
        output = actor(torch.zeros(4, 16))
        self.assertEqual(tuple(output.shape), (4, 2))
        self.assertTrue(bool(torch.all(output >= -1.0)))
        self.assertTrue(bool(torch.all(output <= 1.0)))

    def test_visual_cruise_speed_is_enforced(self):
        for distance in (0.3, 0.75, 1.5, 10.0):
            self.assertLessEqual(speed_for_distance(distance, 0.10), 0.10)

    def test_zero_residual_is_exactly_nominal(self):
        nominal = goal_seeking_command(2.0, 0.0)
        composed = compose_residual_command(2.0, 0.0, [0.0, 0.0])
        np.testing.assert_array_equal(composed, nominal)

    def test_policy_output_and_physical_command_are_bounded(self):
        command = compose_residual_command(2.0, 0.0, [100.0, -100.0])
        self.assertGreaterEqual(command[0], 0.0)
        self.assertLessEqual(command[0], MAX_LINEAR_SPEED)
        self.assertGreaterEqual(command[1], -MAX_ANGULAR_SPEED)
        self.assertLessEqual(command[1], MAX_ANGULAR_SPEED)

    def test_residual_can_choose_either_bypass_direction(self):
        left = compose_residual_command(2.0, 0.0, [0.0, 1.0])
        right = compose_residual_command(2.0, 0.0, [0.0, -1.0])
        self.assertGreater(left[1], 0.0)
        self.assertLess(right[1], 0.0)

    def test_large_heading_error_keeps_turn_first_nominal(self):
        command = goal_seeking_command(2.0, math.radians(90.0))
        self.assertEqual(float(command[0]), 0.0)

    def test_invalid_action_is_rejected(self):
        with self.assertRaises(ValueError):
            compose_residual_command(2.0, 0.0, [0.0])
        with self.assertRaises(ValueError):
            compose_residual_command(2.0, 0.0, [np.nan, 0.0])


if __name__ == "__main__":
    unittest.main()
