import unittest

from reward_shaping import CLEARANCE_GAIN, shaped_step_reward


class RewardShapingTest(unittest.TestCase):
    def test_outside_gate_matches_original_reward_exactly(self):
        actual = shaped_step_reward(0.02, -0.03, 0.7, 0.09, False)
        expected = 100.0 * 0.02 - 1.0 + 12.0 * -0.03 + 0.5 * 0.7
        self.assertEqual(actual, expected)

    def test_clearance_increase_is_rewarded_in_takeover(self):
        worse = shaped_step_reward(0.0, 0.0, 0.0, -0.05, True)
        better = shaped_step_reward(0.0, 0.0, 0.0, 0.05, True)
        self.assertAlmostEqual(better - worse, CLEARANCE_GAIN * 0.10)

    def test_takeover_weakens_heading_penalty(self):
        outside = shaped_step_reward(0.0, -0.1, -1.0, 0.0, False)
        inside = shaped_step_reward(0.0, -0.1, -1.0, 0.0, True)
        self.assertGreater(inside, outside)

    def test_clearance_delta_is_clipped(self):
        at_limit = shaped_step_reward(0.0, 0.0, 0.0, 0.10, True)
        extreme = shaped_step_reward(0.0, 0.0, 0.0, 10.0, True)
        self.assertEqual(at_limit, extreme)


if __name__ == '__main__':
    unittest.main()
