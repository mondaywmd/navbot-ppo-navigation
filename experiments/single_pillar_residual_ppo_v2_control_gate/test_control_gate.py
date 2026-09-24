import unittest

import numpy as np

from control_gate import (
    EXIT_CLEAR_SCANS,
    GATED_BASE_LINEAR_MAX,
    LidarControlGate,
    compose_gated_command,
)
from residual_action import goal_seeking_command


class ControlGateTest(unittest.TestCase):
    def test_hysteresis_requires_consecutive_clear_scans(self):
        gate = LidarControlGate()
        self.assertTrue(gate.update(0.79))
        for _ in range(EXIT_CLEAR_SCANS - 1):
            self.assertTrue(gate.update(1.11))
        self.assertFalse(gate.update(1.11))

    def test_non_clear_scan_resets_exit_counter(self):
        gate = LidarControlGate()
        gate.update(0.50)
        gate.update(1.20)
        gate.update(0.90)
        self.assertTrue(gate.update(1.20))

    def test_outside_gate_is_exact_nominal_and_ignores_residual(self):
        nominal = goal_seeking_command(2.0, 0.2)
        command = compose_gated_command(2.0, 0.2, [-1.0, 1.0], False)
        np.testing.assert_array_equal(command, nominal)

    def test_gate_removes_heading_correction_and_caps_base_speed(self):
        command = compose_gated_command(2.0, 0.5, [0.0, 0.0], True)
        self.assertAlmostEqual(float(command[0]), GATED_BASE_LINEAR_MAX, places=6)
        self.assertEqual(float(command[1]), 0.0)

    def test_residual_bounds_are_unchanged(self):
        left = compose_gated_command(2.0, 0.0, [-1.0, 1.0], True)
        right = compose_gated_command(2.0, 0.0, [1.0, -1.0], True)
        np.testing.assert_allclose(left, [0.04, 0.30], atol=1e-6)
        np.testing.assert_allclose(right, [0.14, -0.30], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
