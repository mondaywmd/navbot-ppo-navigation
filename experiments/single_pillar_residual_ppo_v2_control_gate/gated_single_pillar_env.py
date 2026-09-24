"""Fixed single-pillar environment with LiDAR-gated residual authority."""

import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from environment_new import diagonal_dis
from single_pillar_env import SinglePillarEnv
from control_gate import LidarControlGate, compose_gated_command, minimum_valid_range


class GatedSinglePillarEnv(SinglePillarEnv):
    def __init__(self, is_training):
        super().__init__(is_training)
        self.control_gate = LidarControlGate()

    def reset(self):
        self.control_gate.reset()
        return super().reset()

    def step_residual(self, residual_action, past_residual_action):
        clearance = minimum_valid_range(self.latest_scan)
        gate_active = self.control_gate.update(clearance)
        distance = math.hypot(
            self.goal_position.position.x - self.position.x,
            self.goal_position.position.y - self.position.y,
        )
        command = compose_gated_command(
            distance, math.radians(float(self.diff_angle)), residual_action, gate_active
        )

        velocity = Twist()
        velocity.linear.x = float(command[0])
        velocity.angular.z = float(command[1])
        self.pub_cmd_vel.publish(velocity)
        scan = rospy.wait_for_message('scan', LaserScan, timeout=5)
        self.latest_scan = scan
        state, rel_dis, _yaw, _rel_theta, diff_angle, done, arrive = self.getState(scan)
        normalized_scan = [value / 3.5 for value in state]
        indices = [int(index * len(normalized_scan) / 10) for index in range(10)]
        previous = np.asarray(past_residual_action, dtype=np.float32)
        if previous.shape != (2,):
            raise ValueError('past_residual_action must have shape (2,)')
        heading_rad = math.radians(diff_angle)
        observation = np.asarray(
            [normalized_scan[index] for index in indices] + previous.tolist() + [
                rel_dis / diagonal_dis,
                math.sin(heading_rad),
                math.cos(heading_rad),
                diff_angle / 180.0,
            ],
            dtype=np.float32,
        )
        assert observation.shape == (16,)
        reward = self.setReward(done, arrive)
        return observation, reward, done, arrive, command.copy(), gate_active, clearance

