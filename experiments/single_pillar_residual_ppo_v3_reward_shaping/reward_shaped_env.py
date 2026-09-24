"""V3 reward-only extension of the unchanged V2 gated environment."""

import math

from geometry_msgs.msg import Twist

from control_gate import minimum_valid_range
from gated_single_pillar_env import GatedSinglePillarEnv
from reward_shaping import shaped_step_reward


class RewardShapedSinglePillarEnv(GatedSinglePillarEnv):
    def reset(self):
        observation = super().reset()
        self.previous_front_clearance = minimum_valid_range(self.latest_scan)
        return observation

    def setReward(self, done, arrive):
        current_distance = math.hypot(
            self.goal_position.position.x - self.position.x,
            self.goal_position.position.y - self.position.y,
        )
        distance_rate = self.past_distance - current_distance
        heading_alignment = math.cos(math.radians(self.diff_angle))
        heading_improvement = (
            0.0 if self.prev_heading_alignment is None
            else heading_alignment - self.prev_heading_alignment
        )
        clearance = minimum_valid_range(self.latest_scan)
        clearance_delta = clearance - self.previous_front_clearance
        reward = shaped_step_reward(
            distance_rate,
            heading_improvement,
            heading_alignment,
            clearance_delta,
            self.control_gate.active,
        )
        self.past_distance = current_distance
        self.prev_heading_alignment = heading_alignment
        self.previous_front_clearance = clearance
        self.last_reward_terms = {
            'gate_active': self.control_gate.active,
            'clearance_delta': clearance_delta,
            'distance_rate': distance_rate,
            'heading_alignment': heading_alignment,
        }
        if done:
            reward = -300.0
            self.pub_cmd_vel.publish(Twist())
        if arrive:
            reward = 500.0
            self.pub_cmd_vel.publish(Twist())
        return reward

    def step(self, residual_action, past_residual_action):
        observation, reward, done, arrive, _command, _active, _clearance = (
            self.step_residual(residual_action, past_residual_action)
        )
        return observation, reward, done, arrive
