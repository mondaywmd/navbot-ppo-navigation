"""Left/right symmetric PPO reward for goal recovery after obstacle avoidance."""
import math


def transition_reward(previous_distance, distance, heading, linear, collision=False,
                      success=False, timed_out=False):
    """Reward progress while discouraging fast lateral passes near the goal."""
    if collision:
        return -20.0
    if success:
        return 20.0
    progress = previous_distance - distance
    reward = 8.0 * progress - 0.01
    abs_heading = abs(heading)
    # Near the target, a large bearing error means the robot is passing beside
    # it.  Penalize forward speed in that state, but never prefer left or right.
    near_weight = max(0.0, min(1.0, (0.80 - distance) / 0.60))
    lateral_weight = max(0.0, min(1.0, (abs_heading - math.radians(35)) /
                                   math.radians(55)))
    reward -= 1.5 * near_weight * lateral_weight * max(0.0, linear)
    # Explicitly make post-approach regression more costly near the goal.
    if progress < 0.0 and distance < 0.80:
        reward += 8.0 * progress
    if timed_out:
        reward -= 5.0
    return reward
