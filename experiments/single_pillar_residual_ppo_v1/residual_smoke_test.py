"""End-to-end zero-residual smoke test in the fixed Gazebo scene."""

import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from residual_action import (
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    goal_seeking_command,
)
from single_pillar_env import SinglePillarEnv


MAX_STEPS = 40
ZERO_RESIDUAL = np.zeros(2, dtype=np.float32)


def assert_observation(observation):
    if observation.shape != (16,):
        raise AssertionError("expected observation shape (16,), got %r" % (observation.shape,))
    if not np.all(np.isfinite(observation)):
        raise AssertionError("observation contains a non-finite value")


def main():
    rospy.init_node("single_pillar_residual_smoke", anonymous=True)
    env = SinglePillarEnv(False)
    observation = env.reset()
    assert_observation(observation)

    past_residual = ZERO_RESIDUAL.copy()
    total_reward = 0.0
    terminal = "timeout"

    for step in range(1, MAX_STEPS + 1):
        distance = math.hypot(
            env.goal_position.position.x - env.position.x,
            env.goal_position.position.y - env.position.y,
        )
        heading_error = math.radians(float(env.diff_angle))
        nominal = goal_seeking_command(distance, heading_error)

        observation, reward, done, arrive, command = env.step_residual(
            ZERO_RESIDUAL, past_residual
        )
        assert_observation(observation)
        if not math.isfinite(float(reward)):
            raise AssertionError("reward is not finite")
        np.testing.assert_array_equal(command, nominal)
        if not 0.0 <= float(command[0]) <= MAX_LINEAR_SPEED:
            raise AssertionError("unsafe linear command: %r" % float(command[0]))
        if not -MAX_ANGULAR_SPEED <= float(command[1]) <= MAX_ANGULAR_SPEED:
            raise AssertionError("unsafe angular command: %r" % float(command[1]))

        total_reward += float(reward)
        past_residual = ZERO_RESIDUAL.copy()
        if step == 1 or step % 5 == 0 or done or arrive:
            print(
                "SMOKE step=%d obs=%s reward=%.3f cmd=(%.3f, %.3f) done=%s arrive=%s"
                % (step, observation.shape, reward, command[0], command[1], done, arrive),
                flush=True,
            )
        if done or arrive:
            terminal = "arrival" if arrive else "collision"
            break

    env.pub_cmd_vel.publish(Twist())
    if terminal == "timeout":
        raise AssertionError("smoke episode did not terminate within %d steps" % MAX_STEPS)

    print(
        "SMOKE PASS: terminal=%s steps=%d total_reward=%.3f zero_residual_exact=True"
        % (terminal, step, total_reward),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            rospy.Publisher("/cmd_vel", Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
