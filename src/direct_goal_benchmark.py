#!/usr/bin/env python3
import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from environment_new import Env
from hybrid_navigation import clamp


EPISODES = 3
MAX_STEPS = 180
ALIGN_DEG = 2.0


def goal_speed(distance):
    if distance > 1.50:
        return 3.0
    if distance > 0.80:
        return 1.50
    if distance > 0.40:
        return 0.75
    return 0.25


def remove_all_pillars(env):
    rospy.wait_for_service("/gazebo/delete_model")

    for name in ("obstacle_0", "obstacle_1"):
        try:
            env.del_model(name)
        except Exception:
            pass

    env.obstacle_centers = []
    rospy.sleep(0.5)


def run_episode(env, episode_index):
    # Prevent the single-pillar test mode from affecting this baseline.
    env.fixed_single_pillar = False
    env.reset()
    remove_all_pillars(env)

    print(
        "Episode %02d | Target=(%.2f, %.2f)"
        % (
            episode_index,
            env.goal_position.position.x,
            env.goal_position.position.y,
        )
    )

    past_action = [0.0, 0.0]

    for step in range(1, MAX_STEPS + 1):
        distance = math.hypot(
            env.goal_position.position.x - env.position.x,
            env.goal_position.position.y - env.position.y,
        )
        error = float(env.diff_angle)

        if abs(error) > ALIGN_DEG:
            action = np.array(
                [0.0, clamp(error / 60.0, -0.55, 0.55)],
                dtype=np.float32,
            )
            mode = "ALIGN"
        else:
            action = np.array(
                [goal_speed(distance), 0.0],
                dtype=np.float32,
            )
            mode = "DRIVE"

        _, _, collision, arrived = env.step(action, past_action)
        past_action = action

        if arrived:
            print("  SUCCESS in %d steps" % step)
            return "success", step

        if collision:
            print("  COLLISION in %d steps" % step)
            return "collision", step

    env.pub_cmd_vel.publish(Twist())
    print("  TIMEOUT at %d steps" % MAX_STEPS)
    return "timeout", MAX_STEPS


def main():
    rospy.init_node("direct_goal_no_obstacles", anonymous=True)
    env = Env(False)

    results = []
    for episode in range(EPISODES):
        results.append(run_episode(env, episode))

    successes = [steps for status, steps in results if status == "success"]

    print("\n=== Direct Goal / No Obstacles Benchmark ===")
    print("Success: %d/%d" % (len(successes), EPISODES))
    print("Collisions: %d" % sum(status == "collision" for status, _ in results))
    print("Timeouts: %d" % sum(status == "timeout" for status, _ in results))

    if successes:
        print(
            "Mean success length: %.1f steps"
            % (sum(successes) / len(successes))
        )


if __name__ == "__main__":
    main()
