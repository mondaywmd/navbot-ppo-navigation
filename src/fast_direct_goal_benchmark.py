#!/usr/bin/env python3
import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from environment_new import Env
from hybrid_navigation import clamp


EPISODES = 3
MAX_STEPS = 180
ALIGN_DEG = 1.0


def speed_for_distance(distance):
    if distance > 1.20:
        return 5.0
    if distance > 0.65:
        return 2.0
    if distance > 0.35:
        return 0.65
    return 0.20


def remove_pillars(env):
    rospy.wait_for_service("/gazebo/delete_model")
    for name in ("obstacle_0", "obstacle_1"):
        try:
            env.del_model(name)
        except Exception:
            pass

    env.obstacle_centers = []
    rospy.sleep(0.5)


def run_episode(env, index):
    env.fixed_single_pillar = False
    env.reset()
    remove_pillars(env)

    print(
        "Episode %02d | Target=(%.2f, %.2f)"
        % (
            index,
            env.goal_position.position.x,
            env.goal_position.position.y,
        )
    )

    mode = "ALIGN"
    past_action = [0.0, 0.0]

    for step in range(1, MAX_STEPS + 1):
        distance = math.hypot(
            env.goal_position.position.x - env.position.x,
            env.goal_position.position.y - env.position.y,
        )

        if mode == "ALIGN":
            error = float(env.diff_angle)

            if abs(error) > ALIGN_DEG:
                action = np.array(
                    [0.0, clamp(error / 60.0, -0.55, 0.55)],
                    dtype=np.float32,
                )
            else:
                mode = "DRIVE"
                print("  step %d: aligned; locked straight line to goal" % step)
                action = np.array([0.0, 0.0], dtype=np.float32)

        else:
            # No heading recalculation here: locked, strictly straight motion.
            action = np.array(
                [speed_for_distance(distance), 0.0],
                dtype=np.float32,
            )

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
    rospy.init_node("fast_direct_goal_no_obstacles", anonymous=True)
    env = Env(False)

    results = [run_episode(env, i) for i in range(EPISODES)]
    successes = [steps for status, steps in results if status == "success"]

    print("\n=== Fast Direct Goal / No Obstacles ===")
    print("Success: %d/%d" % (len(successes), EPISODES))
    print("Collisions: %d" % sum(s == "collision" for s, _ in results))
    print("Timeouts: %d" % sum(s == "timeout" for s, _ in results))

    if successes:
        print("Mean success length: %.1f steps" % (sum(successes) / len(successes)))


if __name__ == "__main__":
    main()
