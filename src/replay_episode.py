#!/usr/bin/env python3
"""Replay a recorded evaluation episode approximately in Gazebo."""

import argparse
import json
import os
import time

import rospy
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel


ROBOT_MODEL = "turtlebot3_burger"
TARGET_MODEL = "target"
CONTROL_PERIOD = 0.2  # 与训练环境每一步一致


def set_pose(set_state, name, x, y, z=0.01):
    state = ModelState()
    state.model_name = name
    state.pose.position.x = float(x)
    state.pose.position.y = float(y)
    state.pose.position.z = z
    state.pose.orientation.w = 1.0  # 旧记录未保存初始朝向，暂按 0 rad
    state.reference_frame = "world"
    result = set_state(state)
    if not result.success:
        raise RuntimeError(f"Cannot place {name}: {result.status_message}")


GOAL_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "turtlebot3_simulations",
    "turtlebot3_gazebo", "models", "Target", "model.sdf"
))


def ensure_target(goal):
    """Create the recorded target position, replacing any old target."""
    try:
        rospy.wait_for_service("/gazebo/delete_model", timeout=2)
        rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)(TARGET_MODEL)
    except (rospy.ROSException, rospy.ServiceException):
        pass

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    pose = Pose()
    pose.position.x = float(goal["x"])
    pose.position.y = float(goal["y"])
    pose.position.z = 0.01
    pose.orientation.w = 1.0

    with open(GOAL_MODEL_PATH, "r") as f:
        target_xml = f.read()

    result = rospy.ServiceProxy(
        "/gazebo/spawn_sdf_model", SpawnModel
    )(TARGET_MODEL, target_xml, "namespace", pose, "world")

    if not result.success:
        raise RuntimeError("Cannot create target: " + result.status_message)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", help="episode_XXX_success.json")
    args = parser.parse_args()

    with open(args.record, "r") as f:
        episode = json.load(f)

    if not episode.get("trajectory"):
        raise ValueError("Replay record has no trajectory.")

    rospy.init_node("ppo_episode_replay", anonymous=True)
    rospy.wait_for_service("/gazebo/set_model_state")
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

    start = episode["start"]
    goal = episode["goal"]

    # 恢复起点与目标点
    set_pose(set_state, ROBOT_MODEL, start["x"], start["y"])
    ensure_target(goal)
    rospy.sleep(1.0)

    print(
        f'Replaying episode {episode["episode"]}: '
        f'{episode["outcome"]}, {episode["length"]} steps'
    )

    try:
        for frame in episode["trajectory"]:
            if rospy.is_shutdown():
                break

            cmd = Twist()
            # 保存的是网络动作；环境中线速度实际会除以 4
            cmd.linear.x = float(frame["linear_action"]) / 4.0
            cmd.angular.z = float(frame["angular_action"])
            cmd_pub.publish(cmd)
            rospy.sleep(CONTROL_PERIOD)
    finally:
        cmd_pub.publish(Twist())
        print("Replay finished; robot stopped.")


if __name__ == "__main__":
    main()
