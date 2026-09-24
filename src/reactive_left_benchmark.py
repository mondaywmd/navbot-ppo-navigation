import math
import time
from pathlib import Path

import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose

from environment_new import Env
from reactive_left_navigation import ReactiveLeftNavigator

PILLAR_X, PILLAR_Y = 0.75, 0.45
TARGET_X, TARGET_Y = 2.20, 1.80
EPISODES = 3
MAX_STEPS = 150

PILLAR_SDF = """<sdf version="1.6">
  <model name="single_pillar">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><cylinder><radius>0.30</radius><length>0.60</length></cylinder></geometry>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.30</radius><length>0.60</length></cylinder></geometry>
        <material><ambient>1 0.35 0 1</ambient><diffuse>1 0.35 0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def delete_if_present(delete_model, name):
    try:
        delete_model(name)
    except rospy.ServiceException:
        pass


def install_fixed_scene(env, delete_model, spawn_model):
    for name in ("Target", "target", "obstacle_0", "obstacle_1", "single_pillar"):
        delete_if_present(delete_model, name)

    pillar_pose = Pose()
    pillar_pose.position.x = PILLAR_X
    pillar_pose.position.y = PILLAR_Y
    pillar_pose.position.z = 0.30
    spawn_model("single_pillar", PILLAR_SDF, "", pillar_pose, "world")

    target_sdf = Path(
        "/root/catkin_ws/src/turtlebot3_simulations/"
        "turtlebot3_gazebo/models/Target/model.sdf"
    ).read_text()

    target_pose = Pose()
    target_pose.position.x = TARGET_X
    target_pose.position.y = TARGET_Y
    target_pose.position.z = 0.01
    spawn_model("Target", target_sdf, "", target_pose, "world")

    env.goal_position = target_pose
    env.past_distance = math.hypot(
        TARGET_X - env.position.x,
        TARGET_Y - env.position.y,
    )
    time.sleep(0.4)


def main():
    rospy.init_node("tangent_single_pillar_benchmark", anonymous=True)
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")

    delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    env = Env(2)
    env.fixed_single_pillar = True

    successes = collisions = timeouts = 0
    lengths = []

    for episode in range(EPISODES):
        env.reset()

        navigator = ReactiveLeftNavigator()
        past_action = [0.0, 0.0]
        last_mode = None

        print(f"\nEpisode {episode:02d}")
        print(f"Pillar=({PILLAR_X:.2f}, {PILLAR_Y:.2f}) | Target=({TARGET_X:.2f}, {TARGET_Y:.2f})")

        for step in range(1, MAX_STEPS + 1):
            action, _ = navigator.base_action(env)

            if abs(float(action[1])) > 0.01:
                print(
                    f"  step {step}: TURN command | "
                    f"mode={navigator.mode} | angular_action={float(action[1]):+.3f}"
                )

            if navigator.mode != last_mode:
                waypoint = None if navigator.waypoint is None else tuple(
                    round(float(v), 2) for v in navigator.waypoint
                )
                print(f"  step {step}: mode={navigator.mode}, side={navigator.chosen_side}, waypoint={waypoint}")
                last_mode = navigator.mode

            _, _, done, arrive = env.step(action, past_action)
            past_action = action

            if arrive:
                successes += 1
                lengths.append(step)
                print(f"  SUCCESS in {step} steps")
                break
            if done:
                collisions += 1
                print(f"  COLLISION at step {step}")
                break
        else:
            timeouts += 1
            print(f"  TIMEOUT at {MAX_STEPS} steps")

    print("\n=== Reactive LiDAR Fixed-Left Benchmark ===")
    print(f"Success: {successes}/{EPISODES} ({100 * successes / EPISODES:.1f}%)")
    print(f"Collisions: {collisions}")
    print(f"Timeouts: {timeouts}")
    if lengths:
        print(f"Mean success length: {sum(lengths) / len(lengths):.1f} steps")


if __name__ == "__main__":
    main()
