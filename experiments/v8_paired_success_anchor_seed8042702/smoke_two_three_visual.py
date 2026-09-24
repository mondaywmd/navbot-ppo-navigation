"""Visible no-training expert smoke for one fresh two- and three-pillar scene."""
import json
import os

import rospy
from geometry_msgs.msg import Twist

from curriculum_spec import generate
from validate_multilevel_expert import PathExpert, run
from wide_env import WideStaticEnv


SEED = 8042708


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(root, "results", "two_three_visual_seed8042708.json")
    if os.path.exists(output_path):
        raise RuntimeError("refusing overwrite: " + output_path)
    rospy.init_node("v8_two_three_visual_smoke", anonymous=True)
    env = WideStaticEnv()
    rows = []
    try:
        for level in (2, 3):
            scene = generate(level, 1, seed=SEED + level)[0]
            env.reset_wide(scene)
            print("VISUAL_HOLD level=%d target=(%.3f,%.3f) obstacles=%s" %
                  (level, scene["target_x"], scene["target_y"],
                   [(round(p["x"], 3), round(p["y"], 3), round(p["radius"], 3))
                    for p in scene["obstacles"]]), flush=True)
            rospy.sleep(5.0)
            # run() intentionally performs a fresh independent reset.
            row = run(env, scene)
            rows.append(row)
            print("VISUAL_RUN level=%d outcome=%s steps=%d min_lidar=%.3f" %
                  (level, row["outcome"], row["steps"], row["min_lidar"]), flush=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as output:
            json.dump({"seed": SEED, "training": False, "episodes": rows},
                      output, indent=2, sort_keys=True)
    finally:
        env.restore_center()
        env.pub_cmd_vel.publish(Twist())
        print("CENTER_RESTORED_ZERO", flush=True)


if __name__ == "__main__":
    main()
