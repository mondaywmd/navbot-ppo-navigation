"""Six-reset Gazebo smoke for wide one/two/three-pillar scenes and mirrors."""
import json
import os
import rospy
from geometry_msgs.msg import Twist
from curriculum_spec import generate, mirror
from wide_env import WideStaticEnv


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(root, "results", "wide_env_smoke_v3_seed8042703.json")
    if os.path.exists(result_path): raise RuntimeError("refusing overwrite: " + result_path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    rospy.init_node("v8_wide_static_env_smoke", anonymous=True)
    env = WideStaticEnv(); rows = []
    try:
        for level in (1, 2, 3):
            base = generate(level, 1)[0]
            for scene in (base, mirror(base)):
                observation = env.reset_wide(scene)
                target_state = env.get_model_state("target", "world")
                obstacle_state = env.get_model_state(env.wide_model_names[0], "world")
                if not target_state.success or not obstacle_state.success:
                    raise RuntimeError("spawned wide-scene model disappeared")
                row = {"level": level, "name": scene["name"],
                       "obstacle_count": len(scene["obstacles"]),
                       "observation_dim": len(observation),
                       "observation_finite": bool(all(float(x) == float(x) for x in observation)),
                       "target_requested": [scene["target_x"], scene["target_y"]],
                       "target_actual": [target_state.pose.position.x, target_state.pose.position.y],
                       "obstacle0_actual": [obstacle_state.pose.position.x, obstacle_state.pose.position.y],
                       "min_lidar": float(env.previous_front_clearance)}
                rows.append(row)
                print("WIDE_SMOKE level=%d scene=%s obstacles=%d lidar=%.3f" %
                      (level, scene["name"], len(scene["obstacles"]), row["min_lidar"]), flush=True)
        with open(result_path, "w") as output:
            json.dump({"seed": 8042703, "resets": rows, "passed": True,
                       "training": False}, output, indent=2, sort_keys=True)
    finally:
        env.restore_center(); env.pub_cmd_vel.publish(Twist())
        print("CENTER_RESTORED_ZERO", flush=True)


if __name__ == "__main__": main()
