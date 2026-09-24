"""No-training smoke for the persistent-target V8 reset path."""
import json
import os

import rospy
from geometry_msgs.msg import Twist

from curriculum_spec import generate
from wide_env import WideStaticEnv


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(root, "results", "target_visual_smoke_seed8042707.json")
    if os.path.exists(result_path):
        raise RuntimeError("refusing overwrite: " + result_path)
    rospy.init_node("v8_persistent_target_reset_smoke", anonymous=True)
    env = WideStaticEnv()
    rows = []
    try:
        scenes = [generate(level, 1, seed=8042707 + level)[0]
                  for level in (1, 2, 3)]
        for scene in scenes:
            observation = env.reset_wide(scene)
            state = env.get_model_state("target", "world")
            actual = [state.pose.position.x, state.pose.position.y]
            requested = [scene["target_x"], scene["target_y"]]
            error = max(abs(a - b) for a, b in zip(actual, requested))
            print("TARGET_RESET %s requested=%s actual=%s error=%.9f success=%s" %
                  (scene["name"], requested, actual, error, state.success), flush=True)
            if not state.success or error > 1e-5 or observation.shape != (16,):
                raise AssertionError("persistent-target reset mismatch")
            rows.append({"scene": scene["name"], "requested": requested,
                         "actual": actual, "max_error": error})
            rospy.sleep(5.0)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as output:
            json.dump({"seed": 8042707, "passed": True, "training": False,
                       "resets": rows}, output, indent=2, sort_keys=True)
    finally:
        env.restore_center()
        env.pub_cmd_vel.publish(Twist())
        print("CENTER_RESTORED_ZERO", flush=True)


if __name__ == "__main__":
    main()
