"""Fresh closed-loop evaluation of a clean-start Actor; never uses permanent OOD."""
import argparse, json, math, os, random

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from clean_actor import CleanMirrorActor
from clean_curriculum_spec import generate, mirror
from clean_wide_env import CleanWideStaticEnv
from observation78 import build_observation
from potential_escape import PotentialEscapeShield
from safety_shield import valid_scan
from temporal_lidar import sector_minima, signed_range_rate, temporal_features

STEP_LIMIT = 200


def wrap(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def decode_action(action):
    linear = -0.10 + (float(action[0]) + 1.0) * 0.35 / 2.0
    angular = 0.8 * float(action[1])
    return linear, angular


def normalized_action(linear, angular):
    return np.asarray([np.clip(2.0 * (linear + 0.10) / 0.35 - 1.0, -1, 1),
                       np.clip(angular / 0.8, -1, 1)], dtype=np.float32)


def run_episode(env, actor, scene):
    env.reset_wide(scene)
    previous_sector = None
    previous_action = np.zeros(2, dtype=np.float32)
    shield = PotentialEscapeShield()
    minimum = float("inf")
    reasons = {}
    outcome = "timeout"
    trace = []
    for step in range(1, STEP_LIMIT + 1):
        state = env.get_model_state("turtlebot3_burger", "world")
        x, y = state.pose.position.x, state.pose.position.y
        q = state.pose.orientation
        yaw = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)
        distance = math.hypot(scene["target_x"] - x, scene["target_y"] - y)
        if distance <= 0.20:
            outcome = "success"
            break
        scan = rospy.wait_for_message("/scan", LaserScan, timeout=5)
        scan_min = float(valid_scan(scan).min())
        minimum = min(minimum, scan_min)
        if scan_min < 0.20:
            outcome = "collision"
            break
        current = sector_minima(scan.ranges, scan.range_min, scan.range_max)
        rates = (np.zeros(36, dtype=np.float32) if previous_sector is None else
                 signed_range_rate(previous_sector, current, 0.2))
        heading = wrap(math.atan2(scene["target_y"] - y, scene["target_x"] - x) - yaw)
        observation = build_observation(current, rates, previous_action, distance, heading)
        with torch.no_grad():
            raw = actor(observation).cpu().numpy()[0]
        requested_linear, requested_angular = decode_action(raw)
        front_ttc = float("inf")
        if previous_sector is not None and abs(requested_angular) < 0.10:
            _, ttc = temporal_features(previous_sector, current, 0.2)
            front_ttc = float(min(ttc[17], ttc[18]))
        linear, angular, reason = shield.apply(requested_linear, requested_angular, scan,
                                               front_ttc, float("inf"), goal_distance=distance)
        executed = normalized_action(linear, angular)
        reasons[reason] = reasons.get(reason, 0) + 1
        trace.append({"step": step, "distance": distance, "heading": heading,
                      "minimum_lidar": scan_min, "actor_action": raw.tolist(),
                      "command": [linear, angular], "shield_reason": reason})
        command = Twist(); command.linear.x = linear; command.angular.z = angular
        env.pub_cmd_vel.publish(command)
        previous_sector = current
        previous_action = executed
    env.pub_cmd_vel.publish(Twist())
    return {"outcome": outcome, "steps": step, "minimum_lidar": minimum,
            "shield_reasons": reasons, "trace": trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise RuntimeError("refusing overwrite: " + args.output_dir)
    actor = CleanMirrorActor()
    actor.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    actor.eval()
    rng = random.Random(args.seed)
    rospy.init_node("clean_actor_fresh_evaluation", anonymous=True)
    env = CleanWideStaticEnv(); episodes = []
    try:
        for pair_id in range(args.pairs):
            level = rng.randint(1, 5)
            scene_seed = rng.randrange(1, 2**31)
            base = generate(level, 1, scene_seed)[0]
            for side, scene in (("base", base), ("mirror", mirror(base))):
                result = run_episode(env, actor, scene)
                record = {"pair_id": pair_id, "pillar_count": level, "side": side,
                          "scene_seed": scene_seed, "scene": scene, **result}
                episodes.append(record)
                print("CLEAN_ACTOR_EVAL pair=%d pillars=%d side=%s outcome=%s steps=%d min=%.3f" %
                      (pair_id, level, side, result["outcome"], result["steps"],
                       result["minimum_lidar"]), flush=True)
        os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "episodes.json"), "w") as f:
            json.dump(episodes, f, indent=2, sort_keys=True)
        counts = {name: sum(e["outcome"] == name for e in episodes)
                  for name in ("success", "collision", "timeout")}
        by_count = {str(level): {name: sum(e["pillar_count"] == level and e["outcome"] == name
                                      for e in episodes)
                                 for name in counts}
                    for level in range(1, 6)}
        summary = {"seed": args.seed, "pairs": args.pairs, "episodes": len(episodes),
                   "pillar_count_sampling": "uniform_random_1_to_5_per_pair",
                   "step_limit": STEP_LIMIT, "counts": counts, "by_pillar_count": by_count,
                   "mean_steps": float(np.mean([e["steps"] for e in episodes])),
                   "minimum_lidar": min(e["minimum_lidar"] for e in episodes),
                   "permanent_ood_accessed": False, "ppo_started": False}
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("CLEAN_ACTOR_EVAL_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    finally:
        env.restore_center(); env.pub_cmd_vel.publish(Twist())
        print("CENTER_RESTORED_ZERO", flush=True)


if __name__ == "__main__":
    main()
