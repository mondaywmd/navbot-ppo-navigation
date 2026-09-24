"""Fresh non-OOD deterministic acceptance smoke for V5 versus V8 BC."""
import argparse
import csv
import json
import os

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist

from control_gate import minimum_valid_range
from curriculum_spec import generate, mirror
from residual_networks import ResidualActor
from symmetric_actor import MirrorEquivariantActor
from wide_env import WideStaticEnv


SEED = 8042710
MAX_STEPS = 300


def load_actor(path, symmetric=False):
    actor = (MirrorEquivariantActor if symmetric else ResidualActor)(16, 2)
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    actor.eval()
    return actor


def run(env, actor, scene, label):
    observation = env.reset_wide(scene)
    past = np.zeros(2, dtype=np.float32)
    clearance = float("inf"); outcome = "timeout"
    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            action = actor(observation).squeeze(0).cpu().numpy().astype(np.float32)
        observation, _, done, arrive, _, _, before = env.step_residual(action, past)
        clearance = min(clearance, float(before), minimum_valid_range(env.latest_scan))
        past = action
        if arrive:
            outcome = "success"; break
        if done:
            outcome = "collision"; break
    env.pub_cmd_vel.publish(Twist())
    row = {"actor": label, "level": scene["level"], "scene": scene["name"],
           "side": "mirror" if scene["name"].endswith("_mirror") else "base",
           "outcome": outcome, "steps": step, "min_lidar": clearance}
    print("BC_ACCEPT actor=%s level=%d side=%s outcome=%s steps=%d min=%.3f" %
          (label, row["level"], row["side"], outcome, step, clearance), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5", required=True)
    parser.add_argument("--bc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bc-symmetric", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise RuntimeError("refusing overwrite: " + args.output_dir)
    rospy.init_node("v8_bc_fresh_acceptance", anonymous=True)
    actors = {"v5": load_actor(args.v5),
              "v8_bc": load_actor(args.bc, symmetric=args.bc_symmetric)}
    scenes = []
    for level in (1, 2, 3):
        for base in generate(level, 2, seed=args.seed + level):
            scenes.extend((base, mirror(base)))
    env = WideStaticEnv(); rows = []
    try:
        for scene in scenes:
            for label in ("v5", "v8_bc"):
                rows.append(run(env, actors[label], scene, label))
        counts = {}
        for label in actors:
            subset = [r for r in rows if r["actor"] == label]
            counts[label] = {name: sum(r["outcome"] == name for r in subset)
                             for name in ("success", "collision", "timeout")}
        passed = (counts["v8_bc"]["success"] >= counts["v5"]["success"] and
                  counts["v8_bc"]["collision"] <= counts["v5"]["collision"])
        os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "episodes.csv"), "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        summary = {"seed": args.seed, "episodes_per_actor": len(scenes), "counts": counts,
                   "acceptance_rule": "bc success >= v5 and bc collision <= v5",
                   "passed": passed, "ppo_started": False,
                   "permanent_ood_accessed": False}
        with open(os.path.join(args.output_dir, "summary.json"), "w") as output:
            json.dump(summary, output, indent=2, sort_keys=True)
        print("BC_FRESH_ACCEPTANCE " + json.dumps(summary, sort_keys=True), flush=True)
    finally:
        env.restore_center(); env.pub_cmd_vel.publish(Twist())
        print("CENTER_RESTORED_ZERO", flush=True)


if __name__ == "__main__":
    main()
