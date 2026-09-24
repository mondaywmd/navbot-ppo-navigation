"""Evaluate one fixed checkpoint on the existing frozen 12-task OOD exam."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from control_gate import minimum_valid_range
from residual_networks import ResidualActor
from scenario_env import CENTER_SCENE, V5ScenarioEnv

EPISODES = 5
MAX_STEPS = 300


class OODEnv(V5ScenarioEnv):
    def reset(self):
        observation = super().reset()
        self._set_pose("obstacle_0", self.scenario["pillar_x"],
                       self.scenario["pillar_y"], 0.30)
        rospy.sleep(0.5)
        scan = rospy.wait_for_message("/scan", LaserScan, timeout=5)
        self._previous_arrival_position = None
        observation = self._initial_observation(scan)
        self.previous_front_clearance = minimum_valid_range(scan)
        self.control_gate.reset()
        return observation


def summary(rows):
    n = len(rows)
    return {"episodes": n,
            "successes": sum(r["outcome"] == "success" for r in rows),
            "collisions": sum(r["outcome"] == "collision" for r in rows),
            "timeouts": sum(r["outcome"] == "timeout" for r in rows),
            "success_rate": 100.0 * sum(r["outcome"] == "success" for r in rows) / n,
            "collision_rate": 100.0 * sum(r["outcome"] == "collision" for r in rows) / n,
            "timeout_rate": 100.0 * sum(r["outcome"] == "timeout" for r in rows) / n,
            "mean_steps": sum(r["steps"] for r in rows) / float(n),
            "min_lidar": min(r["min_lidar"] for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.results_dir):
        raise RuntimeError("refusing to overwrite: " + args.results_dir)
    os.makedirs(args.results_dir)
    with open(args.tasks, "rb") as source:
        task_bytes = source.read()
    task_data = json.loads(task_bytes.decode("utf-8"))
    tasks = task_data["tasks"]
    if len(tasks) != 12:
        raise ValueError("frozen exam must contain exactly 12 tasks")
    with open(os.path.join(args.results_dir, "tasks_snapshot.json"), "wb") as output:
        output.write(task_bytes)
    rospy.init_node("v7_final_permanent_ood", anonymous=True)
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    actor.eval()
    env = OODEnv()
    rows = []
    try:
        for task in tasks:
            env.scenario = {key: task[key] for key in
                            ("pillar_x", "pillar_y", "robot_yaw_deg", "target_y")}
            for episode in range(1, EPISODES + 1):
                observation = env.reset()
                past = np.zeros(2, dtype=np.float32)
                minimum = float(env.previous_front_clearance)
                outcome = "timeout"
                for step in range(1, MAX_STEPS + 1):
                    with torch.inference_mode():
                        action = actor(observation).squeeze(0).numpy().astype(np.float32)
                    observation, _, done, arrive, _, _, clearance = env.step_residual(action, past)
                    minimum = min(minimum, float(clearance), minimum_valid_range(env.latest_scan))
                    past = action
                    if arrive:
                        outcome = "success"
                        break
                    if done:
                        outcome = "collision"
                        break
                env.pub_cmd_vel.publish(Twist())
                row = {"task_name": task["task_name"], "episode": episode,
                       "outcome": outcome, "steps": step, "min_lidar": minimum}
                rows.append(row)
                print("V7_OOD task=%s ep=%d outcome=%s steps=%d min=%.4f" %
                      (task["task_name"], episode, outcome, step, minimum), flush=True)
        per_task = {task["task_name"]: summary(
            [row for row in rows if row["task_name"] == task["task_name"]]) for task in tasks}
        result = {"checkpoint": os.path.abspath(args.checkpoint),
                  "checkpoint_sha256": hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest(),
                  "tasks_sha256": hashlib.sha256(task_bytes).hexdigest(),
                  "exploration": False, "episodes_per_task": EPISODES,
                  "max_steps": MAX_STEPS, "per_task": per_task,
                  "overall": summary(rows)}
        with open(os.path.join(args.results_dir, "episodes.csv"), "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        with open(os.path.join(args.results_dir, "summary.json"), "w") as output:
            json.dump(result, output, indent=2, sort_keys=True)
        print("V7_OOD_OVERALL " + json.dumps(result["overall"], sort_keys=True), flush=True)
    finally:
        env.scenario = {"pillar_x": 1.0, **CENTER_SCENE}
        env.reset()
        env.pub_cmd_vel.publish(Twist())
        print("V7_OOD_CENTER_RESTORED_ZERO", flush=True)


if __name__ == "__main__":
    main()
