"""Execute complete paired expert trajectories; creates no training dataset."""
import argparse
import csv
import json
import math
import os
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from control_gate import minimum_valid_range
from scenario_env import CENTER_SCENE, V5ScenarioEnv
from scenes import SEED, write

MAX_STEPS = 300
EPISODES_PER_SIDE = 2
REQUIRED_CLEARANCE = 0.205
FIELDS = (["scene", "pair", "side", "episode", "step"] +
          ["obs_%02d" % i for i in range(16)] +
          ["residual_linear", "residual_angular", "cmd_linear", "cmd_angular",
           "gate_active", "min_lidar", "robot_x", "robot_y", "robot_yaw_deg",
           "phase", "outcome"])


class Env(V5ScenarioEnv):
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


class SafeDetourExpert:
    def __init__(self, scene):
        self.scene = scene
        self.committed = False
        self.phase = "approach"

    @staticmethod
    def wrap(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def action(self, env):
        clearance = minimum_valid_range(env.latest_scan)
        if clearance < 0.92:
            self.committed = True
        if not self.committed:
            return np.zeros(2, dtype=np.float32), "approach"
        px, py = self.scene["pillar_x"], self.scene["pillar_y"]
        corridor_y = py + self.scene["detour_sign"] * 0.72
        entry = (px - 0.42, corridor_y)
        exit_point = (px + 0.45, corridor_y)
        position = (float(env.position.x), float(env.position.y))
        if self.phase in ("approach", "entry"):
            self.phase = "entry"
            if math.hypot(position[0] - entry[0], position[1] - entry[1]) < 0.13:
                self.phase = "traverse"
        if self.phase == "traverse" and math.hypot(
                position[0] - exit_point[0], position[1] - exit_point[1]) < 0.14:
            self.phase = "recover"
        if self.phase == "entry": waypoint = entry
        elif self.phase == "traverse": waypoint = exit_point
        else: waypoint = (2.0, self.scene["target_y"])
        phase = self.phase
        desired = math.atan2(waypoint[1] - float(env.position.y),
                             waypoint[0] - float(env.position.x))
        error = self.wrap(desired - math.radians(float(env.yaw)))
        angular = float(np.clip(1.8 * error / 0.30, -1.0, 1.0))
        linear = -0.6 if abs(error) > math.radians(45) else 1.0
        return np.asarray([linear, angular], dtype=np.float32), phase


def run(env, scene, episode):
    env.scenario = dict(scene)
    observation = env.reset()
    expert = SafeDetourExpert(scene)
    past = np.zeros(2, dtype=np.float32)
    rows = []
    minimum = float("inf")
    outcome = "timeout"
    for step in range(1, MAX_STEPS + 1):
        actor_input = observation.copy()
        action, phase = expert.action(env)
        observation, _, done, arrive, command, gate, before = env.step_residual(action, past)
        clearance = min(float(before), minimum_valid_range(env.latest_scan))
        minimum = min(minimum, clearance)
        if arrive: outcome = "success"
        elif done: outcome = "collision"
        elif step == MAX_STEPS: outcome = "timeout"
        row = {"scene": scene["name"], "pair": scene["pair"], "side": scene["side"],
               "episode": episode, "step": step, "residual_linear": float(action[0]),
               "residual_angular": float(action[1]), "cmd_linear": float(command[0]),
               "cmd_angular": float(command[1]), "gate_active": int(gate),
               "min_lidar": clearance, "robot_x": float(env.position.x),
               "robot_y": float(env.position.y), "robot_yaw_deg": float(env.yaw),
               "phase": phase, "outcome": outcome if done or arrive or step == MAX_STEPS else "running"}
        for index, value in enumerate(actor_input): row["obs_%02d" % index] = float(value)
        rows.append(row); past = action
        if done or arrive: break
    env.pub_cmd_vel.publish(Twist())
    return rows, {"scene": scene["name"], "pair": scene["pair"], "side": scene["side"],
                  "episode": episode, "outcome": outcome, "steps": step,
                  "min_lidar": minimum}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.results_dir): raise RuntimeError("refusing overwrite: " + args.results_dir)
    os.makedirs(args.results_dir)
    scenes = write(os.path.join(args.results_dir, "scenes.json"))
    with open(os.path.join(args.results_dir, "protocol.json"), "w") as output:
        json.dump({"seed": SEED, "pairs": 6, "episodes_per_side": 2,
                   "required_clearance": REQUIRED_CLEARANCE, "training": False},
                  output, indent=2, sort_keys=True)
    rospy.init_node("v8_paired_success_expert_validation", anonymous=True)
    env = Env(); traces = []; episodes = []
    try:
        for scene in scenes:
            for episode in range(1, EPISODES_PER_SIDE + 1):
                rows, result = run(env, scene, episode); traces.extend(rows); episodes.append(result)
                print("V8_EXPERT scene=%s ep=%d outcome=%s steps=%d min=%.4f" %
                      (scene["name"], episode, result["outcome"], result["steps"], result["min_lidar"]), flush=True)
        with open(os.path.join(args.results_dir, "steps.csv"), "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS); writer.writeheader(); writer.writerows(traces)
        with open(os.path.join(args.results_dir, "episodes.csv"), "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=episodes[0].keys()); writer.writeheader(); writer.writerows(episodes)
        accepted = []
        for pair in range(1, 7):
            subset = [row for row in episodes if row["pair"] == pair]
            if len(subset) == 4 and all(row["outcome"] == "success" and row["min_lidar"] >= REQUIRED_CLEARANCE for row in subset):
                accepted.append(pair)
        result = {"accepted_pairs": accepted, "accepted_count": len(accepted),
                  "episodes": len(episodes), "successes": sum(r["outcome"] == "success" for r in episodes),
                  "collisions": sum(r["outcome"] == "collision" for r in episodes),
                  "timeouts": sum(r["outcome"] == "timeout" for r in episodes),
                  "dataset_created": False}
        with open(os.path.join(args.results_dir, "summary.json"), "w") as output:
            json.dump(result, output, indent=2, sort_keys=True)
        print("V8_EXPERT_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        env.scenario = {"pillar_x": 1.0, **CENTER_SCENE}; env.reset(); env.pub_cmd_vel.publish(Twist())
        print("CENTER_RESTORED_ZERO", flush=True)


if __name__ == "__main__": main()
