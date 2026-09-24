"""Online fresh paired-mirror V7 scenes; no fixed task list is read."""
import json
import os
import random
import rospy
from sensor_msgs.msg import LaserScan
from control_gate import minimum_valid_range
from scenario_env import V5ScenarioEnv

TRAIN_SEED = 7042701


class OnlinePairSampler:
    def __init__(self, seed=TRAIN_SEED):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.pair_index = 0
        self.pending_mirror = None

    def next_scene(self):
        if self.pending_mirror is not None:
            result, self.pending_mirror = self.pending_mirror, None
            return result
        self.pair_index += 1
        pillar_x = self.rng.uniform(0.72, 1.18)
        pillar_y = self.rng.uniform(-0.30, 0.30)
        target_y = self.rng.uniform(-0.32, 0.32)
        robot_yaw_deg = self.rng.uniform(-10.0, 10.0)
        base = {"pair_index": self.pair_index, "pair_side": "base",
                "pillar_x": pillar_x, "pillar_y": pillar_y,
                "target_y": target_y, "robot_yaw_deg": robot_yaw_deg}
        self.pending_mirror = {
            "pair_index": self.pair_index, "pair_side": "mirror",
            "pillar_x": pillar_x, "pillar_y": -pillar_y,
            "target_y": -target_y, "robot_yaw_deg": -robot_yaw_deg}
        return base


class V7OnlinePairedEnv(V5ScenarioEnv):
    def __init__(self, scene_log, seed=TRAIN_SEED):
        super().__init__()
        self.sampler = OnlinePairSampler(seed)
        self.scene_log = os.path.abspath(scene_log)
        if os.path.exists(self.scene_log):
            raise RuntimeError("refusing to overwrite scene log: " + self.scene_log)

    def reset(self):
        self.scenario = self.sampler.next_scene()
        observation = super().reset()
        self._set_pose("obstacle_0", self.scenario["pillar_x"],
                       self.scenario["pillar_y"], 0.30)
        rospy.sleep(0.5)
        scan = rospy.wait_for_message("/scan", LaserScan, timeout=5)
        self._previous_arrival_position = None
        observation = self._initial_observation(scan)
        self.previous_front_clearance = minimum_valid_range(scan)
        self.control_gate.reset()
        os.makedirs(os.path.dirname(self.scene_log), exist_ok=True)
        with open(self.scene_log, "a") as output:
            output.write(json.dumps({"seed": self.sampler.seed, **self.scenario},
                                    sort_keys=True) + "\n")
        return observation
