"""Deterministic, named held-out perturbation evaluation for the final V4 actor."""

import argparse
import csv
import json
import math
import os

import numpy as np
import rospy
import torch
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from control_gate import minimum_valid_range
from residual_networks import ResidualActor
from reward_shaped_env import RewardShapedSinglePillarEnv


SCENARIOS = (
    {'name': 'pillar_left_10cm',  'pillar_y':  0.10, 'robot_yaw_deg':  0.0, 'target_y':  0.00},
    {'name': 'pillar_right_10cm', 'pillar_y': -0.10, 'robot_yaw_deg':  0.0, 'target_y':  0.00},
    {'name': 'robot_yaw_left_5deg',  'pillar_y': 0.00, 'robot_yaw_deg':  5.0, 'target_y': 0.00},
    {'name': 'robot_yaw_right_5deg', 'pillar_y': 0.00, 'robot_yaw_deg': -5.0, 'target_y': 0.00},
    {'name': 'target_left_10cm',  'pillar_y': 0.00, 'robot_yaw_deg': 0.0, 'target_y':  0.10},
    {'name': 'target_right_10cm', 'pillar_y': 0.00, 'robot_yaw_deg': 0.0, 'target_y': -0.10},
)
CENTER_SCENE = {'name': 'center_restore', 'pillar_y': 0.0, 'robot_yaw_deg': 0.0, 'target_y': 0.0}


class HeldOutEnv(RewardShapedSinglePillarEnv):
    def __init__(self):
        super().__init__(False)
        self.scenario = CENTER_SCENE
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.read_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def _set_pose(self, model_name, x, y, z, yaw_deg=0.0):
        state = ModelState()
        state.model_name = model_name
        state.reference_frame = 'world'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        yaw = math.radians(yaw_deg)
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)
        response = self.set_model_state(state)
        if not response.success:
            raise RuntimeError('set_model_state failed for %s: %s' % (
                model_name, response.status_message
            ))

    def reset(self):
        observation = super().reset()
        scenario = self.scenario
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        self._set_pose('turtlebot3_burger', 0.0, 0.0, 0.0, scenario['robot_yaw_deg'])
        self._set_pose('obstacle_0', 1.0, scenario['pillar_y'], 0.30)
        self._set_pose('target', 2.0, scenario['target_y'], 0.01)
        self.goal_position.position.x = 2.0
        self.goal_position.position.y = scenario['target_y']
        rospy.sleep(0.5)
        scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        self._previous_arrival_position = None
        observation = self._initial_observation(scan)
        self.previous_front_clearance = minimum_valid_range(scan)
        self.control_gate.reset()
        actual = {}
        for model_name in ('turtlebot3_burger', 'obstacle_0', 'target'):
            response = self.read_model_state(model_name, 'world')
            if not response.success:
                raise RuntimeError('could not verify %s' % model_name)
            actual[model_name] = [response.pose.position.x, response.pose.position.y]
        print('HELDOUT_RESET name=%s actual=%s min_lidar=%.6f' % (
            scenario['name'], actual, self.previous_front_clearance
        ), flush=True)
        return observation


def rates(rows):
    count = len(rows)
    successes = sum(row['outcome'] == 'success' for row in rows)
    collisions = sum(row['outcome'] == 'collision' for row in rows)
    timeouts = count - successes - collisions
    return {
        'episodes': count,
        'successes': successes,
        'collisions': collisions,
        'timeouts': timeouts,
        'success_rate': 100.0 * successes / count,
        'collision_rate': 100.0 * collisions / count,
        'timeout_rate': 100.0 * timeouts / count,
        'mean_steps': sum(row['steps'] for row in rows) / float(count),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()
    if os.path.exists(args.results_dir):
        raise RuntimeError('refusing to overwrite results directory: %s' % args.results_dir)
    os.makedirs(args.results_dir)
    rospy.init_node('v4_heldout_generalization_eval', anonymous=True)
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    actor.eval()
    env = HeldOutEnv()
    rows = []

    try:
        for scenario in SCENARIOS:
            env.scenario = scenario
            scenario_rows = []
            for episode in range(1, 11):
                observation = env.reset()
                past = np.zeros(2, dtype=np.float32)
                outcome = 'timeout'
                for step in range(1, 301):
                    with torch.inference_mode():
                        action = actor(observation).squeeze(0).numpy().astype(np.float32)
                    observation, _reward, done, arrive, _command, _active, _clearance = (
                        env.step_residual(action, past)
                    )
                    past = action
                    if done or arrive:
                        outcome = 'success' if arrive else 'collision'
                        break
                env.pub_cmd_vel.publish(Twist())
                row = {'scenario': scenario['name'], 'episode': episode,
                       'outcome': outcome, 'steps': step}
                rows.append(row)
                scenario_rows.append(row)
                print('HELDOUT episode scenario=%s episode=%d outcome=%s steps=%d' % (
                    scenario['name'], episode, outcome, step
                ), flush=True)
            print('HELDOUT_SCENARIO_SUMMARY name=%s metrics=%s' % (
                scenario['name'], rates(scenario_rows)
            ), flush=True)
    finally:
        env.scenario = CENTER_SCENE
        env.reset()
        env.pub_cmd_vel.publish(Twist())
        print('HELDOUT_CENTER_RESTORED', flush=True)

    with open(os.path.join(args.results_dir, 'episodes.csv'), 'w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=('scenario', 'episode', 'outcome', 'steps'))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'checkpoint': args.checkpoint,
        'scenarios': {name: rates([row for row in rows if row['scenario'] == name])
                      for name in [scenario['name'] for scenario in SCENARIOS]},
        'overall': rates(rows),
    }
    with open(os.path.join(args.results_dir, 'summary.json'), 'w') as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    print('HELDOUT_OVERALL_SUMMARY %s' % summary['overall'], flush=True)


if __name__ == '__main__':
    main()
