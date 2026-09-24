"""Reproducible adversarial OOD exam; evaluation only, never training."""

import argparse
import csv
import json
import math
import os

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from control_gate import minimum_valid_range
from residual_action import goal_seeking_command
from residual_networks import ResidualActor
from scenario_env import CENTER_SCENE, V5ScenarioEnv


SEED = 29012
CANDIDATE_POOL_TARGET = 20
EXAM_TASKS = 12
EPISODES_PER_TASK = 5
MAX_STEPS = 300


def candidate_stream(seed):
    rng = np.random.RandomState(seed)
    index = 0
    while True:
        index += 1
        yield {
            'candidate_index': index,
            'pillar_x': float(rng.uniform(0.75, 1.25)),
            'pillar_y': float(rng.uniform(-0.18, 0.18)),
            'robot_yaw_deg': float(rng.uniform(-15.0, 15.0)),
            'target_y': float(rng.uniform(-0.15, 0.15)),
        }


def point_segment_distance(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    t = ((px - x0) * dx + (py - y0) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def legal_candidate(candidate):
    # Arena-safe with ample wall clearance and no initial robot overlap.
    if math.hypot(candidate['pillar_x'], candidate['pillar_y']) <= 0.55:
        return False
    # Keep adversarial candidates meaningfully near the direct route.
    return point_segment_distance(
        candidate['pillar_x'], candidate['pillar_y'],
        0.0, 0.0, 2.0, candidate['target_y'],
    ) <= 0.42


class OODEnv(V5ScenarioEnv):
    def reset(self):
        # V5ScenarioEnv fixes pillar x=1; apply arbitrary x after its safe reset.
        observation = super().reset()
        scene = self.scenario
        self._set_pose('obstacle_0', scene['pillar_x'], scene['pillar_y'], 0.30)
        rospy.sleep(0.5)
        scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        self._previous_arrival_position = None
        observation = self._initial_observation(scan)
        self.previous_front_clearance = minimum_valid_range(scan)
        self.control_gate.reset()
        return observation

    def step_pure_goal_seeking(self):
        distance = math.hypot(
            self.goal_position.position.x - self.position.x,
            self.goal_position.position.y - self.position.y,
        )
        command = goal_seeking_command(distance, math.radians(float(self.diff_angle)))
        velocity = Twist()
        velocity.linear.x = float(command[0])
        velocity.angular.z = float(command[1])
        self.pub_cmd_vel.publish(velocity)
        scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        self.latest_scan = scan
        state, _distance, _yaw, _theta, _diff, done, arrive = self.getState(scan)
        return done, arrive, minimum_valid_range(scan), command


def set_scene(env, candidate):
    env.scenario = {
        'pillar_x': candidate['pillar_x'],
        'pillar_y': candidate['pillar_y'],
        'robot_yaw_deg': candidate['robot_yaw_deg'],
        'target_y': candidate['target_y'],
    }


def baseline_trial(env, candidate):
    set_scene(env, candidate)
    env.reset()
    minimum = float('inf')
    outcome = 'timeout'
    for step in range(1, MAX_STEPS + 1):
        done, arrive, clearance, _command = env.step_pure_goal_seeking()
        minimum = min(minimum, clearance)
        if done or arrive:
            outcome = 'collision' if done and not arrive else 'success'
            break
    env.pub_cmd_vel.publish(Twist())
    return {
        'outcome': outcome,
        'steps': step,
        'min_lidar': minimum,
        'final_robot_x': float(env.position.x),
        'final_robot_y': float(env.position.y),
    }


def actor_trial(env, actor, candidate):
    set_scene(env, candidate)
    observation = env.reset()
    past = np.zeros(2, dtype=np.float32)
    minimum = env.previous_front_clearance
    outcome = 'timeout'
    for step in range(1, MAX_STEPS + 1):
        with torch.inference_mode():
            action = actor(observation).squeeze(0).numpy().astype(np.float32)
        observation, _reward, done, arrive, _command, _active, clearance = (
            env.step_residual(action, past)
        )
        minimum = min(minimum, clearance, minimum_valid_range(env.latest_scan))
        past = action
        if done or arrive:
            outcome = 'success' if arrive else 'collision'
            break
    env.pub_cmd_vel.publish(Twist())
    return {'outcome': outcome, 'steps': step, 'min_lidar': float(minimum)}


def summarize(rows):
    count = len(rows)
    return {
        'episodes': count,
        'successes': sum(row['outcome'] == 'success' for row in rows),
        'collisions': sum(row['outcome'] == 'collision' for row in rows),
        'timeouts': sum(row['outcome'] == 'timeout' for row in rows),
        'success_rate': 100.0 * sum(row['outcome'] == 'success' for row in rows) / count,
        'collision_rate': 100.0 * sum(row['outcome'] == 'collision' for row in rows) / count,
        'timeout_rate': 100.0 * sum(row['outcome'] == 'timeout' for row in rows) / count,
        'mean_steps': sum(row['steps'] for row in rows) / float(count),
        'min_lidar': min(row['min_lidar'] for row in rows),
        'mean_min_lidar': sum(row['min_lidar'] for row in rows) / float(count),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()
    if os.path.exists(args.results_dir):
        raise RuntimeError('refusing to overwrite results: %s' % args.results_dir)
    os.makedirs(args.results_dir)
    rospy.init_node('v5_adversarial_ood_exam', anonymous=True)
    env = OODEnv()
    candidates, retained = [], []
    stream = candidate_stream(SEED)
    try:
        while len(retained) < CANDIDATE_POOL_TARGET:
            candidate = next(stream)
            if not legal_candidate(candidate):
                continue
            evidence = baseline_trial(env, candidate)
            row = {**candidate, **evidence}
            candidates.append(row)
            print('OOD_BASELINE candidate=%d outcome=%s steps=%d min_lidar=%.4f' % (
                candidate['candidate_index'], evidence['outcome'], evidence['steps'],
                evidence['min_lidar']
            ), flush=True)
            if evidence['outcome'] == 'collision' and evidence['min_lidar'] < 0.20:
                retained.append(row)

        chooser = np.random.RandomState(SEED + 1)
        chosen_indices = chooser.choice(len(retained), EXAM_TASKS, replace=False)
        tasks = []
        for task_number, retained_index in enumerate(chosen_indices, 1):
            task = dict(retained[int(retained_index)])
            task['task_name'] = 'ood_seed29012_task%02d' % task_number
            tasks.append(task)
        with open(os.path.join(args.results_dir, 'tasks.json'), 'w') as output:
            json.dump({'seed': SEED, 'tasks': tasks}, output, indent=2, sort_keys=True)
        with open(os.path.join(args.results_dir, 'baseline_candidates.csv'), 'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=candidates[0].keys())
            writer.writeheader()
            writer.writerows(candidates)
        print('OOD_TASKS_SELECTED count=%d retained_pool=%d tested=%d' % (
            len(tasks), len(retained), len(candidates)
        ), flush=True)

        actor = ResidualActor(16, 2)
        actor.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        actor.eval()
        episodes = []
        for task in tasks:
            task_rows = []
            for episode in range(1, EPISODES_PER_TASK + 1):
                result = actor_trial(env, actor, task)
                row = {'task_name': task['task_name'], 'episode': episode, **result}
                episodes.append(row)
                task_rows.append(row)
                print('OOD_ACTOR task=%s episode=%d outcome=%s steps=%d min_lidar=%.4f' % (
                    task['task_name'], episode, result['outcome'], result['steps'],
                    result['min_lidar']
                ), flush=True)
            print('OOD_TASK_SUMMARY task=%s metrics=%s' % (
                task['task_name'], summarize(task_rows)
            ), flush=True)
        with open(os.path.join(args.results_dir, 'episodes.csv'), 'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=episodes[0].keys())
            writer.writeheader()
            writer.writerows(episodes)
        summary = {
            'checkpoint': args.checkpoint,
            'seed': SEED,
            'per_task': {task['task_name']: summarize([
                row for row in episodes if row['task_name'] == task['task_name']
            ]) for task in tasks},
            'overall': summarize(episodes),
        }
        with open(os.path.join(args.results_dir, 'summary.json'), 'w') as output:
            json.dump(summary, output, indent=2, sort_keys=True)
        print('OOD_OVERALL_SUMMARY %s' % summary['overall'], flush=True)
    finally:
        env.scenario = {'pillar_x': 1.0, **CENTER_SCENE}
        env.reset()
        env.pub_cmd_vel.publish(Twist())
        print('OOD_CENTER_RESTORED', flush=True)


if __name__ == '__main__':
    main()
