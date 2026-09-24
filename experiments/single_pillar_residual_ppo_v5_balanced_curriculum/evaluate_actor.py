"""Deterministic validation of the V5 BC Actor on both demonstration scenes."""

import argparse
import csv
import json
import os

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist

from collect_balanced_demos import SCENARIOS
from residual_networks import ResidualActor
from scenario_env import CENTER_SCENE, V5ScenarioEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--episodes-per-direction', type=int, default=5)
    args = parser.parse_args()
    if os.path.exists(args.results_dir):
        raise RuntimeError('refusing to overwrite: %s' % args.results_dir)
    os.makedirs(args.results_dir)
    rospy.init_node('v5_evaluate_balanced_bc_actor', anonymous=True)
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    actor.eval()
    env = V5ScenarioEnv()
    rows = []
    action_min = np.full(2, np.inf)
    action_max = np.full(2, -np.inf)
    try:
        for scenario in SCENARIOS:
            env.scenario = dict(scenario)
            for episode in range(1, args.episodes_per_direction + 1):
                observation = env.reset()
                past = np.zeros(2, dtype=np.float32)
                outcome = 'timeout'
                for step in range(1, 301):
                    with torch.inference_mode():
                        action = actor(observation).squeeze(0).numpy().astype(np.float32)
                    if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.000001):
                        raise AssertionError('Actor action outside normalized bounds')
                    action_min = np.minimum(action_min, action)
                    action_max = np.maximum(action_max, action)
                    observation, _reward, done, arrive, command, _active, _clearance = (
                        env.step_residual(action, past)
                    )
                    if not (-1e-6 <= float(command[0]) <= 0.250001):
                        raise AssertionError('unsafe linear command')
                    if not (-0.600001 <= float(command[1]) <= 0.600001):
                        raise AssertionError('unsafe angular command')
                    past = action
                    if done or arrive:
                        outcome = 'success' if arrive else 'collision'
                        break
                env.pub_cmd_vel.publish(Twist())
                rows.append({'scenario': scenario['name'], 'episode': episode,
                             'outcome': outcome, 'steps': step})
                print('V5_EVAL scene=%s episode=%d outcome=%s steps=%d' % (
                    scenario['name'], episode, outcome, step
                ), flush=True)
    finally:
        env.scenario = CENTER_SCENE.copy()
        env.reset()
        env.pub_cmd_vel.publish(Twist())
    summary = {'action_min': action_min.tolist(), 'action_max': action_max.tolist(),
               'scenarios': {}}
    for scenario in SCENARIOS:
        subset = [row for row in rows if row['scenario'] == scenario['name']]
        summary['scenarios'][scenario['name']] = {
            'episodes': len(subset),
            'successes': sum(row['outcome'] == 'success' for row in subset),
            'collisions': sum(row['outcome'] == 'collision' for row in subset),
            'timeouts': sum(row['outcome'] == 'timeout' for row in subset),
            'mean_steps': sum(row['steps'] for row in subset) / float(len(subset)),
        }
    with open(os.path.join(args.results_dir, 'episodes.csv'), 'w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=('scenario', 'episode', 'outcome', 'steps'))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.results_dir, 'summary.json'), 'w') as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    print('V5_EVAL_SUMMARY %s' % summary, flush=True)
    if any(value['successes'] != value['episodes']
           for value in summary['scenarios'].values()):
        raise AssertionError('V5 BC Actor did not pass every deterministic episode')


if __name__ == '__main__':
    main()
