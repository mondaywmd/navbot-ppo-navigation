"""Search and collect only physically successful demos for both pillar offsets."""

import argparse
import json
import os

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from scenario_env import CENTER_SCENE, V5ScenarioEnv


SCENARIOS = (
    {'name': 'pillar_world_y_positive_06cm', 'pillar_y': 0.06,
     'robot_yaw_deg': 0.0, 'target_y': 0.0},
    {'name': 'pillar_world_y_negative_06cm', 'pillar_y': -0.06,
     'robot_yaw_deg': 0.0, 'target_y': 0.0},
)

# Search independently in each scene; ordering does not encode a side assumption.
CANDIDATES = tuple(
    np.asarray(action, dtype=np.float32) for action in (
        (1.0, -1.0), (1.0, 1.0), (1.0, -0.8), (1.0, 0.8),
        (0.8, -1.0), (0.8, 1.0), (0.6, -1.0), (0.6, 1.0),
    )
)


def run_episode(env, action, record=False):
    observation = env.reset()
    past = np.zeros(2, dtype=np.float32)
    observations, actions = [], []
    outcome = 'timeout'
    action_min = np.full(2, np.inf)
    action_max = np.full(2, -np.inf)
    for step in range(1, 301):
        if record:
            observations.append(observation.copy())
            actions.append(action.copy())
        action_min = np.minimum(action_min, action)
        action_max = np.maximum(action_max, action)
        observation, _reward, done, arrive, _command, _active, _clearance = (
            env.step_residual(action, past)
        )
        past = action
        if done or arrive:
            outcome = 'success' if arrive else 'collision'
            break
    env.pub_cmd_vel.publish(Twist())
    return {
        'outcome': outcome, 'steps': step,
        'observations': observations, 'actions': actions,
        'action_min': action_min.tolist(), 'action_max': action_max.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes-per-direction', type=int, default=3)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metadata', required=True)
    args = parser.parse_args()
    for path in (args.output, args.metadata):
        if os.path.exists(path):
            raise RuntimeError('refusing to overwrite: %s' % path)
    rospy.init_node('v5_collect_balanced_demonstrations', anonymous=True)
    env = V5ScenarioEnv()
    all_observations, all_actions, all_directions, all_episode_ids = [], [], [], []
    metadata = {'candidate_search': {}, 'directions': {}}
    global_episode = 0
    try:
        for direction_id, scenario in enumerate(SCENARIOS):
            env.scenario = dict(scenario)
            trials = []
            successful = []
            for candidate in CANDIDATES:
                result = run_episode(env, candidate, record=False)
                trial = {'action': candidate.tolist(), 'outcome': result['outcome'],
                         'steps': result['steps'], 'actual_xy': env.actual_xy()}
                trials.append(trial)
                print('V5_SEARCH scene=%s action=%s outcome=%s steps=%d' % (
                    scenario['name'], candidate.tolist(), result['outcome'], result['steps']
                ), flush=True)
                if result['outcome'] == 'success':
                    successful.append((result['steps'], candidate.copy()))
            if not successful:
                raise AssertionError('no successful expert action for %s' % scenario['name'])
            successful.sort(key=lambda item: item[0])
            selected = successful[0][1]
            metadata['candidate_search'][scenario['name']] = trials
            episode_rows = []
            for local_episode in range(1, args.episodes_per_direction + 1):
                result = run_episode(env, selected, record=True)
                print('V5_DEMO scene=%s episode=%d action=%s outcome=%s steps=%d' % (
                    scenario['name'], local_episode, selected.tolist(),
                    result['outcome'], result['steps']
                ), flush=True)
                if result['outcome'] != 'success':
                    raise AssertionError('selected action failed collection validation')
                all_observations.extend(result['observations'])
                all_actions.extend(result['actions'])
                all_directions.extend([direction_id] * result['steps'])
                all_episode_ids.extend([global_episode] * result['steps'])
                episode_rows.append({'episode': local_episode, 'outcome': result['outcome'],
                                     'steps': result['steps']})
                global_episode += 1
            metadata['directions'][scenario['name']] = {
                'selected_action': selected.tolist(),
                'episodes': episode_rows,
                'success_rate': 1.0,
                'samples': sum(row['steps'] for row in episode_rows),
                'action_min': selected.tolist(),
                'action_max': selected.tolist(),
            }
    finally:
        env.scenario = CENTER_SCENE.copy()
        env.reset()
        env.pub_cmd_vel.publish(Twist())

    np.savez_compressed(
        args.output,
        observations=np.asarray(all_observations, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.float32),
        direction_ids=np.asarray(all_directions, dtype=np.int64),
        episode_ids=np.asarray(all_episode_ids, dtype=np.int64),
        pillar_y=np.asarray([
            SCENARIOS[direction]['pillar_y'] for direction in all_directions
        ], dtype=np.float32),
        dataset_role=np.asarray('v5_training_demonstration'),
    )
    with open(args.metadata, 'w') as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
    print('V5_DEMOS_SAVED samples=%d dataset=%s metadata=%s' % (
        len(all_actions), args.output, args.metadata
    ), flush=True)


if __name__ == '__main__':
    main()
