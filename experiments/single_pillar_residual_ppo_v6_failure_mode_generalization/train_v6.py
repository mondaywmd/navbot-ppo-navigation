"""Opt-in V6 PPO entry.  Merely importing this module cannot start training."""

import argparse
import os

import numpy as np
import rospy
import torch

from net_critic import NetCritic
from ppo import PPO
from residual_networks import ResidualActor
from scenario_spec import DEFAULT_RUN_NAME, TRAIN_SEED
from training_env import V6TrainingEnv


V5_ACTOR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'single_pillar_residual_ppo_v5_balanced_curriculum',
    'runs', 'v5_v4init_pillar_y_curriculum_10k_seed29', 'checkpoints',
    'actor_iter0016_step00010421.pth',
))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, required=True)
    parser.add_argument('--run-name', default=DEFAULT_RUN_NAME)
    args = parser.parse_args()
    if not 1 <= args.timesteps <= 20000:
        raise ValueError('timesteps must be in [1,20000]')
    experiment = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(experiment, 'runs')
    run_dir = os.path.join(output_dir, args.run_name)
    if os.path.exists(run_dir):
        raise RuntimeError('refusing to overwrite run: %s' % run_dir)
    rospy.init_node('v6_failure_mode_generalization', anonymous=True)
    env = V6TrainingEnv(seed=TRAIN_SEED)
    agent = PPO(policy_class=ResidualActor, value_func=NetCritic, env=env,
                state_dim=16, action_dim=2, output_dir=output_dir,
                method_name=args.run_name, native_residual=True,
                normalize_returns=True, timesteps_per_batch=500,
                max_timesteps_per_episode=300, n_updates_per_iteration=5,
                save_freq=1, seed=TRAIN_SEED)
    state = torch.load(V5_ACTOR, map_location='cpu')
    agent.actor.load_state_dict(state)
    for name, value in agent.actor.state_dict().items():
        if not torch.equal(value.cpu(), state[name].cpu()):
            raise AssertionError('V5 Actor initialization mismatch at %s' % name)
    print('[V6] verified final V5 Actor: %s' % V5_ACTOR, flush=True)
    print('[V6] seed=%d run=%s normalize_returns=True' % (
        TRAIN_SEED, args.run_name
    ), flush=True)
    agent.learn(args.timesteps, np.zeros(2, dtype=np.float32))


if __name__ == '__main__':
    main()
