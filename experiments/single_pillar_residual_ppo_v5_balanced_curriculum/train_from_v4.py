"""Opt-in V5 PPO fine-tune initialized exactly from the final V4 Actor."""

import argparse
import os

import numpy as np
import rospy
import torch

from net_critic import NetCritic
from ppo import PPO
from residual_networks import ResidualActor
from training_env import V5CurriculumEnv


DEFAULT_V4_ACTOR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..',
    'single_pillar_residual_ppo_v4_behavior_clone', 'runs',
    'v4_bc_initialized_ppo_finetune_5k_seed7', 'checkpoints',
    'actor_iter0008_step00005299.pth',
))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--checkpoint', default=DEFAULT_V4_ACTOR)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--level1-episodes', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.timesteps <= 10000:
        raise ValueError('timesteps must be in [1, 10000]')
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(experiment_dir, 'runs')
    run_dir = os.path.join(output_dir, args.run_name)
    if os.path.exists(run_dir):
        raise RuntimeError('refusing to overwrite existing run: %s' % run_dir)
    rospy.init_node('single_pillar_v5_v4_initialized_finetune', anonymous=True)
    env = V5CurriculumEnv(seed=args.seed, level1_episodes=args.level1_episodes)
    agent = PPO(
        policy_class=ResidualActor,
        value_func=NetCritic,
        env=env,
        state_dim=16,
        action_dim=2,
        output_dir=output_dir,
        method_name=args.run_name,
        native_residual=True,
        normalize_returns=True,
        timesteps_per_batch=500,
        max_timesteps_per_episode=300,
        n_updates_per_iteration=5,
        save_freq=1,
        seed=args.seed,
    )
    v4_state = torch.load(args.checkpoint, map_location='cpu')
    agent.actor.load_state_dict(v4_state)
    for name, value in agent.actor.state_dict().items():
        if not torch.equal(value.cpu(), v4_state[name].cpu()):
            raise AssertionError('V4 Actor load verification failed at %s' % name)
    print('[V5] final V4 Actor loaded and verified: %s' % args.checkpoint, flush=True)
    print('[V5] pillar-y curriculum: level1 +/-0.03 m, level2 +/-0.06 m', flush=True)
    print('[V5] held-out pillar +/-0.10 m is excluded; normalize_returns=True', flush=True)
    agent.learn(args.timesteps, np.zeros(2, dtype=np.float32))


if __name__ == '__main__':
    main()
