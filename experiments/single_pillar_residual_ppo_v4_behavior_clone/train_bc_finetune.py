"""Five-thousand-step PPO fine-tune initialized strictly from the BC actor."""

import os

import numpy as np
import rospy
import torch

from net_critic import NetCritic
from ppo import PPO
from residual_networks import ResidualActor
from reward_shaped_env import RewardShapedSinglePillarEnv


RUN_NAME = 'v4_bc_initialized_ppo_finetune_5k_seed7'


def main():
    rospy.init_node('single_pillar_v4_bc_ppo_finetune', anonymous=True)
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(experiment_dir, 'runs')
    run_dir = os.path.join(output_dir, RUN_NAME)
    if os.path.exists(run_dir):
        raise RuntimeError('refusing to overwrite existing run: %s' % run_dir)
    bc_checkpoint = os.path.join(experiment_dir, 'actor_bc_fixed_pillar.pth')
    env = RewardShapedSinglePillarEnv(True)
    agent = PPO(
        policy_class=ResidualActor,
        value_func=NetCritic,
        env=env,
        state_dim=16,
        action_dim=2,
        output_dir=output_dir,
        method_name=RUN_NAME,
        native_residual=True,
        normalize_returns=True,
        timesteps_per_batch=500,
        max_timesteps_per_episode=300,
        n_updates_per_iteration=5,
        save_freq=1,
        seed=7,
    )
    bc_state = torch.load(bc_checkpoint, map_location='cpu')
    agent.actor.load_state_dict(bc_state)
    for name, value in agent.actor.state_dict().items():
        if not torch.equal(value.cpu(), bc_state[name].cpu()):
            raise AssertionError('BC actor load verification failed at %s' % name)
    print('[V4] BC actor loaded and verified: %s' % bc_checkpoint, flush=True)
    print('[V4] normalize_returns=True; advantages are normalized by PPO.', flush=True)
    agent.learn(5000, np.zeros(2, dtype=np.float32))


if __name__ == '__main__':
    main()
