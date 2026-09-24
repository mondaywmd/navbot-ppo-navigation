"""Fixed 5k-step training entry for V3 after tests and smoke pass."""

import os

import numpy as np
import rospy

from net_critic import NetCritic
from ppo import PPO
from residual_networks import ResidualActor
from reward_shaped_env import RewardShapedSinglePillarEnv


def main():
    rospy.init_node('single_pillar_v3_train', anonymous=True)
    env = RewardShapedSinglePillarEnv(True)
    agent = PPO(
        policy_class=ResidualActor,
        value_func=NetCritic,
        env=env,
        state_dim=16,
        action_dim=2,
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
        method_name='v3_reward_shaping_5k_seed7',
        native_residual=True,
        timesteps_per_batch=500,
        max_timesteps_per_episode=300,
        n_updates_per_iteration=5,
        save_freq=1,
        seed=7,
    )
    agent.learn(5000, np.zeros(2, dtype=np.float32))


if __name__ == '__main__':
    main()
