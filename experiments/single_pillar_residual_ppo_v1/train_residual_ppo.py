"""Explicit entry point for the later 5k--10k fixed-scene training stage.

This file defines the training wiring but never starts unless ``--timesteps``
is deliberately supplied.
"""

import argparse
import os

import numpy as np
import rospy

from net_critic import NetCritic
from ppo import PPO
from residual_networks import ResidualActor
from single_pillar_env import SinglePillarEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        help="Explicit training budget; stage 3 should use 5000--10000.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.timesteps <= 10000:
        parser.error("--timesteps must be between 1 and 10000 for this stage")
    return args


def main():
    args = parse_args()
    rospy.init_node("single_pillar_residual_ppo_train", anonymous=True)
    env = SinglePillarEnv(True)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    agent = PPO(
        policy_class=ResidualActor,
        value_func=NetCritic,
        env=env,
        state_dim=16,
        action_dim=2,
        output_dir=output_dir,
        method_name="fixed_single_pillar_residual_ppo",
        native_residual=True,
        timesteps_per_batch=500,
        max_timesteps_per_episode=200,
        n_updates_per_iteration=5,
        save_freq=1,
        seed=args.seed,
    )
    agent.learn(
        total_timesteps=args.timesteps,
        past_action=np.zeros(2, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
