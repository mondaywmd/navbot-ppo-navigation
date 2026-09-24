"""Safety smoke for the BC actor under PPO's initial action distribution."""

import argparse

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from torch.distributions import MultivariateNormal

from residual_networks import ResidualActor
from reward_shaped_env import RewardShapedSinglePillarEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(7)
    rospy.init_node('v4_bc_ppo_noise_smoke', anonymous=True)
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    actor.eval()
    covariance = torch.diag(torch.full((2,), 0.05))
    env = RewardShapedSinglePillarEnv(False)
    results, sampled_actions, physical_commands = [], [], []

    for episode in range(1, args.episodes + 1):
        observation = env.reset()
        past = np.zeros(2, dtype=np.float32)
        outcome = 'timeout'
        for step in range(1, 301):
            with torch.inference_mode():
                mean = actor(observation).squeeze(0)
                sampled = MultivariateNormal(mean, covariance).sample()
                action = torch.clamp(sampled, -1.0, 1.0).numpy().astype(np.float32)
            observation, _reward, done, arrive, command, _active, _clearance = (
                env.step_residual(action, past)
            )
            sampled_actions.append(action.copy())
            physical_commands.append(command.copy())
            past = action
            if done or arrive:
                outcome = 'success' if arrive else 'collision'
                break
        env.pub_cmd_vel.publish(Twist())
        results.append((outcome, step))
        print('V4_NOISE episode=%d outcome=%s steps=%d' % (episode, outcome, step), flush=True)

    actions = np.asarray(sampled_actions)
    commands = np.asarray(physical_commands)
    successes = sum(outcome == 'success' for outcome, _ in results)
    collisions = sum(outcome == 'collision' for outcome, _ in results)
    print(
        'V4_NOISE_SUMMARY success=%d/%d collision=%d timeout=%d mean_steps=%.2f '
        'residual0=[%.5f,%.5f] residual1=[%.5f,%.5f] '
        'linear=[%.5f,%.5f] angular=[%.5f,%.5f]'
        % (successes, len(results), collisions, len(results)-successes-collisions,
           sum(step for _, step in results)/len(results),
           actions[:,0].min(), actions[:,0].max(), actions[:,1].min(), actions[:,1].max(),
           commands[:,0].min(), commands[:,0].max(), commands[:,1].min(), commands[:,1].max()),
        flush=True,
    )
    if successes * 2 <= len(results):
        raise AssertionError('BC actor did not succeed in a majority of noisy episodes')


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            rospy.Publisher('/cmd_vel', Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
