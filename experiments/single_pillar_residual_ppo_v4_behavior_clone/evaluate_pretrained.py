"""Deterministic Gazebo validation of the behavior-cloned actor."""

import argparse

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist

from residual_networks import ResidualActor
from reward_shaped_env import RewardShapedSinglePillarEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    rospy.init_node('v4_evaluate_pretrained_actor', anonymous=True)
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    actor.eval()
    env = RewardShapedSinglePillarEnv(False)
    results = []

    for episode in range(1, args.episodes + 1):
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
        results.append((outcome, step))
        print(
            'V4_EVAL episode=%d outcome=%s steps=%d final_action=(%.5f,%.5f)'
            % (episode, outcome, step, action[0], action[1]),
            flush=True,
        )
    successes = sum(outcome == 'success' for outcome, _ in results)
    print(
        'V4_EVAL_SUMMARY successes=%d/%d mean_steps=%.2f'
        % (successes, len(results), sum(step for _, step in results) / len(results)),
        flush=True,
    )
    if successes != len(results):
        raise AssertionError('pretrained actor did not succeed in every validation episode')


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            rospy.Publisher('/cmd_vel', Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
