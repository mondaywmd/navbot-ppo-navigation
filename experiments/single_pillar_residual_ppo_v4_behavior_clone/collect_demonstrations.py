"""Collect successful fixed-scene expert trajectories for behavior cloning."""

import argparse

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from reward_shaped_env import RewardShapedSinglePillarEnv


EXPERT_ACTION = np.asarray([1.0, -1.0], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rospy.init_node('v4_collect_expert_demonstrations', anonymous=True)
    env = RewardShapedSinglePillarEnv(False)
    observations, actions, episode_ids = [], [], []

    for episode in range(args.episodes):
        observation = env.reset()
        past = np.zeros(2, dtype=np.float32)
        outcome = 'timeout'
        for step in range(1, 301):
            observations.append(observation.copy())
            actions.append(EXPERT_ACTION.copy())
            episode_ids.append(episode)
            observation, _reward, done, arrive, _command, _active, _clearance = (
                env.step_residual(EXPERT_ACTION, past)
            )
            past = EXPERT_ACTION
            if done or arrive:
                outcome = 'success' if arrive else 'collision'
                break
        env.pub_cmd_vel.publish(Twist())
        print('V4_DEMO episode=%d outcome=%s steps=%d' % (episode + 1, outcome, step), flush=True)
        if outcome != 'success':
            raise AssertionError('expert demonstration did not succeed')

    np.savez_compressed(
        args.output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
    )
    print('V4_DEMO_SAVED samples=%d path=%s' % (len(actions), args.output), flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            rospy.Publisher('/cmd_vel', Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
