"""V3 smoke: zero residual collides and maximum negative omega succeeds."""

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from reward_shaped_env import RewardShapedSinglePillarEnv


def main():
    rospy.init_node('single_pillar_v3_reward_smoke', anonymous=True)
    env = RewardShapedSinglePillarEnv(False)
    for label, action, expected in (
        ('zero', np.asarray([0.0, 0.0], dtype=np.float32), 'collision'),
        ('max_negative_omega', np.asarray([1.0, -1.0], dtype=np.float32), 'success'),
    ):
        observation = env.reset()
        past = np.zeros(2, dtype=np.float32)
        total_reward = 0.0
        outcome = 'timeout'
        clearance_reward_steps = 0
        for step in range(1, 301):
            observation, reward, done, arrive, command, active, _clearance = env.step_residual(
                action, past
            )
            terms = env.last_reward_terms
            clearance_reward_steps += int(active and terms['clearance_delta'] > 0.0)
            if observation.shape != (16,) or not np.all(np.isfinite(observation)):
                raise AssertionError('invalid observation')
            if not np.isfinite(reward) or not np.all(np.isfinite(command)):
                raise AssertionError('non-finite reward or command')
            total_reward += float(reward)
            past = action
            if done or arrive:
                outcome = 'success' if arrive else 'collision'
                break
        env.pub_cmd_vel.publish(Twist())
        print(
            'V3_SMOKE case=%s outcome=%s steps=%d clearance_increase_steps=%d return=%.3f'
            % (label, outcome, step, clearance_reward_steps, total_reward),
            flush=True,
        )
        if outcome != expected:
            raise AssertionError('%s expected %s, got %s' % (label, expected, outcome))
    print('V3_SMOKE_PASS', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            rospy.Publisher('/cmd_vel', Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
