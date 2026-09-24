"""Run fixed maximum left/right turn residuals through the v2 control gate."""

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from gated_single_pillar_env import GatedSinglePillarEnv


MAX_STEPS = 300


def main():
    rospy.init_node('single_pillar_v2_manual_gate_validation', anonymous=True)
    env = GatedSinglePillarEnv(False)
    results = []
    for label, action in (
        ('left', np.asarray([1.0, 1.0], dtype=np.float32)),
        ('right', np.asarray([1.0, -1.0], dtype=np.float32)),
    ):
        observation = env.reset()
        past = np.zeros(2, dtype=np.float32)
        active_steps = 0
        max_abs_y = 0.0
        total_reward = 0.0
        outcome = 'timeout'
        for step in range(1, MAX_STEPS + 1):
            observation, reward, done, arrive, command, active, clearance = env.step_residual(
                action, past
            )
            active_steps += int(active)
            max_abs_y = max(max_abs_y, abs(float(env.position.y)))
            total_reward += float(reward)
            past = action
            if step == 1 or step % 10 == 0 or done or arrive:
                print(
                    'MANUAL_V2 %s step=%d pos=(%.3f,%.3f) gate=%s clearance=%.3f '
                    'cmd=(%.3f,%.3f) reward=%.3f'
                    % (label, step, env.position.x, env.position.y, active,
                       clearance, command[0], command[1], reward),
                    flush=True,
                )
            if done or arrive:
                outcome = 'success' if arrive else 'collision'
                break
        env.pub_cmd_vel.publish(Twist())
        result = (label, outcome, step, active_steps, max_abs_y, total_reward)
        results.append(result)
        print(
            'MANUAL_V2_END direction=%s outcome=%s steps=%d gate_steps=%d '
            'max_abs_y=%.3f return=%.3f'
            % result,
            flush=True,
        )
    print('MANUAL_V2_SUMMARY %r' % (results,), flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            rospy.Publisher('/cmd_vel', Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
