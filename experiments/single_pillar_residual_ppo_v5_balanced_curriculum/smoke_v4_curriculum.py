"""No-update deterministic smoke of final V4 Actor on both V5 levels."""

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist

from curriculum import LEVELS, seeded_scenarios
from residual_networks import ResidualActor
from scenario_env import CENTER_SCENE, V5ScenarioEnv
from train_from_v4 import DEFAULT_V4_ACTOR


def main():
    rospy.init_node('v5_v4_curriculum_smoke', anonymous=True)
    actor = ResidualActor(16, 2)
    source_state = torch.load(DEFAULT_V4_ACTOR, map_location='cpu')
    actor.load_state_dict(source_state)
    for name, value in actor.state_dict().items():
        if not torch.equal(value.cpu(), source_state[name].cpu()):
            raise AssertionError('V4 Actor mismatch at %s' % name)
    actor.eval()
    env = V5ScenarioEnv()
    action_min = np.full(2, np.inf)
    action_max = np.full(2, -np.inf)
    results = []
    try:
        for index, level in enumerate(LEVELS):
            env.scenario = seeded_scenarios(level, 1, 290 + index)[0]
            observation = env.reset()
            if np.asarray(observation).shape != (16,):
                raise AssertionError('observation must remain exactly 16-D')
            actual = env.actual_xy()
            past = np.zeros(2, dtype=np.float32)
            outcome = 'timeout'
            for step in range(1, 301):
                with torch.inference_mode():
                    action = actor(observation).squeeze(0).numpy().astype(np.float32)
                action_min = np.minimum(action_min, action)
                action_max = np.maximum(action_max, action)
                if np.any(np.abs(action) > 1.000001):
                    raise AssertionError('normalized residual outside [-1,1]')
                observation, _reward, done, arrive, command, _active, _clearance = (
                    env.step_residual(action, past)
                )
                if not (-1e-6 <= float(command[0]) <= 0.250001):
                    raise AssertionError('unsafe physical linear command')
                if not (-0.600001 <= float(command[1]) <= 0.600001):
                    raise AssertionError('unsafe physical angular command')
                past = action
                if done or arrive:
                    outcome = 'success' if arrive else 'collision'
                    break
            env.pub_cmd_vel.publish(Twist())
            results.append((level.name, outcome, step, dict(env.scenario), actual))
            print('V5_SMOKE level=%s scene=%s actual=%s outcome=%s steps=%d' % (
                level.name, env.scenario, actual, outcome, step
            ), flush=True)
    finally:
        env.scenario = CENTER_SCENE.copy()
        env.reset()
        env.pub_cmd_vel.publish(Twist())
    print('V5_SMOKE_SUMMARY results=%s action_min=%s action_max=%s no_updates=True' % (
        results, action_min.tolist(), action_max.tolist()
    ), flush=True)


if __name__ == '__main__':
    main()
