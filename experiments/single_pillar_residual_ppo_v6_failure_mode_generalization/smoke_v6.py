"""One deterministic, no-update Gazebo episode from each V6 mode."""

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist

from residual_networks import ResidualActor
from scenario_spec import MODE_RANGES, TRAIN_SEED, V6ScenarioSampler, load_permanent_tasks
from train_v6 import V5_ACTOR
from training_env import CENTER_SCENE, PERMANENT_TASKS_PATH, V6TrainingEnv


def main():
    rospy.init_node('v6_preparation_smoke', anonymous=True)
    tasks = load_permanent_tasks(PERMANENT_TASKS_PATH)
    sampler = V6ScenarioSampler(tasks, TRAIN_SEED)
    scenes = [sampler.sample(mode) for mode in MODE_RANGES]
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(V5_ACTOR, map_location='cpu'))
    actor.eval()
    results = []
    env = None
    try:
        for scene in scenes:
            env = V6TrainingEnv(seed=TRAIN_SEED, fixed_scene=scene)
            observation = env.reset()
            if observation.shape != (16,):
                raise AssertionError('Actor observation changed from 16-D')
            actual = env.actual_scene()
            past = np.zeros(2, dtype=np.float32)
            outcome = 'timeout'
            for step in range(1, 301):
                with torch.inference_mode():
                    action = actor(observation).squeeze(0).numpy().astype(np.float32)
                observation, _reward, done, arrive, command, _active, _clearance = (
                    env.step_residual(action, past)
                )
                if np.any(np.abs(action) > 1.000001):
                    raise AssertionError('residual action out of bounds')
                if not (-1e-6 <= command[0] <= 0.250001 and
                        -0.600001 <= command[1] <= 0.600001):
                    raise AssertionError('physical command out of bounds')
                past = action
                if done or arrive:
                    outcome = 'success' if arrive else 'collision'
                    break
            env.pub_cmd_vel.publish(Twist())
            results.append({'scene': scene, 'actual': actual,
                            'outcome': outcome, 'steps': step})
            print('V6_SMOKE %s' % results[-1], flush=True)
    finally:
        if env is not None:
            env.fixed_scene = CENTER_SCENE
            env.reset()
            env.pub_cmd_vel.publish(Twist())
    print('V6_SMOKE_SUMMARY results=%s no_ppo_updates=True' % results, flush=True)


if __name__ == '__main__':
    main()
