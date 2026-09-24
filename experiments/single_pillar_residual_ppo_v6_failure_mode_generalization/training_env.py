"""Online-reset V6 environment; world parameters never enter observations."""

import math
import os

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from sensor_msgs.msg import LaserScan

from control_gate import minimum_valid_range
from reward_shaped_env import RewardShapedSinglePillarEnv
from scenario_spec import TRAIN_SEED, V6ScenarioSampler, load_permanent_tasks


PERMANENT_TASKS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'single_pillar_residual_ppo_v5_adversarial_ood_eval',
    'results', 'v5_final_seed29012_12tasks_5eps', 'tasks.json',
))
CENTER_SCENE = {'pillar_x': 1.0, 'pillar_y': 0.0,
                'robot_yaw_deg': 0.0, 'target_y': 0.0}


class V6TrainingEnv(RewardShapedSinglePillarEnv):
    def __init__(self, seed=TRAIN_SEED, fixed_scene=None):
        super().__init__(True)
        self.permanent_tasks = load_permanent_tasks(PERMANENT_TASKS_PATH)
        self.sampler = V6ScenarioSampler(self.permanent_tasks, seed)
        self.episode_index = 0
        self.fixed_scene = fixed_scene
        self.current_scene = CENTER_SCENE.copy()
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.read_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def _set_pose(self, model, x, y, z, yaw_deg=0.0):
        state = ModelState()
        state.model_name = model
        state.reference_frame = 'world'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        yaw = math.radians(yaw_deg)
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)
        response = self.set_model_state(state)
        if not response.success:
            raise RuntimeError('set_model_state failed for %s' % model)

    def reset(self):
        super().reset()
        scene = (dict(self.fixed_scene) if self.fixed_scene is not None
                 else self.sampler.sample_balanced(self.episode_index))
        self.episode_index += 1
        self.current_scene = scene
        self._set_pose('turtlebot3_burger', 0.0, 0.0, 0.0, scene['robot_yaw_deg'])
        self._set_pose('obstacle_0', scene['pillar_x'], scene['pillar_y'], 0.30)
        self._set_pose('target', 2.0, scene['target_y'], 0.01)
        self.goal_position.position.x = 2.0
        self.goal_position.position.y = scene['target_y']
        rospy.sleep(0.5)
        scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        self._previous_arrival_position = None
        observation = self._initial_observation(scan)
        self.previous_front_clearance = minimum_valid_range(scan)
        self.control_gate.reset()
        return observation

    def actual_scene(self):
        result = {}
        for name in ('turtlebot3_burger', 'obstacle_0', 'target'):
            response = self.read_model_state(name, 'world')
            q = response.pose.orientation
            yaw = math.degrees(math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            ))
            result[name] = {'x': response.pose.position.x,
                            'y': response.pose.position.y, 'yaw_deg': yaw}
        return result

