"""Gazebo reset support for V5; obstacle coordinates remain outside observations."""

import math

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from sensor_msgs.msg import LaserScan

from control_gate import minimum_valid_range
from reward_shaped_env import RewardShapedSinglePillarEnv


CENTER_SCENE = {'pillar_y': 0.0, 'robot_yaw_deg': 0.0, 'target_y': 0.0}


class V5ScenarioEnv(RewardShapedSinglePillarEnv):
    def __init__(self):
        super().__init__(False)
        self.scenario = CENTER_SCENE.copy()
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def _set_pose(self, model_name, x, y, z, yaw_deg=0.0):
        state = ModelState()
        state.model_name = model_name
        state.reference_frame = 'world'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        yaw = math.radians(yaw_deg)
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)
        response = self.set_model_state(state)
        if not response.success:
            raise RuntimeError('set_model_state failed for %s: %s' % (
                model_name, response.status_message
            ))

    def reset(self):
        super().reset()
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        scene = self.scenario
        self._set_pose('turtlebot3_burger', 0.0, 0.0, 0.0, scene['robot_yaw_deg'])
        self._set_pose('obstacle_0', 1.0, scene['pillar_y'], 0.30)
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

    def actual_xy(self):
        result = {}
        for name in ('turtlebot3_burger', 'obstacle_0', 'target'):
            response = self.get_model_state(name, 'world')
            if not response.success:
                raise RuntimeError('get_model_state failed for %s' % name)
            result[name] = [response.pose.position.x, response.pose.position.y]
        return result

