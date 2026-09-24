"""Deterministic Gazebo scene for the single-pillar residual-PPO experiment.

The saved direct-goal controller remains untouched.  This module only replaces
the random scene reset with one reproducible blocked route:

    robot (0, 0)  ->  pillar (1, 0)  ->  target (2, 0)
"""

import math

import numpy as np
import rospy
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan

from environment_new import Env, diagonal_dis, goal_model_dir
from residual_action import compose_residual_command


PILLAR_X = 1.00
PILLAR_Y = 0.00
TARGET_X = 2.00
TARGET_Y = 0.00
PILLAR_RADIUS = 0.30


PILLAR_SDF = """
<sdf version='1.6'>
  <model name='single_pillar'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry>
          <cylinder><radius>0.30</radius><length>0.60</length></cylinder>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <cylinder><radius>0.30</radius><length>0.60</length></cylinder>
        </geometry>
        <material>
          <ambient>1 0.35 0 1</ambient>
          <diffuse>1 0.35 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


class SinglePillarEnv(Env):
    """The existing LiDAR/reward environment with a fixed blocked reset."""

    def _initial_observation(self, scan):
        """Build the same 16-value observation returned by ``Env.reset``."""
        self.latest_scan = scan
        state, goal_distance, _yaw, _theta, diff_angle, _done, _arrive = self.getState(scan)
        self.goal_distance = self.getGoalDistace()
        self.prev_heading_alignment = math.cos(math.radians(diff_angle))

        normalized_scan = [value / 3.5 for value in state]
        indices = [int(index * len(normalized_scan) / 10) for index in range(10)]
        lidar_features = [normalized_scan[index] for index in indices]
        heading_rad = math.radians(diff_angle)
        observation = lidar_features + [
            0.0,
            0.0,
            goal_distance / diagonal_dis,
            math.sin(heading_rad),
            math.cos(heading_rad),
            diff_angle / 180.0,
        ]
        assert len(observation) == 16
        return np.asarray(observation, dtype=np.float32)

    def step_residual(self, residual_action, past_residual_action):
        """Execute one bounded residual-control step in physical units.

        The policy action is a correction in ``[-1, 1]^2``.  The returned
        observation keeps the existing 16-value contract, with the previous
        residual action occupying the two action-history entries.
        """
        distance = math.hypot(
            self.goal_position.position.x - self.position.x,
            self.goal_position.position.y - self.position.y,
        )
        heading_error = math.radians(float(self.diff_angle))
        command = compose_residual_command(distance, heading_error, residual_action)

        velocity = Twist()
        velocity.linear.x = float(command[0])
        velocity.angular.z = float(command[1])
        self.pub_cmd_vel.publish(velocity)

        scan = rospy.wait_for_message('scan', LaserScan, timeout=5)
        self.latest_scan = scan
        state, rel_dis, _yaw, _rel_theta, diff_angle, done, arrive = self.getState(scan)

        normalized_scan = [value / 3.5 for value in state]
        indices = [int(index * len(normalized_scan) / 10) for index in range(10)]
        lidar_features = [normalized_scan[index] for index in indices]
        previous_residual = np.asarray(past_residual_action, dtype=np.float32)
        if previous_residual.shape != (2,):
            raise ValueError('past_residual_action must have shape (2,)')

        heading_rad = math.radians(diff_angle)
        observation = lidar_features + previous_residual.tolist() + [
            rel_dis / diagonal_dis,
            math.sin(heading_rad),
            math.cos(heading_rad),
            diff_angle / 180.0,
        ]
        observation = np.asarray(observation, dtype=np.float32)
        assert observation.shape == (16,)

        reward = self.setReward(done, arrive)
        self.last_residual_command = command.copy()
        if not hasattr(self, 'residual_command_audit'):
            self.residual_command_audit = []
        self.residual_command_audit.append(
            (np.asarray(residual_action, dtype=np.float32).copy(), command.copy())
        )
        return observation, reward, done, arrive, command.copy()

    def step(self, residual_action, past_residual_action):
        """PPO-compatible entry point; actions are normalized residuals."""
        observation, reward, done, arrive, _command = self.step_residual(
            residual_action, past_residual_action
        )
        return observation, reward, done, arrive

    def reset(self):
        self._previous_arrival_position = None
        self.prev_heading_alignment = None
        self.latest_scan = None

        rospy.wait_for_service('/gazebo/delete_model')
        for model_name in ('target', 'Target', 'obstacle_0', 'obstacle_1'):
            try:
                self.del_model(model_name)
            except rospy.ServiceException:
                pass

        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_world()
        rospy.sleep(0.5)

        rospy.wait_for_service('/gazebo/spawn_sdf_model')

        pillar_pose = Pose()
        pillar_pose.position.x = PILLAR_X
        pillar_pose.position.y = PILLAR_Y
        pillar_pose.position.z = 0.30
        pillar_pose.orientation.w = 1.0
        pillar_response = self.goal(
            'obstacle_0', PILLAR_SDF, 'namespace', pillar_pose, 'world'
        )
        if not pillar_response.success:
            raise RuntimeError('Could not spawn fixed pillar: %s' % pillar_response.status_message)

        self.obstacle_centers = [(PILLAR_X, PILLAR_Y)]
        self.goal_position.position.x = TARGET_X
        self.goal_position.position.y = TARGET_Y
        self.goal_position.position.z = 0.01
        self.goal_position.orientation.w = 1.0
        with open(goal_model_dir, 'r') as target_file:
            target_sdf = target_file.read()
        target_response = self.goal(
            'target', target_sdf, 'namespace', self.goal_position, 'world'
        )
        if not target_response.success:
            raise RuntimeError('Could not spawn fixed target: %s' % target_response.status_message)

        print(
            '[SinglePillarEnv] robot=(0.00, 0.00), pillar=(%.2f, %.2f), target=(%.2f, %.2f)'
            % (PILLAR_X, PILLAR_Y, TARGET_X, TARGET_Y),
            flush=True,
        )

        while not rospy.is_shutdown():
            try:
                scan = rospy.wait_for_message('scan', LaserScan, timeout=5)
                return self._initial_observation(scan)
            except rospy.ROSException:
                rospy.logwarn('Waiting for /scan in SinglePillarEnv.reset')

        raise RuntimeError('ROS shutdown while waiting for /scan')
