#!/usr/bin/env python3
import os

import roslaunch
import rospy
import numpy as np
import math
from math import pi
import random
import time
# import tf.transformations
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState
from pick_laser import Pick
# from tf.transformations import euler_from_quaternion
from wall_penalty import pen_wall
diagonal_dis = math.sqrt(2) * (3.8 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')
len_batch = 36  # 360 laser points / 36 = 10 picked laser features

class Env():
    def __init__(self, is_training, use_vision=False, vision_dim=64):
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.past_distance = 0.
        self.prev_heading_alignment = None
        self.latest_scan = None  # Most recent LiDAR scan for hybrid navigation.
        self.sum1 = 0
        self.sum2 = 0
        # Keep training and evaluation success criteria identical.
        self.threshold_arrive = 0.2
        
        # Vision setup
        self.use_vision = use_vision
        self.vision_dim = vision_dim
        self.latest_image = None
        self.image_ok = False
        
        if self.use_vision:
            print(f"[Env] Initializing vision mode (raw image only, no encoder in env)", flush=True)
            # Subscribe to camera topic
            # Real Gazebo camera topic: /camera/rgb/image_raw (published by /gazebo plugin)
            self.image_topic = '/camera/rgb/image_raw'
            self.sub_camera = rospy.Subscriber(self.image_topic, Image, self.imageCallback)
            print(f"[Env] Subscribed to {self.image_topic}", flush=True)
            
            # Wait briefly for first image
            print(f"[Env] Waiting for first image on {self.image_topic}...", flush=True)
            timeout = rospy.Time.now() + rospy.Duration(5.0)
            while self.latest_image is None and rospy.Time.now() < timeout and not rospy.is_shutdown():
                rospy.sleep(0.1)
            
            if self.latest_image is not None:
                self.image_ok = True
                print(f"[Env] First image received! encoding={self.latest_image.encoding}, "
                      f"size={self.latest_image.width}x{self.latest_image.height}, "
                      f"timestamp={self.latest_image.header.stamp.to_sec()}", flush=True)
            else:
                print(f"[Env] WARNING: No image received on {self.image_topic} within timeout", flush=True)
                print(f"[Env] Vision features will be zeros. Consider starting fake_camera_publisher.py", flush=True)

    def imageCallback(self, msg):
        """Store latest image from camera as numpy array"""
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            # Convert ROS Image to numpy uint8 HxWx3 (BGR)
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert BGR to RGB
            import cv2
            self.latest_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            if not self.image_ok:
                self.image_ok = True
                print(f"[Env] Image callback received: {msg.encoding}, {msg.width}x{msg.height}, converted to RGB numpy array", flush=True)
        except Exception as e:
            if not hasattr(self, '_img_convert_error_logged'):
                self._img_convert_error_logged = True
                print(f"[Env] Image conversion error: {e}", flush=True)
    
    def getLatestImage(self):
        """Return latest image as numpy uint8 HxWx3 RGB, or None if unavailable"""
        if not self.use_vision or self.latest_image is None:
            return None
        return self.latest_image
    
    def getVisionFeatures(self):
        """Deprecated - vision features now computed in policy network"""
        # Return zeros for backward compatibility
        if not self.use_vision:
            return np.zeros(self.vision_dim)
        return np.zeros(self.vision_dim)

    # def close(self):
    #     """
    #     Close environment. No other method calls possible afterwards.
    #     """
    #     self.roslaunch.shutdown()
    #     time.sleep(10)

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance

        return goal_distance
    
    def verify_target_pose(self, label=""):
        """Verify target model exists and get its pose"""
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=1.0)
            resp = self.get_model_state('target', 'world')
            if resp.success:
                pos = resp.pose.position
                print(f"[Env] verify_target_pose [{label}]: target at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})", flush=True)
                return True, pos
            else:
                print(f"[Env] verify_target_pose [{label}]: get_model_state failed - {resp.status_message}", flush=True)
                return False, None
        except Exception as e:
            print(f"[Env] verify_target_pose [{label}]: Exception - {e}", flush=True)
            return False, None

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        # Signed error: positive means the target is to the left.
        rel_dis_x = self.goal_position.position.x - self.position.x
        rel_dis_y = self.goal_position.position.y - self.position.y
        rel_theta = math.degrees(math.atan2(rel_dis_y, rel_dis_x)) % 360.0
        diff_angle = (rel_theta - yaw + 180.0) % 360.0 - 180.0
        rel_theta = round(rel_theta, 2)
        diff_angle = round(diff_angle, 2)

        # print(diff_angle)
        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getState(self, scan):
        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2
        done = False
        arrive = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_position = np.array(
            [self.position.x, self.position.y],
            dtype=np.float32,
        )
        target_position = np.array(
            [
                self.goal_position.position.x,
                self.goal_position.position.y,
            ],
            dtype=np.float32,
        )
        current_distance = float(
            np.linalg.norm(target_position - current_position)
        )

        # Normal arrival: the current odometry sample is inside success circle.
        arrive = current_distance <= self.threshold_arrive

        # Crossing arrival: the robot may move across the circle between
        # consecutive LiDAR/odometry samples. Count that as success too.
        previous_position = getattr(self, "_previous_arrival_position", None)
        if not arrive and previous_position is not None:
            segment = current_position - previous_position
            segment_length_sq = float(np.dot(segment, segment))

            if segment_length_sq > 1e-9:
                t = float(
                    np.dot(target_position - previous_position, segment)
                    / segment_length_sq
                )
                t = max(0.0, min(1.0, t))
                closest = previous_position + t * segment

                if float(np.linalg.norm(target_position - closest)) <= self.threshold_arrive:
                    arrive = True
                    print("[Env] Arrival: trajectory crossed the success circle.", flush=True)

        self._previous_arrival_position = current_position

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive):
        current_distance = math.hypot(
            self.goal_position.position.x - self.position.x,
            self.goal_position.position.y - self.position.y
        )
        distance_rate = self.past_distance - current_distance

        heading_alignment = math.cos(math.radians(self.diff_angle))
        heading_improvement = 0.0 if self.prev_heading_alignment is None else (
            heading_alignment - self.prev_heading_alignment
        )

        # Prefer both turning toward the goal and keeping the robot
        # pointed at the goal while moving forward.
        reward = (
            100.0 * distance_rate - 1.0
            + 12.0 * heading_improvement
            + 0.5 * heading_alignment
        )
        self.past_distance = current_distance
        self.prev_heading_alignment = heading_alignment

        if done:
            reward = -300.0  # Collision is worse than a timeout.
            self.pub_cmd_vel.publish(Twist())

        if arrive:
            reward = 500.0
            self.pub_cmd_vel.publish(Twist())

        return reward

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        # Turn first, but avoid crawling when the target starts behind.
        # Far away: keep aggressive motion. Near the goal: slow down to settle.
        distance_to_goal = math.hypot(
            self.goal_position.position.x - self.position.x,
            self.goal_position.position.y - self.position.y
        )
        near_goal_scale = min(1.0, max(0.35, distance_to_goal / 0.75))
        turn_fraction = min(abs(self.diff_angle) / 180.0, 1.0)
        linear_scale = max(0.25, 1.0 - 0.75 * turn_fraction) * near_goal_scale

        # Do not weaken a necessary large turn when the target is behind/at the side.
        heading_settle_scale = near_goal_scale if abs(self.diff_angle) < 70.0 else 1.0
        angular_scale = 0.45 + 0.55 * heading_settle_scale

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.30 * linear_vel * linear_scale
        vel_cmd.angular.z = 1.5 * ang_vel * angular_scale
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        self.latest_scan = data
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]
        
        # Uniform sampling: select 10 evenly-spaced samples from normalized scan
        L = len(state)
        indices = [int(i * L / 10) for i in range(10)]
        lidar_features = [state[idx] for idx in indices]
        
        # Build base state: 10 lidar + 2 past_action + 4 goalpose
        assert len(lidar_features) == 10, f"LiDAR features must be 10, got {len(lidar_features)}"
        base_state = lidar_features.copy()
        for pa in past_action:
            base_state.append(pa)
        heading_rad = math.radians(diff_angle)
        base_state = base_state + [
            rel_dis / diagonal_dis,
            math.sin(heading_rad),
            math.cos(heading_rad),
            diff_angle / 180,
        ]
        
        # Assertion: Verify base state is exactly 16-d (10 lidar + 2 past_action + 4 goalpose)
        assert len(base_state) == 16, f"Base state must be 16-d, got {len(base_state)}"
        
        # Vision mode: return base state vector only (NO vision features appended here)
        # Image will be handled separately in PPO rollout
        
        reward = self.setReward(done, arrive)
        return np.asarray(base_state), reward, done, arrive

    def reset(self):
        # A new episode must not use the previous episode's trajectory.
        self._previous_arrival_position = None

        # Reset the env
        
        # Delete old target if it exists
        rospy.wait_for_service('/gazebo/delete_model')
        for model_name in ('target', 'obstacle_0', 'obstacle_1'):
            try:
                self.del_model(model_name)
            except:
                pass
        
        # Reset world (resets robot pose and physics)
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world()
        except (rospy.ServiceException) as e:
            pass
        rospy.sleep(0.5)

        # Randomly spawn two pillars on two different-radius circles,
        # then spawn one large target clear of both pillars.
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            from geometry_msgs.msg import Pose

            pillar_sdf = """
<sdf version='1.6'>
  <model name='random_pillar'>
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

            if getattr(self, 'fixed_single_pillar', False):
                # Deterministic one-pillar scene for tangent-navigation tests.
                obstacle_x, obstacle_y = 0.75, 0.45

                pillar_pose = Pose()
                pillar_pose.position.x = obstacle_x
                pillar_pose.position.y = obstacle_y
                pillar_pose.position.z = 0.30
                pillar_pose.orientation.w = 1.0

                response = self.goal(
                    'obstacle_0',
                    pillar_sdf,
                    'namespace',
                    pillar_pose,
                    'world'
                )
                if not response.success:
                    raise RuntimeError(
                        "Could not spawn fixed single pillar: %s"
                        % response.status_message
                    )

                self.obstacle_centers = [(obstacle_x, obstacle_y)]

                goal_x, goal_y = 2.20, 1.80
                goal_urdf = open(goal_model_dir, "r").read()

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
                self.goal_position.position.z = 0.01
                self.goal_position.orientation.w = 1.0

                spawn_response = self.goal(
                    'target',
                    goal_urdf,
                    'namespace',
                    self.goal_position,
                    'world'
                )
                if not spawn_response.success:
                    raise RuntimeError(
                        "Could not spawn fixed target: %s"
                        % spawn_response.status_message
                    )
            else:
                # One random angle on each circle, centred at the robot reset pose.
                # Two different-radius rings, close enough that a large
                # target can still fit behind a pillar inside the arena.
                PILLAR_RADII = (0.65, 0.90)
                obstacle_centers = []

                for index, radius in enumerate(PILLAR_RADII):
                    spawned = False

                    for attempt in range(100):
                        # Random angle near a diagonal direction. This leaves
                        # enough room for the large target behind the pillar.
                        diagonal_base = random.choice((
                            math.pi / 4,
                            3 * math.pi / 4,
                            5 * math.pi / 4,
                            7 * math.pi / 4,
                        ))
                        angle = diagonal_base + random.uniform(-0.10, 0.10)
                        obstacle_x = radius * math.cos(angle)
                        obstacle_y = radius * math.sin(angle)

                        # Pillars must not overlap each other.
                        if any(
                            math.hypot(obstacle_x - ox, obstacle_y - oy) < 1.10
                            for ox, oy in obstacle_centers
                        ):
                            continue

                        pillar_pose = Pose()
                        pillar_pose.position.x = obstacle_x
                        pillar_pose.position.y = obstacle_y
                        pillar_pose.position.z = 0.30
                        pillar_pose.orientation.w = 1.0

                        response = self.goal(
                            'obstacle_%d' % index,
                            pillar_sdf,
                            'namespace',
                            pillar_pose,
                            'world'
                        )

                        if response.success:
                            obstacle_centers.append((obstacle_x, obstacle_y))
                            spawned = True
                            break

                    if not spawned:
                        raise RuntimeError("Could not place random pillar %d" % index)

                self.obstacle_centers = obstacle_centers

                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'
                target.model_xml = goal_urdf

                for attempt in range(1000):
                    goal_x = random.uniform(-2.2, 2.2)
                    goal_y = random.uniform(-2.2, 2.2)

                    in_blocked_zone = (
                        (1.7 <= goal_x <= 2.3 and -1.2 <= goal_y <= 1.2)
                        or (-2.3 <= goal_x <= -1.7 and -1.2 <= goal_y <= 1.2)
                        or (-1.2 <= goal_x <= 1.2 and 1.7 <= goal_y <= 2.3)
                        or (-1.2 <= goal_x <= 1.2 and -2.3 <= goal_y <= -1.7)
                    )

                    start_distance = math.hypot(goal_x, goal_y)
                    target_clear_of_pillars = all(
                        math.hypot(goal_x - ox, goal_y - oy) >= 1.90
                        for ox, oy in obstacle_centers
                    )

                    # The goal must sit in the shadow of at least one pillar:
                    # a straight route from the reset robot at (0, 0) is blocked.
                    segment_len_sq = goal_x * goal_x + goal_y * goal_y
                    direct_path_blocked = False

                    for ox, oy in obstacle_centers:
                        projection = (
                            ox * goal_x + oy * goal_y
                        ) / segment_len_sq

                        if 0.0 < projection < 1.0:
                            closest_x = projection * goal_x
                            closest_y = projection * goal_y
                            path_clearance = math.hypot(
                                ox - closest_x,
                                oy - closest_y
                            )

                            # Pillar radius 0.30 m plus robot safety margin.
                            if path_clearance <= 0.50:
                                direct_path_blocked = True
                                break

                    if (
                        start_distance >= 0.8
                        and target_clear_of_pillars
                        and direct_path_blocked
                        and not in_blocked_zone
                    ):
                        break
                else:
                    raise RuntimeError("Could not place a target clear of both pillars")

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
                self.goal_position.position.z = 0.01
                self.goal_position.orientation.w = 1.0

                spawn_response = self.goal(
                    target.model_name,
                    target.model_xml,
                    'namespace',
                    self.goal_position,
                    'world'
                )
                if not spawn_response.success:
                    rospy.logerr("Target spawn failed: %s", spawn_response.status_message)

        except (rospy.ServiceException) as e:
            rospy.logerr("Gazebo service error: %s", e)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        self.goal_distance = self.getGoalDistace()
        
        self.latest_scan = data
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        self.prev_heading_alignment = math.cos(math.radians(diff_angle))
        state = [i / 3.5 for i in state]
        
        # Uniform sampling: select 10 evenly-spaced samples from normalized scan
        L = len(state)
        indices = [int(i * L / 10) for i in range(10)]
        lidar_features = [state[idx] for idx in indices]
        
        # Build base state: 10 lidar + 2 past_action + 4 goalpose
        assert len(lidar_features) == 10, f"LiDAR features must be 10, got {len(lidar_features)}"
        base_state = lidar_features.copy()
        base_state.append(0)  # past_action[0]
        base_state.append(0)  # past_action[1]
        heading_rad = math.radians(diff_angle)
        base_state = base_state + [
            rel_dis / diagonal_dis,
            math.sin(heading_rad),
            math.cos(heading_rad),
            diff_angle / 180,
        ]

        # Assertion: Verify base state is exactly 16-d (10 lidar + 2 past_action + 4 goalpose)
        assert len(base_state) == 16, f"Base state must be 16-d, got {len(base_state)}"

        # Vision mode: return base state vector only (NO vision features appended here)
        # Image will be handled separately in PPO rollout

        return np.asarray(base_state)
