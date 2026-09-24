"""Isolated variable-radius, one-to-three-pillar Gazebo reset environment."""
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan
from control_gate import minimum_valid_range
from scenario_env import CENTER_SCENE, V5ScenarioEnv


def pillar_sdf(name, radius):
    radius = float(radius)
    if not 0.18 <= radius <= 0.45:
        raise ValueError("pillar radius outside prepared range")
    return """<sdf version='1.6'><model name='%s'><static>true</static>
<link name='link'><collision name='collision'><geometry><cylinder><radius>%.9f</radius><length>0.60</length></cylinder></geometry></collision>
<visual name='visual'><geometry><cylinder><radius>%.9f</radius><length>0.60</length></cylinder></geometry>
<material><ambient>1 0.35 0 1</ambient><diffuse>1 0.35 0 1</diffuse></material></visual></link></model></sdf>""" % (name, radius, radius)


class WideStaticEnv(V5ScenarioEnv):
    def __init__(self):
        super().__init__()
        self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.wide_scene = None
        self.generation = 0
        self.wide_model_names = []
        self._movable_target_ready = False

    def _delete_if_present(self, name):
        try:
            self.delete_model(name)
        except rospy.ServiceException:
            pass
        for _ in range(50):
            if not self.get_model_state(name, "world").success:
                return
            rospy.sleep(0.02)
        raise RuntimeError("timed out deleting " + name)

    def _spawn_pillar(self, name, obstacle):
        pose = Pose(); pose.position.x = obstacle["x"]; pose.position.y = obstacle["y"]
        pose.position.z = 0.30; pose.orientation.w = 1.0
        response = self.spawn_model(name, pillar_sdf(name, obstacle["radius"]),
                                    "v8", pose, "world")
        if not response.success:
            raise RuntimeError("spawn failed for %s: %s" % (name, response.status_message))
        for _ in range(50):
            if self.get_model_state(name, "world").success:
                return
            rospy.sleep(0.02)
        raise RuntimeError("spawned model not observable: " + name)

    def _movable_target_sdf(self):
        return """<sdf version='1.6'><model name='target'><static>false</static>
<link name='target_link'><gravity>false</gravity><kinematic>true</kinematic>
<visual name='blue'><pose>0 0 0.015 0 0 0</pose><geometry><cylinder><radius>1.5</radius><length>0.03</length></cylinder></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Blue</name></script></material><cast_shadows>false</cast_shadows></visual>
<visual name='red'><pose>0 0 0.025 0 0 0</pose><geometry><cylinder><radius>1.0</radius><length>0.04</length></cylinder></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Red</name></script></material><cast_shadows>false</cast_shadows></visual>
<visual name='yellow'><pose>0 0 0.035 0 0 0</pose><geometry><cylinder><radius>0.5</radius><length>0.05</length></cylinder></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Yellow</name></script></material><cast_shadows>false</cast_shadows></visual>
</link></model></sdf>"""

    def _ensure_target(self, x, y):
        """One-time static-to-movable migration, then pose-only updates."""
        if (not self._movable_target_ready or
                not self.get_model_state("target", "world").success):
            if self.get_model_state("target", "world").success:
                self._delete_if_present("target")
            pose = Pose()
            pose.position.x = x; pose.position.y = y; pose.position.z = 0.01
            pose.orientation.w = 1.0
            response = self.spawn_model("target", self._movable_target_sdf(),
                                        "namespace", pose, "world")
            if not response.success:
                raise RuntimeError("could not create movable target: " + response.status_message)
            for _ in range(50):
                if self.get_model_state("target", "world").success:
                    break
                rospy.sleep(0.02)
            else:
                raise RuntimeError("movable target disappeared after spawn")
            self._movable_target_ready = True
        else:
            self._set_pose("target", x, y, 0.01)

    def _finish_reset(self, target_x, target_y):
        """Reset episode state without invoking the legacy delete/spawn reset."""
        self.goal_position.position.x = target_x
        self.goal_position.position.y = target_y
        self._previous_arrival_position = None
        self.prev_heading_alignment = None
        self.latest_scan = None
        self.control_gate.reset()
        rospy.sleep(0.5)
        scan = rospy.wait_for_message("/scan", LaserScan, timeout=5)
        observation = self._initial_observation(scan)
        self.previous_front_clearance = minimum_valid_range(scan)
        if observation.shape != (16,):
            raise AssertionError("Actor observation changed")
        return observation

    def reset_wide(self, scene):
        if not 1 <= len(scene["obstacles"]) <= 3:
            raise ValueError("scene must contain one to three obstacles")
        self.scenario = {"pillar_y": 0.0, "target_y": scene["target_y"],
                         "robot_yaw_deg": scene["robot_yaw_deg"]}
        self.pub_cmd_vel.publish(Twist())
        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/delete_model")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self._set_pose("turtlebot3_burger", 0.0, 0.0, 0.0,
                       scene["robot_yaw_deg"])
        old_names = list(self.wide_model_names)
        for index in range(3): self._delete_if_present("obstacle_%d" % index)
        for name in old_names: self._delete_if_present(name)
        self.generation += 1
        self.wide_model_names = ["v8_pillar_g%04d_%d" % (self.generation, index)
                                 for index in range(len(scene["obstacles"]))]
        for name, obstacle in zip(self.wide_model_names, scene["obstacles"]):
            self._spawn_pillar(name, obstacle)
        self._ensure_target(scene["target_x"], scene["target_y"])
        observation = self._finish_reset(scene["target_x"], scene["target_y"])
        self.wide_scene = scene
        return observation

    def restore_center(self):
        for name in list(self.wide_model_names): self._delete_if_present(name)
        self.wide_model_names = []
        for index in range(3): self._delete_if_present("obstacle_%d" % index)
        self._spawn_pillar("obstacle_0", {"x": 1.0, "y": 0.0, "radius": 0.30})
        self._set_pose("turtlebot3_burger", 0.0, 0.0, 0.0, 0.0)
        self._ensure_target(2.0, 0.0)
        self.scenario = CENTER_SCENE.copy()
        self.wide_scene = None
        observation = self._finish_reset(2.0, 0.0)
        self.pub_cmd_vel.publish(Twist())
        return observation
