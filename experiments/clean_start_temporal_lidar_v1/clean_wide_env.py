"""Clean-start Gazebo environment with a process-stable movable target."""
import math
import time

import rospy
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelStates

from wide_env import WideStaticEnv


class CleanWideStaticEnv(WideStaticEnv):
    """Never delete/re-spawn an existing target merely because Python restarted."""

    def _ensure_target(self, x, y):
        state=self.get_model_state("target","world")
        if not state.success:
            pose=Pose();pose.position.x=x;pose.position.y=y;pose.position.z=0.01
            pose.orientation.w=1.0
            response=self.spawn_model("target",self._movable_target_sdf(),"",pose,"world")
            if not response.success:
                raise RuntimeError("could not create movable target: "+response.status_message)
        self._movable_target_ready=True
        # Pose updates are retried and verified against gzserver.  Crucially,
        # the existing model is never deleted, so gzclient receives a normal
        # continuous model-state update across independent Python processes.
        for _ in range(5):
            self._set_pose("target",x,y,0.01)
            time.sleep(0.05)
            state=self.get_model_state("target","world")
            if (state.success and abs(state.pose.position.x-x)<=0.002 and
                    abs(state.pose.position.y-y)<=0.002):
                return
        raise RuntimeError("target pose verification failed for (%.3f, %.3f)"%(x,y))

    def _brake_before_reset(self):
        for _ in range(5):
            self.pub_cmd_vel.publish(Twist());time.sleep(0.02)

    def _remove_all_dynamic_pillars(self):
        """Clean models left by any earlier process, not only this instance."""
        states=rospy.wait_for_message("/gazebo/model_states",ModelStates,timeout=5)
        stale=[name for name in states.name if name.startswith(("v8_pillar_","clean_pillar_"))]
        for name in stale:self._delete_if_present(name)
        self.wide_model_names=[]

    def reset_wide(self, scene):
        if not 1<=len(scene["obstacles"])<=5:
            raise ValueError("clean scene must contain one to five obstacles")
        self._brake_before_reset()
        self._remove_all_dynamic_pillars()
        self.scenario={"pillar_y":0.0,"target_y":scene["target_y"],
                       "robot_yaw_deg":scene["robot_yaw_deg"]}
        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/delete_model")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self._set_pose("turtlebot3_burger",0.0,0.0,0.0,scene["robot_yaw_deg"])
        for index in range(5):self._delete_if_present("obstacle_%d"%index)
        self.generation+=1
        self.wide_model_names=["clean_pillar_g%04d_%d"%(self.generation,index)
                               for index in range(len(scene["obstacles"]))]
        for name,obstacle in zip(self.wide_model_names,scene["obstacles"]):
            self._spawn_pillar(name,obstacle)
        self._ensure_target(scene["target_x"],scene["target_y"])
        result=self._finish_reset(scene["target_x"],scene["target_y"])
        self.wide_scene=scene
        self.validate_live_scene(scene)
        return result

    def validate_live_scene(self,scene):
        states=rospy.wait_for_message("/gazebo/model_states",ModelStates,timeout=5)
        live=[name for name in states.name if name.startswith(("v8_pillar_","clean_pillar_"))]
        if len(live)!=len(scene["obstacles"]):
            raise RuntimeError("live pillar count %d != requested %d"%(len(live),len(scene["obstacles"])))
        target=self.get_model_state("target","world")
        if (not target.success or abs(target.pose.position.x-scene["target_x"])>0.002 or
                abs(target.pose.position.y-scene["target_y"])>0.002):
            raise RuntimeError("live target pose does not match scene")
        for name,expected in zip(self.wide_model_names,scene["obstacles"]):
            state=self.get_model_state(name,"world")
            if (not state.success or abs(state.pose.position.x-expected["x"])>0.002 or
                    abs(state.pose.position.y-expected["y"])>0.002):
                raise RuntimeError("live pillar pose does not match scene: "+name)
            yellow_gap=math.hypot(state.pose.position.x-target.pose.position.x,
                                  state.pose.position.y-target.pose.position.y)-expected["radius"]-0.50
            if yellow_gap<0:
                raise RuntimeError("pillar overlaps yellow target: "+name)
        return True

    def restore_center(self):
        self._brake_before_reset()
        self._remove_all_dynamic_pillars()
        result=super().restore_center()
        self._brake_before_reset()
        return result
