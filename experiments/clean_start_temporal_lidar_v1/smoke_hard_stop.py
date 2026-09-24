"""Drive toward the center pillar and verify the full-scan shield stops safely."""
import json
import os
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from safety_shield import filter_command, front_minimum, valid_scan

def set_robot(service):
    state = ModelState(); state.model_name = "turtlebot3_burger"; state.reference_frame = "world"
    state.pose.orientation.w = 1.0
    response = service(state)
    if not response.success: raise RuntimeError(response.status_message)

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "results", "hard_stop_smoke.json")
    if os.path.exists(path): raise RuntimeError("refusing overwrite: " + path)
    rospy.init_node("clean_start_hard_stop_smoke", anonymous=True)
    rospy.wait_for_service("/gazebo/set_model_state")
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1); pub.publish(Twist())
    set_robot(set_state); rospy.sleep(0.5)
    minimum = float("inf"); first_intervention = None; reasons = {}
    try:
        for step in range(1, 51):
            scan = rospy.wait_for_message("/scan", LaserScan, timeout=5)
            minimum = min(minimum, float(valid_scan(scan).min()))
            linear, angular, reason = filter_command(0.25, 0.0, scan)
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason != "clear" and first_intervention is None: first_intervention = step
            command = Twist(); command.linear.x = linear; command.angular.z = angular
            pub.publish(command)
        pub.publish(Twist()); rospy.sleep(0.4)
        final_scan = rospy.wait_for_message("/scan", LaserScan, timeout=5)
        minimum = min(minimum, float(valid_scan(final_scan).min()))
        result = {"passed": minimum >= 0.30 and first_intervention is not None,
                  "training": False, "steps": 50, "requested_linear": 0.25,
                  "minimum_lidar": minimum, "final_front": front_minimum(final_scan),
                  "first_intervention_step": first_intervention, "reasons": reasons}
        if not result["passed"]: raise AssertionError("hard-stop smoke failed: %r" % result)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as output: json.dump(result, output, indent=2, sort_keys=True)
        print("HARD_STOP_PASS " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        pub.publish(Twist()); set_robot(set_state); pub.publish(Twist())
        print("ROBOT_RESET_ZERO", flush=True)

if __name__ == "__main__": main()
