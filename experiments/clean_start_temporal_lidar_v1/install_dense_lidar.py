"""One-time running-model migration from 10 front rays to 360 full-circle rays."""
import json
import os
import rospy
from gazebo_msgs.srv import DeleteModel, GetModelState, SpawnModel
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

BACKUP_PARAM = "/clean_start/original_robot_description"

def dense_description(original):
    replacements = {
        "<samples>10</samples>": "<samples>360</samples>",
        "<min_angle>-1.5707975</min_angle>": "<min_angle>-3.14159265</min_angle>",
        "<max_angle>1.5707975</max_angle>": "<max_angle>3.14159265</max_angle>",
    }
    result = original
    for old, new in replacements.items():
        if old not in result: raise AssertionError("missing legacy field: " + old)
        result = result.replace(old, new, 1)
    return result

def main():
    raise RuntimeError("disabled: runtime replacement crashed Gazebo; use an isolated startup launch")
    root = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(root, "results", "dense_lidar_install.json")
    if os.path.exists(result_path): raise RuntimeError("refusing overwrite: " + result_path)
    rospy.init_node("clean_start_dense_lidar_install", anonymous=True)
    original = rospy.get_param("/robot_description"); modified = dense_description(original)
    if not rospy.has_param(BACKUP_PARAM): rospy.set_param(BACKUP_PARAM, original)
    get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    delete = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    spawn = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
    for service in ("/gazebo/get_model_state", "/gazebo/delete_model", "/gazebo/spawn_urdf_model"):
        rospy.wait_for_service(service)
    state = get_state("turtlebot3_burger", "world")
    if not state.success: raise RuntimeError("running robot not found")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1); pub.publish(Twist())
    deleted = delete("turtlebot3_burger")
    if not deleted.success: raise RuntimeError("remove old robot failed: " + deleted.status_message)
    rospy.set_param("/robot_description", modified)
    response = spawn("turtlebot3_burger", modified, "clean_start", state.pose, "world")
    if not response.success:
        rospy.set_param("/robot_description", original)
        restore = spawn("turtlebot3_burger", original, "restore", state.pose, "world")
        raise RuntimeError("dense spawn failed; legacy restore=%s: %s" % (restore.success, response.status_message))
    scan = rospy.wait_for_message("/scan", LaserScan, timeout=10)
    observed = {"points": len(scan.ranges), "angle_min": scan.angle_min,
                "angle_max": scan.angle_max, "angle_increment": scan.angle_increment,
                "range_min": scan.range_min, "range_max": scan.range_max}
    if observed["points"] != 360 or abs(observed["angle_increment"]) > 0.018:
        raise AssertionError("dense LiDAR mismatch: %r" % observed)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as output:
        json.dump({"passed": True, "training": False, "legacy_files_modified": False,
                   "scan": observed}, output, indent=2, sort_keys=True)
    pub.publish(Twist()); print("DENSE_LIDAR_PASS " + json.dumps(observed, sort_keys=True), flush=True)

if __name__ == "__main__": main()
