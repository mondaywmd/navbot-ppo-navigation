import math
import time
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

pose = None


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def on_odom(message):
    global pose
    p = message.pose.pose.position
    q = message.pose.pose.orientation
    yaw = math.degrees(math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    ))
    pose = (p.x, p.y, yaw)


def main():
    rospy.init_node("fast_straight_calibration", anonymous=True)
    rospy.Subscriber("/odom", Odometry, on_odom)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    rospy.wait_for_service("/gazebo/reset_simulation")
    rospy.ServiceProxy("/gazebo/reset_simulation", Empty)()
    time.sleep(1.0)

    rate = rospy.Rate(10)
    while pose is None and not rospy.is_shutdown():
        rate.sleep()

    x0, y0, yaw0 = pose
    print("START pos=(%.3f, %.3f), yaw=%.2f deg" % pose, flush=True)

    cmd = Twist()
    cmd.linear.x = 0.30
    cmd.angular.z = 0.0

    for step in range(1, 31):  # 3 seconds, safely short of the wall
        pub.publish(cmd)
        if step % 5 == 0:
            x, y, yaw = pose
            print(
                "t=%.1fs pos=(%.3f, %.3f) yaw=%.2f yaw_delta=%+.2f cmd=(0.30, 0.00)"
                % (step / 10.0, x, y, yaw, wrap_deg(yaw - yaw0)),
                flush=True,
            )
        rate.sleep()

    pub.publish(Twist())
    x, y, yaw = pose
    print(
        "END displacement=%.3f m, yaw_delta=%+.2f deg"
        % (math.hypot(x - x0, y - y0), wrap_deg(yaw - yaw0)),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        rospy.Publisher("/cmd_vel", Twist, queue_size=1).publish(Twist())
