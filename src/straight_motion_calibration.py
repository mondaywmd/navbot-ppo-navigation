import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

latest = None


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def on_odom(message):
    global latest
    p = message.pose.pose.position
    q = message.pose.pose.orientation
    yaw = math.degrees(math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    ))
    latest = (p.x, p.y, yaw)


def main():
    rospy.init_node("straight_motion_calibration", anonymous=True)
    rospy.Subscriber("/odom", Odometry, on_odom)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(10)

    for _ in range(30):
        if latest is not None:
            break
        rate.sleep()

    if latest is None:
        raise RuntimeError("No /odom received.")

    start_x, start_y, start_yaw = latest
    print("START pos=(%.3f, %.3f), yaw=%.2f deg" % latest, flush=True)

    command = Twist()
    command.linear.x = 0.10
    command.angular.z = 0.0

    for step in range(1, 101):  # 10 seconds
        pub.publish(command)

        if step % 10 == 0 and latest is not None:
            x, y, yaw = latest
            print(
                "t=%2ds pos=(%.3f, %.3f) yaw=%.2f deg yaw_delta=%+.2f deg "
                "cmd=(0.10, 0.00)"
                % (step // 10, x, y, yaw, wrap_deg(yaw - start_yaw)),
                flush=True,
            )
        rate.sleep()

    pub.publish(Twist())
    rospy.sleep(0.2)

    x, y, yaw = latest
    print(
        "END displacement=%.3f m, yaw_delta=%+.2f deg"
        % (math.hypot(x - start_x, y - start_y), wrap_deg(yaw - start_yaw)),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            rospy.Publisher("/cmd_vel", Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
