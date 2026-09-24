import argparse
import math
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from single_pillar_env import SinglePillarEnv


ARRIVE_RADIUS = 0.20
ALIGN_TOLERANCE_DEG = 1.0
SETTLE_SECONDS = 0.8
CONTROL_HZ = 10.0
MAX_STEPS = 350


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def clamp(value, low, high):
    return max(low, min(high, value))


def distance_to_segment(point, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return math.hypot(point[0] - start[0], point[1] - start[1])

    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = clamp(t, 0.0, 1.0)
    closest_x = start[0] + t * dx
    closest_y = start[1] + t * dy
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def speed_for_distance(distance, cruise_speed=0.25):
    # Conservative verified cruise speed; brake continuously near the target.
    if distance > 1.0:
        return min(0.25, cruise_speed)
    if distance > 0.50:
        return min(0.10 + (distance - 0.50) * 0.30, cruise_speed)
    if distance > ARRIVE_RADIUS:
        return min(0.05 + (distance - ARRIVE_RADIUS) * 0.166667, cruise_speed)
    return 0.0

def publish(pub, linear, angular):
    msg = Twist()
    msg.linear.x = linear
    msg.angular.z = angular
    pub.publish(msg)



latest_angular_z = None
latest_odom_yaw_deg = None


def on_odom_twist(message):
    global latest_angular_z, latest_odom_yaw_deg
    latest_angular_z = float(message.twist.twist.angular.z)

    q = message.pose.pose.orientation
    latest_odom_yaw_deg = math.degrees(math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    ))


def wait_until_angular_stable(env, pub, rate):
    """Wait for the actual physical angular velocity to become zero."""
    stable_samples = 0

    while not rospy.is_shutdown():
        publish(pub, 0.0, 0.0)

        if latest_angular_z is not None and abs(latest_angular_z) < 0.01:
            stable_samples += 1
            if stable_samples >= 10:  # 1.0 second of real zero angular velocity
                print(
                    "ANGULARLY STABLE: odom angular.z=%.4f rad/s"
                    % latest_angular_z,
                    flush=True,
                )
                return
        else:
            stable_samples = 0

        rate.sleep()

def align_to_current_target(env, pub, rate, target):
    """Rotate, stop fully, then return the fresh target bearing."""
    while not rospy.is_shutdown():
        dx = target[0] - float(env.position.x)
        dy = target[1] - float(env.position.y)
        heading = math.degrees(math.atan2(dy, dx))
        error = wrap_deg(heading - float(env.yaw))

        if abs(error) <= ALIGN_TOLERANCE_DEG:
            publish(pub, 0.0, 0.0)
            wait_until_angular_stable(env, pub, rate)
            return heading

        publish(pub, 0.0, clamp(math.radians(error) * 0.8, -0.25, 0.25))
        rate.sleep()


def lidar_collision():
    try:
        scan = rospy.wait_for_message("/scan", LaserScan, timeout=0.2)
    except rospy.ROSException:
        return False

    valid = [
        value for value in scan.ranges
        if math.isfinite(value) and scan.range_min <= value <= scan.range_max
    ]
    return bool(valid) and min(valid) < 0.20

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cruise-speed", type=float, default=0.25,
        help="Maximum straight-line speed in m/s (default: 0.25).",
    )
    parser.add_argument(
        "--post-reset-pause", type=float, default=0.0,
        help="Seconds to keep the reset scene stationary before alignment.",
    )
    args = parser.parse_args()
    if not 0.0 < args.cruise_speed <= 0.25:
        parser.error("--cruise-speed must be in (0, 0.25]")
    if args.post_reset_pause < 0.0:
        parser.error("--post-reset-pause must be non-negative")
    return args


def main():
    args = parse_args()
    rospy.init_node("raw_cmdvel_direct_goal", anonymous=True)
    rospy.Subscriber("/odom", Odometry, on_odom_twist)
    env = SinglePillarEnv(False)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(CONTROL_HZ)

    env.reset()

    publish(pub, 0.0, 0.0)
    if args.post_reset_pause:
        print(
            "VISUAL CHECK: reset complete; holding still for %.1f seconds."
            % args.post_reset_pause,
            flush=True,
        )
        rospy.sleep(args.post_reset_pause)

    target = (float(env.goal_position.position.x), float(env.goal_position.position.y))
    print("Blocked target=(%.2f, %.2f); pillar=(1.00, 0.00)" % target, flush=True)

    # Phase 1: rotate, wait for physical rotation to stop, then re-aim.
    align_to_current_target(env, pub, rate, target)
    locked_heading = align_to_current_target(env, pub, rate, target)

    print(
        "ALIGNED AND STABLE: yaw=%.1f deg, locked_heading=%.1f deg"
        % (float(env.yaw), locked_heading),
        flush=True,
    )

    print("DRIVE: locked_heading=%.1f deg; slowing down near goal" % locked_heading,
          flush=True)

    # Low-speed launch: expose any wheel mismatch before committing to the line.
    print("LAUNCH CHECK: low-speed straight motion, then one micro re-alignment.",
          flush=True)
    for launch_step in range(1, 11):
        if lidar_collision():
            publish(pub, 0.0, 0.0)
            print("COLLISION during launch check.", flush=True)
            return

        publish(pub, min(args.cruise_speed, 0.05 + 0.01 * launch_step), 0.0)
        rate.sleep()

    # The vehicle is stopped before any correction; it never steers while moving.
    publish(pub, 0.0, 0.0)
    wait_until_angular_stable(env, pub, rate)
    locked_heading = align_to_current_target(env, pub, rate, target)
    print(
        "MICRO-ALIGNED: yaw=%.1f deg, final_heading=%.1f deg"
        % (float(env.yaw), locked_heading),
        flush=True,
    )

    previous = (float(env.position.x), float(env.position.y))
    drive_start_time = time.monotonic()

    # Phase 2: no steering; smoothly accelerate into the locked straight line.
    for step in range(1, MAX_STEPS + 1):
        position = (float(env.position.x), float(env.position.y))
        distance = math.hypot(target[0] - position[0], target[1] - position[1])

        if lidar_collision():
            publish(pub, 0.0, 0.0)
            print("COLLISION: LiDAR distance < 0.20 m, steps=%d" % step, flush=True)
            return

        # Count both entering and crossing the arrival circle as success.
        crossed_goal = (
            distance_to_segment(target, previous, position) <= ARRIVE_RADIUS
        )
        if distance <= ARRIVE_RADIUS or crossed_goal:
            publish(pub, 0.0, 0.0)
            print(
                "SUCCESS: distance=%.3f m, crossed=%s, steps=%d"
                % (distance, crossed_goal, step),
                flush=True,
            )
            return

        # Gently preserve the initial straight-line heading while moving.
        # No route replanning: only compensate physical yaw drift.
        current_yaw = (
            latest_odom_yaw_deg
            if latest_odom_yaw_deg is not None
            else float(env.yaw)
        )
        heading_error = wrap_deg(locked_heading - current_yaw)

        if abs(heading_error) < 0.5:
            angular = 0.0
        else:
            angular = clamp(math.radians(heading_error) * 0.8, -0.08, 0.08)

        target_speed = speed_for_distance(distance, args.cruise_speed)
        linear = min(target_speed, args.cruise_speed)

        if abs(heading_error) > 6.0:
            linear = min(linear, 0.15)

        publish(pub, linear, angular)

        if step % 10 == 0:
            print(
                "DRIVE step=%d pos=(%.2f, %.2f) yaw=%.1f "
                "cmd=(%.2f, %.3f) distance=%.3f"
                % (
                    step, position[0], position[1], float(env.yaw),
                    linear, angular, distance,
                ),
                flush=True,
            )

        previous = position
        rate.sleep()

    publish(pub, 0.0, 0.0)
    print("TIMEOUT: distance=%.3f m" % distance, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            rospy.Publisher("/cmd_vel", Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
