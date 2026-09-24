#!/usr/bin/env python3
import argparse
import math

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from environment_new import Env


ALIGN_DEG = 15.0
DRIVE_REALIGN_DEG = 25.0
TARGET_OUTER_RADIUS = 1.5

PILLAR_DIAMETER = 0.60
ESCAPE_DISTANCE = 1.5 * PILLAR_DIAMETER  # 0.90 m
ESCAPE_LINEAR_ACTION = 2.0
GOAL_LINEAR_ACTION = 5.0
EMERGENCY_FRONT_DISTANCE = 0.45


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def sector_nearest(scan, center_deg, half_width_deg, fallback=3.5):
    best_distance = fallback
    best_bearing = center_deg

    for index, distance in enumerate(scan.ranges):
        if not math.isfinite(distance):
            continue
        if distance < scan.range_min or distance > scan.range_max:
            continue

        bearing_deg = math.degrees(
            scan.angle_min + index * scan.angle_increment
        )

        if (
            abs(wrap_deg(bearing_deg - center_deg)) <= half_width_deg
            and distance < best_distance
        ):
            best_distance = distance
            best_bearing = bearing_deg

    return best_distance, best_bearing


def estimate_cluster_center_bearing(
    scan, hit_bearing_deg, range_jump=0.18, max_beams=45
):
    """Estimate a cylinder-centre direction from its contiguous LiDAR cluster."""
    count = len(scan.ranges)

    def valid(distance):
        return (
            math.isfinite(distance)
            and scan.range_min <= distance <= scan.range_max
        )

    # Find the beam that produced the blocking hit.
    hit_index = min(
        range(count),
        key=lambda i: abs(
            wrap_deg(
                math.degrees(
                    scan.angle_min + i * scan.angle_increment
                ) - hit_bearing_deg
            )
        ),
    )

    if not valid(scan.ranges[hit_index]):
        return hit_bearing_deg

    cluster = [hit_index]

    # Expand on both sides while ranges belong to the same smooth cylinder arc.
    for direction in (-1, 1):
        previous_distance = scan.ranges[hit_index]
        index = hit_index

        for _ in range(max_beams):
            next_index = index + direction
            if next_index < 0 or next_index >= count:
                break

            next_distance = scan.ranges[next_index]
            if (
                not valid(next_distance)
                or abs(next_distance - previous_distance) > range_jump
            ):
                break

            cluster.append(next_index)
            previous_distance = next_distance
            index = next_index

    # The circular mean of the whole cluster is the visible cylinder centre.
    sin_sum = 0.0
    cos_sum = 0.0

    for index in cluster:
        angle = scan.angle_min + index * scan.angle_increment
        sin_sum += math.sin(angle)
        cos_sum += math.cos(angle)

    return math.degrees(math.atan2(sin_sum, cos_sum))


def target_bearing_deg(env):
    # Target direction in the LiDAR frame.
    return wrap_deg(env.rel_theta - env.yaw)


def turn_error_for_heading(env, desired_heading):
    """Express a desired world heading using this project's proven turn sign."""
    yaw_minus_goal = wrap_deg(env.yaw - env.rel_theta)
    goal_minus_yaw = wrap_deg(env.rel_theta - env.yaw)

    # Determine which convention environment_new.py currently uses by
    # comparing it with the already proven env.diff_angle controller.
    if abs(wrap_deg(env.diff_angle - yaw_minus_goal)) <= abs(
        wrap_deg(env.diff_angle - goal_minus_yaw)
    ):
        return wrap_deg(env.yaw - desired_heading)

    return wrap_deg(desired_heading - env.yaw)


def target_path_blocked(env, scan):
    """Check whether LiDAR enters the physical corridor to the target.

    Points outside +/-30 degrees never affect goal driving. Within that cone,
    a point matters only if it is also inside the 0.50 m-wide route corridor.
    """
    goal_distance = math.hypot(
        env.goal_position.position.x - env.position.x,
        env.goal_position.position.y - env.position.y,
    )
    goal_bearing = target_bearing_deg(env)

    GOAL_CONE_HALF_ANGLE = 30.0
    CORRIDOR_HALF_WIDTH = 0.50

    # The target's own outer blue ring is not an obstacle.
    route_end = goal_distance - TARGET_OUTER_RADIUS - 0.10

    if route_end <= 0.30:
        return False, goal_bearing, goal_distance

    blocked = False
    blocking_bearing = goal_bearing
    nearest_along_route = float("inf")

    for index, distance in enumerate(scan.ranges):
        if not math.isfinite(distance):
            continue
        if distance < scan.range_min or distance > scan.range_max:
            continue

        beam_bearing = math.degrees(
            scan.angle_min + index * scan.angle_increment
        )
        delta_deg = wrap_deg(beam_bearing - goal_bearing)

        # Ignore all LiDAR points outside the target's +/-30 degree cone.
        if abs(delta_deg) > GOAL_CONE_HALF_ANGLE:
            continue

        delta_rad = math.radians(delta_deg)
        along_route = distance * math.cos(delta_rad)
        lateral_offset = distance * math.sin(delta_rad)

        # Only objects physically crossing the robot-to-target corridor block it.
        if (
            0.0 < along_route < route_end
            and abs(lateral_offset) <= CORRIDOR_HALF_WIDTH
            and along_route < nearest_along_route
        ):
            blocked = True
            blocking_bearing = beam_bearing
            nearest_along_route = along_route

    if blocked:
        blocking_bearing = estimate_cluster_center_bearing(
            scan, blocking_bearing
        )

    return blocked, blocking_bearing, goal_distance

def choose_perpendicular_escape(env, scan, obstacle_bearing):
    """Choose the freer perpendicular direction around the blocking pillar."""
    left_distance, _ = sector_nearest(
        scan, obstacle_bearing + 90.0, half_width_deg=20.0
    )
    right_distance, _ = sector_nearest(
        scan, obstacle_bearing - 90.0, half_width_deg=20.0
    )

    relative_escape = (
        obstacle_bearing + 90.0
        if left_distance > right_distance
        else obstacle_bearing - 90.0
    )

    # Convert the selected LiDAR-frame direction to a world heading.
    escape_heading = wrap_deg(env.yaw + relative_escape)

    return escape_heading, left_distance, right_distance


def run_episode(env, max_steps):
    env.reset()
    past_action = [0.0, 0.0]

    state = "CHECK_GOAL"
    escape_heading = None
    escape_start = None

    print(
        "Pillars=%s | Target=(%.2f, %.2f)"
        % (
            env.obstacle_centers,
            env.goal_position.position.x,
            env.goal_position.position.y,
        )
    )

    for step in range(max_steps):
        scan = rospy.wait_for_message("scan", LaserScan, timeout=3.0)
        blocked, obstacle_bearing, _ = target_path_blocked(env, scan)

        if state == "CHECK_GOAL":
            # LiDAR checks the full target vector before the robot drives.
            if blocked:
                escape_heading, left_clearance, right_clearance = (
                    choose_perpendicular_escape(
                        env, scan, obstacle_bearing
                    )
                )
                state = "TURN_TO_ESCAPE"

                print(
                    "  step %d: target blocked; side clearances "
                    "L=%.2f R=%.2f"
                    % (step + 1, left_clearance, right_clearance)
                )
                action = [0.0, 0.0]

            elif abs(env.diff_angle) > ALIGN_DEG:
                # Rotate only: align the car body with the target.
                action = [
                    0.0,
                    clamp(env.diff_angle / 90.0, -1.0, 1.0),
                ]

            else:
                state = "GO_STRAIGHT"
                action = [GOAL_LINEAR_ACTION, 0.0]

        elif state == "TURN_TO_ESCAPE":
            # Do not keep turning for an escape that LiDAR no longer needs.
            if not blocked:
                state = "CHECK_GOAL"
                print(
                    "  step %d: target corridor already clear; "
                    "cancelling escape turn"
                    % (step + 1)
                )
                action = [0.0, 0.0]

            else:
                # Rotate only until the robot faces a perpendicular escape vector.
                turn_error = turn_error_for_heading(env, escape_heading)

                if abs(turn_error) > ALIGN_DEG:
                    action = [
                        0.0,
                        clamp(turn_error / 90.0, -0.60, 0.60),
                    ]
                else:
                    escape_start = (env.position.x, env.position.y)
                    state = "STRAIGHT_ESCAPE"
                    print(
                        "  step %d: facing escape direction; "
                        "driving straight %.2f m"
                        % (step + 1, ESCAPE_DISTANCE)
                    )
                    action = [ESCAPE_LINEAR_ACTION, 0.0]

        elif state == "STRAIGHT_ESCAPE":
            # No angular motion here: a deliberately aggressive straight escape.
            escape_distance = math.hypot(
                env.position.x - escape_start[0],
                env.position.y - escape_start[1],
            )
            front_distance, front_bearing = sector_nearest(
                scan, 0.0, half_width_deg=16.0
            )

            if escape_distance >= ESCAPE_DISTANCE:
                state = "CHECK_GOAL"
                print(
                    "  step %d: escaped %.2f m; checking target again"
                    % (step + 1, escape_distance)
                )
                action = [0.0, 0.0]

            elif front_distance < EMERGENCY_FRONT_DISTANCE:
                # Another obstacle lies directly on the selected escape line.
                # Stop straight motion, choose a new perpendicular direction.
                escape_heading, left_clearance, right_clearance = (
                    choose_perpendicular_escape(
                        env, scan, front_bearing
                    )
                )
                state = "TURN_TO_ESCAPE"
                print(
                    "  step %d: new obstacle on escape line; "
                    "choosing another side"
                    % (step + 1)
                )
                action = [0.0, 0.0]

            else:
                action = [ESCAPE_LINEAR_ACTION, 0.0]

        else:  # GO_STRAIGHT
            # Keep a strict straight line. Re-check LiDAR every action step.
            if blocked:
                escape_heading, left_clearance, right_clearance = (
                    choose_perpendicular_escape(
                        env, scan, obstacle_bearing
                    )
                )
                state = "TURN_TO_ESCAPE"
                print(
                    "  step %d: route blocked again; detouring"
                    % (step + 1)
                )
                action = [0.0, 0.0]

            # While cruising, tolerate small heading drift so the robot
            # commits to a longer straight segment instead of zig-zagging.
            elif abs(env.diff_angle) > DRIVE_REALIGN_DEG:
                state = "CHECK_GOAL"
                action = [0.0, 0.0]

            else:
                action = [GOAL_LINEAR_ACTION, 0.0]

        _, _, collision, arrive = env.step(action, past_action)
        past_action = action

        if arrive:
            return "SUCCESS", step + 1
        if collision:
            return "COLLISION", step + 1

    return "TIMEOUT", max_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=150)
    args = parser.parse_args()

    rospy.init_node("perpendicular_lidar_detour", anonymous=True)
    env = Env(False, use_vision=False, vision_dim=1280)

    successes = []
    collisions = 0
    timeouts = 0

    try:
        for episode in range(args.episodes):
            print("\nEpisode %02d" % episode)
            outcome, steps = run_episode(env, args.max_steps)

            if outcome == "SUCCESS":
                successes.append(steps)
            elif outcome == "COLLISION":
                collisions += 1
            else:
                timeouts += 1

            print("  %s in %d steps" % (outcome, steps))

    finally:
        env.pub_cmd_vel.publish(Twist())

    print("\n=== Perpendicular LiDAR Detour Benchmark ===")
    print(
        "Success: %d/%d (%.1f%%)"
        % (
            len(successes),
            args.episodes,
            100.0 * len(successes) / args.episodes,
        )
    )
    print("Collisions:", collisions)
    print("Timeouts:", timeouts)

    if successes:
        print(
            "Mean success length: %.1f steps"
            % (sum(successes) / len(successes))
        )
        print(
            "Fastest / slowest success: %d / %d steps"
            % (min(successes), max(successes))
        )


if __name__ == "__main__":
    main()
