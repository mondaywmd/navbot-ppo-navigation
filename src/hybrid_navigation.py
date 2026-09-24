"""Deterministic LiDAR navigation base for residual PPO."""

import math
import numpy as np


ALIGN_DEG = 15.0
DRIVE_REALIGN_DEG = 25.0
TARGET_OUTER_RADIUS = 1.5
GOAL_LINEAR_ACTION = 5.0  # Same proven speed as lidar_detour_benchmark.py
ESCAPE_LINEAR_ACTION = 2.0  # Same proven escape speed as benchmark
ESCAPE_DISTANCE = 0.90
MAX_ESCAPE_SEGMENTS = 3  # Continue outward up to 2.70 m before replanning.


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def target_bearing_deg(env):
    return wrap_deg(env.rel_theta - env.yaw)


def turn_error_for_heading(env, desired_heading):
    """Use the project's existing, proven turn-sign convention."""
    yaw_minus_goal = wrap_deg(env.yaw - env.rel_theta)
    goal_minus_yaw = wrap_deg(env.rel_theta - env.yaw)

    if abs(wrap_deg(env.diff_angle - yaw_minus_goal)) <= abs(
        wrap_deg(env.diff_angle - goal_minus_yaw)
    ):
        return wrap_deg(env.yaw - desired_heading)
    return wrap_deg(desired_heading - env.yaw)


def sector_nearest(scan, center_deg, half_width_deg, fallback=3.5):
    best = fallback
    for index, distance in enumerate(scan.ranges):
        if not math.isfinite(distance):
            continue
        if distance < scan.range_min or distance > scan.range_max:
            continue

        bearing = math.degrees(scan.angle_min + index * scan.angle_increment)
        if abs(wrap_deg(bearing - center_deg)) <= half_width_deg:
            best = min(best, distance)
    return best



def estimate_cluster_center_bearing(
    scan, hit_bearing_deg, range_jump=0.18, max_beams=45
):
    """Estimate the visible cylinder-centre bearing from a LiDAR arc."""
    count = len(scan.ranges)

    def valid(distance):
        return (
            math.isfinite(distance)
            and scan.range_min <= distance <= scan.range_max
        )

    hit_index = min(
        range(count),
        key=lambda i: abs(wrap_deg(
            math.degrees(scan.angle_min + i * scan.angle_increment)
            - hit_bearing_deg
        )),
    )

    if not valid(scan.ranges[hit_index]):
        return hit_bearing_deg

    cluster = [hit_index]

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

    sin_sum = 0.0
    cos_sum = 0.0
    for index in cluster:
        angle = scan.angle_min + index * scan.angle_increment
        sin_sum += math.sin(angle)
        cos_sum += math.cos(angle)

    return math.degrees(math.atan2(sin_sum, cos_sum))

def target_path_blocked(env, scan):
    """Only accept obstacles within target ±30° and its physical route corridor."""
    goal_distance = math.hypot(
        env.goal_position.position.x - env.position.x,
        env.goal_position.position.y - env.position.y,
    )
    goal_bearing = target_bearing_deg(env)
    route_end = goal_distance - TARGET_OUTER_RADIUS - 0.10

    if route_end <= 0.30:
        return False, goal_bearing

    nearest_along_route = float("inf")
    blocking_bearing = goal_bearing

    for index, distance in enumerate(scan.ranges):
        if not math.isfinite(distance):
            continue
        if distance < scan.range_min or distance > scan.range_max:
            continue

        bearing = math.degrees(scan.angle_min + index * scan.angle_increment)
        delta_deg = wrap_deg(bearing - goal_bearing)

        if abs(delta_deg) > 30.0:
            continue

        delta_rad = math.radians(delta_deg)
        along_route = distance * math.cos(delta_rad)
        lateral_offset = distance * math.sin(delta_rad)

        if (
            0.0 < along_route < route_end
            and abs(lateral_offset) <= 0.50
            and along_route < nearest_along_route
        ):
            nearest_along_route = along_route
            blocking_bearing = bearing

    blocked = nearest_along_route < float("inf")
    if blocked:
        blocking_bearing = estimate_cluster_center_bearing(
            scan, blocking_bearing
        )

    return blocked, blocking_bearing


class HybridNavigator:
    """Safety-first state machine used as PPO's base policy."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "CHECK_GOAL"
        self.escape_heading = None
        self.escape_start = None
        self.escape_segments = 0

    @staticmethod
    def compose(base_action, residual_action):
        """Keep neural corrections deliberately small."""
        return np.array([
            clamp(base_action[0] + 0.05 * residual_action[0], 0.0, 5.0),
            clamp(base_action[1] + 0.10 * residual_action[1], -1.0, 1.0),
        ], dtype=np.float32)

    def base_action(self, env):
        """Return (action, safety_locked).

        safety_locked=True means the deterministic controller owns the action.
        PPO may only correct clear, straight-to-goal driving.
        """
        scan = env.latest_scan
        if scan is None:
            return np.array([0.0, 0.0], dtype=np.float32), True

        blocked, obstacle_bearing = target_path_blocked(env, scan)

        if self.state == "CHECK_GOAL":
            if blocked:
                left = sector_nearest(scan, obstacle_bearing + 90.0, 20.0)
                right = sector_nearest(scan, obstacle_bearing - 90.0, 20.0)
                relative_escape = (
                    obstacle_bearing + 90.0 if left > right
                    else obstacle_bearing - 90.0
                )
                self.escape_heading = wrap_deg(env.yaw + relative_escape)
                self.state = "TURN_TO_ESCAPE"
                return np.array([0.0, 0.0], dtype=np.float32), True

            if abs(env.diff_angle) > ALIGN_DEG:
                return np.array([
                    0.0, clamp(env.diff_angle / 90.0, -0.60, 0.60)
                ], dtype=np.float32), True

            self.state = "GO_STRAIGHT"
            return np.array([GOAL_LINEAR_ACTION, 0.0], dtype=np.float32), False

        if self.state == "TURN_TO_ESCAPE":
            if not blocked:
                self.state = "CHECK_GOAL"
                return np.array([0.0, 0.0], dtype=np.float32), True

            error = turn_error_for_heading(env, self.escape_heading)
            if abs(error) > ALIGN_DEG:
                return np.array([
                    0.0, clamp(error / 90.0, -0.60, 0.60)
                ], dtype=np.float32), True

            self.escape_start = (env.position.x, env.position.y)
            self.escape_segments = 1
            self.state = "STRAIGHT_ESCAPE"
            return np.array([ESCAPE_LINEAR_ACTION, 0.0], dtype=np.float32), True

        if self.state == "STRAIGHT_ESCAPE":
            moved = math.hypot(
                env.position.x - self.escape_start[0],
                env.position.y - self.escape_start[1],
            )
            if moved >= ESCAPE_DISTANCE:
                # If the target corridor remains blocked, keep moving outward
                # along the same proven perpendicular heading before replanning.
                if blocked and self.escape_segments < MAX_ESCAPE_SEGMENTS:
                    self.escape_segments += 1
                    self.escape_start = (env.position.x, env.position.y)
                    return np.array(
                        [ESCAPE_LINEAR_ACTION, 0.0], dtype=np.float32
                    ), True

                self.state = "CHECK_GOAL"
                return np.array([0.0, 0.0], dtype=np.float32), True

            return np.array([ESCAPE_LINEAR_ACTION, 0.0], dtype=np.float32), True

        # GO_STRAIGHT
        if blocked or abs(env.diff_angle) > DRIVE_REALIGN_DEG:
            self.state = "CHECK_GOAL"
            return np.array([0.0, 0.0], dtype=np.float32), True

        return np.array([GOAL_LINEAR_ACTION, 0.0], dtype=np.float32), False
