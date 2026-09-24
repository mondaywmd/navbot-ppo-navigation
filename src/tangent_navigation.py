import math
import numpy as np

from hybrid_navigation import clamp, turn_error_for_heading


PILLAR_RADIUS = 0.30
SAFE_RADIUS = 0.70          # 柱子 + 机器人 + 安全余量
PATH_CLEARANCE = 0.55
WAYPOINT_TOLERANCE = 0.20
ALIGN_DEG = 12.0
GO_LINEAR_ACTION = 5.0
BYPASS_LINEAR_ACTION = 2.2


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def point_to_segment_distance(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < 1e-9:
        return float(np.linalg.norm(point - start))

    progress = float(np.dot(point - start, segment) / length_sq)
    progress = max(0.0, min(1.0, progress))
    closest = start + progress * segment
    return float(np.linalg.norm(point - closest))


def target_bearing_deg(env):
    dx = env.goal_position.position.x - env.position.x
    dy = env.goal_position.position.y - env.position.y
    world_bearing = math.degrees(math.atan2(dy, dx))
    return wrap_deg(world_bearing - math.degrees(env.yaw))


def find_blocking_pillar(env, scan):
    """Use LiDAR points near the robot-to-target corridor to estimate one pillar."""
    if scan is None or not getattr(scan, "ranges", None):
        return None

    goal_distance = math.hypot(
        env.goal_position.position.x - env.position.x,
        env.goal_position.position.y - env.position.y,
    )
    route_end = goal_distance - 1.60  # Ignore the target's own large visual rings.
    if route_end <= 0.20:
        return None

    goal_bearing = target_bearing_deg(env)
    valid = []

    for index, raw_range in enumerate(scan.ranges):
        if not math.isfinite(raw_range) or raw_range <= 0.05 or raw_range > 3.5:
            continue

        beam_deg = math.degrees(scan.angle_min + index * scan.angle_increment)
        relative_deg = wrap_deg(beam_deg - goal_bearing)

        # Express this LiDAR hit in coordinates aligned with the target direction.
        forward = raw_range * math.cos(math.radians(relative_deg))
        lateral = raw_range * math.sin(math.radians(relative_deg))

        # Only an object physically close to the target line blocks the route.
        if 0.10 < forward < route_end and abs(lateral) <= PATH_CLEARANCE:
            valid.append((raw_range, beam_deg, forward, lateral))

    if not valid:
        return None

    # Nearest blocking surface.  Average neighbouring LiDAR rays to estimate
    # the cylinder centre bearing more stably than one single laser beam.
    nearest_range, nearest_bearing, _, _ = min(valid, key=lambda item: item[0])
    neighbours = [
        item for item in valid
        if abs(wrap_deg(item[1] - nearest_bearing)) <= 24.0
        and item[0] <= nearest_range + 0.40
    ]

    weights = [1.0 / max(item[0], 0.05) for item in neighbours]
    center_bearing = sum(
        weight * item[1] for weight, item in zip(weights, neighbours)
    ) / sum(weights)

    # LiDAR sees the cylinder surface, so add its radius to approximate centre.
    center_range = nearest_range + PILLAR_RADIUS
    world_angle = env.yaw + math.radians(center_bearing)

    center = np.array([
        env.position.x + center_range * math.cos(world_angle),
        env.position.y + center_range * math.sin(world_angle),
    ], dtype=np.float32)

    return center


class TangentNavigator:
    """
    A LiDAR-only controller for one pillar:
    - Detect a pillar blocking the direct target corridor.
    - Construct a short diagonal waypoint on each side of its safe circle.
    - Keep the shorter valid route.
    - Drive straight to that waypoint, then directly to the target.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "CHECK"
        self.waypoint = None
        self.chosen_side = None

    def _make_bypass_waypoint(self, env, pillar_center):
        start = np.array([env.position.x, env.position.y], dtype=np.float32)
        goal = np.array([
            env.goal_position.position.x,
            env.goal_position.position.y,
        ], dtype=np.float32)

        direction = goal - start
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            return None, None

        direction /= distance
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        candidates = []
        for side in (-1, 1):
            # A diagonal point just before and beside the inflated obstacle.
            # This creates route 1 / route 2 rather than a long perpendicular exit.
            waypoint = (
                pillar_center
                + side * SAFE_RADIUS * normal
                - 0.25 * direction
            )

            start_clearance = point_to_segment_distance(
                pillar_center, start, waypoint
            )
            goal_clearance = point_to_segment_distance(
                pillar_center, waypoint, goal
            )

            if min(start_clearance, goal_clearance) >= PATH_CLEARANCE:
                route_length = (
                    float(np.linalg.norm(waypoint - start))
                    + float(np.linalg.norm(goal - waypoint))
                )
                candidates.append((route_length, side, waypoint))

        if not candidates:
            return None, None

        _, side, waypoint = min(candidates, key=lambda item: item[0])
        return waypoint, side

    def _drive_to(self, env, point, linear_action):
        dx = float(point[0] - env.position.x)
        dy = float(point[1] - env.position.y)
        desired_heading = math.degrees(math.atan2(dy, dx))
        heading_error = turn_error_for_heading(env, desired_heading)

        # First align, then drive a straight diagonal segment.
        if abs(heading_error) > ALIGN_DEG:
            return np.array([
                0.0,
                clamp(heading_error / 70.0, -0.75, 0.75),
            ], dtype=np.float32)

        return np.array([linear_action, 0.0], dtype=np.float32)

    def base_action(self, env):
        scan = getattr(env, "latest_scan", None)
        pillar_center = find_blocking_pillar(env, scan)

        goal = np.array([
            env.goal_position.position.x,
            env.goal_position.position.y,
        ], dtype=np.float32)

        if self.mode == "CHECK":
            if pillar_center is None:
                self.mode = "GO"
                return self._drive_to(env, goal, GO_LINEAR_ACTION), False

            waypoint, side = self._make_bypass_waypoint(env, pillar_center)
            if waypoint is None:
                # Conservative fallback: wait; do not drive into an uncertain obstacle.
                return np.array([0.0, 0.0], dtype=np.float32), True

            self.waypoint = waypoint
            self.chosen_side = side
            self.mode = "BYPASS"
            return self._drive_to(env, self.waypoint, BYPASS_LINEAR_ACTION), True

        if self.mode == "BYPASS":
            waypoint_distance = float(np.linalg.norm(
                self.waypoint - np.array([env.position.x, env.position.y])
            ))

            if waypoint_distance <= WAYPOINT_TOLERANCE:
                self.mode = "CHECK"
                return np.array([0.0, 0.0], dtype=np.float32), True

            # Replan from the current position if the target line is still blocked.
            if pillar_center is not None:
                waypoint, side = self._make_bypass_waypoint(env, pillar_center)
                if waypoint is not None:
                    self.waypoint = waypoint
                    self.chosen_side = side

            return self._drive_to(env, self.waypoint, BYPASS_LINEAR_ACTION), True

        # Direct target phase. PPO can later adjust this phase.
        if pillar_center is not None:
            self.mode = "CHECK"
            return np.array([0.0, 0.0], dtype=np.float32), True

        return self._drive_to(env, goal, GO_LINEAR_ACTION), False
