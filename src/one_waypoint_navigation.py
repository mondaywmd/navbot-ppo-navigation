import math
import numpy as np

from hybrid_navigation import clamp, turn_error_for_heading


# Cylinder radius + TurtleBot footprint + a small execution margin.
SAFE_RADIUS = 0.60  # Calibrated from observed robot collision  # 0.30 m pillar + robot footprint + small margin
WAYPOINT_TOLERANCE = 0.15
ALIGN_DEG = 4.0
FINAL_CROSS_TRACK_LIMIT = 0.18
MAX_FINAL_CORRECTIONS = 1
ARENA_LIMIT = 2.20


def point_to_segment_distance(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))

    if length_sq < 1e-9:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, segment) / length_sq)
    t = max(0.0, min(1.0, t))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def segment_is_safe(start, end, pillar_center):
    return (
        point_to_segment_distance(pillar_center, start, end)
        >= SAFE_RADIUS
    )


class OneWaypointNavigator:
    """
    One-turn geometric controller:
        start -> one safe waypoint -> target

    The waypoint is outside the safety circle. Both straight-line segments
    must be clear, and the total path length is minimized.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "CHECK"
        self.waypoint = None
        self.chosen_side = "one_waypoint"
        self.goal_committed = False
        self.final_heading = None
        self.final_aligned = False
        self.final_start = None
        self.final_corrections = 0

    def _positions(self, env):
        start = np.array([env.position.x, env.position.y], dtype=np.float32)
        target = np.array([
            env.goal_position.position.x,
            env.goal_position.position.y,
        ], dtype=np.float32)
        return start, target

    def _blocking_pillar(self, start, target, env):
        for x, y in getattr(env, "obstacle_centers", []):
            center = np.array([x, y], dtype=np.float32)
            if not segment_is_safe(start, target, center):
                return center
        return None

    def _find_best_waypoint(self, start, target, center):
        # Search only outside the safety circle.  This produces one corner
        # point W with two clear straight segments: start->W and W->target.
        best = None

        for radius in np.linspace(SAFE_RADIUS + 0.05, 1.85, 19):
            for angle_deg in range(0, 360, 2):
                angle = math.radians(angle_deg)
                waypoint = center + radius * np.array(
                    [math.cos(angle), math.sin(angle)],
                    dtype=np.float32,
                )

                if (
                    abs(float(waypoint[0])) > ARENA_LIMIT
                    or abs(float(waypoint[1])) > ARENA_LIMIT
                ):
                    continue

                if not segment_is_safe(start, waypoint, center):
                    continue
                if not segment_is_safe(waypoint, target, center):
                    continue

                total_length = (
                    float(np.linalg.norm(waypoint - start))
                    + float(np.linalg.norm(target - waypoint))
                )

                if best is None or total_length < best[0]:
                    best = (total_length, waypoint)

        if best is None:
            return None

        return best[1]

    def _drive_to(self, env, point, is_final=False):
        dx = float(point[0] - env.position.x)
        dy = float(point[1] - env.position.y)
        distance = math.hypot(dx, dy)
        desired_heading = math.degrees(math.atan2(dy, dx))
        error = turn_error_for_heading(env, desired_heading)

        if distance <= (0.20 if is_final else WAYPOINT_TOLERANCE):
            return np.array([0.0, 0.0], dtype=np.float32)

        if abs(error) > ALIGN_DEG:
            return np.array([
                0.0,
                clamp(error / 60.0, -0.55, 0.55),
            ], dtype=np.float32)

        if not is_final:
            # Fast straight segment, slow down only near W.
            linear_action = 3.0 if distance > 0.50 else 1.20
        else:
            # Blue -> red -> yellow target rings.
            if distance > 1.50:
                linear_action = 5.00
            elif distance > 1.00:
                linear_action = 2.50
            elif distance > 0.50:
                linear_action = 1.20
            else:
                linear_action = 0.60

        return np.array([linear_action, 0.0], dtype=np.float32)

    def _enter_final_straight(self, env):
        self.mode = "GO_TARGET"
        self.goal_committed = True
        self.final_heading = None
        self.final_aligned = False
        self.final_start = np.array(
            [env.position.x, env.position.y],
            dtype=np.float32,
        )
        self.final_corrections = 0
        return self._drive_final_straight(env)

    def _drive_final_straight(self, env):
        # Set the target heading once, then do not oscillate left/right.
        target_x = env.goal_position.position.x
        target_y = env.goal_position.position.y
        dx = float(target_x - env.position.x)
        dy = float(target_y - env.position.y)
        distance = math.hypot(dx, dy)

        if distance <= 0.20:
            return np.array([0.0, 0.0], dtype=np.float32)

        if self.final_heading is None:
            self.final_heading = math.degrees(math.atan2(dy, dx))

        if not self.final_aligned:
            error = turn_error_for_heading(env, self.final_heading)

            if abs(error) > 3.0:
                return np.array([
                    0.0,
                    clamp(error / 60.0, -0.55, 0.55),
                ], dtype=np.float32)

            self.final_aligned = True

        # Keep the chosen straight line. If Gazebo drift moves the robot
        # more than 18 cm away from it, allow exactly one corrective alignment.
        if (
            self.final_start is not None
            and self.final_corrections < MAX_FINAL_CORRECTIONS
        ):
            current = np.array(
                [env.position.x, env.position.y],
                dtype=np.float32,
            )
            target = np.array([target_x, target_y], dtype=np.float32)
            line = target - self.final_start
            line_length = float(np.linalg.norm(line))

            if line_length > 1e-6:
                offset = current - self.final_start
                cross_track = abs(
                    line[0] * offset[1] - line[1] * offset[0]
                ) / line_length

                if cross_track > FINAL_CROSS_TRACK_LIMIT:
                    self.final_heading = math.degrees(math.atan2(dy, dx))
                    self.final_aligned = False
                    self.final_start = current
                    self.final_corrections += 1
                    return np.array([0.0, 0.0], dtype=np.float32)

        # Continue straight toward the target center.
        if distance > 1.50:
            linear_action = 5.00
        elif distance > 1.00:
            linear_action = 2.50
        elif distance > 0.50:
            linear_action = 1.20
        else:
            linear_action = 0.60

        return np.array([linear_action, 0.0], dtype=np.float32)

    def base_action(self, env):
        start, target = self._positions(env)

        if self.mode == "CHECK":
            center = self._blocking_pillar(start, target, env)

            if center is None:
                return self._enter_final_straight(env), False

            self.waypoint = self._find_best_waypoint(start, target, center)
            if self.waypoint is None:
                return np.array([0.0, 0.0], dtype=np.float32), True

            self.mode = "GO_WAYPOINT"
            return self._drive_to(env, self.waypoint), True

        if self.mode == "GO_WAYPOINT":
            if float(np.linalg.norm(start - self.waypoint)) <= WAYPOINT_TOLERANCE:
                return self._enter_final_straight(env), False

            return self._drive_to(env, self.waypoint), True

        # No obstacle rechecking and no repeated heading search here.
        return self._drive_final_straight(env), False
