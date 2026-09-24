import math
import numpy as np

from hybrid_navigation import turn_error_for_heading


PILLAR_RADIUS = 0.30
SAFE_RADIUS = 0.60
CLEARANCE = SAFE_RADIUS + 0.02
WAYPOINT_TOLERANCE = 0.10
GOAL_TOLERANCE = 0.20
ALIGN_TOLERANCE_DEG = 1.0


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def point_to_segment_distance(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < 1e-9:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, segment) / length_sq)
    t = clamp(t, 0.0, 1.0)
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def segment_is_clear(start, end, center):
    return point_to_segment_distance(center, start, end) >= CLEARANCE


class StrictOneWaypointNavigator:
    """
    Hard four-phase controller:

      ROTATE_W    : rotate once toward cached waypoint W
      DRIVE_W     : strictly straight, angular action is always 0
      ROTATE_GOAL : rotate once toward the target center
      DRIVE_GOAL  : strictly straight, angular action is always 0
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "PLAN"
        self.waypoint = None
        self.chosen_side = "strict_one_waypoint"
        self.locked_heading = None

    @staticmethod
    def _position(env):
        return np.array(
            [float(env.position.x), float(env.position.y)],
            dtype=np.float32,
        )

    @staticmethod
    def _target(env):
        return np.array(
            [
                float(env.goal_position.position.x),
                float(env.goal_position.position.y),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _heading_to(start, end):
        delta = end - start
        return math.degrees(math.atan2(float(delta[1]), float(delta[0])))

    @staticmethod
    def _yaw_deg(env):
        # environment_new.py stores yaw in degrees.
        return float(env.yaw)

    def _heading_error(self, env):
        if self.mode == "ROTATE_GOAL":
            return float(env.diff_angle)
        return float(turn_error_for_heading(env, self.locked_heading))

    def _rotation_action(self, env):
        error = self._heading_error(env)
        if abs(error) <= ALIGN_TOLERANCE_DEG:
            return None

        return np.array(
            [0.0, clamp(error / 60.0, -0.55, 0.55)],
            dtype=np.float32,
        )

    def _find_blocking_center(self, start, target, env):
        for x, y in getattr(env, "obstacle_centers", []):
            center = np.array([float(x), float(y)], dtype=np.float32)
            if not segment_is_clear(start, target, center):
                return center
        return None

    def _find_nearest_valid_waypoint(self, start, target, center):
        """
        Literal implementation of the requested rule:
        choose the valid W closest to the starting point.
        Both straight segments must stay outside the safety circle.
        """
        candidates = []

        for radius in np.linspace(CLEARANCE + 0.02, 1.50, 24):
            for degrees in range(360):
                angle = math.radians(degrees)
                waypoint = center + float(radius) * np.array(
                    [math.cos(angle), math.sin(angle)],
                    dtype=np.float32,
                )

                if not segment_is_clear(start, waypoint, center):
                    continue
                if not segment_is_clear(waypoint, target, center):
                    continue

                start_distance = float(np.linalg.norm(waypoint - start))
                total_distance = (
                    start_distance
                    + float(np.linalg.norm(target - waypoint))
                )
                candidates.append((start_distance, total_distance, waypoint))

        if not candidates:
            return None

        # First criterion: nearest to start. Second: shorter total route.
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def _straight_speed(distance):
        # Slow down near W or the goal; angular action remains strictly zero.
        if distance > 1.50:
            return 1.40
        if distance > 0.80:
            return 1.00
        if distance > 0.40:
            return 0.60
        return 0.25

    def _begin_rotation(self, env, destination, next_mode):
        if next_mode == "ROTATE_GOAL":
            self.locked_heading = float(env.rel_theta)
        else:
            self.locked_heading = self._heading_to(self._position(env), destination)
        self.mode = next_mode
        return np.array([0.0, 0.0], dtype=np.float32), True

    def base_action(self, env):
        robot = self._position(env)
        target = self._target(env)

        if self.mode == "PLAN":
            center = self._find_blocking_center(robot, target, env)

            if center is None:
                self.waypoint = None
                return self._begin_rotation(env, target, "ROTATE_GOAL")

            self.waypoint = self._find_nearest_valid_waypoint(
                robot, target, center
            )
            if self.waypoint is None:
                return np.array([0.0, 0.0], dtype=np.float32), True

            return self._begin_rotation(env, self.waypoint, "ROTATE_W")

        if self.mode == "ROTATE_W":
            action = self._rotation_action(env)
            if action is not None:
                return action, True

            self.mode = "DRIVE_W"
            return np.array([0.0, 0.0], dtype=np.float32), True

        if self.mode == "DRIVE_W":
            distance = float(np.linalg.norm(self.waypoint - robot))
            if distance <= WAYPOINT_TOLERANCE:
                return self._begin_rotation(env, target, "ROTATE_GOAL")

            # Strictly straight: angular action is exactly zero.
            return np.array(
                [self._straight_speed(distance), 0.0],
                dtype=np.float32,
            ), True

        if self.mode == "ROTATE_GOAL":
            action = self._rotation_action(env)
            if action is not None:
                return action, False

            self.mode = "DRIVE_GOAL"
            return np.array([0.0, 0.0], dtype=np.float32), False

        if self.mode == "DRIVE_GOAL":
            distance = float(np.linalg.norm(target - robot))
            if distance <= GOAL_TOLERANCE:
                return np.array([0.0, 0.0], dtype=np.float32), False

            # Strictly straight: no scan, no replanning, no correction turn.
            return np.array(
                [self._straight_speed(distance), 0.0],
                dtype=np.float32,
            ), False

        raise RuntimeError(f"Unknown navigation mode: {self.mode}")
