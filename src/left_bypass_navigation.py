import math
import numpy as np

from hybrid_navigation import clamp, sector_nearest, wrap_deg


GOAL_ALIGN_DEG = 2.0
LEFT_ALIGN_DEG = 2.0
BLOCK_HALF_WIDTH_DEG = 18.0
BLOCK_DISTANCE = 0.75
LEFT_ESCAPE_DISTANCE = 0.80


class LeftBypassNavigator:
    """
    Pure rule-based navigation.

    Goal clear  -> align to goal once, then drive straight.
    Goal blocked -> turn to the left of the goal direction, drive 0.8 m
                    straight, then check the goal direction again.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "PLAN"
        self.waypoint = None
        self.chosen_side = "LEFT"
        self.locked_heading = None
        self.escape_start = None
        self.escape_count = 0

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
    def _distance(a, b):
        return float(np.linalg.norm(a - b))

    def _goal_is_blocked(self, env):
        scan = getattr(env, "latest_scan", None)
        if scan is None:
            return False

        nearest = sector_nearest(
            scan,
            float(env.diff_angle),
            BLOCK_HALF_WIDTH_DEG,
        )
        return nearest < BLOCK_DISTANCE

    @staticmethod
    def _speed(distance):
        if distance > 1.5:
            return 1.4
        if distance > 0.8:
            return 1.0
        if distance > 0.4:
            return 0.6
        return 0.25

    @staticmethod
    def _turn_action(error):
        return np.array(
            [0.0, clamp(error / 60.0, -0.55, 0.55)],
            dtype=np.float32,
        )

    def _start_goal_turn(self, env):
        self.mode = "TURN_GOAL"
        self.waypoint = None
        return np.array([0.0, 0.0], dtype=np.float32), False

    def _start_left_bypass(self, env):
        robot = self._position(env)

        # "Left" is defined relative to the robot -> goal direction.
        self.locked_heading = wrap_deg(float(env.rel_theta) + 90.0)
        heading_rad = math.radians(self.locked_heading)

        self.escape_start = robot.copy()
        self.waypoint = robot + LEFT_ESCAPE_DISTANCE * np.array(
            [math.cos(heading_rad), math.sin(heading_rad)],
            dtype=np.float32,
        )

        self.escape_count += 1
        self.mode = "TURN_LEFT"
        return np.array([0.0, 0.0], dtype=np.float32), True

    def base_action(self, env):
        robot = self._position(env)
        target = self._target(env)

        if self.mode == "PLAN":
            if self._goal_is_blocked(env):
                return self._start_left_bypass(env)
            return self._start_goal_turn(env)

        if self.mode == "TURN_GOAL":
            # environment_new.py defines this exactly as goal_heading - yaw.
            error = float(env.diff_angle)
            if abs(error) > GOAL_ALIGN_DEG:
                return self._turn_action(error), False

            self.mode = "DRIVE_GOAL"
            return np.array([0.0, 0.0], dtype=np.float32), False

        if self.mode == "DRIVE_GOAL":
            if self._goal_is_blocked(env):
                return self._start_left_bypass(env)

            distance = self._distance(target, robot)
            if distance <= 0.20:
                return np.array([0.0, 0.0], dtype=np.float32), False

            # Strictly straight toward the already-aligned target.
            return np.array(
                [self._speed(distance), 0.0],
                dtype=np.float32,
            ), False

        if self.mode == "TURN_LEFT":
            error = wrap_deg(self.locked_heading - float(env.yaw))
            if abs(error) > LEFT_ALIGN_DEG:
                return self._turn_action(error), True

            self.mode = "DRIVE_LEFT"
            return np.array([0.0, 0.0], dtype=np.float32), True

        if self.mode == "DRIVE_LEFT":
            travelled = self._distance(robot, self.escape_start)
            if travelled >= LEFT_ESCAPE_DISTANCE:
                self.mode = "PLAN"
                return np.array([0.0, 0.0], dtype=np.float32), True

            # Strictly straight to the left-bypass point.
            remaining = LEFT_ESCAPE_DISTANCE - travelled
            return np.array(
                [5.0, 0.0],
                dtype=np.float32,
            ), True

        raise RuntimeError(f"Unknown rule-navigation mode: {self.mode}")
