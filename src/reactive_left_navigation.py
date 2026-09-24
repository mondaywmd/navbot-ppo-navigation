import math
import numpy as np

from hybrid_navigation import clamp, sector_nearest, wrap_deg


GOAL_ALIGN_DEG = 2.0
LEFT_ALIGN_DEG = 2.0
GOAL_CORRIDOR_HALF_WIDTH_DEG = 18.0
GOAL_BLOCK_DISTANCE = 0.75
MIN_LEFT_PASS_DISTANCE = 0.45


class ReactiveLeftNavigator:
    """
    Pure reactive rule-based navigation.

    No waypoint, no obstacle ground truth, no path search:
      - drive toward goal when its LiDAR corridor is clear;
      - when blocked, turn left 90 degrees;
      - drive left until the goal corridor is clear again;
      - re-align to goal and repeat.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "PLAN"
        self.chosen_side = "LEFT"
        self.waypoint = None  # Only kept for benchmark display compatibility.
        self.left_heading = None
        self.left_start = None

    @staticmethod
    def _position(env):
        return np.array(
            [float(env.position.x), float(env.position.y)],
            dtype=np.float32,
        )

    @staticmethod
    def _goal_distance(env):
        return math.hypot(
            env.goal_position.position.x - env.position.x,
            env.goal_position.position.y - env.position.y,
        )

    def _goal_corridor_blocked(self, env):
        scan = getattr(env, "latest_scan", None)
        if scan is None:
            return False

        nearest = sector_nearest(
            scan,
            float(env.diff_angle),
            GOAL_CORRIDOR_HALF_WIDTH_DEG,
        )
        return nearest < GOAL_BLOCK_DISTANCE

    @staticmethod
    def _turn_action(error):
        return np.array(
            [0.0, clamp(error / 60.0, -0.55, 0.55)],
            dtype=np.float32,
        )

    @staticmethod
    def _goal_speed(distance):
        if distance > 1.5:
            return 3.0
        if distance > 0.8:
            return 1.5
        if distance > 0.4:
            return 0.75
        return 0.25

    def _start_left_turn(self, env):
        # Left relative to the car's current forward direction.
        self.left_heading = wrap_deg(float(env.yaw) + 90.0)
        self.left_start = self._position(env)
        self.mode = "TURN_LEFT"
        return np.array([0.0, 0.0], dtype=np.float32), True

    def base_action(self, env):
        if self.mode == "PLAN":
            if self._goal_corridor_blocked(env):
                return self._start_left_turn(env)

            self.mode = "TURN_GOAL"
            return np.array([0.0, 0.0], dtype=np.float32), False

        if self.mode == "TURN_GOAL":
            # environment_new.py: diff_angle = goal_heading - robot_yaw.
            error = float(env.diff_angle)
            if abs(error) > GOAL_ALIGN_DEG:
                return self._turn_action(error), False

            self.mode = "DRIVE_GOAL"
            return np.array([0.0, 0.0], dtype=np.float32), False

        if self.mode == "DRIVE_GOAL":
            if self._goal_corridor_blocked(env):
                return self._start_left_turn(env)

            distance = self._goal_distance(env)
            if distance <= 0.20:
                return np.array([0.0, 0.0], dtype=np.float32), False

            # Straight only: angular command is exactly zero.
            return np.array(
                [self._goal_speed(distance), 0.0],
                dtype=np.float32,
            ), False

        if self.mode == "TURN_LEFT":
            error = wrap_deg(self.left_heading - float(env.yaw))
            if abs(error) > LEFT_ALIGN_DEG:
                return self._turn_action(error), True

            self.mode = "PASS_LEFT"
            return np.array([0.0, 0.0], dtype=np.float32), True

        if self.mode == "PASS_LEFT":
            travelled = float(
                np.linalg.norm(self._position(env) - self.left_start)
            )

            # Do not leave the bypass immediately; first clear the obstacle.
            if (
                travelled >= MIN_LEFT_PASS_DISTANCE
                and not self._goal_corridor_blocked(env)
            ):
                self.mode = "TURN_GOAL"
                return np.array([0.0, 0.0], dtype=np.float32), True

            # Straight only: angular command is exactly zero.
            return np.array([5.0, 0.0], dtype=np.float32), True

        raise RuntimeError(f"Unknown reactive-navigation mode: {self.mode}")
