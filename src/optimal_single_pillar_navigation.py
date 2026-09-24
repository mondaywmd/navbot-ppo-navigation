import math
import numpy as np

from hybrid_navigation import clamp, turn_error_for_heading


PILLAR_RADIUS = 0.30
SAFE_RADIUS = 0.80  # Extra margin for discrete robot control
ARC_STEP_DEG = 35.0
WAYPOINT_TOLERANCE = 0.22
ALIGN_DEG = 10.0
GO_LINEAR_ACTION = 5.0
ARC_LINEAR_ACTION = 3.0
ARC_TURN_ACTION = 0.72
ENTRY_ALIGN_DEG = 3.0
ENTRY_TOLERANCE = 0.08


def point_to_segment_distance(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < 1e-9:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, segment) / length_sq)
    t = max(0.0, min(1.0, t))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def direct_path_is_clear(start, goal, center):
    return point_to_segment_distance(center, start, goal) >= SAFE_RADIUS


def tangent_points(point, center):
    vector = point - center
    distance = float(np.linalg.norm(vector))

    if distance <= SAFE_RADIUS + 0.03:
        return []

    base = math.atan2(vector[1], vector[0])
    offset = math.acos(SAFE_RADIUS / distance)

    return [
        center + SAFE_RADIUS * np.array(
            [math.cos(base + offset), math.sin(base + offset)],
            dtype=np.float32,
        ),
        center + SAFE_RADIUS * np.array(
            [math.cos(base - offset), math.sin(base - offset)],
            dtype=np.float32,
        ),
    ]


def angle_of(point, center):
    return math.atan2(point[1] - center[1], point[0] - center[0])


def make_arc(center, entry, exit_point, direction):
    entry_angle = angle_of(entry, center)
    exit_angle = angle_of(exit_point, center)

    if direction == "CCW":
        angle_change = (exit_angle - entry_angle) % (2.0 * math.pi)
    else:
        angle_change = -((entry_angle - exit_angle) % (2.0 * math.pi))

    count = max(
        1,
        int(math.ceil(abs(math.degrees(angle_change)) / ARC_STEP_DEG)),
    )

    points = [entry]
    for index in range(1, count + 1):
        angle = entry_angle + angle_change * index / count
        points.append(
            center + SAFE_RADIUS * np.array(
                [math.cos(angle), math.sin(angle)],
                dtype=np.float32,
            )
        )

    return points, abs(angle_change)


class OptimalSinglePillarNavigator:
    """
    Exact shortest-path planner for one known circular obstacle.

    It compares every valid:
        robot tangent -> circular arc -> target tangent
    route and keeps the shortest one for the whole bypass.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "CHECK"
        self.pillar_center = None
        self.waypoints = []
        self.waypoint = None
        self.chosen_side = None
        self.goal_committed = False
        self.final_aligned = False
        self.final_heading = None
        self.arc_started = False

    def _positions(self, env):
        robot = np.array([env.position.x, env.position.y], dtype=np.float32)
        target = np.array([
            env.goal_position.position.x,
            env.goal_position.position.y,
        ], dtype=np.float32)
        return robot, target

    def _blocking_pillar(self, robot, target, env):
        centers = getattr(env, "obstacle_centers", [])
        for x, y in centers:
            center = np.array([x, y], dtype=np.float32)
            if not direct_path_is_clear(robot, target, center):
                return center
        return None

    def _shortest_route(self, robot, target, center):
        # There are exactly two valid homotopy classes around one circle:
        # clockwise and counter-clockwise.  For each class, choose only
        # tangency points whose incoming and outgoing directions are continuous.
        robot_tangents = tangent_points(robot, center)
        target_tangents = tangent_points(target, center)
        candidates = []

        for direction in ("CCW", "CW"):
            best_entry = None
            best_entry_score = -float("inf")

            for entry in robot_tangents:
                radial = (entry - center) / SAFE_RADIUS
                tangent = (
                    np.array([-radial[1], radial[0]], dtype=np.float32)
                    if direction == "CCW"
                    else np.array([radial[1], -radial[0]], dtype=np.float32)
                )

                incoming = entry - robot
                incoming /= max(float(np.linalg.norm(incoming)), 1e-6)
                score = float(np.dot(incoming, tangent))

                if score > best_entry_score:
                    best_entry_score = score
                    best_entry = entry

            best_exit = None
            best_exit_score = -float("inf")

            for exit_point in target_tangents:
                radial = (exit_point - center) / SAFE_RADIUS
                tangent = (
                    np.array([-radial[1], radial[0]], dtype=np.float32)
                    if direction == "CCW"
                    else np.array([radial[1], -radial[0]], dtype=np.float32)
                )

                outgoing = target - exit_point
                outgoing /= max(float(np.linalg.norm(outgoing)), 1e-6)
                score = float(np.dot(outgoing, tangent))

                if score > best_exit_score:
                    best_exit_score = score
                    best_exit = exit_point

            # A real tangent must point in essentially the same direction.
            if best_entry_score < 0.999 or best_exit_score < 0.999:
                continue

            arc_points, arc_angle = make_arc(
                center,
                best_entry,
                best_exit,
                direction,
            )

            route_length = (
                float(np.linalg.norm(robot - best_entry))
                + SAFE_RADIUS * arc_angle
                + float(np.linalg.norm(target - best_exit))
            )
            candidates.append((route_length, direction, arc_points))

        if not candidates:
            return None, None

        _, direction, points = min(candidates, key=lambda item: item[0])
        return points, direction

    def _drive_to(self, env, point, linear_action):
        dx = float(point[0] - env.position.x)
        dy = float(point[1] - env.position.y)
        desired_heading = math.degrees(math.atan2(dy, dx))
        error = turn_error_for_heading(env, desired_heading)

        if abs(error) > ALIGN_DEG:
            return np.array([
                0.0,
                clamp(error / 65.0, -0.75, 0.75),
            ], dtype=np.float32)

        return np.array([linear_action, 0.0], dtype=np.float32)

    def _drive_to_entry(self, env, point):
        # Reach the entry tangent accurately before beginning the circle.
        dx = float(point[0] - env.position.x)
        dy = float(point[1] - env.position.y)
        distance = math.hypot(dx, dy)
        desired_heading = math.degrees(math.atan2(dy, dx))
        error = turn_error_for_heading(env, desired_heading)

        if abs(error) > ENTRY_ALIGN_DEG:
            return np.array([
                0.0,
                clamp(error / 65.0, -0.55, 0.55),
            ], dtype=np.float32)

        linear_action = 1.0 if distance <= 0.35 else ARC_LINEAR_ACTION
        return np.array([linear_action, 0.0], dtype=np.float32)

    def _follow_circle(self, env):
        # Physical v = 0.30 * 3.0 = 0.90 m/s.
        # Physical w = 1.50 * 0.72 = 1.08 rad/s.
        # Executed turn radius is about 0.83 m, safely outside 0.80 m.
        turn = ARC_TURN_ACTION if self.chosen_side == "CCW" else -ARC_TURN_ACTION
        return np.array([ARC_LINEAR_ACTION, turn], dtype=np.float32)

    def _goal_action(self, env):
        # The pillar has already been passed. Track only the target direction;
        # do not recheck or replan obstacles during this final approach.
        goal = np.array([
            env.goal_position.position.x,
            env.goal_position.position.y,
        ], dtype=np.float32)
        robot = np.array([env.position.x, env.position.y], dtype=np.float32)
        distance = float(np.linalg.norm(goal - robot))

        if distance <= 0.20:
            return np.array([0.0, 0.0], dtype=np.float32)

        dx = float(goal[0] - env.position.x)
        dy = float(goal[1] - env.position.y)
        desired_heading = math.degrees(math.atan2(dy, dx))
        error = turn_error_for_heading(env, desired_heading)

        # Re-align only for a meaningful error, preventing left-right jitter.
        if abs(error) > 8.0:
            return np.array([
                0.0,
                clamp(error / 65.0, -0.55, 0.55),
            ], dtype=np.float32)

        # Speed bands match the visible target rings:
        # outside blue -> fast; blue -> medium; red -> slow; yellow -> precise.
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
        robot, target = self._positions(env)

        if self.mode == "CHECK":
            center = self._blocking_pillar(robot, target, env)

            if center is None:
                self.mode = "GO"
                return self._goal_action(env), False

            points, direction = self._shortest_route(robot, target, center)
            if not points:
                return np.array([0.0, 0.0], dtype=np.float32), True

            self.mode = "ARC"
            self.pillar_center = center
            self.waypoints = points
            self.waypoint = self.waypoints[0]
            self.chosen_side = direction
            self.arc_started = False

            # Reach the entry tangent with one straight segment.
            return self._drive_to_entry(env, self.waypoint), True

        if self.mode == "ARC":
            # First reach the entry tangent point.
            if not self.arc_started:
                if float(np.linalg.norm(robot - self.waypoint)) > ENTRY_TOLERANCE:
                    return self._drive_to_entry(env, self.waypoint), True

                self.waypoints.pop(0)
                self.arc_started = True

                if not self.waypoints:
                    self.mode = "GO"
                    self.waypoint = None
                    self.goal_committed = True
                    return self._goal_action(env), False

                self.waypoint = self.waypoints[0]

            # Then keep one continuous fixed-curvature arc.
            if float(np.linalg.norm(robot - self.waypoint)) <= WAYPOINT_TOLERANCE:
                self.waypoints.pop(0)

                if not self.waypoints:
                    self.mode = "GO"
                    self.waypoint = None
                    self.goal_committed = True
                    return self._goal_action(env), False

                self.waypoint = self.waypoints[0]

            return self._follow_circle(env), True

        # The obstacle has been passed; only track the target from here.
        return self._goal_action(env), False
