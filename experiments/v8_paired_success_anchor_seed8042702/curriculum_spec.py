"""Wide static-obstacle curriculum specification; pure generation only."""
import math
import random

SEED = 8042703
ARENA_LIMIT = 3.55
RADIUS_RANGE = (0.18, 0.45)
START_CLEARANCE = 0.55
GOAL_CLEARANCE = 0.55
INTER_PILLAR_CLEARANCE = 0.25
ROBOT_RADIUS = 0.105

LEVELS = {
    1: {"obstacles": 1, "target_x": (1.4, 3.0), "target_abs_y": 1.2,
        "path_lateral": 0.75},
    2: {"obstacles": 2, "target_x": (2.0, 3.2), "target_abs_y": 1.2,
        "path_lateral": 0.80},
    3: {"obstacles": 3, "target_x": (2.6, 3.3), "target_abs_y": 1.1,
        "path_lateral": 0.85},
}


def mirror(scene):
    return {**scene, "name": scene["name"] + "_mirror",
            "target_y": -scene["target_y"],
            "robot_yaw_deg": -scene["robot_yaw_deg"],
            "obstacles": [{"x": p["x"], "y": -p["y"], "radius": p["radius"]}
                          for p in scene["obstacles"]]}


def legal(scene):
    tx, ty = scene["target_x"], scene["target_y"]
    if max(abs(tx), abs(ty)) > ARENA_LIMIT or math.hypot(tx, ty) < 1.2:
        return False
    obstacles = scene["obstacles"]
    for p in obstacles:
        if max(abs(p["x"]), abs(p["y"])) > ARENA_LIMIT:
            return False
        if not RADIUS_RANGE[0] <= p["radius"] <= RADIUS_RANGE[1]:
            return False
        if math.hypot(p["x"], p["y"]) < p["radius"] + START_CLEARANCE:
            return False
        if math.hypot(p["x"] - tx, p["y"] - ty) < p["radius"] + GOAL_CLEARANCE:
            return False
    separated = all(math.hypot(a["x"] - b["x"], a["y"] - b["y"])
               >= a["radius"] + b["radius"] + INTER_PILLAR_CLEARANCE
               for i, a in enumerate(obstacles)
               for b in obstacles[i + 1:])
    return separated and direct_path_blocked(scene)


def point_segment_distance(px, py, x1, y1):
    scale = max(0.0, min(1.0, (px * x1 + py * y1) / (x1 * x1 + y1 * y1)))
    return math.hypot(px - scale * x1, py - scale * y1)


def direct_path_blocked(scene):
    return any(point_segment_distance(p["x"], p["y"],
                                      scene["target_x"], scene["target_y"])
               <= p["radius"] + ROBOT_RADIUS for p in scene["obstacles"])


def generate(level, count, seed=SEED):
    cfg = LEVELS[level]; rng = random.Random(seed + level); result = []
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 1000: raise RuntimeError("could not generate legal scenes")
        tx = rng.uniform(*cfg["target_x"]); ty = rng.uniform(-cfg["target_abs_y"], cfg["target_abs_y"])
        obstacles = []
        fractions = [(i + 1.0) / (cfg["obstacles"] + 1.0) for i in range(cfg["obstacles"])]
        blocker_index = rng.randrange(cfg["obstacles"])
        for obstacle_index, fraction in enumerate(fractions):
            # Position along the start-target segment plus a large perpendicular offset.
            length = math.hypot(tx, ty); nx, ny = -ty / length, tx / length
            radius = rng.uniform(*RADIUS_RANGE)
            lateral_limit = radius + ROBOT_RADIUS - 0.02
            lateral = (rng.uniform(-lateral_limit, lateral_limit)
                       if obstacle_index == blocker_index else
                       rng.uniform(-cfg["path_lateral"], cfg["path_lateral"]))
            along = fraction + rng.uniform(-0.06, 0.06)
            obstacles.append({"x": tx * along + nx * lateral,
                              "y": ty * along + ny * lateral,
                              "radius": radius})
        scene = {"name": "level%d_%05d" % (level, len(result) + 1),
                 "level": level, "target_x": tx, "target_y": ty,
                 "robot_yaw_deg": rng.uniform(-15.0, 15.0),
                 "obstacles": obstacles}
        if legal(scene) and legal(mirror(scene)): result.append(scene)
    return result
