"""Fresh paired validation scenes, independent of every test set."""
import json
import math
import os
import random

SEED = 8042702
PAIR_COUNT = 6


def generate():
    rng = random.Random(SEED)
    scenes = []
    for pair in range(1, PAIR_COUNT + 1):
        pillar_x = rng.uniform(0.82, 1.12)
        pillar_y = rng.uniform(-0.14, 0.14)
        target_y = rng.uniform(-0.18, 0.18)
        yaw = rng.uniform(-6.0, 6.0)
        base = {"name": "pair_%02d_base" % pair, "pair": pair,
                "side": "base", "detour_sign": 1.0,
                "pillar_x": pillar_x, "pillar_y": pillar_y,
                "target_y": target_y, "robot_yaw_deg": yaw}
        mirror = {"name": "pair_%02d_mirror" % pair, "pair": pair,
                  "side": "mirror", "detour_sign": -1.0,
                  "pillar_x": pillar_x, "pillar_y": -pillar_y,
                  "target_y": -target_y, "robot_yaw_deg": -yaw}
        scenes.extend((base, mirror))
    return scenes


def validate(scenes):
    assert len(scenes) == 2 * PAIR_COUNT
    for a, b in zip(scenes[0::2], scenes[1::2]):
        assert a["pair"] == b["pair"]
        assert a["pillar_x"] == b["pillar_x"]
        assert a["pillar_y"] == -b["pillar_y"]
        assert a["target_y"] == -b["target_y"]
        assert a["robot_yaw_deg"] == -b["robot_yaw_deg"]
        assert a["detour_sign"] == -b["detour_sign"]
        assert math.hypot(a["pillar_x"], a["pillar_y"]) > 0.70
    return True


def write(path):
    if os.path.exists(path):
        raise RuntimeError("refusing overwrite: " + path)
    scenes = generate(); validate(scenes)
    with open(path, "w") as output:
        json.dump({"seed": SEED, "scenes": scenes}, output, indent=2, sort_keys=True)
    return scenes

