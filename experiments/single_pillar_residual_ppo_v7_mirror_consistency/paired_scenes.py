"""Generate fresh legal training-only scene pairs without reading any dataset."""
import argparse
import json
import math
import os
import random

SEED = 7042604
PAIR_COUNT = 24
PILLAR_RADIUS = 0.30
ROBOT_RADIUS = 0.105
STATIC_MARGIN = 0.08


def mirror(scene, name):
    return {"name": name, "pair": scene["pair"], "side": "negative",
            "pillar_x": scene["pillar_x"], "pillar_y": -scene["pillar_y"],
            "target_x": scene["target_x"], "target_y": -scene["target_y"],
            "robot_yaw_deg": -scene["robot_yaw_deg"]}


def legal(scene):
    robot_pillar = math.hypot(scene["pillar_x"], scene["pillar_y"])
    target_pillar = math.hypot(scene["target_x"] - scene["pillar_x"],
                               scene["target_y"] - scene["pillar_y"])
    required = PILLAR_RADIUS + ROBOT_RADIUS + STATIC_MARGIN
    return robot_pillar > required and target_pillar > PILLAR_RADIUS + 0.20


def generate():
    rng = random.Random(SEED)
    scenes = []
    for pair in range(1, PAIR_COUNT + 1):
        base = {"name": "pair_%02d_positive" % pair, "pair": pair,
                "side": "positive", "pillar_x": round(rng.uniform(0.72, 1.18), 6),
                "pillar_y": round(rng.uniform(0.10, 0.34), 6),
                "target_x": 2.0, "target_y": round(rng.uniform(0.08, 0.36), 6),
                "robot_yaw_deg": round(rng.uniform(1.0, 9.0), 6)}
        other = mirror(base, "pair_%02d_negative" % pair)
        if not legal(base) or not legal(other):
            raise AssertionError("generated illegal pair")
        scenes.extend((base, other))
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise RuntimeError("refusing to overwrite: " + args.output)
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    payload = {"seed": SEED, "pair_count": PAIR_COUNT,
               "scene_count": 2 * PAIR_COUNT, "training_started": False,
               "scenes": generate()}
    with open(args.output, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    print("WROTE_PAIRS pairs=%d scenes=%d seed=%d" %
          (PAIR_COUNT, 2 * PAIR_COUNT, SEED))


if __name__ == "__main__":
    main()

