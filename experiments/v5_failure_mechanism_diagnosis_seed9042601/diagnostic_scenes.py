"""Independent mirrored diagnostic scenes; no permanent exam data is read."""
import json
import math
import numpy as np

SEED = 9042601
ROBOT_RADIUS = 0.18
PILLAR_RADIUS = 0.30
ARENA_INNER_LIMIT = 3.95

def segment_distance(px, py, x1, y1):
    t = max(0.0, min(1.0, (px * x1 + py * y1) / (x1*x1 + y1*y1)))
    return math.hypot(px - t*x1, py - t*y1)

def generate_scenes():
    rng = np.random.RandomState(SEED)
    scenes = []
    for pair in range(1, 7):
        focus = "near" if pair <= 3 else "far"
        px = float(rng.uniform(0.78, 0.94) if focus == "near" else rng.uniform(1.08, 1.22))
        py = float(rng.uniform(0.045, 0.135))
        ty = float(rng.uniform(0.035, 0.120))
        yaw = float(rng.uniform(4.0, 12.0))
        for side, sign in (("positive", 1.0), ("negative", -1.0)):
            scenes.append({"name": "diag_pair%02d_%s_%s" % (pair, focus, side),
                "pair": pair, "focus": focus, "mirror_side": side,
                "pillar_x": px, "pillar_y": sign*py,
                "target_y": sign*ty, "robot_yaw_deg": sign*yaw})
    return scenes

def static_checks(scenes):
    assert len(scenes) == 12
    checks = []
    for s in scenes:
        rp = math.hypot(s["pillar_x"], s["pillar_y"])
        pt = math.hypot(2.0-s["pillar_x"], s["target_y"]-s["pillar_y"])
        wall = min(ARENA_INNER_LIMIT-abs(s["pillar_x"]), ARENA_INNER_LIMIT-abs(s["pillar_y"]))-PILLAR_RADIUS
        route = segment_distance(s["pillar_x"], s["pillar_y"], 2.0, s["target_y"])
        legal = rp > ROBOT_RADIUS+PILLAR_RADIUS+0.20 and pt > ROBOT_RADIUS+PILLAR_RADIUS+0.20 and wall > 0.50 and route <= 0.42
        if not legal: raise AssertionError("illegal scene: "+s["name"])
        checks.append({"name":s["name"], "robot_pillar_margin":rp-ROBOT_RADIUS-PILLAR_RADIUS,
            "pillar_target_margin":pt-ROBOT_RADIUS-PILLAR_RADIUS,
            "pillar_wall_margin":wall, "direct_route_offset":route, "legal":legal})
    for n in range(1, 7):
        p=[s for s in scenes if s["pair"]==n]; a=next(s for s in p if s["mirror_side"]=="positive"); b=next(s for s in p if s["mirror_side"]=="negative")
        assert math.isclose(a["pillar_x"],b["pillar_x"],abs_tol=1e-12)
        for key in ("pillar_y","target_y","robot_yaw_deg"): assert math.isclose(a[key],-b[key],abs_tol=1e-12)
    return checks

def write_manifest(path):
    scenes=generate_scenes(); checks=static_checks(scenes)
    with open(path,"w") as f: json.dump({"seed":SEED,"generator":{"near_pillar_x":[0.78,0.94],"far_pillar_x":[1.08,1.22],"pillar_y_magnitude":[0.045,0.135],"target_y_magnitude":[0.035,0.120],"robot_yaw_magnitude_deg":[4.0,12.0],"mirror_fields":["pillar_y","target_y","robot_yaw_deg"]},"scenes":scenes,"static_checks":checks},f,indent=2,sort_keys=True)
    return scenes,checks
