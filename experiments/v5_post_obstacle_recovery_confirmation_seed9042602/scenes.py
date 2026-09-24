"""New fixed-seed scenes, independent of all permanent examinations."""
import json, math
import numpy as np
SEED=9042602; ROBOT_RADIUS=.18; PILLAR_RADIUS=.30; WALL=3.95
def segment_distance(px,py,tx,ty):
    t=max(0.,min(1.,(px*tx+py*ty)/(tx*tx+ty*ty)));return math.hypot(px-t*tx,py-t*ty)
def generate():
    rng=np.random.RandomState(SEED);out=[]
    for pair in range(1,7):
        px=float(rng.uniform(1.06,1.24));py=float(rng.uniform(.035,.145));ty=float(rng.uniform(.045,.160));yaw=float(rng.uniform(5.,13.))
        for side,sign in (("positive",1.),("negative",-1.)):
            out.append({"name":"recovery_pair%02d_%s"%(pair,side),"pair":pair,"mirror_side":side,"pillar_x":px,"pillar_y":sign*py,"target_y":sign*ty,"robot_yaw_deg":sign*yaw})
    return out
def validate(items):
    checks=[]
    if len(items)!=12:raise AssertionError("expected 12 scenes")
    for s in items:
        rp=math.hypot(s["pillar_x"],s["pillar_y"]);pt=math.hypot(2-s["pillar_x"],s["target_y"]-s["pillar_y"]);wm=min(WALL-abs(s["pillar_x"]),WALL-abs(s["pillar_y"]))-PILLAR_RADIUS;route=segment_distance(s["pillar_x"],s["pillar_y"],2.,s["target_y"])
        legal=rp>ROBOT_RADIUS+PILLAR_RADIUS+.20 and pt>ROBOT_RADIUS+PILLAR_RADIUS+.20 and wm>.5 and route<=.42
        if not legal:raise AssertionError("illegal "+s["name"])
        checks.append({"name":s["name"],"robot_pillar_margin":rp-ROBOT_RADIUS-PILLAR_RADIUS,"pillar_target_margin":pt-ROBOT_RADIUS-PILLAR_RADIUS,"pillar_wall_margin":wm,"direct_route_offset":route,"legal":legal})
    for n in range(1,7):
        a=next(s for s in items if s["pair"]==n and s["mirror_side"]=="positive");b=next(s for s in items if s["pair"]==n and s["mirror_side"]=="negative")
        assert math.isclose(a["pillar_x"],b["pillar_x"],abs_tol=1e-12)
        for k in("pillar_y","target_y","robot_yaw_deg"):assert math.isclose(a[k],-b[k],abs_tol=1e-12)
    return checks
def write(path):
    items=generate();checks=validate(items)
    with open(path,"w")as f:json.dump({"seed":SEED,"generator":{"pillar_x":[1.06,1.24],"pillar_y_magnitude":[.035,.145],"target_y_magnitude":[.045,.160],"yaw_magnitude_deg":[5.,13.],"mirror_fields":["pillar_y","target_y","robot_yaw_deg"]},"scenes":items,"static_checks":checks},f,indent=2,sort_keys=True)
    return items,checks
