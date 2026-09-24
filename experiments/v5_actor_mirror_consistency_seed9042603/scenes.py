"""Six new legal near-pillar mirror pairs from an independent seed."""
import json,math
import numpy as np
SEED=9042603;RR=.18;PR=.30;WALL=3.95
def route_distance(px,py,ty):
    t=max(0.,min(1.,(2*px+ty*py)/(4+ty*ty)));return math.hypot(px-2*t,py-ty*t)
def generate():
    rng=np.random.RandomState(SEED);out=[]
    for pair in range(1,7):
        px=float(rng.uniform(.80,.95));py=float(rng.uniform(.045,.140));ty=float(rng.uniform(.035,.125));yaw=float(rng.uniform(4.,12.))
        for side,sign in(("positive",1.),("negative",-1.)):out.append({"name":"mirror_pair%02d_%s"%(pair,side),"pair":pair,"mirror_side":side,"pillar_x":px,"pillar_y":sign*py,"target_y":sign*ty,"robot_yaw_deg":sign*yaw})
    return out
def validate(items):
    checks=[]
    for s in items:
        rp=math.hypot(s["pillar_x"],s["pillar_y"]);pt=math.hypot(2-s["pillar_x"],s["target_y"]-s["pillar_y"]);wall=min(WALL-abs(s["pillar_x"]),WALL-abs(s["pillar_y"]))-PR;route=route_distance(s["pillar_x"],s["pillar_y"],s["target_y"]);legal=rp>RR+PR+.20 and pt>RR+PR+.20 and wall>.5 and route<=.42
        if not legal:raise AssertionError("illegal "+s["name"])
        checks.append({"name":s["name"],"robot_pillar_margin":rp-RR-PR,"pillar_target_margin":pt-RR-PR,"wall_margin":wall,"route_offset":route,"legal":legal})
    for n in range(1,7):
        a=next(s for s in items if s["pair"]==n and s["mirror_side"]=="positive");b=next(s for s in items if s["pair"]==n and s["mirror_side"]=="negative");assert a["pillar_x"]==b["pillar_x"]
        for k in("pillar_y","target_y","robot_yaw_deg"):assert a[k]==-b[k]
    return checks
def write(path):
    s=generate();c=validate(s)
    with open(path,"w")as f:json.dump({"seed":SEED,"generator":{"pillar_x":[.80,.95],"pillar_y_magnitude":[.045,.140],"target_y_magnitude":[.035,.125],"yaw_magnitude_deg":[4.,12.]},"scenes":s,"checks":c},f,indent=2,sort_keys=True)
    return s,c
