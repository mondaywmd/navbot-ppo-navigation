"""Fresh mirror-paired 200-step audit with a shielded waypoint controller."""
import argparse
import csv
import heapq
import json
import math
import os

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from clean_curriculum_spec import generate, mirror
from safety_shield import filter_command, valid_scan
from temporal_lidar import sector_minima, temporal_features
from potential_escape import PotentialEscapeShield
from tangent_escape import nearest_bearing
from clean_wide_env import CleanWideStaticEnv

SEED = 8042715
GRID = 0.10
INFLATION = 0.40

def blocked(x, y, scene):
    if not (-0.45 <= x <= 3.65 and -3.45 <= y <= 3.45): return True
    goal_distance=math.hypot(scene["target_x"]-x,scene["target_y"]-y)
    local_clearance=0.25+0.15*min(1.0,goal_distance/0.50)
    return any(math.hypot(x-p["x"], y-p["y"]) <= p["radius"] + local_clearance
               for p in scene["obstacles"])

def plan(scene):
    start=(0,0); goal=(round(scene["target_x"]/GRID),round(scene["target_y"]/GRID))
    queue=[(0.0,start)]; cost={start:0.0}; parent={}
    moves=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while queue:
        _,node=heapq.heappop(queue)
        if node==goal: break
        for dx,dy in moves:
            nxt=(node[0]+dx,node[1]+dy); x,y=nxt[0]*GRID,nxt[1]*GRID
            if blocked(x,y,scene): continue
            candidate=cost[node]+math.hypot(dx,dy)
            if candidate<cost.get(nxt,float("inf")):
                cost[nxt]=candidate;parent[nxt]=node
                heapq.heappush(queue,(candidate+math.hypot(nxt[0]-goal[0],nxt[1]-goal[1]),nxt))
    if goal not in cost: raise RuntimeError("no inflated path")
    nodes=[goal]
    while nodes[-1]!=start:nodes.append(parent[nodes[-1]])
    return [(x*GRID,y*GRID) for x,y in reversed(nodes)]

def wrap(value): return (value+math.pi)%(2*math.pi)-math.pi

def step_limit(scene):
    return 200

def audit_episode(env, scene, record_trace=False, diagnostic_limit=None):
    env.reset_wide(scene); path=plan(scene); index=0; previous_sector=None;shield=PotentialEscapeShield()
    official_limit=step_limit(scene);limit=diagnostic_limit or official_limit
    positions=[];trace=[]; minimum=float("inf"); distance_at_limit=None; reasons={}; outcome="timeout"
    for step in range(1,limit+1):
        state=env.get_model_state("turtlebot3_burger","world")
        x,y=state.pose.position.x,state.pose.position.y
        q=state.pose.orientation; yaw=math.atan2(2*q.w*q.z,1-2*q.z*q.z)
        positions.append((x,y)); distance=math.hypot(scene["target_x"]-x,scene["target_y"]-y)
        if distance<=0.20:
            outcome="success";break
        nearest=min(range(index,len(path)),key=lambda i:math.hypot(x-path[i][0],y-path[i][1]))
        index=max(index,nearest); waypoint=path[min(len(path)-1,index+5)]
        error=wrap(math.atan2(waypoint[1]-y,waypoint[0]-x)-yaw)
        requested_linear=0.20 if abs(error)<math.radians(35) else 0.05
        requested_angular=float(np.clip(1.8*error,-0.8,0.8))
        scan=rospy.wait_for_message("/scan",LaserScan,timeout=5)
        current=sector_minima(scan.ranges,scan.range_min,scan.range_max)
        front_ttc=float("inf");turn_side_ttc=float("inf")
        if previous_sector is not None and abs(requested_angular)<0.10:
            _,ttc=temporal_features(previous_sector,current,0.2)
            front_ttc=float(min(ttc[17],ttc[18]))
        previous_sector=current
        was_active=shield.active
        obstacle_bearing,obstacle_distance=nearest_bearing(scan)
        linear,angular,reason=shield.apply(requested_linear,requested_angular,scan,front_ttc,turn_side_ttc,
                                           goal_distance=distance)
        reasons[reason]=reasons.get(reason,0)+1
        minimum=min(minimum,float(valid_scan(scan).min()))
        command=Twist();command.linear.x=linear;command.angular.z=angular;env.pub_cmd_vel.publish(command)
        if record_trace:
            trace.append({"step":step,"x":x,"y":y,"yaw":yaw,"goal_distance":distance,
                "nearest_distance":obstacle_distance,"nearest_bearing":obstacle_bearing,
                "requested_linear":requested_linear,"requested_angular":requested_angular,
                "final_linear":linear,"final_angular":angular,"reason":reason,
                "escape_active_before":was_active,"escape_active_after":shield.active})
        if step==limit: distance_at_limit=distance
        if minimum<0.20: outcome="collision";break
    env.pub_cmd_vel.publish(Twist())
    final_distance=math.hypot(scene["target_x"]-positions[-1][0],scene["target_y"]-positions[-1][1])
    if outcome=="timeout":
        tail=positions[-100:];travel=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(tail,tail[1:]))
        displacement=math.hypot(tail[-1][0]-tail[0][0],tail[-1][1]-tail[0][1])
        outcome="circling_timeout" if travel>1.0 and displacement<0.30 else ("stalled_timeout" if travel<0.20 else "progress_timeout")
    result={"level":scene["level"],"scene":scene["name"],"side":"mirror" if scene["name"].endswith("_mirror") else "base",
            "outcome":outcome,"steps":step,"step_limit":limit,"official_step_limit":official_limit,
            "diagnostic_extension":diagnostic_limit is not None,"minimum_lidar":minimum,"distance_at_limit":distance_at_limit,
            "final_distance":final_distance,"shield_reasons":json.dumps(reasons,sort_keys=True)}
    if record_trace:result["trace"]=trace
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--pairs-per-level",type=int,default=2)
    parser.add_argument("--levels",default="1,2,3,4,5")
    parser.add_argument("--output-dir",default=None);args=parser.parse_args()
    root=os.path.dirname(os.path.abspath(__file__));outdir=args.output_dir or os.path.join(root,"results","horizon_audit_side_shield_seed%d"%args.seed)
    if os.path.exists(outdir):raise RuntimeError("refusing overwrite: "+outdir)
    rospy.init_node("clean_start_horizon_audit",anonymous=True);env=CleanWideStaticEnv();rows=[]
    try:
        levels=tuple(int(value) for value in args.levels.split(","))
        for level in levels:
            for base in generate(level,args.pairs_per_level,seed=args.seed+level):
                for scene in (base,mirror(base)):
                    row=audit_episode(env,scene);rows.append(row)
                    print("HORIZON level=%d side=%s outcome=%s steps=%d min=%.3f"%(level,row["side"],row["outcome"],row["steps"],row["minimum_lidar"]),flush=True)
        counts={name:sum(r["outcome"]==name for r in rows) for name in sorted(set(r["outcome"] for r in rows))}
        os.makedirs(outdir)
        with open(os.path.join(outdir,"episodes.csv"),"w",newline="")as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        with open(os.path.join(outdir,"summary.json"),"w")as f:
            json.dump({"seed":args.seed,"episodes":len(rows),"counts":counts,"training":False,
                       "permanent_ood_accessed":False},f,indent=2,sort_keys=True)
        print("HORIZON_COMPLETE "+json.dumps(counts,sort_keys=True),flush=True)
    finally:
        env.restore_center();env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO",flush=True)

if __name__=="__main__":main()
