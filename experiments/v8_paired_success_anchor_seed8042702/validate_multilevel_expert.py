"""A* waypoint expert feasibility smoke under unchanged residual bounds."""
import csv
import heapq
import json
import math
import os
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from control_gate import minimum_valid_range
from curriculum_spec import generate, mirror
from wide_env import WideStaticEnv

GRID = 0.10
INFLATION = 0.34
MAX_STEPS = 300


def blocked(x, y, scene):
    if not (-0.45 <= x <= 3.65 and -3.45 <= y <= 3.45): return True
    return any(math.hypot(x-p["x"], y-p["y"]) <= p["radius"] + INFLATION
               for p in scene["obstacles"])


def plan(scene):
    goal = (round(scene["target_x"]/GRID), round(scene["target_y"]/GRID))
    start = (0, 0); queue = [(0.0, start)]; cost = {start: 0.0}; parent = {}
    moves = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while queue:
        _, node = heapq.heappop(queue)
        if node == goal: break
        for dx, dy in moves:
            nxt = (node[0]+dx, node[1]+dy); x, y = nxt[0]*GRID, nxt[1]*GRID
            if blocked(x, y, scene): continue
            candidate = cost[node] + math.hypot(dx, dy)
            if candidate < cost.get(nxt, float("inf")):
                cost[nxt] = candidate; parent[nxt] = node
                heuristic = math.hypot(nxt[0]-goal[0], nxt[1]-goal[1])
                heapq.heappush(queue, (candidate+heuristic, nxt))
    if goal not in cost: raise RuntimeError("no inflated-grid path")
    nodes = [goal]
    while nodes[-1] != start: nodes.append(parent[nodes[-1]])
    nodes.reverse(); return [(x*GRID, y*GRID) for x,y in nodes]


class PathExpert:
    def __init__(self, scene): self.path = plan(scene); self.index = 0
    @staticmethod
    def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
    def action(self, env):
        x,y=float(env.position.x),float(env.position.y)
        nearest=min(range(self.index,len(self.path)),key=lambda i:math.hypot(x-self.path[i][0],y-self.path[i][1]))
        self.index=max(self.index,nearest); target=self.path[min(len(self.path)-1,self.index+4)]
        error=self.wrap(math.atan2(target[1]-y,target[0]-x)-math.radians(float(env.yaw)))
        return np.asarray([-0.7 if abs(error)>math.radians(45) else 1.0,
                           np.clip(1.6*error/0.30,-1,1)],dtype=np.float32)


def run(env,scene):
    obs=env.reset_wide(scene); expert=PathExpert(scene); past=np.zeros(2,dtype=np.float32)
    minimum=float("inf"); outcome="timeout"
    for step in range(1,MAX_STEPS+1):
        action=expert.action(env); obs,_,done,arrive,_,_,before=env.step_residual(action,past)
        minimum=min(minimum,float(before),minimum_valid_range(env.latest_scan));past=action
        if arrive: outcome="success";break
        if done: outcome="collision";break
    env.pub_cmd_vel.publish(Twist());return {"name":scene["name"],"level":scene["level"],
        "side":"mirror" if scene["name"].endswith("_mirror") else "base",
        "outcome":outcome,"steps":step,"min_lidar":minimum,"path_points":len(expert.path)}


def main():
    root=os.path.dirname(os.path.abspath(__file__));outdir=os.path.join(root,"results","multilevel_expert_smoke_seed8042703")
    if os.path.exists(outdir):raise RuntimeError("refusing overwrite: "+outdir)
    os.makedirs(outdir);rospy.init_node("v8_multilevel_expert_smoke",anonymous=True);env=WideStaticEnv();rows=[]
    try:
        for level in (1,2,3):
            base=generate(level,1)[0]
            for scene in (base,mirror(base)):
                row=run(env,scene);rows.append(row);print("MULTI_EXPERT level=%d side=%s outcome=%s steps=%d min=%.3f"%(level,row["side"],row["outcome"],row["steps"],row["min_lidar"]),flush=True)
        with open(os.path.join(outdir,"episodes.csv"),"w",newline="")as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        with open(os.path.join(outdir,"summary.json"),"w")as f:json.dump({"episodes":rows,"successes":sum(r["outcome"]=="success"for r in rows),"training":False,"dataset_created":False},f,indent=2,sort_keys=True)
    finally:env.restore_center();env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO",flush=True)


if __name__=="__main__":main()
