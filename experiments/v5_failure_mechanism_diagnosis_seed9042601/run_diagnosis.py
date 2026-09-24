"""Deterministic V5 diagnosis only; contains no optimizer or training path."""
import argparse,csv,json,math,os
import numpy as np
import rospy,torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from control_gate import minimum_valid_range
from diagnostic_scenes import SEED,write_manifest
from residual_networks import ResidualActor
from scenario_env import CENTER_SCENE,V5ScenarioEnv

MAX_STEPS=300; ACTOR_EPISODES=3; RULE_EPISODES=3; EARLY_STEP=30
FIELDS=["controller","scene","pair","focus","mirror_side","episode","step"]+["obs_%02d"%i for i in range(16)]+["residual_linear","residual_angular","cmd_linear","cmd_angular","gate_active","min_lidar","robot_x","robot_y","robot_yaw_deg","goal_distance","goal_relative_angle_deg","post_obstacle","terminal_reason"]

class Env(V5ScenarioEnv):
    def reset(self):
        obs=super().reset(); self._set_pose("obstacle_0",self.scenario["pillar_x"],self.scenario["pillar_y"],0.30); rospy.sleep(0.5)
        scan=rospy.wait_for_message("/scan",LaserScan,timeout=5); self._previous_arrival_position=None; obs=self._initial_observation(scan); self.previous_front_clearance=minimum_valid_range(scan); self.control_gate.reset(); return obs

class Actor:
    def __init__(self,actor): self.actor=actor
    def __call__(self,obs,env):
        del env
        with torch.inference_mode(): return self.actor(obs).squeeze(0).numpy().astype(np.float32)

class Reactive:
    """Fixed LiDAR/relative-goal-only bounded residual state machine."""
    def __init__(self): self.sign=None
    def sector(self,scan,lo,hi):
        values=[]
        for i,v in enumerate(scan.ranges):
            angle=(math.degrees(scan.angle_min+i*scan.angle_increment)+180)%360-180
            if lo<=angle<=hi and math.isfinite(v): values.append(min(float(v),float(scan.range_max)))
        return sum(values)/len(values) if values else 0.0
    def __call__(self,obs,env):
        del obs
        clear=minimum_valid_range(env.latest_scan)
        if self.sign is None and clear<0.80: self.sign=1.0 if self.sector(env.latest_scan,15,105)>=self.sector(env.latest_scan,-105,-15) else -1.0
        if not env.control_gate.active and clear>=0.80: return np.zeros(2,dtype=np.float32)
        if clear<0.58: return np.asarray([-1.0,self.sign or 1.0],dtype=np.float32)
        heading=float(env.diff_angle)
        if abs(heading)>8: return np.asarray([-0.25,np.clip(heading/24.0,-1,1)],dtype=np.float32)
        return np.asarray([1.0,np.clip(heading/30.0,-1,1)],dtype=np.float32)

def run_episode(env,scene,episode,source,name):
    env.scenario=dict(scene); obs=env.reset(); past=np.zeros(2,dtype=np.float32); traces=[]; minimum=float("inf"); outcome="timeout"
    for step in range(1,MAX_STEPS+1):
        actor_input=obs.copy(); residual=source(obs,env); obs,_,done,arrive,command,gate,before=env.step_residual(residual,past); clear=min(float(before),minimum_valid_range(env.latest_scan)); minimum=min(minimum,clear)
        outcome="success" if arrive else ("collision" if done else "timeout"); terminal=outcome if done or arrive or step==MAX_STEPS else "running"; distance=math.hypot(env.goal_position.position.x-env.position.x,env.goal_position.position.y-env.position.y)
        row={"controller":name,"scene":scene["name"],"pair":scene["pair"],"focus":scene["focus"],"mirror_side":scene["mirror_side"],"episode":episode,"step":step,"residual_linear":float(residual[0]),"residual_angular":float(residual[1]),"cmd_linear":float(command[0]),"cmd_angular":float(command[1]),"gate_active":int(bool(gate)),"min_lidar":clear,"robot_x":float(env.position.x),"robot_y":float(env.position.y),"robot_yaw_deg":float(env.yaw),"goal_distance":distance,"goal_relative_angle_deg":float(env.diff_angle),"post_obstacle":int(float(env.position.x)>scene["pillar_x"]+0.30),"terminal_reason":terminal}
        for i,v in enumerate(actor_input): row["obs_%02d"%i]=float(v)
        traces.append(row); past=residual
        if done or arrive: break
    env.pub_cmd_vel.publish(Twist()); return traces,{"controller":name,"scene":scene["name"],"pair":scene["pair"],"focus":scene["focus"],"mirror_side":scene["mirror_side"],"episode":episode,"outcome":outcome,"steps":step,"min_lidar":minimum}

def representatives(scenes,episodes):
    selected=[]
    for side in ("negative","positive"):
        ranked=[]
        for scene in scenes:
            if scene["mirror_side"]!=side: continue
            n=sum(e["outcome"]=="collision" and e["steps"]<=EARLY_STEP for e in episodes if e["scene"]==scene["name"])
            if n: ranked.append((-n,scene["name"],scene))
        if ranked: selected.append(sorted(ranked)[0][2])
    return selected

def write_csv(path,fields,rows):
    with open(path,"w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--results-dir",required=True); a=p.parse_args()
    if os.path.exists(a.results_dir): raise RuntimeError("refusing to overwrite results: "+a.results_dir)
    os.makedirs(a.results_dir); scenes,checks=write_manifest(os.path.join(a.results_dir,"scenes.json"))
    with open(os.path.join(a.results_dir,"protocol.json"),"w") as f: json.dump({"seed":SEED,"actor_episodes_per_scene":3,"rule_episodes_per_representative":3,"max_steps":300,"early_collision_step":30,"deterministic_actor":True,"exploration":False,"training":False,"world_coordinates_in_actor_input":False},f,indent=2,sort_keys=True)
    print("STATIC_CHECKS_PASS scenes=%d seed=%d"%(len(checks),SEED),flush=True); rospy.init_node("v5_failure_mechanism_diagnosis",anonymous=True)
    actor=ResidualActor(16,2); actor.load_state_dict(torch.load(a.checkpoint,map_location="cpu")); actor.eval(); env=Env(); traces=[]; episodes=[]
    try:
        source=Actor(actor)
        for scene in scenes:
            for episode in range(1,4):
                t,s=run_episode(env,scene,episode,source,"v5_actor"); traces+=t; episodes.append(s); print("DIAG_ACTOR scene=%s ep=%d outcome=%s steps=%d min=%.4f"%(scene["name"],episode,s["outcome"],s["steps"],s["min_lidar"]),flush=True)
        selected=representatives(scenes,episodes)
        with open(os.path.join(a.results_dir,"rule_representatives.json"),"w") as f: json.dump({"selection_rule":"highest early-collision count per mirror side; name tie-break","scenes":selected},f,indent=2,sort_keys=True)
        for scene in selected:
            for episode in range(1,4):
                t,s=run_episode(env,scene,episode,Reactive(),"reactive_template"); traces+=t; episodes.append(s); print("DIAG_RULE scene=%s ep=%d outcome=%s steps=%d min=%.4f"%(scene["name"],episode,s["outcome"],s["steps"],s["min_lidar"]),flush=True)
        write_csv(os.path.join(a.results_dir,"step_trace.csv"),FIELDS,traces); write_csv(os.path.join(a.results_dir,"episodes.csv"),["controller","scene","pair","focus","mirror_side","episode","outcome","steps","min_lidar"],episodes); print("DIAGNOSIS_RUN_COMPLETE actor=36 rule=%d"%(len(episodes)-36),flush=True)
    finally:
        env.scenario={"pillar_x":1.0,**CENTER_SCENE}; env.reset(); env.pub_cmd_vel.publish(Twist()); rospy.sleep(0.2); env.pub_cmd_vel.publish(Twist()); print("CENTER_RESTORED_ZERO_VELOCITY",flush=True)
if __name__=="__main__": main()
