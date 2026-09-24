"""Deterministic final-V5 trace collection; no optimizer or training path."""
import argparse,csv,json,math,os
import numpy as np
import rospy,torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from control_gate import minimum_valid_range
from residual_networks import ResidualActor
from scenario_env import CENTER_SCENE,V5ScenarioEnv
from scenes import SEED,write
MAX_STEPS=300
BASE=["scene","pair","mirror_side","episode","step"]
FIELDS=BASE+["obs_%02d"%i for i in range(16)]+["residual_linear","residual_angular","cmd_linear","cmd_angular","gate_active","gate_switches_cumulative","residual_angular_sign_switches_cumulative","cmd_angular_sign_switches_cumulative","min_lidar","robot_x","robot_y","robot_yaw_deg","goal_distance","goal_relative_angle_deg","entered_040","terminal_reason"]
class Env(V5ScenarioEnv):
    def reset(self):
        obs=super().reset();self._set_pose("obstacle_0",self.scenario["pillar_x"],self.scenario["pillar_y"],.30);rospy.sleep(.5);scan=rospy.wait_for_message("/scan",LaserScan,timeout=5);self._previous_arrival_position=None;obs=self._initial_observation(scan);self.previous_front_clearance=minimum_valid_range(scan);self.control_gate.reset();return obs
def sign(v):return 1 if v>.02 else(-1 if v<-.02 else 0)
def episode(env,actor,scene,ep):
    env.scenario=dict(scene);obs=env.reset();past=np.zeros(2,dtype=np.float32);rows=[];minimum=float("inf");prev_gate=False;prev_rs=0;prev_cs=0;gate_sw=rs_sw=cs_sw=0;entered=False;outcome="general_timeout"
    for step in range(1,MAX_STEPS+1):
        actor_input=obs.copy()
        with torch.inference_mode():residual=actor(actor_input).squeeze(0).numpy().astype(np.float32)
        obs,_,done,arrive,cmd,gate,before=env.step_residual(residual,past);clear=min(float(before),minimum_valid_range(env.latest_scan));minimum=min(minimum,clear);distance=math.hypot(env.goal_position.position.x-env.position.x,env.goal_position.position.y-env.position.y);entered=entered or distance<.40
        if step>1 and bool(gate)!=prev_gate:gate_sw+=1
        rsign,csign=sign(float(residual[1])),sign(float(cmd[1]))
        if prev_rs and rsign and rsign!=prev_rs:rs_sw+=1
        if prev_cs and csign and csign!=prev_cs:cs_sw+=1
        if arrive:outcome="success"
        elif done:outcome="early_collision" if step<=30 else "late_collision"
        elif step==MAX_STEPS:outcome="near_goal_safe_timeout" if .20<=distance<=.30 else "general_timeout"
        else:outcome="running"
        row={"scene":scene["name"],"pair":scene["pair"],"mirror_side":scene["mirror_side"],"episode":ep,"step":step,"residual_linear":float(residual[0]),"residual_angular":float(residual[1]),"cmd_linear":float(cmd[0]),"cmd_angular":float(cmd[1]),"gate_active":int(gate),"gate_switches_cumulative":gate_sw,"residual_angular_sign_switches_cumulative":rs_sw,"cmd_angular_sign_switches_cumulative":cs_sw,"min_lidar":clear,"robot_x":float(env.position.x),"robot_y":float(env.position.y),"robot_yaw_deg":float(env.yaw),"goal_distance":distance,"goal_relative_angle_deg":float(env.diff_angle),"entered_040":int(entered),"terminal_reason":outcome}
        for i,v in enumerate(actor_input):row["obs_%02d"%i]=float(v)
        rows.append(row);past=residual;prev_gate=bool(gate);prev_rs=rsign or prev_rs;prev_cs=csign or prev_cs
        if done or arrive:break
    env.pub_cmd_vel.publish(Twist());return rows,{"scene":scene["name"],"pair":scene["pair"],"mirror_side":scene["mirror_side"],"episode":ep,"classification":outcome,"steps":step,"min_lidar":minimum,"final_goal_distance":distance,"ever_entered_040":int(entered),"gate_switches":gate_sw,"residual_angular_sign_switches":rs_sw,"cmd_angular_sign_switches":cs_sw}
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--results-dir",required=True);a=p.parse_args()
    if os.path.exists(a.results_dir):raise RuntimeError("refusing overwrite: "+a.results_dir)
    os.makedirs(a.results_dir);scenes,checks=write(os.path.join(a.results_dir,"scenes.json"))
    protocol={"seed":SEED,"scenes":12,"episodes_per_scene":3,"max_steps":300,"deterministic":True,"exploration":False,"training":False,"success_threshold_unchanged_m":.20,"near_goal_safe_timeout_final_distance_m":[.20,.30],"early_collision_max_step":30,"comparison_window":"later of first distance < 0.40 m and final 100 steps","systematically_higher":"timeout median > success median and at least 75% of timeout values > success median","repeated_nonconvergence":"at least two thirds of near-goal timeouts have >=4 distance-trend reversals and final-minus-minimum distance >=0.02 m","support_requires_near_goal_timeouts":3}
    with open(os.path.join(a.results_dir,"protocol.json"),"w")as f:json.dump(protocol,f,indent=2,sort_keys=True)
    print("STATIC_PASS scenes=%d seed=%d"%(len(checks),SEED),flush=True);rospy.init_node("v5_recovery_confirmation",anonymous=True);actor=ResidualActor(16,2);actor.load_state_dict(torch.load(a.checkpoint,map_location="cpu"));actor.eval();env=Env();traces=[];summaries=[]
    try:
        for scene in scenes:
            for ep in range(1,4):
                r,s=episode(env,actor,scene,ep);traces+=r;summaries.append(s);print("CONFIRM scene=%s ep=%d class=%s steps=%d final=%.4f min=%.4f"%(scene["name"],ep,s["classification"],s["steps"],s["final_goal_distance"],s["min_lidar"]),flush=True)
        with open(os.path.join(a.results_dir,"step_trace.csv"),"w",newline="")as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(traces)
        with open(os.path.join(a.results_dir,"episodes.csv"),"w",newline="")as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
        near=[r for r in traces if int(r["entered_040"])];
        with open(os.path.join(a.results_dir,"after_first_040_trace.csv"),"w",newline="")as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(near)
        print("CONFIRMATION_RUN_COMPLETE episodes=36",flush=True)
    finally:
        env.scenario={"pillar_x":1.,**CENTER_SCENE};env.reset();env.pub_cmd_vel.publish(Twist());rospy.sleep(.2);env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO_VELOCITY",flush=True)
if __name__=="__main__":main()
