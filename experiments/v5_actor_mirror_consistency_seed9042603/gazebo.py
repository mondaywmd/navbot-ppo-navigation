"""Closed-loop paired mirror audit using unchanged final V5 Actor."""
import argparse,csv,json,math,os
import numpy as np,rospy,torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from control_gate import minimum_valid_range
from residual_networks import ResidualActor
from scenario_env import CENTER_SCENE,V5ScenarioEnv
from scenes import SEED,write
MAX_STEPS=300
class Env(V5ScenarioEnv):
    def reset(self):
        obs=super().reset();self._set_pose("obstacle_0",self.scenario["pillar_x"],self.scenario["pillar_y"],.30);rospy.sleep(.5);scan=rospy.wait_for_message("/scan",LaserScan,timeout=5);self._previous_arrival_position=None;obs=self._initial_observation(scan);self.previous_front_clearance=minimum_valid_range(scan);self.control_gate.reset();return obs
def run(env,actor,scene,ep):
    env.scenario=dict(scene);obs=env.reset();past=np.zeros(2,dtype=np.float32);trace=[];minimum=float("inf");outcome="timeout"
    for step in range(1,MAX_STEPS+1):
        actor_input=obs.copy()
        with torch.inference_mode():action=actor(actor_input).squeeze(0).numpy().astype(np.float32)
        obs,_,done,arrive,cmd,gate,before=env.step_residual(action,past);clear=min(float(before),minimum_valid_range(env.latest_scan));minimum=min(minimum,clear)
        if step<=20:
            row={"scene":scene["name"],"pair":scene["pair"],"mirror_side":scene["mirror_side"],"episode":ep,"step":step,"residual_linear":float(action[0]),"residual_angular":float(action[1]),"cmd_linear":float(cmd[0]),"cmd_angular":float(cmd[1]),"gate_active":int(gate),"min_lidar":clear,"robot_x":float(env.position.x),"robot_y":float(env.position.y),"robot_yaw_deg":float(env.yaw)}
            for i,v in enumerate(actor_input):row["obs_%02d"%i]=float(v)
            trace.append(row)
        past=action
        if arrive:outcome="success";break
        if done:outcome="early_collision"if step<=30 else"late_collision";break
    env.pub_cmd_vel.publish(Twist());return trace,{"scene":scene["name"],"pair":scene["pair"],"mirror_side":scene["mirror_side"],"episode":ep,"outcome":outcome,"steps":step,"min_lidar":minimum}
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--results-dir",required=True);a=p.parse_args()
    if os.path.exists(a.results_dir):raise RuntimeError("refusing overwrite: "+a.results_dir)
    os.makedirs(a.results_dir);scenes,checks=write(os.path.join(a.results_dir,"scenes.json"));protocol={"seed":SEED,"pairs":6,"episodes_per_side":3,"max_steps":300,"early_window_steps":20,"early_collision_max_step":30,"exploration":False,"training":False,"pair_violation_thresholds":{"mean_linear_error":.05,"mean_angular_error":.10},"systematic_pair_count":4,"one_sided_early_collision_pair_count":2,"associated_pair_angular_error_median":.10,"offline_thresholds":{"linear_median":.05,"linear_p90":.10,"angular_median":.10,"angular_p90":.20}}
    with open(os.path.join(a.results_dir,"protocol.json"),"w")as f:json.dump(protocol,f,indent=2,sort_keys=True)
    print("STATIC_PASS scenes=%d seed=%d"%(len(checks),SEED),flush=True);rospy.init_node("v5_actor_mirror_closed_loop",anonymous=True);actor=ResidualActor(16,2);actor.load_state_dict(torch.load(a.checkpoint,map_location="cpu"));actor.eval();env=Env();traces=[];episodes=[]
    try:
        for scene in scenes:
            for ep in range(1,4):
                t,s=run(env,actor,scene,ep);traces+=t;episodes.append(s);print("MIRROR scene=%s ep=%d outcome=%s steps=%d"%(scene["name"],ep,s["outcome"],s["steps"]),flush=True)
        fields=list(traces[0]);
        with open(os.path.join(a.results_dir,"first20.csv"),"w",newline="")as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(traces)
        with open(os.path.join(a.results_dir,"episodes.csv"),"w",newline="")as f:w=csv.DictWriter(f,fieldnames=list(episodes[0]));w.writeheader();w.writerows(episodes)
        print("GAZEBO_COMPLETE episodes=36",flush=True)
    finally:
        env.scenario={"pillar_x":1.,**CENTER_SCENE};env.reset();env.pub_cmd_vel.publish(Twist());rospy.sleep(.2);env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO_VELOCITY",flush=True)
if __name__=="__main__":main()
