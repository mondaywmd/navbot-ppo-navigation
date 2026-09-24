"""Collect new shielded waypoint demonstrations using only the 78-D contract."""
import argparse
import csv
import json
import math
import os

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from clean_curriculum_spec import generate, mirror
from horizon_audit import plan, wrap
from observation78 import build_observation
from safety_shield import filter_command, valid_scan
from temporal_lidar import sector_minima, signed_range_rate, temporal_features
from potential_escape import PotentialEscapeShield
from clean_wide_env import CleanWideStaticEnv

MIN_CLEARANCE=0.20; MAX_CANDIDATES_PER_LEVEL=40

def step_limit(scene):
    return 200

def normalized_action(linear,angular):
    return np.asarray([np.clip(2.0*(linear+0.10)/0.35-1.0,-1,1),
                       np.clip(angular/0.8,-1,1)],dtype=np.float32)

def run(env,scene):
    env.reset_wide(scene);route=plan(scene);index=0;previous_sector=None;shield=PotentialEscapeShield()
    previous_action=np.zeros(2,dtype=np.float32);observations=[];actions=[]
    minimum=float("inf");reasons={};outcome="timeout"
    for step in range(1,step_limit(scene)+1):
        state=env.get_model_state("turtlebot3_burger","world")
        x,y=state.pose.position.x,state.pose.position.y;q=state.pose.orientation
        yaw=math.atan2(2*q.w*q.z,1-2*q.z*q.z)
        distance=math.hypot(scene["target_x"]-x,scene["target_y"]-y)
        if distance<=0.20:outcome="success";break
        scan=rospy.wait_for_message("/scan",LaserScan,timeout=5)
        current=sector_minima(scan.ranges,scan.range_min,scan.range_max)
        rates=(np.zeros(36,dtype=np.float32) if previous_sector is None else
               signed_range_rate(previous_sector,current,0.2))
        heading=wrap(math.atan2(scene["target_y"]-y,scene["target_x"]-x)-yaw)
        observation=build_observation(current,rates,previous_action,distance,heading)
        nearest=min(range(index,len(route)),key=lambda i:math.hypot(x-route[i][0],y-route[i][1]))
        index=max(index,nearest);waypoint=route[min(len(route)-1,index+5)]
        error=wrap(math.atan2(waypoint[1]-y,waypoint[0]-x)-yaw)
        requested_linear=0.20 if abs(error)<math.radians(35) else 0.05
        requested_angular=float(np.clip(1.8*error,-0.8,0.8));front_ttc=float("inf");turn_side_ttc=float("inf")
        if previous_sector is not None and abs(requested_angular)<0.10:
            _,ttc=temporal_features(previous_sector,current,0.2);front_ttc=float(min(ttc[17],ttc[18]))
        linear,angular,reason=shield.apply(requested_linear,requested_angular,scan,front_ttc,turn_side_ttc,
                                           goal_distance=distance)
        executed=normalized_action(linear,angular)
        observations.append(observation);actions.append(executed);reasons[reason]=reasons.get(reason,0)+1
        previous_sector=current;previous_action=executed;minimum=min(minimum,float(valid_scan(scan).min()))
        command=Twist();command.linear.x=linear;command.angular.z=angular;env.pub_cmd_vel.publish(command)
        if minimum<0.20:outcome="collision";break
    env.pub_cmd_vel.publish(Twist())
    return {"outcome":outcome,"steps":step,"minimum_lidar":minimum,
            "observations":np.asarray(observations,dtype=np.float32),
            "actions":np.asarray(actions,dtype=np.float32),"shield_reasons":reasons}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--pairs-per-level",type=int,required=True);parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    if os.path.exists(args.output_dir):raise RuntimeError("refusing overwrite: "+args.output_dir)
    rospy.init_node("clean_start_bc_collection",anonymous=True);env=CleanWideStaticEnv()
    saved_obs=[];saved_actions=[];sample_levels=[];sample_pairs=[];sample_sides=[];episodes=[];audit=[]
    global_pair=0
    try:
        for level in (1,2,3,4,5):
            accepted=0;candidates=generate(level,MAX_CANDIDATES_PER_LEVEL,seed=args.seed+level)
            for candidate_index,base in enumerate(candidates,1):
                if accepted>=args.pairs_per_level:break
                pair_runs=[]
                for side,scene in ((0,base),(1,mirror(base))):
                    result=run(env,scene);pair_runs.append((side,scene,result))
                    print("CLEAN_BC level=%d candidate=%d side=%d outcome=%s steps=%d min=%.3f"%
                          (level,candidate_index,side,result["outcome"],result["steps"],result["minimum_lidar"]),flush=True)
                keep=all(r["outcome"]=="success" and r["minimum_lidar"]>=MIN_CLEARANCE for _,_,r in pair_runs)
                audit.append({"level":level,"candidate":candidate_index,"accepted":int(keep),
                              "base_outcome":pair_runs[0][2]["outcome"],"base_min":pair_runs[0][2]["minimum_lidar"],
                              "mirror_outcome":pair_runs[1][2]["outcome"],"mirror_min":pair_runs[1][2]["minimum_lidar"]})
                if not keep:continue
                for side,scene,result in pair_runs:
                    count=len(result["observations"]);saved_obs.append(result["observations"]);saved_actions.append(result["actions"])
                    sample_levels.extend([level]*count);sample_pairs.extend([global_pair]*count);sample_sides.extend([side]*count)
                    episodes.append({"pair_id":global_pair,"level":level,"side":"base" if side==0 else "mirror",
                                     "scene":scene,"steps":result["steps"],"minimum_lidar":result["minimum_lidar"],
                                     "shield_reasons":result["shield_reasons"]})
                global_pair+=1;accepted+=1
            if accepted!=args.pairs_per_level:raise RuntimeError("insufficient accepted pairs at level %d"%level)
        os.makedirs(args.output_dir)
        np.savez_compressed(os.path.join(args.output_dir,"clean_bc.npz"),observations=np.concatenate(saved_obs),
                            actions=np.concatenate(saved_actions),levels=np.asarray(sample_levels,dtype=np.int16),
                            pair_ids=np.asarray(sample_pairs,dtype=np.int16),mirror_sides=np.asarray(sample_sides,dtype=np.int8),
                            seed=np.asarray(args.seed),observation_contract=np.asarray("temporal78"))
        with open(os.path.join(args.output_dir,"episodes.json"),"w")as f:json.dump(episodes,f,indent=2,sort_keys=True)
        with open(os.path.join(args.output_dir,"candidate_audit.csv"),"w",newline="")as f:
            w=csv.DictWriter(f,fieldnames=audit[0].keys());w.writeheader();w.writerows(audit)
        summary={"seed":args.seed,"pairs":global_pair,"episodes":len(episodes),"samples":sum(len(x)for x in saved_obs),
                 "minimum_clearance":MIN_CLEARANCE,"observation_dim":78,"training":False,"old_bc_used":False,"old_weights_used":False}
        with open(os.path.join(args.output_dir,"summary.json"),"w")as f:json.dump(summary,f,indent=2,sort_keys=True)
        print("CLEAN_BC_COMPLETE "+json.dumps(summary,sort_keys=True),flush=True)
    finally:env.restore_center();env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO",flush=True)

if __name__=="__main__":main()
