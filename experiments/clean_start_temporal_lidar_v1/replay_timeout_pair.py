"""Replay the exact rejected formal-collection pair after trend-aware release."""
import json
import os
import argparse
import rospy
from geometry_msgs.msg import Twist
from curriculum_spec import generate, mirror
from horizon_audit import audit_episode
from clean_wide_env import CleanWideStaticEnv

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--side",choices=("base","both"),default="both")
    parser.add_argument("--tag",default="tangent_escape_v2_hard_envelope")
    parser.add_argument("--diagnostic-limit",type=int,default=None)
    args=parser.parse_args()
    root=os.path.dirname(os.path.abspath(__file__));path=os.path.join(root,"results","replay_seed9042702_level2_candidate2_%s_%s.json"%(args.tag,args.side))
    if os.path.exists(path):raise RuntimeError("refusing overwrite: "+path)
    rospy.init_node("clean_start_timeout_replay",anonymous=True);env=CleanWideStaticEnv();rows=[]
    base=generate(2,2,seed=9042704)[1]
    try:
        scenes=(base,) if args.side=="base" else (base,mirror(base))
        for scene in scenes:
            row=audit_episode(env,scene,record_trace=True,diagnostic_limit=args.diagnostic_limit);rows.append(row)
            print("TIMEOUT_REPLAY side=%s outcome=%s steps=%d min=%.3f reasons=%s"%
                  (row["side"],row["outcome"],row["steps"],row["minimum_lidar"],row["shield_reasons"]),flush=True)
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"w")as f:json.dump({"source_seed":9042702,"level":2,"candidate":2,
            "prior_outcomes":{"base":"timeout300","mirror":"timeout300"},"rows":rows,
            "training":False},f,indent=2,sort_keys=True)
    finally:env.restore_center();env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO",flush=True)

if __name__=="__main__":main()
