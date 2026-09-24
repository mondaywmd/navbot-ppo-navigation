"""One-shot exact replay of the fresh seed8042721 level-1 mirror failure."""
import json
import os
import time

import rospy
from geometry_msgs.msg import Twist

from clean_wide_env import CleanWideStaticEnv
from curriculum_spec import generate, mirror
from horizon_audit import audit_episode


def main():
    root=os.path.dirname(os.path.abspath(__file__))
    path=os.path.join(root,"results","replay_seed8042721_level1_pair1_mirror_trace.json")
    if os.path.exists(path):raise RuntimeError("refusing overwrite: "+path)
    # horizon_audit called generate(level, ..., seed=audit_seed+level).
    scene=mirror(generate(1,2,seed=8042722)[0])
    rospy.init_node("replay_seed8042721_mirror_failure",anonymous=True)
    env=CleanWideStaticEnv()
    try:
        row=audit_episode(env,scene,record_trace=True)
        target=env.get_model_state("target","world")
        payload={"audit_seed":8042721,"generation_seed_argument":8042722,
                 "level":1,"pair_index":1,"side":"mirror","scene":scene,
                 "server_target_after_episode":[target.pose.position.x,target.pose.position.y],
                 "row":row,"training":False,"permanent_ood_accessed":False}
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"w") as f:json.dump(payload,f,indent=2,sort_keys=True)
        print("EXACT_REPLAY outcome=%s steps=%d min=%.3f final_distance=%.3f target=(%.3f,%.3f) HOLD_SECONDS=15"%
              (row["outcome"],row["steps"],row["minimum_lidar"],row["final_distance"],
               target.pose.position.x,target.pose.position.y),flush=True)
        env.pub_cmd_vel.publish(Twist());time.sleep(15)
    finally:
        env.restore_center();env.pub_cmd_vel.publish(Twist())
        print("CENTER_RESTORED_ZERO",flush=True)

if __name__=="__main__":main()
