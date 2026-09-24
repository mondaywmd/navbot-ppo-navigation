"""Visual replay of previously logged clean-Actor timeout episodes."""
import argparse, json, os
import rospy, torch
from geometry_msgs.msg import Twist

from clean_actor import CleanMirrorActor
from clean_wide_env import CleanWideStaticEnv
from evaluate_clean_actor import run_episode


def main():
    p=argparse.ArgumentParser();p.add_argument("--episodes",required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--output-dir",required=True)
    a=p.parse_args()
    if os.path.exists(a.output_dir):raise RuntimeError("refusing overwrite: "+a.output_dir)
    source=json.load(open(a.episodes));timeouts=[e for e in source if e["outcome"]=="timeout"]
    actor=CleanMirrorActor();actor.load_state_dict(torch.load(a.checkpoint,map_location="cpu"));actor.eval()
    rospy.init_node("clean_timeout_visual_replay",anonymous=True);env=CleanWideStaticEnv();results=[]
    try:
        for index,e in enumerate(timeouts,1):
            print("REPLAY_START %d/%d pair=%d pillars=%d side=%s"%
                  (index,len(timeouts),e["pair_id"],e["pillar_count"],e["side"]),flush=True)
            r=run_episode(env,actor,e["scene"])
            results.append({"source_pair_id":e["pair_id"],"pillar_count":e["pillar_count"],
                            "side":e["side"],"scene":e["scene"],**r})
            print("REPLAY_END outcome=%s steps=%d min=%.3f"%
                  (r["outcome"],r["steps"],r["minimum_lidar"]),flush=True)
        os.makedirs(a.output_dir)
        with open(os.path.join(a.output_dir,"replays.json"),"w") as f:json.dump(results,f,indent=2,sort_keys=True)
    finally:
        env.restore_center();env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO",flush=True)

if __name__=="__main__":main()
