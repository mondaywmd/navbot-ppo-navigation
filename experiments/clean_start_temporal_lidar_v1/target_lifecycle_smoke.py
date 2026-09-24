"""Move the persistent target once; intended to be invoked in separate processes."""
import argparse
import rospy

from clean_wide_env import CleanWideStaticEnv


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--x",type=float,required=True)
    parser.add_argument("--y",type=float,required=True);args=parser.parse_args()
    rospy.init_node("clean_target_lifecycle_smoke",anonymous=True)
    env=CleanWideStaticEnv();before=env.get_model_state("target","world")
    env._ensure_target(args.x,args.y);after=env.get_model_state("target","world")
    print("TARGET_MOVE before=(%.6f,%.6f) requested=(%.6f,%.6f) after=(%.6f,%.6f) error=(%.9f,%.9f)"%
          (before.pose.position.x,before.pose.position.y,args.x,args.y,
           after.pose.position.x,after.pose.position.y,
           after.pose.position.x-args.x,after.pose.position.y-args.y),flush=True)

if __name__=="__main__":main()
