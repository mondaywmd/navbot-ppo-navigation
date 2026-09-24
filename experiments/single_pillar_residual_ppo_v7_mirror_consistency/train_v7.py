"""One isolated 10k V7 run initialized from final V5; refuses overwrite."""
import os
import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from net_critic import NetCritic
from online_paired_env import TRAIN_SEED, V7OnlinePairedEnv
from ppo_mirror import PPO
from residual_networks import ResidualActor
from scenario_env import CENTER_SCENE

ROOT = os.path.dirname(os.path.abspath(__file__))
V5_RUN = os.path.abspath(os.path.join(ROOT, "..", "single_pillar_residual_ppo_v5_balanced_curriculum", "runs", "v5_v4init_pillar_y_curriculum_10k_seed29"))
V5_ACTOR = os.path.join(V5_RUN, "checkpoints", "actor_iter0016_step00010421.pth")
V5_CRITIC = os.path.join(V5_RUN, "checkpoints", "critic_iter0016_step00010421.pth")
RUN_NAME = "v7_v5init_online_mirror_10k_seed7042701"


def main():
    run_dir = os.path.join(ROOT, "runs", RUN_NAME)
    if os.path.exists(run_dir):
        raise RuntimeError("refusing to overwrite run: " + run_dir)
    rospy.init_node("v7_online_paired_mirror_10k", anonymous=True)
    env = V7OnlinePairedEnv(os.path.join(run_dir, "online_scenes.jsonl"), TRAIN_SEED)
    agent = PPO(policy_class=ResidualActor, value_func=NetCritic, env=env,
                state_dim=16, action_dim=2, output_dir=os.path.join(ROOT, "runs"),
                method_name=RUN_NAME, native_residual=True, normalize_returns=True,
                timesteps_per_batch=500, max_timesteps_per_episode=300,
                n_updates_per_iteration=5, save_freq=1, seed=TRAIN_SEED,
                mirror_loss_coef=0.10)
    agent.actor.load_state_dict(torch.load(V5_ACTOR, map_location="cpu"))
    agent.critic.load_state_dict(torch.load(V5_CRITIC, map_location="cpu"))
    print("[V7] loaded final V5 Actor/Critic; online paired seed=%d" % TRAIN_SEED, flush=True)
    try:
        agent.learn(10000, np.zeros(2, dtype=np.float32))
    finally:
        env.scenario = {"pillar_x": 1.0, **CENTER_SCENE}
        env.reset()
        env.pub_cmd_vel.publish(Twist())
        print("[V7] center restored and zero velocity published", flush=True)


if __name__ == "__main__":
    main()
