"""
Evaluation utilities for a trained navigation policy.

Each test episode is recorded as JSON so successful and failed
trajectories can be inspected after the run.
"""
import json
import os
from datetime import datetime


def _log_summary(ep_len, ep_ret, ep_num, outcome, record_path):
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Outcome: {outcome}", flush=True)
    print(f"Episodic Length: {round(ep_len, 2)}", flush=True)
    print(f"Episodic Return: {round(ep_ret, 2)}", flush=True)
    print(f"Replay record: {record_path}", flush=True)
    print("------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy, env, render=False, max_steps=100, record_dir=None):
    """Run evaluation episodes and yield metrics plus a trajectory record."""
    episode_num = 0

    while True:
        obs = env.reset()
        done = False
        arrive = False
        past_action = [0.0, 0.0]
        ep_ret = 0.0
        trajectory = []

        goal = {
            "x": float(env.goal_position.position.x),
            "y": float(env.goal_position.position.y),
        }
        start = {
            "x": float(env.position.x),
            "y": float(env.position.y),
        }

        for t in range(1, max_steps + 1):
            if render:
                env.render()

            action = policy(obs).detach().cpu().numpy().squeeze(0)
            obs, rew, done, arrive = env.step(action, past_action)
            past_action = action
            ep_ret += float(rew)

            trajectory.append({
                "step": t,
                "robot_x": float(env.position.x),
                "robot_y": float(env.position.y),
                "linear_action": float(action[0]),
                "angular_action": float(action[1]),
                "reward": float(rew),
            })

            done = bool(done or arrive)
            if done:
                break

        ep_len = len(trajectory)
        if arrive:
            outcome = "SUCCESS"
        elif done:
            outcome = "COLLISION"
        else:
            outcome = "TIMEOUT"

        if outcome == "TIMEOUT":
            timeout_penalty = -150.0
            ep_ret += timeout_penalty
            trajectory[-1]["reward"] += timeout_penalty

        record_path = ""
        if record_dir:
            record_path = os.path.join(
                record_dir, f"episode_{episode_num:03d}_{outcome.lower()}.json"
            )
            with open(record_path, "w") as f:
                json.dump({
                    "episode": episode_num,
                    "outcome": outcome,
                    "length": ep_len,
                    "return": ep_ret,
                    "start": start,
                    "goal": goal,
                    "trajectory": trajectory,
                }, f, indent=2)

        yield ep_len, ep_ret, outcome, record_path
        episode_num += 1


def eval_policy(policy, env, render=False, max_episodes=10, max_steps=100):
    """Evaluate a policy and save replay records for every episode."""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = os.path.join(project_dir, "evaluation_replays", stamp)
    os.makedirs(record_dir, exist_ok=True)
    print(f"[eval] Replay records: {record_dir}", flush=True)

    for ep_num, (ep_len, ep_ret, outcome, record_path) in enumerate(
        rollout(policy, env, render, max_steps, record_dir)
    ):
        _log_summary(ep_len, ep_ret, ep_num, outcome, record_path)
        if ep_num + 1 >= max_episodes:
            break
