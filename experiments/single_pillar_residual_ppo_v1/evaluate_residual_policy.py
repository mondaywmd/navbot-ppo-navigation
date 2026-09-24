"""Deterministic fixed-scene evaluation for a trained residual actor."""

import argparse
import csv
import os

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist

from residual_networks import ResidualActor
from single_pillar_env import SinglePillarEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    return args


def main():
    args = parse_args()
    rospy.init_node("single_pillar_residual_deterministic_eval", anonymous=True)
    env = SinglePillarEnv(False)
    actor = ResidualActor(16, 2)
    actor.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    actor.eval()

    rows = []
    for episode in range(1, args.episodes + 1):
        observation = env.reset()
        past_residual = np.zeros(2, dtype=np.float32)
        total_reward = 0.0
        outcome = "timeout"

        for step in range(1, args.max_steps + 1):
            with torch.inference_mode():
                # Deterministic policy mean only: no distribution and no sample().
                residual = actor(observation).squeeze(0).cpu().numpy()
            residual = np.clip(residual, -1.0, 1.0).astype(np.float32)
            observation, reward, done, arrive, command = env.step_residual(
                residual, past_residual
            )
            if observation.shape != (16,) or not np.all(np.isfinite(observation)):
                raise AssertionError("invalid observation during evaluation")
            if not np.isfinite(reward) or not np.all(np.isfinite(command)):
                raise AssertionError("non-finite reward or command during evaluation")
            total_reward += float(reward)
            past_residual = residual
            if done or arrive:
                outcome = "success" if arrive else "collision"
                break

        env.pub_cmd_vel.publish(Twist())
        rows.append((episode, outcome, step, total_reward))
        print(
            "EVAL episode=%d outcome=%s steps=%d return=%.3f"
            % (episode, outcome, step, total_reward),
            flush=True,
        )

    successes = sum(outcome == "success" for _, outcome, _, _ in rows)
    collisions = sum(outcome == "collision" for _, outcome, _, _ in rows)
    mean_steps = sum(step for _, _, step, _ in rows) / float(len(rows))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["episode", "outcome", "steps", "return"])
        writer.writerows(rows)
    print(
        "EVAL SUMMARY: episodes=%d successes=%d collisions=%d timeouts=%d "
        "success_rate=%.1f%% collision_rate=%.1f%% mean_steps=%.2f"
        % (
            len(rows), successes, collisions, len(rows) - successes - collisions,
            100.0 * successes / len(rows), 100.0 * collisions / len(rows), mean_steps,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            rospy.Publisher("/cmd_vel", Twist, queue_size=1).publish(Twist())
        except Exception:
            pass
