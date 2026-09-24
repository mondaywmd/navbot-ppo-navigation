"""Run one rollout through the real PPO collection path without training."""

import tempfile

import numpy as np
import rospy

from net_critic import NetCritic
from ppo import PPO
from residual_action import MAX_ANGULAR_SPEED, MAX_LINEAR_SPEED
from residual_networks import ResidualActor
from single_pillar_env import SinglePillarEnv


ROLLOUT_STEPS = 20


def main():
    rospy.init_node("single_pillar_untrained_ppo_check", anonymous=True)
    env = SinglePillarEnv(False)

    with tempfile.TemporaryDirectory(prefix="single_pillar_untrained_ppo_") as output_dir:
        agent = PPO(
            policy_class=ResidualActor,
            value_func=NetCritic,
            env=env,
            state_dim=16,
            action_dim=2,
            output_dir=output_dir,
            method_name="untrained_check",
            native_residual=True,
            timesteps_per_batch=ROLLOUT_STEPS,
            max_timesteps_per_episode=ROLLOUT_STEPS,
            seed=7,
        )
        rollout = agent.rollout(past_action=np.zeros(2, dtype=np.float32), t_so_far=0)

    observations, residuals, _log_probs, returns, lengths = rollout[:5]
    observations = observations.detach().cpu().numpy()
    residuals = residuals.detach().cpu().numpy()
    returns = returns.detach().cpu().numpy()

    if observations.ndim != 2 or observations.shape[1] != 16:
        raise AssertionError("PPO did not receive N x 16 observations: %r" % (observations.shape,))
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(returns)):
        raise AssertionError("non-finite observation or reward-to-go")
    if residuals.shape[1:] != (2,) or np.max(np.abs(residuals)) > 1.0:
        raise AssertionError("PPO residual escaped [-1, 1]: %r" % (residuals,))
    if len(env.residual_command_audit) != len(residuals):
        raise AssertionError("not every PPO residual reached the environment")

    commands = np.asarray([entry[1] for entry in env.residual_command_audit])
    if np.min(commands[:, 0]) < 0.0 or np.max(commands[:, 0]) > MAX_LINEAR_SPEED:
        raise AssertionError("unsafe physical linear command")
    if np.max(np.abs(commands[:, 1])) > MAX_ANGULAR_SPEED:
        raise AssertionError("unsafe physical angular command")

    episode_count = len(lengths)
    if episode_count < 1 or sum(lengths) != len(residuals):
        raise AssertionError("PPO termination accounting is inconsistent")
    print(
        "UNTRAINED PPO PASS: transitions=%d episodes=%d obs_dim=16 "
        "residual=[%.3f, %.3f] linear=[%.3f, %.3f] angular=[%.3f, %.3f]"
        % (
            len(residuals), episode_count,
            float(np.min(residuals)), float(np.max(residuals)),
            float(np.min(commands[:, 0])), float(np.max(commands[:, 0])),
            float(np.min(commands[:, 1])), float(np.max(commands[:, 1])),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
