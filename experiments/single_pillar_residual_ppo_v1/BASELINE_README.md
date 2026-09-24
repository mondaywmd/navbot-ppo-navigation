# Direct Goal V1 + Heading Compensation

Verified no-obstacle controller.

- Cruise speed: 0.25 m/s
- Brakes near target
- Confirms physical angular velocity is stable before driving
- Small heading-hold compensation only; it does not re-plan routes
- Uses LiDAR collision stop and 0.20 m arrival radius

Run once:
docker exec -it navbot-ppo bash /root/catkin_ws/src/project_ppo/baselines/direct_goal_v1_heading_comp_v1/restore_and_run_once.sh

## Residual action interface

`residual_action.py` is the first isolated residual-PPO component.  A policy
action in `[-1, 1]^2` adds at most `0.05 m/s` linear and `0.30 rad/s` angular
correction to a direct-to-goal command.  Zero residual exactly reproduces the
nominal command.  `SinglePillarEnv.step_residual` connects those commands to
Gazebo while preserving the 16-value observation contract, and
`residual_smoke_test.py` verifies a complete zero-residual short episode.  PPO
training is intentionally not started yet.

`untrained_ppo_check.py` exercises PPO's real rollout collector without any
optimizer step.  `train_residual_ppo.py` is the explicit stage-3 entry point;
it requires `--timesteps` and limits the current stage to at most 10,000.
