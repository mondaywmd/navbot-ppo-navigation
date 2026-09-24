# V5 balanced demonstrations and progressive curriculum

> **Current best model (confirmed 2026-09-04).** The selected checkpoint is
> `runs/v5_v4init_pillar_y_curriculum_10k_seed29/checkpoints/actor_iter0016_step00010421.pth`.
> The permanent 12-task OOD exam remains frozen and evaluation-only; it must
> never be used for training, tuning, curriculum construction, or checkpoint
> selection. See the project-level `PROJECT_STATUS.md`.

This directory is isolated from V1--V4.  The active V5 route initializes
directly from the final V4 PPO Actor; behavior cloning and demonstration NPZ
generation were cancelled before any trainable dataset was written.

The Actor remains `ResidualActor(16, 2)` and receives only the established
LiDAR/relative-goal/action-history observation.  Gazebo world coordinates are
used only to reset named scenarios.

Training curriculum (for a later explicitly authorized PPO run):

- Level 1: independently seeded uniform pillar-y in `[-0.03,+0.03] m`.
- Level 2: independently seeded uniform pillar-y in `[-0.06,+0.06] m`.
- Robot yaw remains `0 deg` and target y remains `0 m` in both levels.

The six existing held-out cases at pillar `y=+/-0.10 m`, robot yaw `+/-5 deg`,
and target `y=+/-0.10 m` remain excluded from training.  The optional training
entry requires an explicit timestep budget and new run name, refuses overwrite,
and loads `actor_iter0008_step00005299.pth` exactly before PPO starts.
