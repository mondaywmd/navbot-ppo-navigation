# NavBot PPO Navigation

[Website](https://mondaywmd.github.io/monday-robotics-universe/) · [Project & experiments](https://mondaywmd.github.io/monday-robotics-universe/navbot.html) · [About Monday](https://mondaywmd.github.io/monday-robotics-universe/about.html)

An AI-assisted robot-learning project for differential-drive navigation in ROS Noetic and Gazebo Classic. The repository records the full path from basic motion calibration and procedural obstacle generation to behavior cloning, PPO, frozen evaluation, failure diagnosis, and a clean-start temporal-LiDAR policy.

The goal is not only to train a controller, but to build a reproducible experimental loop:

`procedural scene → LiDAR observation → policy → safety layer → robot action → replay and evaluation`

## Final clean-start result

The latest clean-start experiment does not inherit the earlier V5 policy. It uses freshly generated mirrored training scenes, temporal LiDAR features, a structurally mirror-equivariant Actor, behavior cloning, PPO, and a deterministic safety layer.

| Evaluation | Success | Collision | Timeout |
|---|---:|---:|---:|
| BC-only, 30 matched fresh episodes | 27/30 | 1/30 | 2/30 |
| PPO, same 30 episodes | **29/30** | 1/30 | **0/30** |

PPO recovered two difficult four-obstacle timeouts. Both models still failed the same mirrored five-obstacle case, so the result is an improvement—not a claim of solved navigation or sim-to-real readiness.

## What the robot observes

The clean-start Actor receives a 78-dimensional observation:

- 36 LiDAR sector minimum distances;
- 36 signed changes from the previous LiDAR frame;
- 2 values for the previously executed action;
- 4 values describing goal distance and relative direction.

World coordinates, obstacle count, and obstacle radius are used for scene generation and evaluation only; they are not given to the Actor.

## Key engineering decisions

- Calibrated motion before training; speeds above roughly 3 m/s produced unacceptable simulated drift, so later experiments used a conservative 2.5 m/s ceiling.
- Generated legal 1–5 obstacle scenes with fixed seeds, mirrored pairs, clearance checks, and route-feasibility tests.
- Replaced single-frame LiDAR with temporal distance trends.
- Enforced left/right symmetry in the Actor architecture instead of relying only on data augmentation.
- Separated training monitoring from deterministic, exploration-off acceptance tests.
- Froze evaluation sets and refused result-directory overwrites to reduce test leakage.
- Preserved failed V1–V8 branches and diagnosis reports because negative results were part of the learning process.

## Repository map

| Path | Purpose |
|---|---|
| `experiments/clean_start_temporal_lidar_v1/` | Latest clean-start pipeline, datasets, training and matched evaluation |
| `experiments/single_pillar_residual_ppo_v1...v8/` | Earlier residual-PPO iterations and controlled comparisons |
| `src/` | Original ROS/Gazebo PPO environment, networks, controllers and benchmarks |
| `launch/` | ROS launch files |
| `baselines/` | Protected hand-written controller baselines |
| `runs/` | Lightweight configs and text records from historical runs |
| `evaluation_replays/` | Episode traces used for failure analysis |
| `PROJECT_STATUS.md` | Chronological technical decision log |
| `docs/2026-09-05_clean_start_robot_learning_portfolio_summary.md` | Detailed Chinese project summary |

## Environment

The original work was run in a Linux/WSL workflow with ROS Noetic, Gazebo Classic and Python 3.8-era ML dependencies. See `src/requirements.txt` for the historical package snapshot. It is intentionally preserved as an experiment record and may require version adaptation on a current machine.

Typical ROS workspace placement:

```text
catkin_ws/
└── src/
    └── project_ppo/
```

Then build the workspace with Catkin and source the generated setup file before using the launch files or scripts. Exact experiment commands and constraints are documented inside each experiment directory and in `PROJECT_STATUS.md`.

## Public-repository scope

This repository publishes source code, experiment definitions, lightweight metrics, evaluation traces and documentation. Generated caches, TensorBoard event binaries, videos, ROS bags and repeated model checkpoints are excluded from Git. Paths and hashes for important historical checkpoints remain in the reports; the original research archive is retained locally.

## Limitations

- Simulation only; no sim-to-real evidence.
- Gazebo Classic and ROS Noetic rather than ROS 2 / modern Gazebo.
- Primarily static cylindrical obstacles.
- One difficult five-obstacle mirrored case remains unsafe.
- Some early experiment branches contain exploratory code and are retained for historical comparison, not as a polished API.
- Development was AI-assisted. Experimental direction, safety constraints, observations and acceptance decisions were defined and reviewed by the project owner.

## Documentation

- [Chronological project status](PROJECT_STATUS.md)
- [Clean-start portfolio summary](docs/2026-09-05_clean_start_robot_learning_portfolio_summary.md)
- [Robotics Universe case study](https://mondaywmd.github.io/monday-robotics-universe/navbot.html)

## License

No open-source license has been selected yet. The code is publicly viewable, but reuse rights are not granted until a license is added.
