# V7 mirror-consistency experiment

This isolated directory contains the completed V7 experiment without modifying
V1--V6. It was motivated by final-V5 raw Actor non-equivariance. The
control-chain audit found zero mirror error in Goal-Seeking, gate logic, and
final command composition; it did not establish collision causality.

Prepared components:

- `mirror.py`: exact 16-D observation and 2-D residual mirror transforms.
- `paired_scenes.py`: 24 fresh fixed-seed base scenes and their exact mirrors.
- `consistency.py`: differentiable Actor consistency objective.
- `test_v7_preparation.py`: involution, scene-pair, objective, and gradient tests.
- `zero_update_smoke.py`: loads final V5, performs forward/backward only, and
  verifies every parameter is bitwise unchanged.

The frozen permanent OOD exam is not read, imported, or referenced by these
programs. Seed `7042604` is new and scene bounds are declared locally. World
coordinates are reset metadata only and never Actor inputs.

Executed training objective:

`L_actor_total = L_PPO_actor + lambda_mirror * L_mirror`

where `L_mirror = mean((v(M(o))-v(o))^2 + (w(M(o))+w(o))^2)`, with coefficient
`0.10`. Online seed `7042701` generated base/mirror resets without reading the
fixed 48-scene preparation manifest. See `FINAL_REPORT.md` for the failed
acceptance result and the paired-exposure batching defect.

Preparation commands (safe; no parameter update):

```bash
python3 paired_scenes.py --output artifacts/paired_scenes_seed7042604.json
python3 -m unittest -v test_v7_preparation.py
python3 zero_update_smoke.py --checkpoint /path/to/final_v5_actor.pth
```
