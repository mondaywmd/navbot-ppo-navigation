# V6 failure-mode generalization preparation

> **Archived — failed final acceptance (2026-09-04).** This experiment is
> retained as a comparison/control result and must not be promoted as the
> current best model. Its permanent-OOD result was 43/60 successes, 12/60
> collisions, and 5/60 timeouts. Do not continue training from the frozen
> permanent OOD exam; see the project-level `PROJECT_STATUS.md`.

V6 initializes only from the final V5 Actor and samples scenes online; it does
not create a formal training dataset.  Seed `61129` is new and the two modes
are sampled 50/50 by alternating episode index.

- `positive_target_positive_yaw`: pillar x `[0.78,1.18] m`, pillar y
  `[-0.16,+0.16] m`, target y `[+0.04,+0.15] m`, yaw `[+4,+15] deg`.
- `far_pillar_negative_target_recovery`: pillar x `[1.10,1.25] m`, pillar y
  `[-0.16,+0.16] m`, target y `[-0.15,-0.04] m`, yaw `[-12,+12] deg`.

Every candidate must remain a legal blocked-route scene.  It is rejected if it
falls inside the joint L-infinity neighborhood of any permanent OOD task:
pillar-x `0.08 m`, pillar-y `0.06 m`, target-y `0.06 m`, and yaw `4 deg`.
In other words, all four coordinate differences must be within their radius to
trigger rejection.  These radii are deliberately conservative relative to the
robot/pillar footprint and sensor/reset variability, while retaining enough
space to cover both failure-mode families without reproducing a test question.

The prepared run name is `v6_v5init_failure_modes_10k_seed61129`.  The entry
refuses to overwrite an existing run, keeps the 16-D observation, residual
action limits, 300-step episode cap, and return/advantage normalization.
No training is started during preparation.
