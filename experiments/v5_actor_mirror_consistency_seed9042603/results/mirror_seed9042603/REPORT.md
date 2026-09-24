# Final V5 Actor mirror-consistency diagnosis

## Transform

LiDAR 0..9 -> 9..0; index 11, 13, and 15 negate; indices 10, 12, and 14 remain unchanged. Unit tests and double-mirror checks passed.

## Offline errors

| Group | N | Linear mean/median/P90/max | Angular mean/median/P90/max |
|---|---:|---|---|
| all | 486 | 0.4095 / 0.3499 / 0.5905 / 1.3645 | 1.7183 / 1.7417 / 1.8375 / 1.9124 |
| early_collision | 86 | 0.4763 / 0.3329 / 1.1165 / 1.3645 | 1.6770 / 1.8184 / 1.9118 / 1.9124 |
| success | 200 | 0.4020 / 0.3454 / 0.5739 / 1.1104 | 1.7271 / 1.7434 / 1.8335 / 1.9097 |
| timeout | 200 | 0.3883 / 0.3737 / 0.5656 / 0.8559 | 1.7272 / 1.7159 / 1.8243 / 1.8950 |

## Gazebo mirror pairs

| Pair | Positive outcomes | Negative outcomes | Paired steps | Mean linear error | Mean angular error | Violated | One-sided early collision |
|---:|---|---|---:|---:|---:|---|---|
| 01 | {'success': 3} | {'success': 3} | 60 | 0.0109 | 1.9394 | True | False |
| 02 | {'early_collision': 2, 'success': 1} | {'early_collision': 3} | 56 | 0.0097 | 1.9397 | True | False |
| 03 | {'success': 2, 'early_collision': 1} | {'success': 1, 'early_collision': 2} | 58 | 0.0065 | 1.9397 | True | False |
| 04 | {'early_collision': 2, 'success': 1} | {'success': 2, 'early_collision': 1} | 60 | 0.0054 | 1.9395 | True | False |
| 05 | {'early_collision': 3} | {'success': 2, 'early_collision': 1} | 46 | 0.0088 | 1.9410 | True | False |
| 06 | {'success': 3} | {'success': 3} | 60 | 0.0108 | 1.9392 | True | False |

Checks: `{'offline_error_clear': True, 'closed_loop_systematic_violation': True, 'one_sided_early_collision_association': False}`.

Formal verdict: **evidence_insufficient**.

No collision causality is claimed.

Minimal next hypothesis: Cannot decide; next inspect coordinate and action-composition symmetry with a minimal control audit.

## Control and action-history audit

The V5 data path was traced from its actual imports. The Actor emits normalized
residual `(r_v, r_w)`. Goal-Seeking computes a nominal command from goal
distance and signed heading error. With the gate inactive, that nominal command
is published unchanged. With the gate active, nominal linear speed is capped at
0.09 m/s, nominal angular speed is replaced by zero, and
`(0.05*r_v, 0.30*r_w)` is added before clipping to
`[0, 0.25] x [-0.60, 0.60]`.

The V5 environment publishes this composed command directly to `/cmd_vel`.
Observation indices 10--11 receive the previous **raw normalized Actor
residual**, not the nominal or composed physical command. The independent
direct-goal benchmark's post heading-hold compensation is not imported or
called by V5, and no external compensation publisher was active at runtime.

A seed-9042603 audit of 1000 sampled distance/heading/residual tuples covered
gate on/off and clipping. Under `heading -> -heading` and
`(r_v,r_w) -> (r_v,-r_w)`, every maximum nominal and composed-command mirror
error was exactly `0.0` (tolerance `1e-6`). See
`control_symmetry_audit.json`.

Thus Goal-Seeking, gate logic, final action composition, and post-command
history feedback are excluded as explanations for the measured Actor mirror
error. Actor raw-residual non-equivariance remains the strongest actionable
evidence. The collision association still fails the preregistered strict
one-sided criterion, so no collision-causality claim is made.

Next action: prepare, but do not train, V7 with fresh paired mirror training
scenes and an Actor mirror-consistency constraint.
