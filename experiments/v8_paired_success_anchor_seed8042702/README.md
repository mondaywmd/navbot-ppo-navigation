# V8 paired-success anchor preparation

This isolated preparation validates complete left/right mirrored expert
trajectories before any BC or PPO is allowed. It never reads permanent OOD
tasks. Seed `8042702` generates six fresh scene pairs. Both members of a pair
must execute twice, succeed four times total, and satisfy the declared minimum
clearance before the pair can be retained.

The expert may use reset metadata and robot pose to construct labels. Saved
Actor inputs remain the established 16-D observation only; world coordinates
are audit fields and are never appended to observations.

## Formal paired-success BC collection

Seed `8042705` produced the frozen preparation dataset at
`datasets/paired_success_bc_seed8042705/`. It contains 24 accepted mirror
pairs (8 each with one, two, and three independently sized pillars), 48
successful episodes, and 5,617 16-D observation/action samples. A pair was
retained only when both its base and mirror route succeeded and both complete
routes had minimum LiDAR clearance of at least 0.25 m. Rejected candidates
remain in `candidate_audit.csv`; they are not present in the NPZ dataset.

`pair_ids` are local to a curriculum level. The unique pairing key is
`(level, pair_id)`, which yields 24 distinct pairs; consumers must not use
`pair_id` alone. The small left/right sample-count difference (2810 versus
2807) comes only from episode-length differences; episode counts are exactly
24 per side. `keep_v5_mask` selects 883 teacher-anchor samples using the
predeclared safe post-obstacle recovery rule. No optimizer or training was
started during collection.

The target model is moved with `/gazebo/set_model_state`; it is not deleted or
respawned. Direct server inspection during collection confirmed a non-center
target at `(3.1061, 0.6846)` while gzclient still displayed the center target.
That discrepancy is a visualization/reset-intermediate-state issue, not a
scene-generation or data-collection failure.

The V8 reset path was subsequently patched to bypass the legacy single-pillar
reset, which deleted/respawned the target and wrote a center pose before the
final wide-scene pose. `reset_wide()` now keeps the target model persistent and
moves it exactly once to the requested final pose; it only spawns the model if
it is absent on a fresh Gazebo server. Center restoration uses the same
persistent-target path. The no-training smoke at
`results/target_reset_smoke_seed8042706.json` passed three non-center resets
with exact server-side coordinate agreement and restored center/zero velocity.

Server-side pose agreement alone did not fix gzclient: the original Target SDF
was static, so Gazebo Classic retained its old visual pose. V8 now performs one
startup-only migration to a non-static, gravity-disabled, single-link target
with the same three colored discs. Subsequent episode resets are pose-only.
`results/target_visual_smoke_seed8042707.json` passed three resets (10-micrometre
tolerance; observed maximum error 3.1 micrometres). A separate visual hold moved
the same model between `(2.8, +1.1)` and `(2.8, -1.1)` before restoring center.
