# Project Status

## 2026-09-04 — V6 final permanent OOD acceptance

The third and final acceptance phase was completed independently. No training,
scene generation, seed change, parameter change, center-scene rerun, or
held-out perturbation rerun was performed.

- Actor: `experiments/single_pillar_residual_ppo_v6_failure_mode_generalization/runs/v6_v5init_failure_modes_10k_seed61129/checkpoints/actor_iter0017_step00010001.pth`
- Actor SHA-256: `f1765a3277290784e8386ca50d6f3d3078801aa13e3427b77e6eb33d53a05b5d`
- Permanent tasks: `experiments/single_pillar_residual_ppo_v5_adversarial_ood_eval/results/v5_final_seed29012_12tasks_5eps/tasks.json`
- Tasks SHA-256: `c8ee443abc408ab4b19c72e79a8857677f9884b1d9d08acddf3fbf898521cf43`
- Evaluation: deterministic Actor inference, 12 fixed tasks, 5 independent resets per task, 300-step cap, 60 episodes total, no PPO updates
- Results: `experiments/single_pillar_residual_ppo_v6_failure_mode_generalization/results/v6_final_permanent_ood_seed29012_12tasks_5eps/`
- Postcondition: center scene restored and zero velocity published

| Task | V5 S/C/T | V6 S/C/T | V6 minimum LiDAR (m) |
|---|---:|---:|---:|
| 01 | 5/0/0 | 5/0/0 | 0.292 |
| 02 | 1/0/4 | 2/0/3 | 0.259 |
| 03 | 5/0/0 | 4/1/0 | 0.183 |
| 04 | 5/0/0 | 5/0/0 | 0.239 |
| 05 | 3/0/2 | 3/0/2 | 0.383 |
| 06 | 1/4/0 | 2/3/0 | 0.187 |
| 07 | 1/4/0 | 0/5/0 | 0.173 |
| 08 | 3/2/0 | 2/3/0 | 0.181 |
| 09 | 4/1/0 | 5/0/0 | 0.228 |
| 10 | 5/0/0 | 5/0/0 | 0.280 |
| 11 | 5/0/0 | 5/0/0 | 0.230 |
| 12 | 5/0/0 | 5/0/0 | 0.277 |
| **Overall** | **43/11/6** | **43/12/5** | **0.173** |

V6 overall success remained 43/60 (71.67%). Collision increased from 11/60
(18.33%) to 12/60 (20.00%), while timeout decreased from 6/60 (10.00%) to
5/60 (8.33%). Therefore V6 did not lower the permanent-OOD collision rate and
did not trade collisions for more timeouts.

Across the targeted tasks 06–09, the aggregate remained 9 successes and 11
collisions in both V5 and V6. Tasks 06 and 09 improved by one episode each,
while tasks 07 and 08 regressed by one episode each. The targeted group did not
improve in aggregate; failures were redistributed between tasks.

## Model-selection decision after V6 acceptance

- **Current best model: V5.** Keep the final V5 Actor as the selected model:
  `experiments/single_pillar_residual_ppo_v5_balanced_curriculum/runs/v5_v4init_pillar_y_curriculum_10k_seed29/checkpoints/actor_iter0016_step00010421.pth`.
- **V6 status: archived failed-acceptance control experiment.** Preserve the
  complete V6 experiment, checkpoint, metadata, and results in place for
  comparison and reproducibility. Do not promote V6 as the current best model.
- **Permanent OOD exam remains frozen.** The existing 12 tasks in
  `experiments/single_pillar_residual_ppo_v5_adversarial_ood_eval/results/v5_final_seed29012_12tasks_5eps/tasks.json`
  must remain unchanged and must never be used for training, curriculum design,
  checkpoint selection, hyperparameter tuning, or scene generation.
- **Next phase: failure-mechanism diagnosis only.** Analyze the observed failure
  modes at finer resolution before proposing a new experiment. Do not extend
  PPO training or add training steps based on these acceptance results.

The permanent OOD set remains evaluation-only. Any future training scenes must
be created independently without reading, copying, perturbing, or otherwise
deriving examples from the frozen tasks.

## Next phase gate — mechanism diagnosis before any V7 or PPO

V6 showed that oversampling observed failure categories is not sufficient to
improve V5. Do not create V7 and do not start or extend PPO until the following
mechanism diagnosis is complete. Diagnostic scenes must be newly generated for
this purpose and must not be read, copied, perturbed, or derived from the frozen
permanent OOD tasks.

### Diagnosis A: are early-collision scenes controllably solvable?

Run a hand-written/rule-based avoidance controller on a new set of early-
collision diagnostic scenes while retaining the current safety action bounds.

- If the rule controller also cannot pass, treat the primary limitation as
  control authority, speed, gate behavior, or residual-action composition.
  Adjust the controller mechanism first and manually establish feasibility
  before any policy training.
- If the rule controller passes but V5 collides, treat the primary limitation
  as action selection by the learned policy rather than physical feasibility.

Record per episode: scene identifier and parameters, independent reset,
outcome, steps, minimum LiDAR clearance, commanded base action, residual
action, gate state, composed linear/angular velocity, robot pose, and heading
relative to the goal.

### Diagnosis B: where do safe timeouts stall?

For newly generated scenes where V5 avoids the obstacle but times out, record
the post-obstacle trajectory, position, heading, LiDAR state, residual action,
gate state, and final composed velocity. Classify each timeout as one of:

1. persistent circling or oscillation;
2. insufficient heading recovery toward the goal;
3. insufficient forward speed;
4. another mechanism supported by the trace.

### Decision after diagnosis

- **Rule controller succeeds, V5 collides:** collect diverse successful
  demonstrations only from new training-dedicated scenes; use a small amount
  of behavior cloning or DAgger warm-up, then limited PPO fine-tuning. Mix in
  independently defined center and small-perturbation training scenes to
  reduce forgetting.
- **Rule controller cannot pass:** change gate, base speed, or residual-action
  composition first; verify feasibility manually before training any policy.
- **Primary failure is safe timeout:** improve post-obstacle goal recovery in
  the controller and/or reward. Do not respond by merely strengthening obstacle
  penalties.

No permanent OOD result may be used to choose diagnostic coordinates, tune a
controller, construct demonstrations, design a curriculum, select a checkpoint,
or make intermediate training decisions. The frozen 12-task set is reserved
only for the final examination after all development decisions are complete.

## 2026-09-04 — V5 failure-mechanism diagnosis, seed 9042601

An independent diagnostic-only experiment was completed at
`experiments/v5_failure_mechanism_diagnosis_seed9042601/`. It used 12 new
fixed-seed scenes (six exact mirror pairs), three deterministic final-V5 Actor
resets per scene, and no training or demonstration collection. The permanent
OOD task file and permanent result directories were not read or used.

- V5 Actor total: 30 successes, 5 collisions, 1 timeout in 36 episodes.
- All five collisions were early (steps 15–19) and occurred on the positive
  side of near-pillar pairs 02 and 03. The corresponding negative mirror scenes
  succeeded in all six episodes, providing strong evidence of policy-side
  mirror asymmetry.
- The predeclared reactive control was tested three times on the automatically
  selected pair-03 positive representative. It avoided collision in all three
  trials but timed out in all three. This shows that another bounded action
  sequence can avoid the immediate crash, but it does not prove that the full
  task is solvable. Report only: `control_did_not_find_path`; do not claim
  physical infeasibility.
- One safe timeout occurred on far-pillar pair 05 positive. After passing the
  pillar it spent 79 steps within 0.35 m of the goal and ended at 0.2021 m,
  just outside the 0.20 m success radius. Compared with a same-scene success,
  it had larger mean/final goal-angle error, lower mean linear speed, more gate
  activation, and more angular sign changes. The predeclared classifier returns
  `evidence_insufficient`; the observed pattern is consistent with high-
  amplitude post-obstacle angular oscillation rather than a simple lack of
  forward progress.

Minimal next hypothesis: on additional independently generated diagnostic
scenes, test whether repeated post-obstacle gate activation combined with
near-saturated V5 angular residuals causes angular oscillation and near-misses
of the success circle. This does not authorize V7, PPO, BC/DAgger data
collection, or changes to existing policies, rewards, controllers, or scenes.

## 2026-09-04 — Post-obstacle recovery confirmation, seed 9042602

The independent confirmation experiment at
`experiments/v5_post_obstacle_recovery_confirmation_seed9042602/` completed 36
deterministic final-V5 episodes on 12 new fixed-seed recovery scenes. No
training, fine-tuning, demonstrations, model/controller/threshold changes, or
permanent OOD access occurred.

The predeclared classifications were 23 successes, 8 near-goal safe timeouts,
5 general timeouts, and zero collisions. Final-window medians for success
versus near-goal safe timeout were: gate switches 1 versus 7; gate-active
fraction 0.517 versus 0.604; mean absolute angular residual 0.932 versus 0.940;
commanded-angular sign switches 1 versus 7; monotonic distance-decrease
fraction 1.000 versus 0.706; distance-trend reversals 0 versus 1.5; and final
distance rebound above the window minimum 0.000 versus 0.022 m.

The minimum sample requirement and both systematic switching comparisons
passed. The predeclared repeated-nonconvergence condition did not pass: fewer
than two thirds of the near-goal timeouts simultaneously had at least four
distance-trend reversals and at least 0.02 m final rebound.

Formal verdict: **evidence insufficient**. Near-goal timeout is associated with
more gate and turn-command switching in this sample, but the full preregistered
rule does not support the oscillation hypothesis and no causal claim is made.
On this evidence, do not design or run a control ablation. V7, PPO,
fine-tuning, and demonstration collection remain prohibited.

## Diagnostic priority after recovery confirmation

The recovery confirmation was informative because it rejected the tempting
working explanation that gate oscillation alone is the main cause of timeout.
Gate/turn switching was associated with near-goal timeout, but the full
predeclared evidence rule did not pass. Do not prioritize further gate-focused
experiments on the current evidence.

The strongest reproduced mechanism clue remains the left/right asymmetry from
the seed-9042601 diagnosis: positive-side near-pillar scenes produced early
collisions while their exact negative mirrors succeeded. The next diagnostic
priority is therefore **V5 policy mirror consistency**, not additional PPO.

The next diagnostic, when explicitly authorized, should contain two read-only
tests using the unchanged final V5 Actor:

1. **Observation-level equivariance test.** Mirror the same legal 16-D Actor
   input and compare deterministic outputs. Under the established observation
   contract, the ten LiDAR samples must be permuted by the sensor's exact
   left/right index mapping; previous linear residual and goal distance remain
   unchanged; previous angular residual, `sin(goal_heading)`, and normalized
   signed heading error change sign; `cos(goal_heading)` remains unchanged.
   The expected output relation is approximately
   `linear_residual(mirror(obs)) = linear_residual(obs)` and
   `angular_residual(mirror(obs)) = -angular_residual(obs)`.
2. **Closed-loop paired-scene test.** In newly generated legal mirror pairs,
   compare the first 20 steps after independent resets. Check mirrored LiDAR
   features and goal geometry as well as linear-residual equality, angular-
   residual sign reversal, composed linear-speed equality, and composed
   angular-speed sign reversal. Keep stochastic exploration disabled.

This test must predeclare numerical tolerances and aggregation rules before
running. World coordinates may be logged for audit but must not enter the
Actor. Permanent OOD tasks and results remain inaccessible and cannot inform
scene selection or thresholds. No model, controller, reward, gate, action
composition, success threshold, or protected baseline may be modified.

If mirror inconsistency is reproduced, the supported target is policy
left/right generalization bias. This would identify a mechanism to address in
a later separately authorized design step; it does not itself authorize V7,
training, fine-tuning, augmentation, or demonstration collection.

## 2026-09-04 — V5 mirror/control audit and V7 preparation decision

Final-V5 offline inference on 486 fresh non-OOD observations showed large raw
Actor mirror errors: linear mean/median/P90/max
`0.4095/0.3499/0.5905/1.3645`, angular
`1.7183/1.7417/1.8375/1.9124`. All six fresh closed-loop mirror pairs violated
the early angular-action relation (`1.9392--1.9410` mean error). The strict
one-sided collision association did not reproduce, so collision causality
remains **evidence insufficient**.

The actual V5 control chain was then audited. Goal-Seeking, gate logic, and
final residual composition had exactly zero mirror error over 1000 seeded
samples, including gate on/off and clipping (tolerance `1e-6`). V5 publishes
that composed command directly, and observation indices 10--11 contain the
previous raw normalized Actor residual—not a post-compensated command. The
separate direct-goal heading-hold compensation is not imported by V5, and no
external `/cmd_vel` compensation publisher was active.

Therefore controller asymmetry is excluded as the explanation for the
measured Actor-output error. Raw Actor residual mapping asymmetry is the
strongest actionable evidence. Proceed only with V7 preparation: fresh paired
mirror scenes, an Actor mirror-consistency objective, unit tests, and a
zero-update smoke. Do not start PPO or modify V1--V6 or permanent OOD data.

V7 preparation is now complete at
`experiments/single_pillar_residual_ppo_v7_mirror_consistency/`. Seed 7042604
generated 24 base scenes plus their exact mirrors (48 scenes total). All five
preparation tests passed. The final-V5 zero-update smoke evaluated and
backpropagated the proposed consistency loss without creating an optimizer;
the loss was 1.255619 (linear 0.319327, angular 0.936292) and the Actor state
digest remained `4643c506d3651b8d2a685eccaef900e2bc2ec3dc26e0592a82a86657ade6af1f`.
No V7 PPO run, checkpoint, or parameter update exists. The mirror-loss weight
is deliberately unset pending a separately authorized training protocol.

## 2026-09-04 — V7 online-mirror 10k and permanent OOD acceptance

The isolated run `v7_v5init_online_mirror_10k_seed7042701` loaded final V5,
used online seed 7042701, and collected 10,226 steps for the authorized 10k
budget. Final checkpoint: `actor_iter0017_step00010226.pth`. Mirror loss fell
from 3.202129 to 0.194362; training episodes were 45 success, 50 collision, and
6 timeout.

The sampler logged 119 resets (59 complete pairs plus one base) but PPO logged
101 executed episodes. The inherited rollout resets at batch termination and
again at the next batch start, so some paired resets were never executed. Thus
strict online pair generation worked, but strict paired trajectory exposure was
not guaranteed. This is a V7 protocol defect.

The unchanged frozen permanent OOD exam was then run once: V7 scored 23 success
(38.33%), 29 collision (48.33%), and 8 timeout (13.33%), versus V5's 43/11/6.
V7 fails acceptance and is archived as a control experiment; V5 remains best.
No further training was started.

Immediate next hypothesis: anchor learning with complete paired successful
trajectories—especially left-side detours—preserve V5 actions on successful
states, weaken mirror regularization, and fix batching so both pair members are
executed. Longer term, transferable multi/dynamic-obstacle navigation requires
temporal LiDAR, local corridor/waypoint selection, safety-filtered tracking, and
a mixed curriculum from one static obstacle to multiple and moving obstacles.

## 2026-09-04 — V8 diverse paired-success BC dataset collected (no training)

The isolated V8 preparation at `experiments/v8_paired_success_anchor_seed8042702/`
now includes a formal dataset generated with fresh seed `8042705`. Online
scenes varied target x/y, one/two/three pillar x/y positions, and each pillar's
radius. Every scene had at least one direct-path-blocking pillar and passed
start/goal/wall/inter-pillar legality checks. Each base scene was immediately
followed by its exact left/right mirror.

Strict pair acceptance retained 24 pairs (8 per obstacle-count level), 48/48
successful episodes, and 5,617 samples. Both members had to succeed and keep
minimum LiDAR clearance at or above 0.25 m; six otherwise-successful candidate
pairs were rejected for insufficient clearance. The dataset contains 16-D
Actor observations and bounded expert actions, with finite-value checks
passing. There are 883 selectively marked V5 teacher-anchor samples in safe
post-obstacle recovery states. `pair_id` is level-local, so the required unique
training key is `(level, pair_id)`.

No BC, PPO, optimizer step, V8 checkpoint, or permanent-OOD access occurred.
Gazebo was restored to the center scene and zero velocity was published.
Direct `/gazebo/get_model_state` inspection confirmed that target coordinates
do change (observed `(3.1061, 0.6846)`); gzclient's center-target display is a
visualization artifact caused by the intermediate parent reset, not fixed
training geometry. Before any later training, remove that intermediate visual
update without changing the final server-side scene semantics.

The target visualization reset was then fixed only in V8 `wide_env.py`.
`reset_wide()` no longer enters the legacy reset that deleted/respawned target
and briefly wrote the center pose. The target is now persistent and receives a
single final `set_model_state` update (with spawn-only-if-absent support for a
fresh server); center restoration follows the same path. No-training smoke
seed `8042706` checked targets `(2.9576,-0.2009)`, `(2.6772,1.1603)`, and
`(3.2041,-0.8916)` and obtained exact gzserver coordinates each time. The
16-D observation contract remained intact, and center/zero were restored.

Follow-up visual inspection showed that pose-only updates of the original
`<static>true</static>` Target still did not move its gzclient visual despite
correct gzserver coordinates. V8 therefore now performs a one-time startup
migration to a non-static, gravity-disabled, single-link target with the same
three-disc appearance. It is not deleted between episodes; later resets use
only `set_model_state`. Visual smoke seed `8042707` passed three non-center
resets with maximum coordinate error about 3.1 micrometres, then an explicit
display check held the same target at `(2.8,+1.1)` and `(2.8,-1.1)` for ten
seconds each. Center and zero velocity were restored; no training occurred.

A subsequent visible no-training smoke used fresh seed `8042708` for one
two-pillar and one three-pillar scene. The two-pillar route collided at step
114 (minimum LiDAR 0.181 m); the three-pillar route succeeded in 108 steps
with 0.322 m minimum LiDAR. This is preserved as observed, not rerun or filtered
into an all-success result. It confirms that layout geometry, not obstacle
count alone, controls difficulty and that BC collection must continue to admit
only complete successful pairs meeting the clearance threshold. Center and
zero velocity were restored afterward.

V8 then ran a V5-initialized 200-epoch BC candidate (`v8_bc_seed8042709`) on a
pair-level 18-train/6-validation split. Offline validation total fell from
0.6556 to 0.2426, but the selective V5-anchor MSE rose to 0.1635. A predeclared
fresh non-OOD closed-loop acceptance at seed `8042710` rejected it: V5 scored
8 success / 3 collision / 1 timeout while the BC candidate scored 6 / 5 / 1.
The BC candidate especially produced early base-side collisions while mirrors
succeeded. It is retained only as a failed control and must not initialize PPO.
No permanent OOD was accessed and center/zero were restored.

A second V8 BC candidate then discarded V5 completely. It used random seed
`8042711` and an Actor forward pass that enforces exact left/right equivariance
by construction (1000-sample maximum mirror error exactly zero). Pair-held-out
validation MSE fell from 0.5578 to 0.06469. Nevertheless, fresh non-OOD
closed-loop acceptance at seed `8042712` failed: V5 scored 5 success / 5
collision / 2 timeout, while scratch symmetric BC scored 3 / 9 / 0. It is also
rejected and must not initialize PPO. This shows that exact symmetry and low
offline imitation error do not provide collision safety under closed-loop
distribution shift.

No V8 PPO has started. The next evidence-driven mechanism is a separate hard
safety shield using the full LiDAR scan plus command-dependent stopping/swept
turning clearance. The current gate is an authority modifier, not a forward
invariance guarantee. Any shield experiment must first derive its threshold
from robot geometry, control period, and braking behavior, then test whether
collisions become safe stops/timeouts on fresh non-OOD scenes. Do not alter V5
or access permanent OOD while developing it.

## 2026-09-04 — clean-start temporal LiDAR preparation

The independent `experiments/clean_start_temporal_lidar_v1/` line does not
inherit V5/V8 weights, anchors, or BC data. Live inspection established that
the actual `/scan` is only 10 front-facing rays over -90 to +90 degrees at
about 20-degree spacing and 5 Hz; it is not a dense scan later subsampled by
the Actor.

A runtime replacement attempt with a 360-ray URDF caused Gazebo to exit during
the spawn service call. That mechanism was rejected and its installer disabled.
The original launch was restarted and the robot, center pillar, target, 10-ray
scan, and zero velocity were verified/restored. No training or permanent OOD
access occurred. Dense LiDAR must be introduced through a separate startup
launch, never by replacing the sensor on a running server.

The horizon audit is predeclared: at 5 Hz, 300 steps is roughly 60 seconds, and
prior success at step 296 makes the cutoff plausibly tight. Fresh safe episodes
will be evaluated at 300 with diagnostic continuation to 450. The actual limit
will increase only if trajectories systematically converge in steps 301--450
rather than circle or remain stopped.

The safe startup-only dense launch subsequently passed with 360 full-circle
points at about one-degree spacing and 5 Hz. Straight hard-stop smoke retained
0.3087 m minimum clearance. Turning-side protection was then added: it inspects
the commanded turn's inner sector, slows inside 0.50 m, and stops inward motion
at 0.32 m. Fresh non-OOD seed `8042715` produced 12 mirrored one/two/three-
pillar audits; all 12 succeeded within 92--258 steps with zero collision or
timeout and at least 0.321 m clearance. No trajectory required steps 301--450,
so retain the 300-step limit.

The clean-start Actor observation contract is now specified as 78 values: 36
current sector minima, 36 signed inter-frame range rates, previous linear and
angular actions, and four goal-geometry values. Positive rate means approaching
and negative means receding. Exact mirroring reverses both sector blocks and
negates previous angular action plus signed goal-heading components. Unit tests
for dimension/range, approach/recede sign, action mirroring, and double mirror
all pass. No clean-start Actor/Critic, BC, PPO, or permanent OOD access exists.

## 2026-09-05 — clean-start front-corridor safety rule

The clean-start rule controller now follows a simpler contract: target/path
tracking is the default; LiDAR alters steering and speed only when the
candidate motion's 0.22 m half-width swept corridor intersects an obstacle.
A purely lateral obstacle does not reduce straight-line speed, but it prevents
turning into that side.  The normal desired clearance is 0.40 m, relaxed
continuously only within the final 0.50 m of goal distance to a floor of 0.25
m.  The 0.20 m value remains a severe LiDAR-proxy violation threshold, not a
Gazebo contact distance.  Inter-frame range trend remains available for
approach/recede decisions.  Reverse motion is available only when the selected
safe direction lies behind the robot.  The episode limit is currently 500.

An old `collect_clean_bc.py` process from the previously interrupted formal
collection was discovered still publishing `/cmd_vel`.  It was terminated;
no partial `clean_bc_seed9042702` dataset directory exists.  Center restoration
was repeated in the safe order (zero command before pose reset), and `/cmd_vel`
had no remaining publisher afterward.  No output from that accidental process
may be used.

All temporal-LiDAR, 78-D observation, base shield, and front-corridor barrier
unit tests pass.  Fresh non-OOD smoke seed `8042720` used one newly generated
mirrored pair at each of one, two, and three pillars (six episodes total).  All
6 succeeded in 99--181 steps with zero collision or timeout.  Minimum
full-circle LiDAR values were 0.208/0.220 m (one pillar), 0.301/0.223 m (two),
and 0.212/0.215 m (three); these close values occurred beside obstacles and do
not represent an occupied forward swept corridor.  This is encouraging but is
only a small no-learning smoke, not authorization to start BC/PPO.  Gazebo was
restored to the center scene and zero velocity at completion.  Permanent OOD
was not accessed.

The target-visual fix was then hardened across Python process restarts.  The
clean-start scripts now use `CleanWideStaticEnv`: an existing target is moved
and server-verified, never deleted/re-spawned merely because a new process
started.  Two separate processes moved the same target to `(2.8,+1.1)` and
then `(2.8,-1.1)` with zero server-coordinate error.  Reset/restore now brakes
before pose changes as well as afterward.  Final verification restored target
`(2,0)`, robot approximately `(0,0)`, near-zero twist, and no `/cmd_vel`
publisher.

A larger seed `8042721` acceptance was stopped after episode 2 because the
first single-pillar mirror reached 0.198 m LiDAR and met the predeclared severe
violation proxy.  Server geometry for that scene did not overlap the yellow
target: target `(2.380,+0.668)`, pillar centre `(1.228,-0.117)`, pillar radius
`0.450`, centre separation `1.393`, and pillar-surface-to-target-centre distance
`0.943` versus yellow radius `0.500`.  The apparent overlap was stale gzclient
target visualization from the old cross-process lifecycle bug.  Do not resume
broad acceptance or BC until the mirror turning/swept-corridor miss is
reproduced with a full trace.

The apparent collision/target-overlap anomaly was subsequently traced to a
second cross-process lifecycle bug: old `v8_pillar_*` models were tracked only
in the Python instance that spawned them, so interrupted/new processes left
three stale pillars in Gazebo.  The affected acceptance was invalidated and
produced no result directory.  `CleanWideStaticEnv` now enumerates live Gazebo
models before every reset/restore and removes every stale `v8_pillar_*` before
spawning the requested count.  It also verifies live target pose, pillar count,
pillar poses, and yellow-target non-overlap after every reset.  Static replay
of the questioned level-1 mirror contained exactly one pillar and retained a
0.443 m gap between the pillar surface and yellow target edge.

With those lifecycle checks active, fresh non-OOD acceptance seed `8042724`
completed 12/12 successes (two mirrored pairs at each of one, two, and three
pillars), zero collision and zero timeout.  Step counts were
`73/73, 63/63, 80/80, 111/111, 105/104, 92/92`; minimum full-circle LiDAR was
0.305 m.  The maximum clean success was 111 steps.  Per user direction, do not
yet freeze a final episode limit; eventually select it from broader stable
successes plus a 50-step margin.  This remains a no-learning acceptance only;
BC/PPO has not resumed and permanent OOD was not accessed.  Center/zero were
restored.

The startup-only approach then succeeded. `clean_start_dense.launch` loads an
expanded standalone URDF producing 360-degree/360-ray scans at approximately
one-degree spacing and 5 Hz; legacy files remain unchanged. The center scene
was restored with the dense sensor active. Temporal preprocessing uses 36
sector minima. Static two-frame noise produced at most 0.0413 m/s apparent
closing rate, motivating a 0.06 m/s deadband.

A 0.25 m/s straight-at-pillar hard-stop smoke intervened at step 6 and retained
0.3087 m minimum clearance (0.3183 m final front clearance), then reset/zeroed.
Fresh horizon audit seed `8042714` ran a shielded non-learning waypoint
controller in 12 mirrored one/two/three-pillar scenes: all 12 succeeded by step
68--102, with zero collision, timeout, or post-300 completion. Thus this sample
does not support increasing the 300-step horizon. Full-circle clearance reached
0.278 m during a three-pillar turn, so turning-side protection must be added
before new clean-start demonstrations are collected. No learning or permanent
OOD access occurred.

## 2026-09-05 — clean-start BC baseline and fresh closed-loop acceptance

The clean-start episode limit is now 200 steps.  This follows the observed
149-step maximum in the successful one-through-five-pillar expert acceptance,
plus approximately 50 steps of margin.  Both `horizon_audit.py` and
`collect_clean_bc.py` use this limit.

Fresh BC dataset seed `9051001` contains 30 independent mirror pairs (60
successful episodes, 5518 samples), balanced at six pairs for each pillar
count from one through five.  It uses only newly generated scenes; no old BC,
old weights, or permanent OOD material was used.  A randomly initialized
78-256-256-2 Actor with exact architectural left/right equivariance was trained
for 400 epochs under seed `9051002`.  Best pair-held-out validation MSE was
0.0147177 at epoch 346, versus 0.208397 before training.  The frozen BC-only
checkpoint is `runs/clean_bc_1to5_seed9051002/actor_bc_best.pth`.

Fresh non-OOD closed-loop seed `9051003` sampled the pillar count uniformly
from one through five for each of ten newly generated mirror pairs.  The Actor
received only the 78-D temporal-LiDAR/goal/action-history observation; obstacle
count, coordinates, and radii were not Actor inputs.  Results were 17/20
success, 0/20 collision, and 3/20 timeout at the 200-step limit.  One- and
two-pillar episodes were 12/12 successful; the three timeouts occurred in the
sampled four/five-pillar cases.  The random ten-pair draw happened to contain
no three-pillar pair.  Minimum full-circle LiDAR was 0.2111 m, so collision was
avoided but the desired 0.40 m clearance was not strictly maintained.  This is
the frozen BC-only baseline before any clean-start PPO.  Gazebo was restored
to the center scene and zero velocity.  Permanent OOD was not accessed.

## 2026-09-05 — clean-start 10k PPO and matched fresh evaluation

After zero-update and one-update smoke checks, independent clean-start PPO run
seed `9051006` initialized only from the frozen clean BC Actor and completed
10010 environment steps in 12 updates.  Its 98 exploratory training episodes
contained 92 successes, one collision, and five timeouts.  Every episode used
a newly seeded online scene whose pillar count was sampled uniformly from one
through five; the 200-step limit and deterministic LiDAR safety layer remained
unchanged.  The exact mirror Actor structure retained zero mirror error after
updates.  Permanent OOD was not accessed.

Matched fresh evaluation seed `9051007` used the identical 15 newly generated
mirror pairs for both frozen Actors with exploration disabled.  BC-only scored
27/30 success, 1/30 collision, and 2/30 timeout; PPO scored 29/30 success,
1/30 collision, and 0/30 timeout.  PPO converted both four-pillar BC timeouts
into successes at 160/161 steps.  The same five-pillar mirror scene reached the
0.20 m LiDAR violation threshold for both Actors (BC step 39, PPO step 44), so
PPO did not remove that safety failure.  Successful-episode median steps
improved from 93 to 91.  Successful-episode mean rose from 93.26 to 96.66 only
because PPO newly completed the difficult four-pillar pair at mean 160.5 steps;
on pillar counts successful under both Actors, PPO was equal or modestly faster.
Gazebo was restored to the center scene and zero velocity.  These are fresh
non-OOD results, not permanent-OOD certification.

A four-page Chinese portfolio/archive report was generated at
`docs/2026-09-05_clean_start_robot_learning_portfolio_summary.pdf`, with an
editable Markdown source beside it.  The report records the system design,
human-led diagnostic decisions, AI-assisted implementation disclosure,
quantitative BC/PPO results, limitations, job-skill mapping, artifact paths,
and SHA-256 checksums for the key dataset, checkpoints, and evaluation
summaries.
