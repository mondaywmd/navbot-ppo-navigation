# V7 10k training and frozen permanent-OOD result

## Protocol

- Final V5 Actor/Critic initialization; online training seed `7042701`
- 10,000 requested steps; 10,226 collected through a complete episode
- PPO plus `0.10 * Actor mirror-consistency loss`
- Final checkpoint: `actor_iter0017_step00010226.pth`
- Frozen exam: original 12 tasks x 5 deterministic episodes
- No intermediate checkpoint selection or retraining from exam results

Mirror loss decreased from `3.202129` to `0.194362`. Training episodes were 45
success, 50 collision, and 6 timeout.

The sampler logged 119 resets (59 complete reset pairs and one final base),
while PPO logged 101 executed episodes. The inherited rollout resets after a
batch's last episode and again at the next batch start, so some paired resets
were not executed. Online pair generation worked, but strict paired trajectory
exposure was not guaranteed; this is a V7 protocol defect.

## Frozen permanent OOD comparison

| Task | V5 S/C/T | V7 S/C/T | V7 min LiDAR (m) |
|---:|---:|---:|---:|
| 01 | 5/0/0 | 5/0/0 | 0.215 |
| 02 | 1/0/4 | 2/3/0 | 0.192 |
| 03 | 5/0/0 | 1/3/1 | 0.193 |
| 04 | 5/0/0 | 2/2/1 | 0.171 |
| 05 | 3/0/2 | 5/0/0 | 0.297 |
| 06 | 1/4/0 | 0/5/0 | 0.181 |
| 07 | 1/4/0 | 0/5/0 | 0.186 |
| 08 | 3/2/0 | 0/5/0 | 0.182 |
| 09 | 4/1/0 | 1/2/2 | 0.173 |
| 10 | 5/0/0 | 3/2/0 | 0.177 |
| 11 | 5/0/0 | 0/1/4 | 0.194 |
| 12 | 5/0/0 | 4/1/0 | 0.188 |
| **Overall** | **43/11/6** | **23/29/8** | **0.171** |

V7 success fell from 71.67% to 38.33%; collision rose from 18.33% to 48.33%;
timeout rose from 10.00% to 13.33%. V7 fails acceptance and does not replace
V5. Output symmetry improved, but the objective supplied no correct safe-action
anchor and likely disrupted V5's useful one-sided behavior.

## Next hypotheses (not executed)

Near term: use complete paired successful trajectories, especially left-side
detours, preserve V5 behavior on its successful states, weaken the mirror
penalty, and guarantee that both members of each pair are actually executed.
Reward safe clearance together with post-obstacle goal recovery.

Long term: move toward temporal LiDAR, local safe-corridor/waypoint selection,
and a safety-filtered tracking controller. Train a mixed curriculum from one
static obstacle through multiple static obstacles to moving obstacles while
retaining earlier levels. Single-frame LiDAR cannot reliably infer obstacle
velocity, so dynamic-obstacle transfer requires temporal state. No further
training was started.
