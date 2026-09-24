# Diagnostic conclusion

This conclusion uses only the new seed-`9042601` diagnostic run.

## Early collision

V5 produced five early collisions, all on the positive side of near-pillar
mirror pairs 02 and 03. Their negative mirrors completed all six episodes.
The selected pair-03 positive representative collided at steps 15–17 under
V5. The fixed reactive controller produced zero collisions but also zero
successes in three trials; all three timed out with at least 0.395 m LiDAR
clearance.

Therefore there is evidence that a different bounded action sequence can turn
the immediate collision into a safe trajectory, but there is no positive proof
yet of a complete feasible path to the goal. The correct formal result is:
**the reactive control did not find a complete path; physical infeasibility
cannot be claimed.**

## Safe timeout

The only safe Actor timeout was pair-05 far positive, episode 2. A same-scene
success was available for comparison.

| Metric after passing pillar | Success ep. 1 | Timeout ep. 2 |
|---|---:|---:|
| Samples | 63 | 122 |
| Mean absolute goal angle | 40.211 deg | 72.096 deg |
| Final absolute goal angle | 17.960 deg | 86.140 deg |
| Mean linear speed | 0.107 m/s | 0.074 m/s |
| Mean absolute angular speed | 0.393 rad/s | 0.402 rad/s |
| Gate-active fraction | 36.5% | 50.8% |
| Steps within 0.35 m of goal | 11 | 79 |
| Angular sign changes | 6 | 12 |
| Final goal distance | 0.1985 m | 0.2021 m |

The predeclared classifier returns `evidence_insufficient`, because angular
direction persistence is below its 0.80 threshold. The trace nevertheless
rules out a simple no-progress explanation: the robot made 0.957 m of
post-pillar progress and repeatedly approached the goal. The observed pattern
is high-amplitude angular oscillation with frequent gate activation near the
goal, not cleanly one-direction persistent turning, pure weak recovery, or
pure low forward speed.

## Minimal verifiable next hypothesis

On independently generated follow-up diagnostic scenes, test whether repeated
post-obstacle gate activation combined with near-saturated V5 angular residuals
causes angular oscillation and near-misses of the 0.20 m success circle. This is
a diagnosis hypothesis only; it does not authorize controller changes,
demonstration collection, V7, or training.
