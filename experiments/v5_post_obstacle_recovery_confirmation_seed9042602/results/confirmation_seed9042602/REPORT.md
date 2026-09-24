# V5 post-obstacle recovery confirmation

Seed `9042602`; 12 new scenes; 36 deterministic episodes; no training or permanent OOD access.

## Classification

{'general_timeout': 5, 'success': 23, 'near_goal_safe_timeout': 8}

## Scene statistics

S = success, N = near-goal safe timeout, T = general timeout.

| Pair | Positive S/N/T | Negative S/N/T |
|---:|---:|---:|
| 01 | 1/0/2 | 0/2/1 |
| 02 | 3/0/0 | 3/0/0 |
| 03 | 3/0/0 | 3/0/0 |
| 04 | 2/1/0 | 2/1/0 |
| 05 | 2/1/0 | 2/1/0 |
| 06 | 0/1/2 | 2/1/0 |
| **Total** | **11/3/4** | **12/5/1** |

## Final-window comparison

| Group | N | Gate switches | Gate active | Mean abs angular residual | Angular sign switches | Monotonic-distance fraction | Trend reversals | Final-min rebound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| success | 23 | 1.000 | 0.517 | 0.932 | 1.000 | 1.000 | 0.000 | 0.000 |
| near-goal safe timeout | 8 | 7.000 | 0.604 | 0.940 | 7.000 | 0.706 | 1.500 | 0.022 |

## Predeclared support checks

{'near_goal_timeout_count_at_least_3': True, 'gate_switches_systematically_higher': True, 'angular_reversals_systematically_higher': True, 'repeated_nonconvergence': False}

Verdict: **evidence_insufficient**.

This tests association only and makes no causal claim.

Worth designing a separate minimal control ablation next: **no; evidence is insufficient**.
