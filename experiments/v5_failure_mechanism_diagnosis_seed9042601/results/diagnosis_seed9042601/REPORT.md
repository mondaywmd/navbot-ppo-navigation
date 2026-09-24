# V5 failure-mechanism diagnosis

Seed `9042601`; 36 deterministic Actor episodes; no training or permanent exam data.

| Pair | Focus | Positive S/C/T | Negative S/C/T |
|---:|---|---:|---:|
| 01 | near | 3/0/0 | 3/0/0 |
| 02 | near | 1/2/0 | 3/0/0 |
| 03 | near | 0/3/0 | 3/0/0 |
| 04 | far | 3/0/0 | 3/0/0 |
| 05 | far | 2/0/1 | 3/0/0 |
| 06 | far | 3/0/0 | 3/0/0 |

Actor overall S/C/T: **30/5/1**.

## Early-collision feasibility

Early collisions: **5**. Reactive control S/C/T: **0/0/3**. Evidence: **control_did_not_find_path**.

## Safe-timeout post-obstacle diagnosis

| Scene | Ep | Post steps | Mean abs angle | Mean linear | Mean abs angular | Progress | Class |
|---|---:|---:|---:|---:|---:|---:|---|
| diag_pair05_far_positive | 2 | 122 | 72.096 | 0.074 | 0.402 | 0.957 | evidence_insufficient |

### Same-scene successful versus timeout trajectory

| Scene | Case | Post steps | Mean abs angle | Final abs angle | Mean linear | Mean abs angular | Progress |
|---|---|---:|---:|---:|---:|---:|---:|
| diag_pair05_far_positive | timeout ep 2 | 122 | 72.096 | 86.140 | 0.074 | 0.402 | 0.957 |
| diag_pair05_far_positive | success ep 1 | 63 | 40.211 | 17.960 | 0.107 | 0.393 | 0.947 |

## Minimal next hypothesis

The fixed reactive control did not find a path; this is not proof of physical infeasibility. More controller-level feasibility diagnosis is required.

Diagnostic evidence only; this does not authorize training.
