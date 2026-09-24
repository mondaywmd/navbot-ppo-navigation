# Final V5 Actor mirror-consistency diagnosis

Read-only mechanism diagnosis, seed `9042603`. It uses only prior non-OOD
diagnostic traces and six new near-pillar mirror pairs. It contains no training
or model/environment/controller modification path.

Derived 16-D mirror map:

- indices 0..9: reverse, because the Burger ray sensor has 10 samples ordered
  from -pi/2 to +pi/2 and observation construction preserves all 10 indices;
- 10 previous linear residual: unchanged;
- 11 previous angular residual: negate;
- 12 normalized goal distance: unchanged;
- 13 sine of signed goal-heading error: negate;
- 14 cosine of signed goal-heading error: unchanged;
- 15 signed goal-heading error divided by 180: negate.

Expected Actor relation: mirrored linear residual equals the original linear
residual; mirrored angular residual equals its negative.
