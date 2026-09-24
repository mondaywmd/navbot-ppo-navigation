# Clean-start temporal LiDAR navigation

This experiment starts from random Actor/Critic weights and does not consume
V5/V8 checkpoints or BC datasets. Permanent OOD data remains inaccessible.

Preparation order: install an isolated 360-degree/360-ray running model;
validate temporal range rates and a full-scan hard safety shield; assess the
300-step horizon against a diagnostic 450-step continuation; only then design
new demonstrations and learning. No preparation script starts training.

The attempted runtime robot replacement caused Gazebo to exit during URDF
spawn. The original launch, robot, center pillar, target, 10-ray scan, and zero
velocity were restored. Runtime sensor replacement is rejected; dense LiDAR
must be loaded by a separate startup launch.

At 5 Hz, 300 steps is about 60 seconds. Because earlier successful routes took
as many as 296 steps, clean-start evaluation will retain a diagnostic 450-step
continuation. The horizon will only be extended if episodes converge during
steps 301--450 rather than merely circle or remain stopped.

The startup-only dense launch now passes with 360 full-circle points at about
one-degree spacing and 5 Hz. The first straight hard-stop smoke maintained
0.3087 m minimum clearance. A turning-side extension then checked the commanded
turn's inner sector and was evaluated on 12 new mirrored scenes (seed 8042715):
12/12 succeeded within 92--258 steps, with zero collision/timeout and minimum
clearance 0.321 m. No episode required steps 301--450, so 300 remains the limit.

The new learning interface is 78-D: 36 current sector minima, 36 signed
inter-frame range rates (positive approaching, negative receding), previous
linear/angular action, and four goal-geometry values. Its mirror transform
reverses both 36-sector blocks and negates only signed angular quantities.
Dimension/range, approach/recede, and double-mirror tests pass. No model has
been trained with this observation yet.

The separate `clean_start_dense.launch` subsequently succeeded: `/scan` has
360 full-circle points at about one-degree spacing and 5 Hz. Thirty-six
ten-degree features take sector minima. Stationary two-frame smoke measured a
maximum false closing rate of 0.0413 m/s, below the declared 0.06 m/s deadband.

The first hard-stop smoke repeatedly requested 0.25 m/s into the center pillar.
The shield intervened at step 6 and maintained 0.3087 m minimum LiDAR clearance.
A fresh seed-8042714 horizon audit then used shielded waypoint control on 12
mirror-paired one/two/three-pillar scenes. All 12 succeeded within 68--102
steps, with no collision or timeout, so this sample provides no reason to raise
the 300-step limit. Minimum full-circle clearance was 0.278 m; turning-side
clearance remains the next shield item before demonstration collection.
