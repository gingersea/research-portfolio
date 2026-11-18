# Persistent point identity, visibility, location and geometric lifting v1

User approved the next stage after tracking_audit. Previous studies remain read-only.
No new generative-video experiments, human labels, paid APIs or public publication.

## Question and units

Can official CoTracker3 separate observed visible locations from missing or wrongly
associated targets better than the prior per-frame hue detector and foreground
prototype? Then, what geometric information is gained or lost by lifting those SAME
2D predictions with known versus perturbed depths and compensating camera motion?
This second comparison is a geometry/error-propagation ablation, not a full
TAPIP3D/MegaSaM/SpatialTracker model benchmark or a real-depth accuracy claim.

Thirty controlled sequences:10conditions x3seeds.33frames,384x256,24FPS. Development
seed910; heldout seeds1910/2910. No threshold/model/input tuning on heldout output.
Conditions:clean,textureless,hue_change,same_hue,occlusion,disappearance,camera_motion,
crossing,identity_replacement,ambiguous_twins. Render textured circular planar sprites
with known world XYZ, persistent material queries, intrinsics, camera poses, z-buffer
depth and current-frame visibility. This is controlled simulated geometry, not
spherical 3D object reconstruction or natural-video generalization.

Five material points per object: center and four axial offsets of0.4radius. Main
cross-method comparison uses the two centers; auxiliary CoTracker diagnostics use
all10queries. Exclude query frame0 from primary metrics because the official wrapper
forces query-frame outputs to their supplied ground truth. Every method sees the
same RGB frames, initial queries and allowed setup geometry, never future GT.
Labels are renderer truth, not human annotation. Missing frames never disappear
from the denominator. 30sequences are the units; points/frames are repeated measures.

## Frozen inference

CoTracker3 scaled_offline.pth, HF revision bf55ea50d4390e1820a267f131cd6587240fb2c5,
SHA2562670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834.
Official code82e02e8029753ad4ef13cf06be7f4fc5facdda4d. Native CoTrackerPredictor:
offline=True,v2=False,window_len60,6iterations,support_grid6x6,FP32,normal native
resize to model_resolution. Read-only hook captures raw visibility/confidence.
Record both official visibility>.9 and prespecified visibility*confidence>.9.
The second rule is not claimed calibrated merely because it is stricter.
No quality retries, source patches or variable thresholds. Hardware/API failures
are retained and repaired only as software failures, with versioned receipts.

Baselines are unchanged legacy hue components and previous static-background
prototype, applied only to center-query observables. Known initial radius/area and
fiducial geometry are supplied to the prototype, which is an additional baseline
input advantage. Its foreground assumptions will fail under some camera motion.

## Metrics and scientific gates

Use all32post-query frames. Report per-condition/per-sequence visible GT location
errors (all predictions and declared-visible subset), PCK3px, declared-visible
precision and recall, false-visible rate among GT-invisible points, wrong-other-
center claims within5px, ID-association transitions, and longest invalid run.
Position availability and position accuracy are separate. Count return of a finite
coordinate at an invisible frame as a prediction, not a visible observation.
For ambiguous_twins after frame12, object identity is not observationally identified;
exclude those frames from identity correctness and report declared claims as
unsupported identity assertions, never fabricate a unique identifiable history.

Production promotion requires EACH identifiable heldout condition's center-query
PCK3px>=.90, visible precision>=.99, recall>=.90; invisible false-visible<=.01 when
invisible examples exist; wrong-other-center claim fraction<=.01. Additionally
ambiguous post-event identity claims must<=.01 for a measurement system claiming
unambiguous identity. N/A denominators are explicit. Failure means no expansion or
new physical scores. This tiny controlled set cannot establish a population FPR.

## Paired 3D information ablation, after 2D validation

Keep identical XY and visibility outputs. On visible GT queries, compare:
constant depth4m; sampled oracle depth at predicted pixel; oracle depth with a
fixed10% sinusoidal temporal scale perturbation; oracle depth with a fixed20% scale
bias. Report world-coordinate errors with true known camera transform, plus
uncompensated camera coordinates on the camera_motion condition. True depth is an
explicit privileged-information upper bound, not an input available for the old
generated videos. Also compute exact-query oracle lift to verify geometry code.
No truth-based scale fitting, no reconstructed trajectories, no physical priors.
Do not claim 2D tracking improvement: lifting keeps projected XY fixed by design.
Full learned 3D tracking is not promoted when the identity/visibility gate fails.

The previous4generated clips are already-seen development cases and have no human
identity/visibility truth; avoid assigning accuracy to them. This stage's decisive
results come from the new controls and their counterexamples.
