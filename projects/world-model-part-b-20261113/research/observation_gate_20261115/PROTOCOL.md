# Independent image observation gate and actual learned RGB-D tracking comparison

Previous tracking_validation controls are now development material. Prior studies
and scores stay immutable. All new analysis is in observation_gate_20261115.

## Independent gate

The gate sees initial RGB patches/queries and allowed authored initial radii,
current RGB, and proposed center coordinates. It never receives future GT,
depth, network visibility or network confidence. It does not move coordinates.
Grayscale normalized template evidence is searched locally within3px, checked
against the competing initial identity and stronger global matches. Missing image
evidence is rejected. Initially indistinguishable references(similarity>=.95)
are permanently marked ambiguous after an appearance loss or close encounter.
This is a restricted fixed-scale template gate, not general object re-ID.

Calibration uses all30 already-seen prior controls, with12fixed parameter
combinations. Primary config minimizes accepted incorrect+unsupported claims,
then maximizes correct retention, with fixed tie-break. Secondary coverage
operating point maximizes correct retention among configs with development joint
precision>=.99 and no unsupported claims. Both are frozen before new inference;
show both complete results, never choose per-condition winners. Primary is .5
aperture,.9correlation,.15identity margin; secondary .5,.8,.15. Raw scores and
every rejected reason are retained. These are operating rules, not calibrated
identity probabilities.

## New data and endpoint

40new controls:10conditions x4newseeds. Development810; heldout1810/2810/3810.
Same33frames,384x256,24FPS as prior stage, but new per-seed per-object random
material textures. Twins share texture. Renderer retains seed-dependent geometry
in all conditions. RGB pixel hashes must all differ and must not repeat any prior
validation video. Truth trajectories/visibility are separate from model inputs.
Prior renderer supplies exact camera/depth geometry; new textures only change RGB.

Center point0/5 are primary, all10material points auxiliary. Exclude query frame0.
Report raw and gated visibility precision/recall, accepted correct(PCK3px) precision,
visible correct retention, false-visible counts/rates, wrong-other-center claims,
unsupported twin-history claims, and abstention. Missing denominators=N/A. Reject-
everything is not success. Production needs accepted precision>=.99, false-visible
and wrong-other rates<=.01 AND retained correct observations>=.90 in identifiable
conditions; twin uncertainty must be explicit. Tiny simulated sequences do not
establish population error rates or real-I2V accuracy.

## Complete model comparison, after gate verification

Run official CoTracker3 and official TAPIP3D learned checkpoint on the SAME new
clips/queries,6iterations,6x6support grids. Use native author modules, never replace
TAPIP3D with simple lifting or substitute nearest-neighbor kernels. CoTracker3 uses
the prior fixed revision/checkpoint and native predictor. TAPIP3D code revision
4cb7e69a1687f67d56ec3e506768f51f2c581b46 and model revision
08730a588204f258f7a86530df15b2a6426e5f5b; checkpoint SHA256
3a9514d526559838e6158360af2b857da596d064b06a94fae7fd3b85134b2b1e.
Build original pointops2 with local CUDA12.8/SM120 in an isolated environment.
Use input resolution256x384 for TAPIP3D, raw evaluation mode, native6iterations,
FP32 first; preserve any engineering failure and document a compatibility-only
change before retrying. No model-quality retries or heldout threshold tuning.

Both methods get the same known simulated sensor-depth/camera data for 3D scoring:
CoTracker is lifted using scene depth at its own predicted pixels; TAPIP3D uses
depth/camera in its full learned3D point-cloud tracking architecture. Neither gets
future GT point coordinates or visibility labels. Query3D is lifted from initial
query pixels and initial supplied depth. No truth-derived normalization bound.
Native TAPIP3D depth ROI is computed from supplied depth statistics.

Compare 2D projection and world3D error, raw visibility, same independent gate,
allGT-visible denominator, each-method accepted subset AND common accepted subset.
Retain failed/abstained conditions; label privileged-depth comparison clearly.
This is actual learned RGB-D tracking, not a monocular MegaSaM pipeline, real-depth
validation, or physical scoring. Offline research comparison may proceed with a
partially rejecting gate; production/generation expansion remains blocked unless
the scientific observation/coverage requirements pass.
