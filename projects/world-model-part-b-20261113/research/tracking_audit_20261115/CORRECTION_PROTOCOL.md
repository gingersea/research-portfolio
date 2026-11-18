# Exploratory correction v1, fixed before running candidate on generated frames

Original four I2V clips and their gate are immutable. All four clips are already
seen development material; this reanalysis is not an unseen validation set.

Confirmed root causes from diagnosis/summary.json: global marker cardinality
rejects additional green artifacts despite all authored anchors remaining present;
global hue uniqueness loses identity under appearance changes/duplicate hues.
Area changes, ambiguous merges, disappearance and absent contact are separate.

Two separate interventions:
1. Keep the original green threshold/area>=30, associate each authored anchor to
   exactly one component within12px, one-to-one. Reject absent/ambiguous anchors.
   Final camera gate remains ALL97 frames, individual drift<=1.5px.
2. Restricted static-background prototype: use per-row median RGB of the supplied
   initial image as background; foreground is max absolute RGB difference>20.
   Mask only the authored marker squares expanded6px beyond their10px half-width.
   No hue identity criterion. Components>=40px are matched to previous observed
   centers with radius4*input_ball_radius, mutual unique one-to-one matching.
   Missing observations are never filled; tolerate at most2 missing frames for
   maintaining association memory, then terminate identity. Any competing candidate
   or many-to-one conflict terminates the affected identity conservatively.
   Components can update association even when physical area gate fails, but only
   complete components with area/inputarea in[.75,1.25] count eligible.
   These appearance/association constants are fixed once; no generated-output tuning.

Known trajectory controls FIRST: fixed unaltered, hue changes, grayscale, hue
collision, disappearance, ambiguous overlap and clutter. Include controls where
simple spatial matching must abstain. Test the original detector first to retain
the failing regression. Record position error only where an observation is actually
returned, plus false observations during known absence/identity failure. One heldout
control sweep with seeds710..717,8conditions,17frames/sequence precedes production
reanalysis. Stop candidate promotion if controls fail. Original generated clips may
still be inspected diagnostically with any failure prominently retained.

This is a minimal image-observation diagnostic, not CoTracker3/SpatialTracker/
MegaSaM/TAPIP3D reproduction, general video segmentation, material-point tracking,
human labels, or metric 3D. No gravity/ballistic/ARAP prior, depth completion, temporal
interpolation, regeneration, new physical score, or change to the original expansion
gate. New coverage means component/association availability, not accuracy proven
on generated objects. Background/model assumptions must be reported.
