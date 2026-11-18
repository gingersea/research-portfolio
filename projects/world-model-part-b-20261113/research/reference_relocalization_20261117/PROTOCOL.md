# Frozen study protocol before holdout acquisition/inference

## Question and separation

Test reference-target global correspondence as a route to actual relocalization,
not merely rejecting the full-video tracker's positions. Prior12 EgoPoints
sequences/10participants are development/regression only. New validation excludes
ALL their participant IDs, not just overlapping clips. No model/prompt training.

The author's PointSt3R sharing link is expired (anonymous page preserved).
Use the explicitly published MASt3R pretrained baseline instead, if available;
do not claim the dynamic fine-tuned/visibility-head PointSt3R model was run.
Pin implementation/weights before holdout inference. Do not substitute a
descriptor/geometry confidence for a calibrated visibility probability.

## Data and denominators

Extend acquisition of the SAME original author EgoPoints tar to at most its
first3GiB logical prefix, reusing all verified earlier ranges. Source selection
is archive order, excluding prior10participants; complete clips only,64–1200RGB
frames, all original frames preserved. For this reappearance-focused pilot,
eligible clips must contain at least one known annotated OOV→visible coordinate-
valid return. Select the first eligible clip per new participant, up to6new
participants. Labels may define this target stratum, never model results.
If fewer qualify within the fixed acquisition window, keep the actual smaller
sample and report the limitation; do not change criteria after outcomes.
No unseen-return coverage claim if no qualifying external return sample exists.

Keep all initial query points and all requested human-labeled timestamps,
including known occluded/OOV negatives. Unlabeled frames stay unknown.
Inputs:RGB,first-frame query locations,requested timestamp schedule only.
Future human coordinates/visibility remain evaluator-only in a separate file.
512x288RGB storage; normalized error dx*256/W,dy*256/H and strict<4px correctness,
same endpoint as the previous study, not official full-dataset leaderboard.

## Fixed comparison arms

1. Full-video native CoTracker3, same frozen source/weights/querywise support grid,
   six iterations, defaultfeature chunk200 and explicit cache clearing per query.
2. Previous frozen two-frame reverse-CoTracker reciprocal acceptance gate,
   same coordinates as1; no retuning on new validation.
3. Published MASt3R dense descriptors from(reference,target)RGB pairs. Bilinear
   query descriptor, global cosine nearest neighbor, fixed tie handling first
   row-major maximum as in the author's matching code. Coordinate-only/raw arm
   reports forced correspondence without asserting visibility: accept all finite
   in-image candidates so false acceptance on known absent points is visible.
4. MASt3R reciprocal descriptor verification: globally match the selected target
   descriptor back to reference pixels, accept only if back-projection is within
   strict4normalized pixels of the original query. This is correspondence
   self-consistency, not independently calibrated visibility or identity.

Use the author's reference-target inference architecture and preprocessing,
with input_yres288 to match stored images and preserve coordinates. No target
search window based on truth, oracle selection, GT depth or future query reset.
Freeze exact loading/matching details with adapter tests before new predictions.

## Analysis and completion

Primary: correct coordinate recovery on annotated returns and all visible points,
accepted precision, correct retention and false visible/wrong-location acceptances.
Retain failed planned cases as zero accepted/correct with full GT denominators.
Return events are annotated transitions, not dense temporal absence estimates.
Participant is the cluster; bootstrap10000(seed20261117) for paired/descriptive
intervals only when meaningful. Do not use degenerate all-zero resamples as a
population false-positive bound. Show individual counts and abstention.
Report enriched-stratum selection, small sample and prior-model training unknowns.

Old12 development outcomes and new holdout outcomes are strictly separate.
No post-holdout threshold/model selection; any later change needs another test.
Round ends after the bounded comparison, integrity/meaningful regression checks,
honest different-family review, Chinese report and copied-source/cache handoff.
No public upload, paid inference API, Google/gingersea operation or1081 changes.
