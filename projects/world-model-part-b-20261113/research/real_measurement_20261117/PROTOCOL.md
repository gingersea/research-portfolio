# External sparse-label measurement comparison: frozen before inference

## Population, selection and inference

Use the author's original EgoPoints tar, not generated pseudo-labels. List all
archive member names first. Eligible clips have64–1200 RGB frames (resource
bound, not performance selection). Group overlapping clips by their original
EPIC source-video prefix Pxx_yy, choose one per source video by SHA256 of
`real-measurement-20261117:` plus clip ID, order source videos by the same salt,
take the first24 (or all if fewer). Preserve missing/corrupt selected cases,
no replacements. Report participant groups for dependence and resampling.
No new development split: all selected clips are a fixed external test with
no parameter search or model training. Previous DAVIS exposure does not extend
to this dataset, but checkpoint authors' training provenance is not guaranteed.

Keep every source RGB frame, resize to512x288 for storage/inference; retain
original dimensions, image hashes, frame order and timestamp indices. All
queries supplied by the benchmark's reference frame, one query per native
CoTracker3 call with the fixed author6x6 support grid,6iterations, checkpoint
and source already recorded in tracking_validation_20261115/assets.json.
No future truth coordinates or visibility flags reach model inference.

The task requests measurement at the benchmark's annotated timestamps. These
timestamps (without target coordinates/visibility) are provided as scheduling
metadata to ALL methods, including the reciprocal check. This is evaluation of
sparse requested measurements, not a claim to infer dense identity reliability.
Truth masks/coordinates live in separate files, opened only by the evaluator.
Unannotated visibility zeros are never negative labels. Query frame is excluded.

## Fixed three arms

1. Native: author's CoTracker3 visible declaration AND finite in-image coordinates.
2. Frozen confidence: v*c>0.995 AND native AND in-image. 0.995 was chosen on DAVIS
   November15 and is not retuned here. It is a score threshold, not a probability.
3. Reciprocal: native AND in-image AND successful reverse correspondence back to
   the original reference within4 normalized pixels AND reverse native visibility.
   Reverse model receives exactly [candidate target RGB, reference RGB], query at
   its own candidate pixel. Reject invalid/invisible/nonfinite reverse outputs.
   No coordinate is changed; this is a verification gate, not an identity-recovery
   algorithm. Self-consistency can hold for the wrong identity; quantify that risk.

The motivation is query/correspondence self-consistency in point-tracking work;
PointSt3R motivates explicitly comparing reference-target evidence across long
gaps. This gate is NOT a reproduction of PointSt3R/BootsTAP or their training.

## Endpoints, denominators, missingness

Primary: accepted precision (known visible and <4px normalized L2 / all accepted
known-label point-times), correct retention (same numerator / all known-visible
coordinate-valid post-query point-times), and false visible claims on labeled
occluded/out-of-view point-times. Normalization: dx*256/W,dy*256/H; this matches
our previous task convention, not the official EgoPoints full-resolution ranking.
Count separately false visible at out-of-view versus in-view occluded and visible
wrong location. Accepted known-visible points with unavailable coordinates are
counted as unverifiable acceptance, never removed to inflate precision.

Secondary: accuracy at4px on all known-valid visible positions independent of
claims; returned-point recovery after an annotated OOV event; strict recovery
also requires no false visible claim at the prior annotated OOV. Sparse annotations
support these annotated transitions only, not exact loss/recovery times or dense
temporal occlusion windows. Report macro per-video values, pooled counts and
participant-cluster bootstrap95% intervals (10000 draws, seed20261117). Empty
denominators are N/A, never zero error. Do not treat points/frames as independent
experimental replicates. Missing inference contributes zero accepted/correct
and remains in GT positive/negative denominators; report execution coverage.

Exploratory gate for useful measurement: pooled precision>=95%, correct retention
>=20%, at least half the planned clips with>=5 correct accepted labeled point-times.
Even passing these empirical thresholds is not a guarantee in generated physics
videos. No post-hoc threshold search, replacements, score-selected examples or
claim that external point validation calibrates a physical VLM judge.

## End of this round

Execute all selected cases/all arms, audit sparse masks and failures, report
paired precision/coverage effects and whether reciprocity rejects actual errors
without claiming coordinate recovery. Preserve first attempts, exact source,
weights, model inputs, raw predictions and hashes. Decide the next method from
these results; no generator expansion or new physical ranking in this round.
