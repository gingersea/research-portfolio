# Existing-label measurement validation, version 1

Frozen before this study's calibration or diagnostics. Previous paper_benchmark
outputs and raw data remain read-only. No neural inference, new annotation,
generation, training, or public upload. The current question is whether existing
coordinates/visibility can support reliable accepted measurements, and whether
3D error is primarily localization or depth. Surface-point accuracy is not object
identity classification or proof of physical consistency.

## Calibration and evaluation

All 30 DAVIS sequences were previously exposed through baseline scores. Sort by
SHA256("measurement-validation-v1:" + case); first 10 are calibration, remaining
20 validation. This separates this study's fitting labels but is NOT a pristine
unseen benchmark or a pretrained-model data-contamination guarantee. No points
from the same video cross this split. All queries/frames remain included, using
the prior official first-query convention and pinned scoring implementation.

Only fit a scalar threshold on frozen visibility*confidence, on calibration data
only. Grid: 0,.5,.7,.8,.9,.95,.97,.98,.99,.995,.999,.9999; accept score>threshold.
Correct means GT visible and 2D Euclidean error<4px at 256x256. Calibration target:
pooled accepted precision>=.95, correct retention>=.20, and at least 5/10 videos
with accepted count>=10% of GT-visible post-query frames. Among feasible choices,
maximize correct accepted count then minimize threshold. If none reaches target,
explicitly mark failure and select highest precision among coverage-eligible
choices (then largest correct count, then lowest threshold). If even coverage is
infeasible, reject all and record failure. This is empirical selection, not a
statistical 95% guarantee. Score is not a calibrated correctness probability.

Freeze selected threshold receipt before validation labels are loaded. Compare
native visibility, previous fixed score>.9, selected threshold, and selected
threshold with fixed two-frame reactivation confirmation. Confirmation initially
uses the query frame's score; after rejection require two consecutive accepted
scores to reactivate, immediately reject on any low score. It changes declarations
only, not coordinates. It cannot repair a spatially wrong trajectory. Network
predictions are already offline; no online-tracker claim.

Report official video-mean AJ/APD/OA, supplemental pooled precision, retention,
false-visible rate, accepted wrong-location and occluded counts. Bootstrap the
20 validation VIDEOS with replacement, 10000 resamples, seed 20261115, paired
differences versus native; descriptive 95% percentile intervals. Frames/points
are repeated measures, not independent experiments. No hypothesis tests or
multiplicity-adjusted superiority claims. Keep all 363 previously defined return
events; failed events classified into no<4px position in the visible run versus
some<4px positions rejected by visibility. This is localization-versus-declaration
diagnosis, NOT a labeled identity-swap cause analysis. No further rules selected
from validation results in this version.

## 3D diagnostic

Reuse all three prior frozen PStudio 64-frame/8-query pilots (24 queries, 395
post-query visible point-times). Geometry-derived XYZ is evaluation truth, not
pure manual depth. All three are previously seen: diagnostic, no heldout claim.
For each method report raw 3D EPE and evaluation-only oracle ablations: predicted
ray with true Z; true ray with predicted Z; and DA-V2 sampled at true XY with true
ray. These oracles deliberately read GT ONLY inside the evaluator and are NOT
usable systems or model improvements. Use nearest depth sampling identical to
the previous Co+depth path, record bounds/clamping; all visible points included.
No per-sequence fitted scale, no feeding future GT to inference. Validate camera
projection and vector decomposition numerically. Three sequences are too few for
a general 3D superiority claim; report every sequence and equal-video means.

Deliver source snapshots, exact patch from empty new-study baseline, hashes of
read-only dependencies, input-bound receipts, meaningful fixtures, actual output
tables/figures/report and versioned local handoff. Existing README and prior
reports are not edited. Reproduce CPU scoring/calibration from frozen predictions;
do not imply a second neural-inference run.
