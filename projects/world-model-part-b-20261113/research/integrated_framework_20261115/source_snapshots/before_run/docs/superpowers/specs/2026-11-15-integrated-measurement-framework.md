# Integrated measurement framework

User steering: build and run the whole framework before optimizing individual
algorithms. Continue the current physical-consistency research despite the older
README's provisional direction. Existing algorithms/checkpoints/thresholds stay
fixed; no new fitting, recovery heuristics, video generation or model downloads.

One CLI consumes a hashed case manifest and writes a new run directory. Stages:
input verification/decode -> tracking -> RGB observation/visibility -> estimated
depth -> 3D coordinates -> physical relation -> judge calibration/decision ->
label-based evaluation -> complete per-case and dataset-specific aggregation.
Represent executed, reused, not applicable, missing and error explicitly. Never
turn unmeasurable into physical FAIL or quietly drop a failed case. No pooled
leaderboard across incompatible datasets. No future truth enters inference.

Adapters cover all 30 DAVIS clips, all 3 existing PStudio pilots, all 4 native I2V
clips, and the exact 12 existing blinded gravity calibration controls. Cache
replay must bind predictions to input and receipts by hashes. Fresh I2V mode
runs fixed CoTracker3, DA-V2 and TAPIP3D from stored RGB, retaining camera
assumptions and unvalidated scale. Controlled gravity uses the frozen RGB center
tracker/fit; depth is inapplicable to its 2D relation. DAVIS has no 3D/physics GT.
PStudio tests 3D measurement, not gravity. I2V can yield ABSTAIN with reasons.

Recompute the existing control judge gate from its 12 original independent
judgments/receipts. These are reused judgments, not new independent sessions.
The numeric relation judge and optional visual judge are separate channels;
export a blind RGB frame-sheet package for four I2V clips with visual judgment
status pending, never invent a VLM result or call pending human labels complete.

Success: one command replays all 49 inputs through applicable stages, a fresh run
processes all four I2V RGB clips through all three networks and 3D/measurement
decisions, all outcomes retained, tests cover invalid inputs/cache binding/
abstention/aggregation, a readable report and local handoff have real receipts.
Framework completion and scientific validity are reported separately.
