# Integrated Measurement Framework Implementation Plan

> **For agentic workers:** User has directed implementation now. Execute natively in this session without delegation or another approval round.

**Goal:** Run the existing research components as one traceable pipeline before optimization.
**Architecture:** Add a small physcheck.relations package with explicit adapters and stage records. Keep dated studies immutable and import their frozen numeric functions. Separate inference inputs from evaluator-only labels.
**Tech Stack:** Python3.12, NumPy, OpenCV, existing Torch/CUDA environment and checkpoints.
**Spec:** docs/superpowers/specs/2026-11-15-integrated-measurement-framework.md

## Global Constraints

- No algorithm/threshold optimization, new models, generation, or public upload.
- Old studies and README are read-only; new outputs refuse overwrite.
- Every selected case, including failures and abstentions, stays in aggregation.
- Cached predictions bind to inputs; GT is loaded only by evaluation/calibration.
- Missing visual judgments remain pending; 3D camera assumptions remain explicit.

## Review Focus

- Cache from another video must fail even if shapes match (Task1 fixture).
- Missing depth or invalid calibration must not produce physical PASS (Task1 fixture).
- Points outside the image must not become valid by boundary clamping (Task1 fixture).
- One video error must remain a row without erasing the rest (Task2 fixture).
- Re-running into an existing output must refuse overwrite (Task2 fixture).

### Task1: Contracts and frozen stage adapters

Create benchmark/physcheck/relations/{__init__,contracts,backends}.py and tests/test_relations.py.
Interfaces: verify_file(entry)->Path; verify_cache(receipt,input_sha,pred_sha)->None;
lift_points(xy,depth,K)->(xyz,valid); relation_decision(measurement,admissible)->dict.
Write fixtures for listed safety/measurement boundaries, observe RED, implement
minimal adapters, observe GREEN. Fresh backend run(video,queries_tyx,K) returns
native co/tap XY/visibility/XYZ plus DA depth; parameters match prior runs.

### Task2: One-command orchestration and real runs

Create benchmark/physcheck/relations/{pipeline,__main__}.py and
research/integrated_framework_20261115/{prepare,deliver}.py.
Interfaces: run(manifest,output,fresh_i2v=False,gpu=1)->summary with all cases.
Normalize raw/cached tracking, execute RGB gates/relations, evaluate published
labels, replay control-judge calibration, export blind RGB sheets, emit per-case
artifacts and aggregate by data domain. Test error retention/no overwrite, then
run replay49 and fresh4 native I2V with the fixed networks. No threshold fitting.

### Task3: Verification and local delivery

Run focused tests and existing repository tests appropriate to new package. Check
49 rows, stage provenance, fresh4 model receipts, unchanged old input hashes,
matching published baseline metrics, and inspect real visual output. Write report,
source before/after snapshots with exact added-file patch, versioned local handoff.
No git commit/push is part of this task.
