# Part B production and calibration results

This report is artifact-derived. Generated control prompts are not independently verified labels.

Checkpoint root: `run/results_reviewed`.

Calibration valid: **False**. Judge: gpt-6-astra through isolated platform QA sessions.
The actual dated model snapshot is not exposed by the harness. No Claude API request was made.

## Generated matrix

| Model | Calibration clips | Production clips | Decoded frames | Generation status |
|---|---:|---:|---:|---|
| wan2.1_1.3b | 4 | 24 | 2268 | 28/28 decoded and hashed |
| cogvideox_5b | 4 | 24 | 1372 | 28/28 decoded and hashed |

Decoded and hashed means file-level generation completion, not verified prompt compliance or physical correctness.
One seed per calibration prompt; three seeds per production prompt. Eight original-frame-index samples per clip.
Calibration was judged before any production clip. Every clip uses a separate opaque-ID session.

## Calibration observations

| Model | Control prompt | Reported categories | Eligible control? | Notes |
|---|---|---|---|---|
| wan2.1_1.3b | calib_float_person | none | False | The person appears upright with feet supported by the terrain in all eight sampled frames (0, 11, 23, 34, 46, 57, 69, 80). No visible physical contradiction is evident; motion between sampled frames is not assessed. |
| wan2.1_1.3b | calib_floating_glass | none | False | The sampled frames show a stream entering a glass, with a curved submerged feature and small bubble-like shapes beneath it. Their material and depth are visually ambiguous; they could represent entrained air or refracted fluid surfaces. No unambiguous physical contradiction is visible, and motion between samples is not inferred. |
| wan2.1_1.3b | calib_pass_through | none | False | No unambiguous physical contradiction is visible in the sampled frames. Airborne balls in frames 11, 57, and 69 can be consistent with bouncing; their blurred or elongated outlines do not establish rigid-body deformation. The sparse samples do not establish the trajectories or origins of balls entering view. |
| wan2.1_1.3b | calib_uphill | none | False | No clear physical contradiction is visible in sampled frames 0, 11, 23, 34, 46, 57, 69, and 80. The balls appear airborne or partially obscured by the foreground ridge; airborne positions alone do not establish a violation. Blur and occlusion prevent reliable assessment of contact or deformation. |
| cogvideox_5b | calib_float_person | none | False | Inspected the grid containing frames 0, 7, 14, 21, 27, 34, 41, and 48. The person appears airborne in several frames, but these sampled stills do not establish a physical contradiction without inferring motion between them. Ground or water contact is unclear in the later frames, and blurred or doubled contours do not provide clear evidence of a rigid-body violation. |
| cogvideox_5b | calib_floating_glass | none | False | The sampled frames show a glass containing water and an ice-like object, with surface ripples and small droplets. No clear physical contradiction is visible; refraction and the sparse sampling prevent reliable inference about the object's motion or changing appearance. |
| cogvideox_5b | calib_pass_through | none | False | No clear physical contradiction is visible in the sampled frames. A third ball becomes visible near the blue ball, but earlier occlusion cannot be excluded. Changes in ball coloring alone do not establish a physical violation, and the sampled images do not establish the intervening collision dynamics. |
| cogvideox_5b | calib_uphill | none | False | No clear physical contradiction is visible in the sampled frames. The leaning figure in frames 0, 7, and 14 has visible ground contact; the stylized poses and perspective do not establish a support violation. Frames 21, 27, 34, 41, and 48 show a standing figure with feet on the ground. Motion between sampled frames is not assessed. |

A control is not a known positive merely because its prompt describes a violation.
Constant uphill speed and pass-through dynamics may be unidentifiable from this static-frame protocol.
No failed controls were dropped, no labels were fabricated, and no judge was coached to produce a desired answer.

## Production leaderboard

# Part B visible-violation summary

Calibration valid: False. Static-frame judgments only.

| Model | Judged production videos | Valid violation rate |
|---|---:|---:|
| cogvideox_5b | 24 | N/A (invalid calibration) |
| wan2.1_1.3b | 24 | N/A (invalid calibration) |

Diagnostic-only counts are retained in JSON. They are not a valid model ranking.

Detailed diagnostic-only per-video/category counts, if calibration is invalid, are retained in JSON rather than interpreted as a valid model ranking.

## Proposal status

- Task 7: real calibration and production clips sampled; eight-frame grids and individual PNGs archived.
- Task 8: real isolated judgments and strict validation executed. Scientific calibration validity is the gate outcome above, not assumed from code completion.
- Task 10: 48 production clips and all 56 judgments retained; CLI validation and leaderboard aggregation attempted with failure status preserved.
- Remaining: independently verified calibration references (including negative controls), human generated-video labels, false-positive control, reliable smallest-target measurement, valid Part A physical estimates and the complete English paper draft.

## Interpretation and reproducibility

Each video is an evaluation unit; three seeds share a prompt and are not independent scene categories. Counts are descriptive; no P values, causal claim, or inferential confidence interval is reported.
The evaluator sees eight static frames, not full temporal dynamics. An empty violation list means no observed contradiction under this protocol, not physical correctness.
The frozen sampler resizes native frames to 320x180 grid cells without preserving aspect ratio. Original sampled PNGs are also archived. Wan uses 81 frames at 16 fps; CogVideoX uses 49 frames at 8 fps. Native durations differ, so diagnostic counts are not a duration-matched causal comparison.
Filtered session receipts verify unique actual child UUIDs, fork_turns=none, model alias/effort and actual image-input bytes against archived grid hashes. Per-clip read scope is instruction-enforced rather than an operating-system sandbox. No full private transcripts are exported.
Backend routing, rejected/failed controls and code decisions are documented in PROGRESS.md. Generator settings, video hashes, source hashes, package versions and timings remain in per-video sidecars.

Run `generate_part_b.py --model MODEL --gpu GPU --output RUN --models-dir CHECKPOINT_CACHE` once per model. Judge controls in fresh one-clip sessions, export their filtered receipts with `export_judge_receipts.py --partial` and run `assemble_part_b.py --phase calibration` before any production judgment. After completing the production judgments, export all 56 receipts without --partial, run `audit_part_b.py`, `assemble_part_b.py --phase production`, and this renderer. Keep the original early calibration gate unchanged.
The full real generated matrix must exist; scripts never fill missing judgments or generate human labels automatically.

AI assistance: code, experiment orchestration, VLM judgments and report drafting were AI-assisted. Human truth labeling remains incomplete.

Review findings and one TDD fix pass are documented in REVIEW_RESOLUTION.md. The initial checkpoint remains in run/results; the final bound replay is in run/results_reviewed. Both retain the same first judgments and failed scientific gate.
Render the final replay with `summarize_part_b.py --results-dir research/part_b_20261113T114040Z/run/results_reviewed` from the repository root. The amended production assembler accepts the same --results-dir argument and refuses to replace a CLI checkpoint.

## Gate details

```json
{
  "valid": false,
  "reason": "calibration gate failed: v_772567b42b93601624f2 (no gravity violation found); v_b9f673a411654ccd3ed5 (no support violation found); v_1d67264071a97694e1ac (no collision violation found); v_e90f9cee3619bda58167 (no gravity violation found); v_c7aa0d3c0af8330f6446 (no gravity violation found); v_beb926eab2b6b36d2078 (no support violation found); v_2b85fb3039fc61b28649 (no collision violation found); v_aecf44b75913d8ec776b (no gravity violation found)",
  "unverifiable_controls": [
    "v_772567b42b93601624f2",
    "v_b9f673a411654ccd3ed5",
    "v_1d67264071a97694e1ac",
    "v_e90f9cee3619bda58167",
    "v_c7aa0d3c0af8330f6446",
    "v_beb926eab2b6b36d2078",
    "v_2b85fb3039fc61b28649",
    "v_aecf44b75913d8ec776b"
  ]
}
```
