# Closeout: what eight sampled frames can miss

This is a controlled methodology audit, not a video-model ranking or VLM accuracy measurement.
Protocol was fixed locally before creating this run. Existing project outcomes were already known; the study is exploratory.

## Exact window exposure

| Video frames | Samples | Event duration (frames) | Visible / possible windows | Exposure |
|---:|---:|---:|---:|---:|
| 49 | 8 | 1 | 8/49 | 16.33% |
| 49 | 8 | 4 | 26/46 | 56.52% |
| 49 | 8 | 8 | 42/42 | 100.00% |
| 49 | 16 | 1 | 16/49 | 32.65% |
| 49 | 16 | 4 | 46/46 | 100.00% |
| 49 | 16 | 8 | 42/42 | 100.00% |
| 49 | 32 | 1 | 32/49 | 65.31% |
| 49 | 32 | 4 | 46/46 | 100.00% |
| 49 | 32 | 8 | 42/42 | 100.00% |
| 81 | 8 | 1 | 8/81 | 9.88% |
| 81 | 8 | 4 | 26/78 | 33.33% |
| 81 | 8 | 8 | 50/74 | 67.57% |
| 81 | 16 | 1 | 16/81 | 19.75% |
| 81 | 16 | 4 | 58/78 | 74.36% |
| 81 | 16 | 8 | 74/74 | 100.00% |
| 81 | 32 | 1 | 32/81 | 39.51% |
| 81 | 32 | 4 | 78/78 | 100.00% |
| 81 | 32 | 8 | 74/74 | 100.00% |

Exact finite enumeration over legal start positions; no inferential confidence interval.
Exposure assumes uniformly weighted discrete starts and is only a necessary visibility condition, not semantic detection accuracy.
No claim is made that actual generator errors follow this timing distribution.

## Lossless counterexamples

12 pairs / 24 FFV1 videos: source RGB equals decoded RGB; each pair differs at exactly one designated unsampled frame.
All 12 pairs have byte-identical eight-frame PNGs and annotated grids, while full-video states and video hashes differ.
Labels describe deviations from a specified fixed-camera rigid-ball world with no external contact/force, not independently human-labeled natural physics.
If opposite classes have equal priors and the classifier receives only identical images without label leakage, expected image-only discrimination is 50%. This is a conditional information bound, not an observed VLM result.
Dense sampling reveals the programmed differences; it does not by itself certify physical interpretation.

## Closeout decision

The old pipeline is closed as a feasibility/methodology study with explicit negative outcomes. Its Part A SI estimates and Part B calibrated ranking remain unavailable.
Do not spend further generator runs merely to force the old calibration gate to pass. Reuse code, frozen artifacts and controlled audit tools for a differently scoped question.
This experiment demonstrates a known sparse-observation limitation in this specific pipeline; it does not establish a novel general theorem or a publishable research contribution on its own.

## Reproduction

`python research/closeout_20261114/experiment.py --output NEW_DIRECTORY`
The output must not exist. The current repository sampler is invoked without modifying it. No model generation, paid API call, human label or VLM judgment is fabricated.
See coverage.json for all50 conditions, pairs.json for states/interventions and verification.json for observed checks.
