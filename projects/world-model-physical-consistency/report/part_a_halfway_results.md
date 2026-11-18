# Part A halfway results (artifact-derived)

Generated from `benchmark/results/raw/part_a.jsonl`; these are measurement-coverage statistics, not a physical-correctness leaderboard.

| Model | Generated | Tracked | Fitted | Mean g relative error (fitted only) | Mean T relative error (fitted only) |
|---|---:|---:|---:|---:|---:|
| wan2.1_1.3b | 12 | 11 | 0 | N/A | N/A |
| cogvideox_5b | 12 | 5 | 0 | N/A | N/A |

## Per-video measurements

| Model | Prompt | Seed | Track | Fit | Parameters | Reason if unavailable |
|---|---|---:|---|---|---|---|
| cogvideox_5b | fall_a1 | 0 | ok | error | N/A | ValueError: unreliable diameter calibration: elongated/blurred ball |
| cogvideox_5b | fall_a1 | 1 | detection_failure | skipped | N/A | object present in 11/49 frames (< 50%) |
| cogvideox_5b | fall_a1 | 2 | detection_failure | skipped | N/A | object present in 5/49 frames (< 50%) |
| cogvideox_5b | horiz_a4 | 0 | detection_failure | skipped | N/A | object detected in only 0/49 frames |
| cogvideox_5b | horiz_a4 | 1 | detection_failure | skipped | N/A | object present in 8/49 frames (< 50%) |
| cogvideox_5b | horiz_a4 | 2 | detection_failure | skipped | N/A | object detected in only 1/49 frames |
| cogvideox_5b | pend_a3 | 0 | ok | fit_failed | N/A | R2=0.956, RMS_px=10.99 (thresholds 0.6, 3.0) |
| cogvideox_5b | pend_a3 | 1 | ok | fit_failed | N/A | R2=0.169, RMS_px=0.86 (thresholds 0.6, 3.0) |
| cogvideox_5b | pend_a3 | 2 | ok | fit_failed | N/A | R2=0.209, RMS_px=12.85 (thresholds 0.6, 3.0) |
| cogvideox_5b | proj_a2 | 0 | detection_failure | skipped | N/A | object detected in only 0/49 frames |
| cogvideox_5b | proj_a2 | 1 | detection_failure | skipped | N/A | object present in 8/49 frames (< 50%) |
| cogvideox_5b | proj_a2 | 2 | ok | error | N/A | ValueError: unreliable diameter calibration: elongated/blurred ball |
| wan2.1_1.3b | fall_a1 | 0 | ok | error | N/A | ValueError: unreliable diameter calibration: elongated/blurred ball |
| wan2.1_1.3b | fall_a1 | 1 | ok | error | N/A | ValueError: unreliable diameter calibration: elongated/blurred ball |
| wan2.1_1.3b | fall_a1 | 2 | ok | fit_failed | N/A | R2=0.362, RMS_px=84.43 (thresholds 0.6, 3.0) |
| wan2.1_1.3b | horiz_a4 | 0 | ok | error | N/A | ValueError: unreliable diameter calibration: elongated/blurred ball |
| wan2.1_1.3b | horiz_a4 | 1 | detection_failure | skipped | N/A | object present in 15/81 frames (< 50%) |
| wan2.1_1.3b | horiz_a4 | 2 | ok | fit_failed | N/A | R2=0.985, RMS_px=10.92 (thresholds 0.6, 3.0) |
| wan2.1_1.3b | pend_a3 | 0 | ok | error | N/A | ValueError: unreliable diameter calibration: elongated/blurred ball |
| wan2.1_1.3b | pend_a3 | 1 | ok | error | N/A | ValueError: unreliable diameter calibration: elongated/blurred ball |
| wan2.1_1.3b | pend_a3 | 2 | ok | error | N/A | ValueError: unreliable diameter calibration: changing apparent size |
| wan2.1_1.3b | proj_a2 | 0 | ok | fit_failed | N/A | R2=0.991, RMS_px=5.12 (thresholds 0.6, 3.0) |
| wan2.1_1.3b | proj_a2 | 1 | ok | error | N/A | ValueError: unreliable diameter calibration: elongated/blurred ball |
| wan2.1_1.3b | proj_a2 | 2 | ok | fit_failed | N/A | R2=0.588, RMS_px=7.32 (thresholds 0.6, 3.0) |

## Synthetic validation

| Scene | Parameter | Truth | Estimate | Relative error |
|---|---|---:|---:|---:|
| free_fall | g | 9.8 | 9.80529 | 0.054% |
| projectile | g | 9.8 | 9.81588 | 0.162% |
| projectile | v0x | 3 | 3 | 0.000% |
| pendulum | T | 2.00709 | 2.00431 | 0.138% |
| horizontal_motion | v0 | 2 | 2.06565 | 3.282% |
| horizontal_motion | decel | 0.5 | 0.499275 | 0.145% |

Errors are conditional on the nominal FPS and the prompt-stated 0.1 m ball diameter. Rejected fits and missing measurements are not zero error. Regression standard errors do not include calibration/camera uncertainty.
