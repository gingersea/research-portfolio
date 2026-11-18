# Physical Consistency of Open Video Models

Experiments with frozen Wan2.1 1.3B and CogVideoX-5B generators, trajectory measurement, and synthetic ball-segmentation detectors. The active scope is video generation; superseded meteorological proposals have been removed.

**Current conclusion:** measurement validity remains the bottleneck. None of the 24 generated clips passed the complete scale-calibration and physical-fitting gate. This does not mean physical correctness is zero. Multiscale training improves some synthetic conditions but fails its prespecified smallest-target endpoint and has false-positive tradeoffs. No new detector has replaced the frozen physical audit.

## Completed experiments

| Experiment | Evidence | Result and boundary |
|---|---|---|
| Frozen Part A | 2 models x 4 prompts x 3 seeds; 24 MP4s and 1,560 frames | Wan tracked 11/12; CogVideoX tracked 5/12; valid physical fits 0/24 |
| Detector pilot | 29,465-parameter TinyUNet; 2,048/256/256 synthetic train/validation/test images | Archived test IoU 0.97041; CPU recheck 0.97037; generated-video accuracy unvalidated |
| Transfer screening | All 1,560 frames; 120 anonymized annotation frames | Unlabeled output comparisons; manual annotation template remains blank |
| Scale stress | 256 paired scenes x 3 sizes x 3 input strategies x 2 methods; 4,608 records | Medium-target legacy IoU falls from 0.7954 at native resolution to 0.2986 after resizing |
| Multiscale training | 2 arms x 3 seeds; 6 new checkpoints; 23,040 held-out metric records | Smallest-target primary IoU difference -0.02078, conditional 95% CI [-0.02976, -0.01154]; medium-target improvements do not establish general success |
| Part B infrastructure | Fixed judging protocol, frame sampler and regression fixes | Calibrated production judging remains incomplete |

The multiscale CI resamples paired scenes conditional on three trained seeds; it does not include uncertainty over all possible training runs. Synthetic scenes are procedural data, not manually labeled generator outputs.

## Reports

- [Current results and limitations (Chinese)](docs/RESULTS_2026-11-12.zh.md)
- [Matched multiscale training](research/multiscale_20261109T191120Z/REPORT.zh.md), [complete tables](research/multiscale_20261109T191120Z/results/TABLES.md)
- [Scale stress and frame-sampling fixes](research/stress_20261109T144046Z/REPORT.zh.md)
- [Transfer screening](research/transfer_20261109T143307Z/REPORT.zh.md)
- [Frozen audit](report/part_a_halfway_results.md), [previous progress report](report/progress-2026-11-06.zh.md)
- [Three-page course milestone](report/AIST4010_Milestone_JIANG_Huanyun.pdf)

## Download and verify

Git contains source, tests, the original 24 clips, frozen records, seven small detector checkpoints (one legacy and six new), annotation frames, numerical results and figures. Large generator weights are not required for analysis replay.

The [video-results-2026-11-12 release](https://github.com/gingersea/world-model-physical-consistency/releases/tag/video-results-2026-11-12) includes full derived synthetic scenes, masks and predictions in one archive. Redundant verification replays are excluded. Archive and repository checksums are in [delivery_manifest.json](delivery_manifest.json). Historical study checksums also describe local replays; use the delivery manifest for this published snapshot.

Use Python 3.12 with [requirements-analysis.txt](requirements-analysis.txt). Actual export environment versions are in [environment-verified.json](environment-verified.json); historical experiment versions remain in each study's provenance. Install a PyTorch build suitable for your CPU or CUDA device. Exact numerical replay may require matching recorded versions.

```bash
git clone https://github.com/gingersea/world-model-physical-consistency.git
cd world-model-physical-consistency
python -m pip install -r requirements-analysis.txt
python scripts/verify_delivery.py
python -m pytest -q -p no:cacheprovider
```

[Reproduction instructions](docs/REPRODUCING.md) explain how to replay screening, scale stress and checkpoint evaluation, or rerun training into a new directory.

## Files

- `benchmark/physcheck/`: generation, tracking, fitting, sampling and judging protocol.
- `benchmark/results/raw/`: unchanged historical audit; absolute paths record provenance.
- `benchmark/results/training_pilot/`: frozen legacy detector and training metrics.
- `data/generated_videos/`: 24 original clips and sidecars, located by `data/video_manifest.json`.
- `data/frozen_physcheck_source.tar.gz`: exact historical analysis source matching the frozen source hash.
- `research/`: dated scripts, per-scene metrics, figures, new checkpoints and reports.
- `docs/protocols/`: local research protocols, not external preregistrations.
- `docs/superpowers/`: historical designs superseded by dated results where they conflict.

Human labels, reliable smallest-target measurement, false-positive control and calibrated Part B judging remain open tasks. Physical SI-unit estimates also require defensible camera, scale and time assumptions.
