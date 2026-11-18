# World Models: Action-Conditioned Prediction and Planning

The provisional active direction is action-conditioned prediction and closed-loop planning. The earlier text-to-video physical-consistency work is closed as a feasibility and measurement study, with its data and unresolved limitations retained. This change follows the user's request to close the current experiments and change direction; the precise next research question remains adjustable.

**Current stage:** a fully observed, noiseless linear state-world pilot is complete (six least-squares fits, 128 held-out rollouts and 64 matched planning cases). It validates an action/prediction/control interface; it is not a trained visual world model, a novel method or real-world planning evidence. The next study must add visual observations and meaningful baselines, including a reactive controller and a known-dynamics reference.

**Archived conclusion:** measurement validity remained the bottleneck. None of the 24 generated clips passed the complete scale-calibration and physical-fitting gate. This does not mean physical correctness is zero. Multiscale training improves some synthetic conditions but fails its prespecified smallest-target endpoint and has false-positive tradeoffs. No new detector has replaced the frozen physical audit.

The November 13 Part B run completed 48 production clips, 8 controls and 56 isolated judgments. Calibration failed: the generated controls were not verified known positives under the static-frame protocol. Valid Part B rates are N/A; retained diagnostic judgments do not establish a model ranking.

## Completed experiments

| Experiment | Evidence | Result and boundary |
|---|---|---|
| Frozen Part A | 2 models x 4 prompts x 3 seeds; 24 MP4s and 1,560 frames | Wan tracked 11/12; CogVideoX tracked 5/12; valid physical fits 0/24 |
| Detector pilot | 29,465-parameter TinyUNet; 2,048/256/256 synthetic train/validation/test images | Archived test IoU 0.97041; CPU recheck 0.97037; generated-video accuracy unvalidated |
| Transfer screening | All 1,560 frames; 120 anonymized annotation frames | Unlabeled output comparisons; manual annotation template remains blank |
| Scale stress | 256 paired scenes x 3 sizes x 3 input strategies x 2 methods; 4,608 records | Medium-target legacy IoU falls from 0.7954 at native resolution to 0.2986 after resizing |
| Multiscale training | 2 arms x 3 seeds; 6 new checkpoints; 23,040 held-out metric records | Smallest-target primary IoU difference -0.02078, conditional 95% CI [-0.02976, -0.01154]; medium-target improvements do not establish general success |
| Part B production and calibration | 48 production clips + 8 controls; 56 isolated sessions; 448 sampled PNGs | Full matrix decoded and hashed; calibration invalid; valid violation rates N/A |
| Sparse-frame closeout | 50 exact sampling conditions; 12 paired synthetic counterexamples / 24 lossless videos | All 12 pairs have different complete videos but identical eight-frame inputs; a visibility limitation, not measured VLM accuracy |
| Action-conditioned stage 0 | 2 state-model arms x 3 data seeds; 128 test rollouts; 64 matched planning cases | In this intentionally linear world: conditioned fits 64/64 goals each, action-blind 0/64; feasibility only |

The multiscale CI resamples paired scenes conditional on three trained seeds; it does not include uncertainty over all possible training runs. Synthetic scenes are procedural data, not manually labeled generator outputs.

## Reports

- [Closeout and provisional direction change (Chinese)](docs/CLOSEOUT_AND_PIVOT_2026-11-14.zh.md), [sampling audit](research/closeout_20261114/results/REPORT.md), [new stage-0 pilot](research/action_world_stage0_20261114/results/REPORT.md)
- [Part B production and calibration (Chinese)](docs/RESULTS_2026-11-13.zh.md), [artifact-derived English report](research/part_b_20261113T114040Z/REPORT.md), [proposal gap audit](research/part_b_20261113T114040Z/PROPOSAL_AUDIT.zh.md)
- [November 12 results and limitations (Chinese)](docs/RESULTS_2026-11-12.zh.md)
- [Matched multiscale training](research/multiscale_20261109T191120Z/REPORT.zh.md), [complete tables](research/multiscale_20261109T191120Z/results/TABLES.md)
- [Scale stress and frame-sampling fixes](research/stress_20261109T144046Z/REPORT.zh.md)
- [Transfer screening](research/transfer_20261109T143307Z/REPORT.zh.md)
- [Frozen audit](report/part_a_halfway_results.md), [previous progress report](report/progress-2026-11-06.zh.md)
- [Three-page course milestone](report/AIST4010_Milestone_JIANG_Huanyun.pdf)

## Download and verify

Git contains source, tests, the original 24 clips, frozen records, seven small detector checkpoints (one legacy and six new), annotation frames, numerical results and figures. Large generator weights are not required for analysis replay.

The [video-results-2026-11-12 release](https://github.com/gingersea/world-model-physical-consistency/releases/tag/video-results-2026-11-12) includes full derived synthetic scenes, masks and predictions in one archive. Redundant verification replays are excluded. Archive and repository checksums are in [delivery_manifest.json](delivery_manifest.json). Historical study checksums also describe local replays; use the delivery manifest for this published snapshot.

The commands below verify that published November 12 snapshot. This local research branch additionally retains the November 13 Part B study; its checksum receipt is separate from the old delivery manifest.

Use Python 3.12 with [requirements-analysis.txt](requirements-analysis.txt). Actual export environment versions are in [environment-verified.json](environment-verified.json); historical experiment versions remain in each study's provenance. Install a PyTorch build suitable for your CPU or CUDA device. Exact numerical replay may require matching recorded versions.

```bash
git clone https://github.com/gingersea/world-model-physical-consistency.git
cd world-model-physical-consistency
git checkout ac86e9bdd16222c872a6372a9ec2e7f35e1f3ef1
python -m pip install -r requirements-analysis.txt
python scripts/verify_delivery.py
python -m pytest -q -p no:cacheprovider
```

On the Part B research branch, verify the new study with `python research/part_b_20261113T114040Z/verify_snapshot.py`. See its [handoff](research/part_b_20261113T114040Z/HANDOFF.md) for raw videos, fixed inputs, first judgments, actual session receipts and the failed calibration gate.

For the November 14 local closeout and pilot, run `python research/closeout_20261114/verify.py`. This checks retained media, all prediction summaries, matched planning trajectories and the preserved earlier snapshot. The [new handoff](research/closeout_20261114/HANDOFF.md) describes minimal CPU dependencies and reproduction into fresh directories.

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

Human labels, reliable smallest-target measurement, false-positive control and calibrated Part B judging remain recorded gaps of the closed feasibility study. They are not marked solved. Physical SI-unit estimates also require defensible camera, scale and time assumptions. The active provisional work now measures action-conditioned prediction and actual closed-loop outcomes, beginning with controlled synthetic baselines.
