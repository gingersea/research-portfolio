# Reproducing the uploaded experiments

Run commands from the repository root. Historical reports retain original workstation paths and execution evidence. These commands use exported repository paths. Use new output directories to preserve frozen results.

## Integrity and regression checks

```bash
python scripts/verify_delivery.py
python -m pytest -q -p no:cacheprovider
```

The verifier checks repository checksums, decodes 24 clips and 1,560 frames, verifies metadata and detector weights, and reconstructs the frozen analysis-source hash. It does not relabel videos or recompute the old physical-fit audit with later code.

`benchmark/verify_part_a.py` is the historical writing entry point. It expects original paths and source; use the new verifier for read-only exported-snapshot checks.

## Transfer screening (CPU)

```bash
python scripts/reproduce_transfer.py --output outputs/transfer_replay
python research/transfer_20261109T143307Z/plot_screening.py --results outputs/transfer_replay
```

The adapter creates a temporary derived JSONL mapping archived identifiers to exported videos, copies the original training helper and checkpoint, then invokes the unchanged screening script. Archived records remain untouched. Path fields and their hashes differ from historical provenance; numerical outputs can be compared directly. Annotation frames have no human ground-truth labels.

## Synthetic scale stress (CPU)

```bash
python research/stress_20261109T144046Z/run_stress.py \
  --checkpoint benchmark/results/training_pilot/best_model.pt \
  --output outputs/scale_stress
python research/stress_20261109T144046Z/summarize.py --results outputs/scale_stress
```

This regenerates 768 transformed synthetic scenes and produces 4,608 measurements, masks and predictions. The release archive contains the original full outputs; Git contains the original per-scene metrics.

## Multiscale checkpoint evaluation

```bash
python research/multiscale_20261109T191120Z/experiment.py \
  --output research/multiscale_20261109T191120Z/results \
  --old-checkpoint benchmark/results/training_pilot/best_model.pt \
  --eval-only --evaluation-name replay_local
```

This uses six frozen new checkpoints, the old detector and the color baseline. It regenerates 256 positive/64 negative scenes and 23,040 condition/method records. CUDA is selected when available, otherwise CPU. Exact cross-device floating-point equivalence is not guaranteed.

## Rerun the six trainings

```bash
python research/multiscale_20261109T191120Z/experiment.py \
  --output outputs/multiscale_training \
  --old-checkpoint benchmark/results/training_pilot/best_model.pt
python research/multiscale_20261109T191120Z/summarize.py --results outputs/multiscale_training
```

Both arms train for 12 epochs with seeds 7/17/27. Validation selects checkpoints; test data are rendered after all six models are frozen. Shared test scenes are not independent new samples for each training seed.

## Original generator inference

Large generator checkpoints are external dependencies. Original revisions/hashes are in `benchmark/results/raw/*weights_manifest.json`; model identifiers and settings are in `benchmark/physcheck/generate/models.py` and `benchmark/physcheck/ENVIRONMENT.md`. Optional generation also requires the documented diffusers/transformers/accelerate/safetensors/Hugging Face dependencies and sufficient GPU memory. Detector replay does not require the large weights.

## Release archive

Download `video-physics-results-2026-11-12.tar.gz` from the repository release using an authorized account. Check its SHA256 against `delivery_manifest.json`, then extract into a new directory. It contains a `world-model-physical-consistency/` prefix and canonical full outputs, excluding caches and duplicate replays. It is not a generator-weight distribution.
