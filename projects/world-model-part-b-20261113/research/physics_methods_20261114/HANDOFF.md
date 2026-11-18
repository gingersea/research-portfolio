# Reproduce the literature-method pilot

This local follow-up investigates eight primary papers and tries released VPT
inference plus VideoPhy2 public-human-label aggregation. Read `LITERATURE_REVIEW.zh.md`
for paper-specific methods and `TRIAL_PROTOCOL.md` for fixed trial choices.
The old Part B calibration failure is not changed by this study.

## Assets and environment

The large base weights and official source exports are external, not vendored.
Pinned upstream revisions and verified hashes are retained in
`base_model_manifest.json`, `assets_vpt.json`, `source_vpt_manifest.json`,
`source_videophy_manifest.json` and `dataset_manifest.json`.

The local isolated overlay is
`/data/AI_Research/shared/physics_methods_20261114/env_vpt`.
It inherits installed ML dependencies through a `.pth` file pointing to
`/home/room115/pytorch_env/lib/python3.12/site-packages`, with local pins:

```text
transformers==4.57.6
diffusers==0.36.0
peft==0.18.0
huggingface-hub==0.36.0
tokenizers==0.22.2
```

Torch is2.9.0+cu128, GPU is RTX5090. System ffmpeg encodes the MP4s.
This overlay is specific to the workstation, not a portable lockfile; actual
inference/analysis versions are retained by `generation_audit.json`.

## Replay

Run from the repository root. Public asset preparation scripts need network;
aggregation and output verification do not call any model API.

```bash
python research/physics_methods_20261114/analyze_videophy2.py
python research/physics_methods_20261114/plot_public_labels.py
python research/physics_methods_20261114/capture_provenance.py
python research/physics_methods_20261114/audit_outputs.py
```

Use `env_vpt/bin/python` for inference and environment capture. To rerun generation
on this workstation:

```bash
/data/AI_Research/shared/physics_methods_20261114/env_vpt/bin/python research/physics_methods_20261114/run_vpt.py --arm base --gpu 0
/data/AI_Research/shared/physics_methods_20261114/env_vpt/bin/python research/physics_methods_20261114/run_vpt.py --arm vpt --gpu 1
```

Existing video/receipt pairs are skipped only after their recorded hash matches.
For independent resampling, use a copied study directory with an empty generation
subdirectory; keep this study's frozen outputs intact. Exact GPU byte replay is
not promised across software/hardware changes. Smoke outputs use reduced settings
and are excluded from every production count.

`export_sources.py` exports selected text blobs using a local pinned Git tree;
`fetch_ranges.py` verifies downloaded pinned HF bytes; `prepare_dataset.py`
originally captured the dataset revision. For a historical replay use the retained
revision in `dataset_manifest.json`, rather than resolving a new dataset HEAD.

## Evidence boundaries

The public-human-label tables describe existing published model videos with
unequal model coverage; the common-prompt table addresses that coverage difference
on382captions. It does not contain human labels for our new six clips. The three
`[nan]` rule placeholders remain missing data, not physical rules or pass labels.
The MP4/native-frame arrays and contact sheets are new generation outputs;
inspection is unblinded and qualitative. No calibrated judge, new human labels,
physics improvement rate or valid new-model leaderboard is reported.
