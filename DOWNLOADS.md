# Downloads and reproduction

## Interview PDFs

- [00 — Guide and evidence boundaries](interview/00_面试使用说明与证据边界.pdf)
- [01 — Optimization and LoRA](interview/01_优化与LoRA_完整介绍.pdf)
- [02 — Video physical consistency](interview/02_视频物理一致性_完整介绍.pdf)

## Full experiment results

Most code, reports, figures, small numerical arrays, and result tables are directly in the repository. Five large optimization JSON logs are gzip-compressed without changing their bytes. Each has an adjacent derived summary for browsing; restore the complete logs before running scripts that expect the original `.json` names:

```bash
python tools/restore_results.py
```

The script verifies source SHA-256 hashes and refuses to overwrite a different existing file. It needs only Python's standard library.

Large video-result arrays are stored in six ZIP assets attached to the [portfolio-2026-11-18 release](https://github.com/gingersea/research-portfolio/releases/tag/portfolio-2026-11-18):

| Asset | Experiment |
|---|---|
| `world-model-proposal_completion_20261117.zip` | Full-length public geometry predictions |
| `world-model-physics_methods_20261114.zip` | Generated-video numerical frame artifacts |
| `world-model-identity_recovery_20261115.zip` | Depth and tracking result arrays |
| `world-model-principia_i2v_20261115.zip` | Native image-to-video frame arrays |
| `world-model-integrated_framework_20261115.zip` | Integrated pipeline prediction arrays, including preserved versions |
| `world-model-paper_benchmark_20261115.zip` | Public benchmark 3D predictions |

Download all result ZIPs into a directory outside the checkout, then restore the original relative paths:

```bash
gh release download portfolio-2026-11-18 --repo gingersea/research-portfolio --pattern 'world-model-*.zip' --dir ../portfolio-assets
python tools/restore_results.py --assets ../portfolio-assets
```

The archives contain paths relative to the repository root. The restoration tool validates archive hashes and rejects unexpected or unsafe member paths before extraction. `provenance/release-files.json` gives per-file hashes; `provenance/release-assets.json` gives archive hashes and sizes.

## What is not bundled

The [exclusion manifest](provenance/excluded-files.json) records individual omitted checkpoints/caches and excluded directory classes. In particular, almost 1 TB of model and optimizer checkpoints from the optimization workspace is not uploaded. Environment installations, VCS metadata, local assistant metadata, raw video/input caches, and raw review-transport payloads are also excluded. Small experimental detector checkpoints are retained.

For actual execution, follow each project's requirements and dated reproduction instructions, obtain the named upstream model/data assets, and adapt any workstation-specific paths. Restoration of archived results is not equivalent to rerunning neural inference or training. No experiment was rerun merely to create this portfolio.

## Provenance

- `source-files.json`: original copied-file byte counts and hashes.
- `compressed-results.json`: original JSON and gzip paths, hashes, and preview locations.
- `release-files.json` / `release-assets.json`: large result members and archives.
- `portable-link-changes.json`: link-only edits to displayed Markdown copies.
- `compressed-link-changes.json`: navigation redirected to a losslessly compressed result.
- `excluded-files.json`: what was omitted and why.
- `upload-files.json`: final Git snapshot file hashes, including portable copies.

The uploaded project snapshots retain historical reports, provenance, and limitations. The current scientific interpretation is summarized in [EVIDENCE_GUIDE.md](EVIDENCE_GUIDE.md).
