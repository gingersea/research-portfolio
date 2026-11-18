# Reproduce the stated scope

Actual environment: `/data/AI_Research/shared/paper_benchmark_20261115/env/bin/python`.
Versions, the123-test workspace result and compile receipt are in tests_receipt.json.
No dependency install, model training or paid API is required on this workstation.

## Re-score and render from frozen predictions

From the original repository root, with genuinely new output directories:

```bash
PYTHONPATH=benchmark /data/AI_Research/shared/paper_benchmark_20261115/env/bin/python \
  research/real_measurement_20261117/analyze.py --output /tmp/real-measurement-score-new

/data/AI_Research/shared/paper_benchmark_20261115/env/bin/python \
  research/real_measurement_20261117/report.py --output /tmp/real-measurement-report-new
```

The reporter reads this study's frozen `analysis/` and tests receipt; `--output`
only changes its output destination. To report newly scored alternative data,
make a separate study copy and explicitly relocate inputs, rather than silently
overwriting analysis or mixing runs. No scientific threshold search is implemented.

## Fresh inference and source separation

`run_models.py --part0/1 --gpu0/1` uses stored RGB and first-frame queries only,
plus the known requested measurement timestamps. Each query is separately run
through the author's native CoTracker3 predictor; reciprocal candidates use its
two-frame interface. The existing source/checkpoint hashes are verified against
tracking_validation_20261115/assets.json. The runner never opens truth.npz or
original_annot.npz. Only the evaluator opens future coordinates/visibility.

The actual commands have spaces between option and value:

```bash
python research/real_measurement_20261117/run_models.py --part 0 --gpu 0
python research/real_measurement_20261117/run_models.py --part 1 --gpu 1
```

These dated production commands reject existing per-case receipts. Do not run
them in place over this completed study. For a fresh neural repetition, copy the
study, relocate manifest input paths, verify frozen source hashes, preserve
original results separately and use an empty predictions directory. The compact
handoff verifies cached scoring, not a clean-machine model installation/rerun.

## Source data and bounded acquisition

sources/ pins the author's GitHub revision and public Dropbox folder metadata.
The original uncompressed tar is59.48GB. PROTOCOL_v2 changes acquisition/selection
before new EgoPoints outcomes to a bounded archive-prefix pilot; V3 stops at the
same selected cases once complete. This is not a uniform sample or full-dataset
benchmark replication. All requested clips are retained regardless of performance.

The acquired prefix and individual downloaded ranges, including partial attempts,
remain in `/data/AI_Research/shared/real_measurement_20261117`. Original selected
JPEGs are byte-preserved under its extracted/ directory; source_images.json binds
each image's SHA256 and tar offset. Each study data/ folder contains the exact
original annotation, derived inference input, separate derived truth and source
image manifest. Unknown vis_valids frames never become negative labels.

Downloader/preparer scripts have fixed dated outputs and should not be rerun over
frozen evidence. Their preserved failed versions document acquisition history;
they are not additional recommended production entry points. No Google endpoint,
gingersea operation, public upload or1081 configuration change is needed.

## Actual handoff verification

package.py creates a versioned local directory and records its path in
handoff_receipt.json. It retains the original manifest, creates a separate
relocated evaluator manifest, then actually runs copied-source scoring on copied
inputs/truth/predictions. Per-video results and summary must match; Chinese report
and CSV must regenerate byte for byte. Vector figures are also regenerated,
without claiming PDF byte identity across timestamps. All package hashes are
verified and recorded in MANIFEST.json.

The package contains the new sparse scorer/test and recorded full-workspace test
receipt; it is a scoped experiment handoff, not a complete offline reproduction
of every historical study. Original raw archive/JPEGs, model weights and the
existing environment are external dependencies with recorded provenance.
Original source videos/annotations retain their own rights; a local handoff is
not authorization for public redistribution. Review outputs and manual rulings
are included without claiming independent comprehensive approval.
