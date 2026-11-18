# Local reproduction

Existing interpreter:
`/data/AI_Research/shared/paper_benchmark_20261115/env/bin/python`.
Run from the project root. No paid inference API is used. Publisher weights and
EgoPoints material retain their research-use licenses; no public redistribution
is authorized by these commands.

Preserved provenance: `implementation_assets.json`, `checkpoint.json`, the
expired PointSt3R sharing page in `sharepoint_direct/`, and range/transport
attempts in `acquisition*`. Initial failed source requests remain in `sources/`;
`source_api/` is the successful fixed-revision GitHub API fallback. Python source
under the external fixed archive has not been modified.

`selection.json` describes the actual bounded public tar prefix and archive-order
participant-disjoint return-stratum selection. `prepare_validation.py` creates
`manifest_validation.json`, query-only `input.npz`, separate human `truth.npz`,
original annotations and `validation_freeze.json`. Existing experiment outputs
are immutable: commands producing them reject an existing directory. To rerun,
use an isolated copy or a new explicit output name. Acquisition is not rerun for
cached scoring and must not overwrite preserved source bytes.

Actual model commands, after input and source freeze:

```bash
PY=/data/AI_Research/shared/paper_benchmark_20261115/env/bin/python
STUDY=research/reference_relocalization_20261117
$PY $STUDY/run_descriptor.py --manifest research/real_measurement_20261117/manifest.json --output $STUDY/predictions_development --gpu 0
$PY $STUDY/run_native.py --gpu 1
$PY $STUDY/run_descriptor.py --manifest $STUDY/manifest_validation.json --freeze $STUDY/validation_freeze.json --output $STUDY/predictions_descriptor_validation --gpu 0
```

Old CoTracker predictions are reused from
`research/real_measurement_20261117/predictions_memory` with their original model,
input and array hashes. Their reuse is not new independent inference. New native
predictions use the same checkpoint, querywise support grid, six iterations,
feature chunk200, float32 and cache clearing before each full query.
MASt3R uses the official public static checkpoint; there is no PointSt3R dynamic
checkpoint or visibility head. Native/image-pair methods have different temporal
inputs by design. Both use the same prepared RGB and initial queries. Full new
validation inference only runs if at least one qualifying case exists.

Analysis and coordinate-vs-acceptance audit:

```bash
$PY $STUDY/analyze.py --manifest $STUDY/manifest_validation.json --native $STUDY/predictions_native_validation --descriptor $STUDY/predictions_descriptor_validation --output $STUDY/analysis_validation --split new_participant_validation
$PY $STUDY/pairing_audit.py --manifest $STUDY/manifest_validation.json --native $STUDY/predictions_native_validation --descriptor $STUDY/predictions_descriptor_validation --output $STUDY/pairing_validation
$PY $STUDY/audit_integrity.py
$PY -m pytest -q
```

`analyze.py --relocated-data PATH` and `pairing_audit.py --relocated-data PATH`
read per-case copied truth while still enforcing original SHA256. This supports
actual copied-source/cache replay without opening original truth locations.
`create_handoff.py` records exactly what was copied and which outputs replayed
identically. Cached scoring replay is not a fresh GPU inference reproduction.
All-new acquisition is not needed for that check. Full model source/weights are
preserved separately in the bundle for future research reproduction.

Published diagnostics include six meaningful descriptor tests, the full CLI
counterexample, original-author feature/matching comparison, explicit failures
of the independent GLM/Qwen samplers, and separate development/validation counts.
The independent reviewers did not provide a comprehensive pass. No new
generated-video physical validity or calibrated Part B result follows.
