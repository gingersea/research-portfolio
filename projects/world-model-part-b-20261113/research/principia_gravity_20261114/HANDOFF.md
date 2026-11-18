# Reproduce the paired-gravity calibration

Read PROTOCOL.md and REPORT.zh.md first. All new media are procedural controls,
not generated-model results or human-labeled real videos. Old studies are immutable.

## Local outputs

- data/public/:168 anonymous setups, each FFV1 and H264 (336 clips total).
- data/private/:analytic development/heldout truth. Never pass these to judges.
- development_measurements_v1/v2.json and corresponding gates: retained failure
  and correction. source_snapshots/development_v1/ preserves the old instrument.
- heldout_measurements_v2.json, heldout_gate_v2.json: frozen image-only measurement.
- judge_inputs/:exact42 contact sheets and12payload/rubric sets used in judging.
- judgments/:12unchanged first judgments; judge_session_receipts.json: filtered
  session evidence. No conversation text or hidden reasoning is exported.
- judge_gate.json:initial result; judge_gate_v2.json:stronger integrity validation.
- official_crosscheck.json:unchanged pinned official helper counterexamples.
- gravity_calibration.png/pdf/svg and CSV tables:derived from actual heldout data.

## Environment and replay

Executed interpreter:/home/room115/pytorch_env/bin/python, Python3.12.3. No GPU,
model download or paid inference API was used to create/measure the control videos.
Use NumPy, OpenCV, SciPy, Pillow, matplotlib, pytest and system ffmpeg. Exact
analysis environment is retained by the final package metadata. The default
scientific interpreter on this workstation lacks pytest; tests ran in pytorch_env.

Run from repository root:

```bash
python -m pytest -q -p no:cacheprovider research/principia_gravity_20261114/test_validation.py
python research/principia_gravity_20261114/measure.py --split heldout --data research/principia_gravity_20261114/data --output /tmp/gravity_measurements_fresh.json
python research/principia_gravity_20261114/assess_measurement.py --measurements /tmp/gravity_measurements_fresh.json --truth research/principia_gravity_20261114/data/private/heldout_truth.json --output /tmp/gravity_gate_fresh.json
python research/principia_gravity_20261114/verify.py --record /tmp/gravity_verification_fresh.json
```

Choose fresh output paths; scripts refuse existing result files. To rerender,
give render_controls.py --output a fresh directory, first --split development,
then --split heldout. Measurement and private truth are separate explicit inputs.
The measured frames/conditions are paired by scene; don't count codecs or frames
as independent samples.

Historical judge payloads/receipts retain their original absolute workstation
paths and are byte-preserved. Measurement replay accepts a copied data directory;
strict historical judge access/path auditing is scoped to the original workspace.
Recreating judges is a NEW experiment requiring fresh independent sessions and
output directory; do not overwrite these judgments or silently replace failed ones.

To rerun official numerical checks, obtain Principia repo revision
893e003804f0994ea6467b5a756ab657b82d4df9, then pass --source and a new --output to
official_checks.py. No SAM3 checkpoint or paper-level benchmark execution is claimed.

## Decision boundary

Both limited procedural gates passed. A next native-I2V pilot is specified as
2setups×2seeds before broader generation; it has not been executed in this study.
Natural-image tracking, subtle violations, other phenomena, human agreement,
provider-dated judge versions, and real-generator rankings remain unvalidated.
