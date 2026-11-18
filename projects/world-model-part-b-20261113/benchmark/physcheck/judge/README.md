# Part B judging workflow

A vision-capable agent reads sampled frames and answers the fixed template in
`protocol.py`. The CLI validates those retained answers; it does not call a
vision model or create labels. Record the actual backend, available version and
any version limit. The November 13 study uses isolated platform QA sessions
(`gpt-6-astra/high`), not Claude API calls.

## Blinded inputs

Generate calibration and production videos, then sample eight evenly spaced
frames per clip. Give each clip a unique opaque ID that reveals neither model,
prompt, calibration status nor expected answer. Keep the ID-to-model/prompt/seed
manifest outside the judge's allowed reads. A fresh session receives only the
fixed template, its opaque ID, original sampled frame indices and its images.
Retain both the grid and individual sampled PNGs.

Use one fresh session per video. Judge all calibration clips first and record the
gate before production judging. Record real session identifiers and image-viewing
evidence where the harness exposes them. A matching prompt does not establish
that a generated control actually contains a visible physical contradiction:
record control eligibility from the observed images, with uncertain controls
marked unverifiable. Controller/VLM inspection is not human ground truth.

## Outputs and manifest

Each line in `judge_outputs.jsonl` wraps the judge's unchanged JSON answer:

```json
{"video_id":"v_opaque","raw_output":"<the judge's JSON answer>","judge_model":"<actual backend>","judge_version":"<available version and limits>","judge_session":"<unique actual session ID>"}
```

The private manifest maps every video ID to `model`, `prompt_id`, `seed`, `kind`,
`is_calibration`, `expected_violation_kind` and `sampled_frame_indices`. For
controls, use explicit boolean `calibration_observable: true` only after establishing
the visible contradiction. False or an omitted field makes the control ineligible;
nonboolean values are rejected. Prompt text alone cannot establish eligibility.

## Production validation checkpoint

```bash
PYTHONPATH=benchmark python3 -m physcheck judge \
  --input-file PATH/judge_outputs.jsonl \
  --manifest PATH/part_b_manifest.json \
  --output-file PATH/judge_validated.jsonl
```

With `--manifest`, validation requires exact ID coverage, matching embedded IDs,
one recorded judge identity, unique per-clip sessions, schema compliance and
violations referring only to sampled original frames. Use `--calibration-only`
with only the control outputs for the initial gate. Omitting `--manifest` retains
the legacy path and does not provide these production checks.

The status checkpoint binds to the exact canonical manifest SHA256. Reporting
requires that binding and revalidates saved coverage, sessions, judge identity and
calibration against retained judgments. Changed manifests, incomplete/duplicate
rows, nonboolean or inconsistent gates, and missing strict-production status are
rejected. Old unbound checkpoints remain historical evidence; produce a new
checkpoint under a new results root instead of rewriting them.

Successful calibration exits 0 and writes validated JSONL plus a `.status.json`
sidecar. Failed calibration exits 2, preserves parsed diagnostic outputs and gate
details in the sidecar, and does not write validated JSONL. Retain the initial
gate and first judgments. The CLI refuses an existing output or status path;
use a new checkpoint path for a distinct experiment. Do not coach judges,
silently drop failed controls or
repeat judgments until an expected label appears. An amended protocol/control
set needs its own documented experiment.

Set `PHYSCHECK_RESULTS_DIR` to the same results root, then run:

```bash
PYTHONPATH=benchmark python3 -m physcheck report --part b
```

The report consumes the status sidecar. A failed gate produces a JSON/Markdown
leaderboard marked invalid, with valid rates shown as N/A and separate diagnostic
counts. Empty violation lists do not establish physical correctness.

The complete November 13 study workflow, including real session receipt export,
video auditing and artifact assembly, is in
`research/part_b_20261113T114040Z/PROGRESS.md` and its study scripts. Cross-backend
API clients and environment-driven API routing are not implemented here; external
adapters can produce the same retained-answer format. No paid API call is needed
for CLI validation or reporting.
