# Part B study handoff

Start with `REPORT.md` for artifact-derived results and `PROPOSAL_AUDIT.zh.md`
for the proposal gap audit. `PROGRESS.md` records implementation decisions,
backend limitations and test evidence. The Chinese result narrative is
`docs/RESULTS_2026-11-13.zh.md` in the repository.

## Evidence layout

- `run/private/artifacts/videos/`: generated MP4s and generator sidecars.
- `run/private/`: model-to-opaque-ID manifests and controller control-eligibility
  observations. These files were excluded from judges' allowed read scope.
- `run/blind/frames/`: each clip's fixed payload, eight sampled PNGs and grid.
- `run/judgments/`: unchanged first per-clip JSON judgments.
- `run/calibration_gate.json`: initial gate retained before production judging.
- `run/judge_session_receipts.json`: filtered actual session/model/image evidence;
  conversation text, reasoning and complete private transcripts are excluded.
- `run/results/`: unchanged first joined manifest, answers, status and leaderboard
  from the pre-review code. The first status predates manifest binding.
- `run/results_reviewed/`: final bound checkpoint, CLI receipts and N/A leaderboard
  after the two review fixes. It uses the same original judgments and failed gate;
  no video was regenerated or rejudged. Use this checkpoint with current report code.
- `run/verification.json`, `weights_verification.json`, `pytest.xml`,
  `pytest_reviewed.xml`: actual video audit, weight-hash audit, pre-review67 and
  post-fix83 passing regression cases.
- `run/judge_cli_result.json`, `run/report_cli_result.json`: real CLI results.
- `source_changes.patch`: benchmark/test changes relative to starting repository
  commit ac86e9bdd16222c872a6372a9ec2e7f35e1f3ef1.
- `REVIEW_RESOLUTION.md`: independent review findings, TDD fixes and remaining
  limits. The reviewed candidate and the fix commit are identified separately.

The failed calibration is part of the result. A missing validated JSONL is
intentional when the judge exits 2; the status sidecar retains parsed diagnostics.
Do not infer model correctness or a valid ranking from diagnostic counts.

## Verify a copied or extracted snapshot

```bash
python3 verify_snapshot.py
```

This verifies listed files by relative SHA256 paths and prints the actual
calibration status. It works after extraction to a different directory. It checks
retained bytes, not independent scientific validity.

Generation sidecars, original payloads and model manifests preserve original
absolute workstation paths rather than rewriting the inputs after judgment.
The source `audit_part_b.py` and assembler expect those run paths to exist.
Use relative snapshot verification for a relocated archive; an independent new
production experiment needs a new output directory and fresh judge sessions.
Never replace the original gate or judgments to make a replay appear calibrated.

No weights or full private session logs are included. Generation requires the
frozen checkpoint cache and Python dependencies identified in generator metadata.
The fixed grid protocol, native-duration difference, backend alias limitation
and lack of human truth labels remain scientific limits.
