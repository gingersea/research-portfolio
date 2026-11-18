# Part B execution ledger

Plan: docs/superpowers/plans/2026-10-16-video-physical-consistency.md
Spec: docs/superpowers/specs/2026-10-16-video-physical-consistency-design.md
Starting commit: ac86e9bdd16222c872a6372a9ec2e7f35e1f3ef1.

## Proposal gaps before this run

- Tasks 1-6: infrastructure and 24 Part A clips exist, but the scientific physical-parameter target remains unmet (0/24 valid fits).
- Task 7: sampler and calibration prompts exist; generated calibration videos and completed calibration experiment missing.
- Task 8: fixed template/parser/gate exist; no real isolated calibrated judging. Parser does not enforce payload IDs, frame membership, or a single judge version; calibration keys do not match opaque video IDs.
- Task 9: CLI exists; judge currently keys calibration by prompt IDs rather than per-video opaque IDs; `report --part` is parsed but not used.
- Task 10: no completed 48-clip production matrix, calibrated validated judge JSONL, or Part B leaderboard.
- Task 11: complete English research report with Part A/B tables remains unfinished.
- Task 12: weather cleanup and result upload already completed; next Part B result snapshot remains pending.
- README open tasks: human labels, reliable smallest-target measurement, false-positive control and calibrated Part B judging remain open.

## Scope fixed for this run

Two existing frozen generators; 8 production prompts x seeds 0/1/2 x 2 models = 48 clips.
Calibration: 4 existing prompts x seed 0 x 2 models = 8 clips, allowed by Task 10 (at least one seed per prompt).
Original 50-step settings and model-native frames/resolutions remain fixed. Generate calibration first within each GPU job. Eight evenly sampled frames per clip, frozen original template, isolated judge per opaque clip ID.

Ruling: work in a separate research branch/worktree to preserve the published snapshot; main is not modified.
Ruling: local Claude authentication is API-key only. Under the spec's zero API-spend constraint use the current platform's isolated vision sessions, record actual backend/version, and never label these as Claude outputs.
Ruling: calibration labels are hypotheses from prompts until actual images support them. An ambiguous or dynamics-only control cannot be declared a true known-positive by prompt text alone. Record gate failure without coaching the judge or fabricating a pass.
Ruling: model identities and calibration expectations stay in private manifests; judges receive only opaque ID, fixed instructions, sampled images and sampled frame indices. No cross-video context.
Ruling: if calibration fails, retain all production artifacts and diagnostic judgments, but valid leaderboard rates/ranking must be unavailable. Do not silently drop failed controls or rejudge until a desired answer appears.

## Live tasks

1. Complete proposal audit and generation setup: complete; published-source baseline 48 tests passed.
2. Generate and sample all 56 clips: complete; 48 production+8 controls, 3640 decoded frames, 448 sampled PNGs. Fresh audit confirms native settings and exact sampled pixel equality with decoded source.
3. Judge calibration first in independent sessions and evaluate gate: all8 isolated judgments complete; gate executed before any production judgment, valid=false. All8 expected positive flags absent; image review cannot certify these generated clips as known positives. Preserve failure, not a calibrated pass.
4. Judge production, validate schemas/coverage, aggregate only with explicit gate status: complete; all56 actual independent sessions and image-byte receipts verified. Real judge CLI exits2 on invalid calibration; real report exits0 with48 production records and N/A valid rates. No valid ranking or validated JSONL is fabricated.
5. Regression/end-to-end checks and review fix pass complete:83/83 tests, real judge2/report0. Final archive and extracted-copy verification receipt is delivered separately under /data/AI_Research/shared/handoff/part_b_20261113T114040Z/. No video, judgment or review job remains running.

## Code verification and observed controls

Commit f2e13bf: strict opaque-ID manifest joins, exact coverage, unique per-clip sessions,
one configured judge identity, frame-index membership, numeric schema checks and explicit
calibration failure status. CLI `report --part` now respects the selected part; failed
gates produce a clearly invalid N/A leaderboard with separate diagnostic counts.
TDD: initial production module missing; report contract failed before implementation;
boolean-number regression failed before the strict guard. Final suite: 63 passed in 7.24s.

Judge routing: the registered `lazycodex-qa-executor` role uses gpt-6-astra/high.
All successful judge calls use this same role with fork_turns=none; each opaque clip
has a unique `/root/judge_<video_id>` task. The harness exposes a model alias rather
than a dated model snapshot; this limitation must be disclosed in the report.
The first request to an unregistered worker role was blocked before execution and
reissued with the registered QA role; no extra judgment was produced.

Actual first control v_772567b42b93601624f2: prompt asks floating person, grid shows
feet on terrain at all sampled frames. The isolated judge reported no visible violation.
Controller visually confirmed that observation; the generated prompt is not a known-positive
physics reference. Do not relabel this as judge failure or force a violation to pass.

Fixed judge inputs: original JUDGE_TEMPLATE plus the same one-clip visible-evidence
instructions. First two invitations included that brief inline; later calls read its
frozen `run/blind/JUDGE_BRIEF.md` equivalent. No model identities, source prompts,
expected control categories, or other judgments are supplied to judges.

## Session evidence upgrade (without changing judgments)

Filtered local rollout records now confirm actual gpt-6-astra/high metadata,
fork_turns=none for every registered invocation, distinct child UUIDs and real
input_image events. Decoded image-input bytes match each archived grid's SHA256.
No full conversation, reasoning text, encrypted prompt value or credentials are
exported. `export_judge_receipts.py` writes only this experiment's evidence fields.

The initial calibration gate used actual canonical tool task names as session IDs.
Final raw records use the observed rollout UUIDs, with agent_path retained to map
back to those original task names. No judge was rerun or coached. The original
`run/calibration_gate.json` is NOT overwritten: its mtime precedes all production
judgment files and is part of the execution-order check.

Root sampler limitation retained for this frozen protocol: grids resize native
frames to320x180 cells without aspect-ratio preservation. Original sampled PNGs
are archived and available to each judge. Native durations also differ by model.
These are additional reasons not to interpret uncalibrated counts as true physics
error rates or a duration-matched causal model comparison.

Checkpoint preservation regression: an initial successful CLI output followed by
a failed run at the same path overwrote its status while leaving the old JSONL.
The real CLI regression reproduced this before the fix. Production validation
now refuses either existing output/status path before any checkpoint write, so
distinct experiments cannot silently overwrite retained gate evidence.

Strict-schema regression: the legacy parser coerces numeric notes and null
descriptions to strings. Two production tests failed before the new check;
the manifest validation path now rejects these nontext fields rather than
normalizing invalid raw evidence. Legacy parsing remains unchanged.

Commit e4c6eb9 contains the checkpoint preservation, strict text-field checks,
the real CLI success-path regression and corrected workflow documentation.
Full regression suite after these changes: 67 passed in 7.60s; JUnit evidence
retained in pytest.xml. No LSP server is installed; the user previously declined
installation, so validation uses real Python tests and CLI runs.

Generation supervision observation: polling the old Wan exec wrapper (session
6443) returned exit143 with no stdout. Independently, its atomic completion
receipt reports expected28/ok28/records28, its log ends with the final successful
multi_b8 seed2 clip, the generator process is absent and GPU0 has released memory.
No successful shell exit is claimed from that wrapper. Final acceptance uses
the full matrix, retained per-clip metadata and fresh video decode/hash audit.

The Cog wrapper (session12172) likewise returned143 when polled after completion;
its independent completion receipt is also expected28/ok28/records28. Both GPUs
are now idle and released model memory. These observations are distinguished
from the per-video generation and decoding results rather than reported as
successful wrapper process exits.

## Final independent review and one fix pass

Fresh reviewer of exact head60ddfb2 returned two Important findings: H1 stale
status/manifest certification and H2 fail-open control eligibility. Regraded both
as Important by their effect on a reusable scientific checkpoint. Original report
and its SHA256, all rulings and costs are retained in REVIEW_RESOLUTION.md.

Final: fixed H1 — changed-manifest, missing/extra/duplicate rows, nonboolean and
inconsistent gates, missing status regressions RED→GREEN; checkpoint binds exact
canonical manifest and report revalidates saved rows. Final: fixed H2 — omitted,
string, numeric and null eligibility regressions RED→GREEN; only explicit boolean
true qualifies. Added typed manifest/checkpoint boundaries. One fix pass committed
as d5dca1b; final suite83/83 in8.17s, pytest_reviewed.xml. No re-review dispatched.

Final: Ruling: individual classification correctness stays unverified — no human
ground truth and scientific calibration invalid — cost: unknown false-positive
and false-negative rates; no valid model ranking.
Final: Ruling: receipt consistency is the available provenance boundary — private
backend authentication and dated snapshot unavailable — cost: independent backend
authentication and exact provider snapshot replay remain unavailable.
Final: Ruling: retain legacy PartA/transfer/smallest-target/human-label/FPR work as
open — user prioritizes this PartB run — cost: no valid g/T or generated detector accuracy.
Final: Ruling: API adapters/full paper remain open — not claimed in this run —
cost: those proposal deliverables are still incomplete.
Final: Ruling: mtimes are not cryptographic chronology — controller observed early
calibration and kept the initial gate — cost: no independent signed execution-order proof.
Final: minor (deferred): embedded unified-patch context whitespace is retained as
proper patch syntax; git diff --check may flag that artifact, not runnable-source behavior.

Post-fix real CLI replay writes a NEW run/results_reviewed root. Actual judge
return2/report return0,48 production rows, calibration false. Original744 run
files retain identical SHA256; wrappers, first judgments, gate and board match
the first checkpoint exactly. No images or judgments were regenerated/coached.
