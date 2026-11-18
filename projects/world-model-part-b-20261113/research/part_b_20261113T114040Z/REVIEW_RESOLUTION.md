# Independent review and resolution

Reviewed candidate: `60ddfb2523f657a692753a8d1936d78f60df5197`, against
`ac86e9bdd16222c872a6372a9ec2e7f35e1f3ef1`. One fresh read-only reviewer
returned REQUEST_CHANGES for two Important findings. The actual failed-calibration
research result was independently supported; the reusable production path needed
the fixes below. No second reviewer was dispatched.

Original full report remains at
`/home/room115/.omo/evidence/part-b-20261113-code-review.md`, SHA256
`450564f6a777551471eb4aca863df9f5be5ff2bd18d605d9b125e53518ef23f9`.

## Important findings fixed in d5dca1b

1. A saved successful status could certify a changed or partial manifest. Status
   now binds the canonical manifest SHA256. Reporting rejects mismatched manifests,
   missing/extra/duplicate records, nonboolean gates, inconsistent gate judgments,
   changed judge identities and missing strict-production status. Persisted rows
   are revalidated before aggregation. Manifest/checkpoint boundaries have explicit
   Python TypedDict definitions; runtime guards establish the behavior.
2. Missing or malformed control eligibility could pass. Positive eligibility now
   requires explicit boolean true; omission is ineligible, and nonboolean values
   are rejected. Positive fixtures explicitly establish eligibility.

Fifteen new adversarial cases failed before the fixes; an additional missing-status
report case reproduced the legacy fallback before that guard was added. All16
now pass. Full suite: **83 passed in8.17s**, recorded in `pytest_reviewed.xml`.
The independent pre-fix suite was67 passed in8.17s. No lint/typechecker or LSP
success is claimed; LSP is unavailable and installation was previously declined.

The final real CLI replay uses the same56 first judgments and manifests, writing
new evidence only under `run/results_reviewed/`. Judge exits2 and report exits0;
the final board remains invalid/N/A. Byte checks confirmed all744 original run
files unchanged, identical raw answer wrappers, identical parsed judgments/gate
and identical diagnostic/N/A leaderboard. This is a validation replay, not a
new judging or generation experiment.

## Executor rulings and their costs

| Decision, in execution order | Reason | Cost or remaining uncertainty |
|---|---|---|
| Preserve published main in a separate worktree | Keep the November12 receipt intact | New results live on a local research branch until separately published |
| Use platform QA instead of API-only Claude | Respect zero paid API calls | Backend differs from design; dated provider snapshot unavailable |
| Treat prompt controls as hypotheses | Generated images may not enact the requested contradiction | No eligible known-positive set or calibrated pass from this run |
| Give judges only their opaque clip and fixed inputs | Preserve per-video blindness | Read scope is instruction-enforced, without an OS sandbox |
| Keep diagnostics but withhold valid rates after gate failure | Preserve observations without false certification | No scientific model ranking |

The reviewer explicitly declined the following judgments; the executor accepts
each boundary rather than treating it as verified:

- Individual VLM physical correctness: no human truth labels; classifications
  remain diagnostic, with unknown false positives/negatives.
- Independent authentication of private backend internals/transcripts and exact
  dated model snapshot: only filtered receipt/artifact consistency is verified;
  independent backend authentication and exact snapshot replay remain unavailable.
- Re-solving Part A scale/transfer/smallest-target validity and human labels/FPR:
  retained as open work under the user's Part B priority; valid g/T and generated
  detector accuracy remain unavailable.
- API adapters and complete English paper: explicitly incomplete; this delivery
  supplies the Part B experiment report and protocol-independent offline format.
- Cryptographic proof of historical order from portable mtimes: not claimed.
  The controller observed calibration first and retained its early gate, while
  archival mtimes alone cannot independently establish that chronology.

Minor retained: unified-patch context lines in `source_changes.patch` can trigger
`git diff --check` whitespace warnings. Keep the patch's proper context syntax;
no runtime or blanket lint-clean claim is inferred from that artifact.

Remaining scientific work: independently human-verified observable positive and
negative controls, revised frozen calibration, human generated-video labels,
false-positive control, reliable smallest-target measurement, valid Part A
physical estimates and the full English manuscript. The failed gate is not a
0% judge-accuracy estimate or evidence that either generator is physically correct.
