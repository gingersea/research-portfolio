# Controller recovery state

## Authoritative latest state (supersedes historical steps below)

All56 videos, all56 independent judgments, all448 exact sampled PNGs are DONE.
No generator or judge work remains. Both GPU processes finished their complete
28/28 manifests; both old exec wrappers returned143, not a proven clean shell
exit. Fresh audit passed all56 video hashes/3640 decoded frames/native settings
and exact pixel equality of448 samples with source. Complete actual receipts
verify56 distinct child UUIDs, gpt-6-astra/high, fork:none and viewed grid bytes.
Real full-production CLI judge returned2 (calibration invalid) and report returned0
with48 production records and N/A valid rates. Do NOT overwrite initial gate or
rerun the immutable judge checkpoint. Full suite is67 passed in7.60s after code
commit e4c6eb9; generation/data/English+Chinese reports committed as60ddfb2.

The ONE final fresh read-only reviewer finished review of exact head60ddfb2523f657a692753a8d1936d78f60df5197 from baseac86e9b.
Both Important findings were reproduced RED→GREEN and fixed in one pass atd5dca1b:
manifest-bound checkpoints/report revalidation and explicit boolean control
eligibility. Final suite83 passed in8.17s. No re-review is required or running.
The final actual CLI replay is run/results_reviewed (judge2/report0), preserving
all744 original run files and the same judgments/gate/board byte-for-byte or
structurally equal as appropriate. REVIEW_RESOLUTION.md records the evidence,
all declined judgments/rulings and the retained patch-format whitespace minor.
Final packaging/extracted-copy proof is delivered separately under
/data/AI_Research/shared/handoff/part_b_20261113T114040Z. Main remainsac86e9b and is untouched.
No merge/push is required for current experiment-report request. Human labels,
verified calibration references/negative controls, FPR control, smallest-target
measurement, valid PartA parameters and full English manuscript remain OPEN.

The remaining text is historical execution recovery detail, not pending work.

User requested execution of proposal's remaining experiments, prioritizing real Part B generation, calibration first, isolated blinded per-clip judging, strict CLI validation and leaderboard. Do not end the turn with pending production work. A failed scientific gate is an honest result; complete the real production matrix/diagnostic judgments and report invalid scores as N/A.

Active worktree: `/data/AI_Research/projects/world-model-part-b-20261113`, branch `research/part-b-20261113`, starting published main ac86e9b. Code validation commit f2e13bf; suite 63 passed in 7.24s. Published main is untouched.

Generation jobs already running (DO NOT duplicate/restart while running):

- Unified exec session 6443: Wan on physical GPU 0.
- Unified exec session 12172: CogVideoX on physical GPU 1.
- Both use `generate_part_b.py`, output `research/part_b_20261113T114040Z/run`.
- Logs: `generation_wan.log`, `generation_cog.log` in study directory.
- Two jobs run all four calibration prompts seed0 first, then eight production prompts seeds0/1/2. 28 videos each, total56. Each video takes about180–205s; full run about1.5h. Both GPUs nearly100% utilized. No reduced settings.
- Frozen cache: `/home/room115/jhy/world-model-physical-consistency/.worktrees/video-bench/benchmark/physcheck/models_cache`. `weights_verification.json`: all13 large upstream weight hashes freshly passed.

Watch `run/private/{wan2.1_1.3b,cogvideox_5b}.json` for completed clips. Do not include `control_eligibility.json` in the per-model manifest loop. A ready clip has status=ok, video_id, is_calibration and grid_path. Run/judgments/{video_id}.json means its isolated judgment completed. Missing judgment may be an active child: check collaboration.list_agents before re-dispatching.

Judge protocol:

- Local Claude auth is API-key only, so no Claude paid API calls under zero-API-spend spec. Use registered `lazycodex-qa-executor` role, fork_turns=none, one new agent for ONE video. This role is fixed gpt-6-astra/high; model snapshot beyond alias is undisclosed and must be noted.
- Original fixed payload is `run/blind/frames/{vid}/payload.json`, image `grid.png`, optional individual frames in same directory.
- Shared safe brief: `run/blind/JUDGE_BRIEF.md`. Agent may read ONLY brief, that payload and those images. It must view the image with view_image, not infer from prompt. No identities, private manifest, expected labels or other judgments.
- Actual canonical task name MUST be `/root/judge_{vid}` (task_name argument `judge_{vid}`); assembler uses that true unique session identifier. No reusing an agent/session for a second clip.
- Write ONLY `run/judgments/{vid}.json` using payload schema. No extra raw metadata in that file. Prompt: “Judge exactly one opaque clip {vid}. Follow [full safe brief path]. Allowed reads only brief and this clip directory's payload.json/grid.png/frame_*.png. Inspect with view_image. Write only [judgment path] in payload schema. Do not inspect identities, metadata or other judgments. You are not alone; preserve others' files. Stop after this one visually grounded judgment; return output and viewed image.”
- Maximum3 child slots alongside root. Dispatch independent ready clips concurrently. All calibration eight must be judged and gate recorded BEFORE any production judgment; production may generate in parallel.
- First7 completed calibration judgments reported no unambiguous violations. First Wan floating-person control actually shows feet on ground (root also viewed); do not force labels or confuse control-generation failure with judge false negatives.

Next:

1. Finish eight calibration judgments; inspect actual control outputs. Record image eligibility in `run/private/control_eligibility.json` keyed opaqueID with {eligible: bool, reason: str}. This is controller visual assessment, NOT human ground truth. Unknown/dynamics-only controls are ineligible; uphill constant-speed cannot be certified under static-frame-only protocol. Keep all8 controls, don't silently drop/relabel/rejudge until desired pass.
2. Execute `assemble_part_b.py --phase calibration`. It records gate and does not require production judgments. A failed gate is expected given generated controls often don't enact violation.
3. Continue generating all48 production clips; judge every ready production clip once in new isolated agents (after calibration gate). Even if gate fails, retain as unvalidated diagnostics; no valid ranking.
4. After all56 generation/jobs and judge files finish: `audit_part_b.py` verifies video/frame/matrix hashes; `assemble_part_b.py --phase production` invokes actual `physcheck judge --manifest...` (return2 expected on failed gate) and actual `physcheck report --part b`. Invalid gate writes `.status.json`, NO validated JSONL, and leaderboard JSON/Markdown with valid rates None/N/A plus separate diagnostic counts.
5. `summarize_part_b.py` creates artifact-derived English REPORT.md and hashes. Add Chinese handoff/report, update live ledger and proposal gap status accurately, retain all video/grid/raw judgments/gate/results and backend limits. No human labels or new Part A measurements fabricated.
6. Final regression/end-to-end validation and one fresh registered code reviewer at final exact commit per executing-plans skill. Fix important findings by tests. Commit artifacts on research branch per plan; don't merge/push shared main without a separate need. Current user asked report after completion, not new publication.

Important pre-existing protocol issues now addressed in new production path: opaque-ID calibration mapping, full coverage/duplicate rejection, embedded-ID mismatch, sampled original frame labels, fixed configured judge identity, independent session names, bool-not-number, calibration failure preserved, `--part` respected. User-visible open tasks remain human labels, false-positive control, smallest-target measurement, true calibrated validity, English full manuscript.

## Later recovery update

All8 calibration judgments and controller visual eligibility review are complete.
`run/calibration_gate.json` was written before first production judge and is valid=false.
DO NOT rerun/overwrite that file now; audit checks its timestamp precedes production.
Production generation/judging is underway; run `status_part_b.py` for authoritative
counts and ready IDs rather than trusting a past count. Last exported session receipts
covered33 completed judgments; more are expected. Always use new agents per clip.

Actual model/session/image evidence is now available, not merely inferred names:

`python3 research/part_b_20261113T114040Z/export_judge_receipts.py --parent-log /home/room115/.codex/sessions/2026/11/09/rollout-2026-11-09T22-19-49-01a1c0ad-fad7-7aa3-9ca1-8511cc3eaeb8.jsonl --child-log-dir /home/room115/.codex/sessions/2026/11/13 --partial`

Use without --partial at the end (asserts56). It exports only safe metadata, verifies
fork:none, actual gpt-6-astra/high, child UUID uniqueness, actual image events and byte
identity with each archived grid. Final assembler now REQUIRES those receipts and uses
observed UUIDs; regenerate final receipt file BEFORE production assembly. The old
calibration-gate session names map via receipt.agent_path to those same actual UUIDs.

An additional real CLI happy-path regression was added after commit f2e13bf; targeted
suite16 passed, full suite expected64 but must actually run final and report real result.
Study entry points now include status_part_b.py and export_judge_receipts.py in addition
to generator, assembler, audit and renderer. Sampler grid aspect warp and native-duration
differences are documented as retained limitations; don't alter judge inputs mid-run.
