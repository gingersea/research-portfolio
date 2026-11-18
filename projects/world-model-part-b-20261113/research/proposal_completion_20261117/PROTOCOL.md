# Proposal completion execution, frozen before new scores

Execute original Tasks1-12 plus the November16 roadmap. Existing experiments,
labels and failed gates stay immutable. Completion of an experiment includes a
negative result, but failed scientific acceptance criteria are NEVER marked met.
Do not access gingersea or Google, alter1081, use paid APIs or publish materials.
The latest AGENTS account restriction supersedes the old plan's remote target;
prepare a reviewable local release and explicitly leave remote delivery unfulfilled.

## Work packages

1. Add fresh public-data inference to the unified CLI. DAVIS: rerun all30/650
   first-visible queries with the fixed native CoTracker3, one query at a time;
   no changes to coordinates/scorer/threshold. PStudio: same3 public examples,
   extend from64 to all150 frames; keep the previously selected8 IDs, querywise
   CoTracker3+same DA-V2 and native TAPIP3D. Real supplied K/static camera.
   Known selected data: engineering/full-length extension, not new unseen data.
   Report per-video raw3D EPE,2D error,visibility and all failures. No GT input
   beyond official initial queries; query3D uses ESTIMATED depth at query frame.

2. Finish an external-human-label physical-judging test with actual source videos.
   Fixed VideoPhy2 CSV revision already downloaded; select24 from existing URLs
   with pc>=4 or pc<=2,12 per class, from non-Google public source hosts. Sort
   each stratum by SHA256('proposal20261117:'+video_url). First4 each calibrate,
   remaining8 each evaluate. No replacements for unavailable videos. Resolve
   original URLs without sending private data. This balanced probe does not
   estimate generator population quality; use balanced accuracy/precision and
   recall, per-video error/abstention and explicit missing denominator.

3. Use locally executed Qwen2.5-VL-7B-Instruct, frozen HF revision
   cc594898137f460bfe9f0759e9844b3ce807cfb5, code/runtime receipts. All data stays
   local, no change to shared Ollama models. Fixed deterministic decoding. For
   VideoPhy2 use32 evenly spaced source frames (the author's evaluator's sampling
   count), images only without prompt/model/labels, ask pc1-5 and visible rule
   evidence. First calibration: all clear-violation references(pc<=2) must be
   flagged(score<4), all consistent controls(pc>=4) must avoid false violation.
   Missing/decode/schema failure invalidates gate. Then fixed test regardless of
   outcome, clearly diagnostic if calibration invalid. No rejudging to pass.
   This is a new sampled-video PC protocol, NOT retroactive validation of the
   original static8-frame Part B or a reproduction of VideoPhy-AutoEval.

4. Independently apply this new local visual model to all4 old I2V RGB packages,
   with the prior gravity applicability rubric, without old predictions/AI
   opinions/reference labels in the model context. Reuse all12 prior procedural
   gravity controls for THIS model's own calibration first, not the old GPT
   calibration result. All first replies/inputs retained, no per-case tuning.
   They are AI opinions, not human GT; protocol validity follows actual new gate.

5. Finish Task11 English report/rendered tables and a Chinese findings report,
   task-by-task requirements ledger, local release manifest/reproduction. Include
   Part A0/24 valid fits and original Part B invalidity as results, not numerical
   estimates or rankings. Literature source claims are rechecked in primary
   documents; do not repeat unverified novelty/open-versus-closed assertions.
   Final independent text/code review: local Qwen-family via Ollama, GPT is the
   producer, separate context/raw evidence and actual findings recorded. No
   assertion of independent human validation or clean-machine reproducibility.

## Analysis and stop rules

Keep all planned clips. Video/scene is unit, queries/frames and codec variants
are dependent repeated measurements. Report descriptive video-level CIs where
meaningful; n3 geometry stays individual/descriptive. Existing30 DAVIS baseline
exposure is disclosed. Calibration and evaluation identities remain disjoint for
the new human-PC probe. No training or local optimization. New generation is
conditional on valid measurement/judge gates; failed gates stop expansion.
If weights/media cannot be obtained, record attempts/checksums and exact missing
work; do not substitute unvalidated aliases, AI-created human labels or claim all
scientific goals complete. No new publication or forbidden-account operation.
