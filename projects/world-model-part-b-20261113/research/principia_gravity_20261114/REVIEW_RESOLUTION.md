# Design review and decisions before held-out evaluation

Independent read-only reviewer: fresh agent `principia_design_audit`, no history
fork. Review covered the written protocol and two official source files; it did
not certify implementation, held-out data or judge success. Parent reproduced the
source counterexamples in `official_crosscheck.json` before held-out execution.

1. Ordering loss: official `ratio()` min/max-normalizes height/time independently.
   Heights(2,1), gravities(4,1) give native score~1, identity-preserving score0.4.
   Directed primary was already in the measurement code; clarified protocol and
   retained native diagnostic separately. This does not invalidate every paper
   result or allege that the authors used this counterexample.
2. Release origin and contact: documented constant-acceleration, rest, common-scale
   assumptions; fitted vertex is an estimate. Added numeric release/duration error
   gates. Official strict center>threshold needs an interior contact threshold,
   not blindly the physical ground line. Unchanged source checks preserve both
   behaviors and demonstrate common pre-release delay sensitivity.
3. Definitions and denominators: froze color component/area/moving observation/
   stable-contact definitions. Every admissible assigned case belongs to the
   numeric-error denominator; missing measurement fails rather than disappearing.
4. Blindness: separate public payloads, anonymous IDs, first-result-only rule,
   actual tool/image receipt audit required. Shared filesystem remains a limitation;
   no claim of OS sandboxing or independent human labels.

Development v1 failed H264 camera detection, while lossless passed. The issue was
pixel-area weighting across stationary fiducials; v2 uses equal component-centroid
weights. Thresholds did not change. v1 source/outputs retained; v2 development gate
passed both encodings on all4scene families. Held-out data had not been rendered
or evaluated at this correction. The source snapshot `heldout_v2` records the
subsequently frozen measurement, renderer and protocol. The rubric is bound
separately by the original twelve per-case input manifests, not by that snapshot.

## Final implementation review and hardening

The same independent read-only reviewer verified the frozen sources, gate
recomputation, all12judgment/payload/rubric hashes, and actual raw-session tool/
image events (50custom calls,42image hashes). No literal outside-input path or
private-label read was found; this is recorded-call verification, not OS isolation.

Two reusable-gate defects were exposed with in-memory altered inputs: changing a
judgment reason while retaining its old receipt, and repeating one manifest case
12times, could still pass the initial gate. The actual archived inputs were intact.
Preserved initial gate/source in `judge_gate.json` and `source_snapshots/judge_gate_v1`.
Hardened the gate to bind first-result/payload/image/rubric hashes, exact unique
2scene×6condition assignments, model/role, frame coverage and each condition's
2/2successes. No judgment, label, measurement threshold or input was changed.
The stronger gate is `judge_gate_v2.json`; it also passes all12 original cases.

Parent ran13targeted pytest checks. Independent reviewer reran the former exploits
through the complete gate in memory: modified judgment/payload fail, duplicate
assignments reject, unchanged inputs reproduce v2 exactly. Its default scientific
interpreter lacked pytest, so independent pytest execution is not claimed.
This resolves the two material gate findings. It does not validate natural-image
tracking, subtle gravity deviations, other physics phenomena or generator rankings.
