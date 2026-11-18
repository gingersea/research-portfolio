# Different-family review and A/B comparison

Producer: GPT. Independent local contexts: GLM-4.7-Flash and Qwen3-32B.
Requests, original replies, model digests, usage and latency are preserved in
`review_matcher/` and `review_protocol/`. No paid or external inference.

- GLM correctly calculated the two literal forward-match fixtures. Its proposed
  sampler uses an invalid 3D grid for 4D input and incorrectly invents a half-pixel
  normalization in the supplied author sampler. Actual execution failed. Its
  discussion of `samp.py` containing argmax is also false: argmax is in the author
  evaluation script. No implementation or review pass follows.
- Qwen returned agreement statements as "concrete_findings". Its assertion that
  inference_mode could be a protocol deviation is unfounded; it was already in
  the frozen native baseline. The protocol explicitly rejects degenerate zero
  false-positive resamples as population bounds. These objections are rejected.
- Qwen lacked the acquisition script and therefore could not establish the
  provenance of the excluded participant list. That is a real coverage limit
  of its review input, not proof of selection leakage. The preparer asserts each
  exclusion; the final audit must independently recompute prior-vs-new participant
  disjointness from both manifests and report actual IDs.
- Qwen's independent sampler allocates Nx2 output for NxD feature vectors.
  `audit_reviews.py` runs it verbatim against the preserved real 24D descriptors;
  the exception/result is preserved rather than repaired on the model's behalf.
- Producer implementation is checked with meaningful literal tests and the
  original author's PIPs sampler plus per-query cosine implementation on real
  model features (`matching_audit/`). This is deterministic evidence, not a
  replacement claim of successful independent family review.

These few samples show concrete limitations of both local reviewers on this
task. They do not rank model families generally. The two alternative
implementations are preserved; neither is silently used in production. Final
scientific claims require actual outcomes and denominator audits, not consensus.

The separate GLM numerical/writing check in `review_interpretation/` also failed:
it substituted all accepted point-times138/165 for accepted return events12/15.
Its opening "15/16" applies only to return coordinates, not all visible-point
coordinate accuracy. Its wording that the new method added no correct coordinates
is too broad: new return coordinates=0, while all-visible pairings include10
MASt3R-only correct locations and35 native-only correct locations. These omissions
are rejected against the raw count JSON. The alternative prose is retained as
a failed same-topic writing comparison, not copied into the report. The request
occurred before any new participant predictions; it was not a review of new
validation results.

The final Qwen check in `review_new_results/` read the new raw JSON and the
diagnostic note independently. It correctly extracted28/42 vs21/42 coordinates,
zero new return coordinates, one lost return coordinate,19 wrong accepted
locations and one false-visible point. However, even with explicit field
definitions it substituted total accepts29/41 for correct accepted returns6/7.
Its assertion that the P10 result shows the note "mixing sparse labels and
physical violations" is unsupported: the note explicitly separates those
endpoints. Its P35/small-sample statement agrees with an already-present limit
and is not a newly found discrepancy. No significance test supports any implied
statistical reading of its alternative wording. The raw alternative is retained
and not adopted; this narrow numerical check also failed overall. No further
model calls were used to seek agreement. Final conclusions use the verified
counts, source comparison, participant audit and copied-cache replay.
