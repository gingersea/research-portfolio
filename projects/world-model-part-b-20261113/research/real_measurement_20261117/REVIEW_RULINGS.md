# Different-family design/code review and rulings

Actual local calls are retained in review_design/ (GLM) and review_code/ (Qwen),
with raw inputs, output, model family/digest, token counts and stop reason.
Production is GPT; neither call received the production conversation or prior
review. Both completed without truncation. No source uploads or paid APIs.

The GLM output contains three unsupported allegations:

1. It objects that precision includes accepted negatives. Reject: those false
   acceptances MUST be in precision's denominator; excluding them inflates it.
   Protocol and implementation both count all accepted known-label point-times.
2. It objects that ground-truth returned-point denominator is independent of
   predicted OOV. Reject: reference events must not depend on predictions.
   Strict recovery is separately evaluated using the prior predicted claim.
3. It claims normalization is fixed256 in both dimensions. Reject: the code
   explicitly multiplies by `[256/width,256/height]`.

The Qwen reviewer correctly computed the listed counts for the unknown-frame/OOV
fixture: positive1,negative1,accepted2,correct1,false-visible1,return1,recovered1,
strict-recovered0 (the output lists these fields). Its three code findings are
unsupported:

1. It objects to asserting confidence_accepted is a subset of native. Reject:
   the protocol explicitly defines a verification gate retaining coordinates.
2. It claims reverse query must be t,y,x. Reject: the actual frozen author's
   CoTrackerPredictor.forward documents **t,x,y**. The stored q is t,y,x, converted
   using q[[0,2,1]] for native inference; reverse_query=[0,x,y] is already native.
   reciprocal_accept receives XY positions, not three-coordinate queries.
3. It confuses storage rescaling with evaluation normalization. Reject: preparing
   original coords with512/originalW,288/originalH and then normalizing their
   errors with256/512,256/288 composes to256/originalW,256/originalH.
   The aspect/resize-invariance regression fixture checks this property directly.

No measurements were changed to satisfy these outputs. The new counterexample
fixture also demonstrates that a cycle-consistent wrong correspondence can pass
the reciprocal gate while remaining a false accepted location against human GT.
This is a scientific limitation, explicitly preserved in evaluation, not a claim
that reciprocity solves identity recovery. Actual data/inference/replay and
automated checks, rather than reviewer agreement, support the reported result.

The reviews do not establish runtime correctness, human physical labeling,
image inspection or comprehensive independent approval. Preserve that limitation.
Both calls predate the CUDA-cache correction. They do not cover its source;
the corrected12-case execution and7-case exact-array comparison provide the
actual evidence for that separate engineering change.
