# Full-length completeness correction (after first inference)

The full150-frame basketball example produced85 GT-visible post-query points
with CoTracker coordinates outside the image. The depth lift correctly marked
them unavailable; the old metric attempted to serialize NaN and produced a
case error. Original inference, partial result and error remain unchanged.

Scoring now preserves the full denominator: raw full-denominator3D EPE is null
when any required point has no valid3D prediction. A finite-subset diagnostic is
separately named and accompanied by valid fraction/invalid count. Macro rawEPE
stays null if a video is incomplete; the video is not silently removed. This is
an engineering/scoring-contract correction, not a tracking or depth optimization.
Fixtures reproduce invalid points and incomplete-video macro accounting.
Re-score the same frozen new predictions in a new output directory; no new
model inference or sample selection is needed. All original attempts retained.
