# Principia-inspired gravity calibration (fixed before rendering)

Authorization: user requested controlled paired-object relational measurement,
measurement and judge validation first, then larger generation only if valid.
This is a procedural calibration study, not the full Principia benchmark.

## Question, implementation and alternatives

Start with gravity only: identical-radius/material balls, one image plane,
different release heights, same annotated floor. The invariant is
q=(h_left/h_right)/(T_left/T_right)^2=1; Principia score compares height ratio
and squared flight-time ratio by1-|a-b|/(|a|+|b|). No SI acceleration claims.
Primary score preserves left/right object identity. The released native event
module independently min/max-normalizes both ratios; its score is a separate
diagnostic because this can lose the ordering and miss violations.
Pendulum adds length/projection/period assumptions; full SAM3/real-data inference
needs unavailable gated assets. Both are deferred. A colored-circle image tracker
is a deliberately limited calibration instrument, not a replacement SAM3 claim.

## Fixed sample and data separation

Four development scenes (seed71000..71003),24 held-out scenes(seed81000..81023).
Each scene has six paired conditions: equal gravity; one ball's gravity multiplied
by4; eight-frame release delay; pre-impact occlusion; moving camera; truncation
before either impact. Colors/height side/geometry vary across scene seeds.
96frames,30FPS,512x384 (truncated clips12frames). Both balls share a nominal
release at0 except the explicit asynchronous case. No bounce: only first fall.
Positive and anti-physics clips have matched appearance; no outcome text in pixels.
Drawn green fiducials allow camera stability checks. Renderer uses analytic
positions and records private ground truth; analyzer receives decoded video plus
public setup annotation only. Private truth is not imported by measurement code.

Every clip is encoded losslessly with FFV1 and with H264 CRF23; these encodings
are technical repeats, not additional independent scenes.28scene families x6
conditions x2encodings =336clips. All retained; no output-driven sample changes.

## Prespecified measurement

Track red/blue connected components; reject absent/ambiguous objects, unstable
area, camera fiducial movement>1.5px, incomplete contact, or release difference
>1.5frames. Floor/radius are public controlled-setup geometry; no true landing
time/acceleration/condition is available to the tracker. First observed center
within2px of floor-radius followed by three stable frames defines contact.
Fit quadratic precontact motion only to estimate release time and assess fit
residual, not to read renderer acceleration or construct a true impact time.
Reject residual RMS>1px, nonpositive curvature or too few moving observations.
Numerical definitions: exactly one color component of area>=40pixels per ball;
per-track maximum/minimum area<=1.25; >=8moving precontact observations, defined
as displacement>2px from initial center. Four consecutive contact-near observations
span three stable frame intervals with peak-to-peak y<=0.6px. Known setup floor
and radius are annotations, not truth dynamics. Fit-vertex release must lie between
-1.5frames and contact-8frames. For this controlled setup the balls start from
rest, acceleration is constant during the first fall, image scale is shared and
uniform. Vertex-based release is a model-based estimate, not observed proof of
zero initial velocity; applicability outside these conditions is unvalidated.
Use measured height and measured first-contact time minus fitted release time.
PASS if symmetric score>=0.90; FAIL if<=0.85; otherwise ABSTAIN. These are local
large-violation calibration thresholds, not Principia's published binary cutoffs.

Primary held-out gates, each encoding separately: all24equal-gravity clips PASS,
all24factor4 clips FAIL, all96assumption-invalid clips ABSTAIN. For admissible
clips: contact event error<=1.5frames; height error<=2px; score error<=0.06.
Contact error is against the first sampled contact ceil(continuous impact frame),
not the continuous event itself. Also require fitted release error<=1frame and
corrected flight-time error<=1.5frames. Missing numeric measurements in ANY of
the48truth-admissible clips per codec fail the gate, rather than being dropped.
All denominators retained; abstentions never count as physically correct.
Report maximum error and paired per-scene results; no model-ranking hypothesis
test, no treating frames/codecs as independent samples. Clopper-Pearson95%CIs
for per-condition scene success fractions, acknowledging conditional generator.
If a development repair is necessary, version it before evaluating held-out data.
No threshold changes after held-out results.

## VLM judge calibration

Freeze rubric before any judgment. Use the first two held-out scene seeds,
all6conditions, lossless decoded frames:12anonymous cases. Each gets the same
physical setup description, nominal FPS, and ALL native frames via chronological
contact sheets (24frames/sheet). No condition, truth label, filenames revealing
condition, measurements, original task history or expected class counts.
One fresh isolated agent session per video, fork_turns=none. Fixed model role
lazycodex-qa-executor, actual model/version alias and image-view receipts recorded.
No paid model API. Judges see only their assigned public payload and rubric;
the parent assembles calibration after all first judgments. No retries/tuning.
Return strict PASS/FAIL/ABSTAIN, observed release/contact evidence, reason.
Context isolation is not a filesystem sandbox. Instruct judges to read only their
assigned public payload/rubric/sheets, then verify actual file/image tool receipts;
if private files, measurement results, study code or other cases were read, mark
that judgment contaminated and the gate invalid. Do not replace that first result.

Judge gate: both positives PASS (false positives0/2), both factor4violations FAIL
(misses0/2), all8assumption-invalid cases ABSTAIN. Every obvious violation must
be correctly flagged. This is a small calibration gate, not general VLM accuracy.
Use three-valued target: asynchronous motion can be physically possible but is
inadmissible for this synchronized-release test. Missing evidence is not FAIL.

## Expansion and integrity boundary

Development correction v2, before held-out rendering/evaluation: the initial global
green-pixel centroid moved when H264 changed relative marker areas, falsely
rejecting all compressed development clips as camera motion. Preserve v1 source
and outputs. Locate exactly four green components(area>=30pixels) and average
their individual centroids with equal weights instead. The1.5px drift gate and
all other thresholds remain unchanged. This is an instrument correction based on
development data, not a held-out threshold relaxation.

Only if measurement and judge gates both pass may a bounded new generator pilot
be considered. A failed gate means no expanded generation or physical ranking.
Even passed procedural gates do not validate real-image tracking, SAM3 or all
physical laws. Public source is Principia arXiv2609.04200v1 and official repo
893e003804f0994ea6467b5a756ab657b82d4df9. Reuse unchanged official scoring helpers
as a numerical cross-check; separately report its event detector/release-origin
assumptions. Do not fabricate human labels or claim independent external GT:
procedural controls are known by construction under their analytic model.
