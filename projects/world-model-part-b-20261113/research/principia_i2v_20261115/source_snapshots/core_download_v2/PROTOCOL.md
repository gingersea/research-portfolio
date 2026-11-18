# Native I2V feasibility pilot: fixed two setups x two seeds

User authorization: continue the agreed native-I2V pilot, validate initial-state
preservation and tracking coverage before expanding; learn from existing papers.
Prior procedural measurement/judge gates passed, but are not natural-image
validation. Old snapshots remain immutable. No paid inference API or public upload.

## Published methods and model choice

Principia(arXiv2609.04200v1): condition on a specified initial frame, preserve
object identity in hL/hR versus(TL/TR)^2, check basic motion before physics scoring.
Physics-IQ Verified(arXiv2606.18943v1): describe setup/action/camera without giving
the expected outcome; retain per-case denominators and camera/artifact failures.
LTX-Video(arXiv2501.00103): native jointly trained I2V/T2V model; use unchanged
Diffusers0.36.0 LTXImageToVideoPipeline, not a patched T2V model or pasted first frame.
Pinned core release:Lightricks/LTX-Video-0.9.5 at
e58e28c39631af4d1468ee57a853764e11c1d37e. Selected for existing32GB hardware and
available compatible inference stack, not for observed physics quality.
Existing cached T5-v1.1-XXL text encoder may be reused as an explicit component:
verify architecture, local weight hashes and predefined upstream numerical
sentinels before inference; do not assert full byte equivalence from sparse checks.
Native model/core inference remains unchanged; component provenance is reported.

## Fixed inputs and outputs (before production generation)

Two procedural controlled initial states at768x512: two equal-radius32px balls,
same material, x=224/544, floor_y=464, colors red/blue. Drop heights160/288px,
swapped between initial states A/B. Four green fiducials support camera checks.
These are synthetic setup images, not Principia real recordings or AI-edited photos.
No image-generation API, inferred mass labels or hidden outcome text.
Use seeds42and43 per setup, four videos total.97frames at24FPS,50Euler steps,
guidance3, decode_timestep0.0, no output selection, prompt enhancement or retries
for model quality. Save actual generated first frame; never replace it with input.
All outputs stored as native uint8 frame arrays and MP4. Software smoke outputs
use reduced settings and are excluded. Failed attempts remain recorded separately.

Shared prompt specifies same-material balls initially at rest, simultaneous
release, same floor/image plane, fixed-camera real-time scientific demonstration,
and only the existing setup. It does not specify acceleration, arrival ordering,
flight-time ratio, rebound count, or desired physics score.
Negative prompt fixed:"worst quality, blurry, jittery, distorted".

## Input and instrument checks

Save exact geometry/masks/setup and input hashes. Before interpreting generated
videos, test the frozen prior image-only measurement on known same-gravity and
factor4anti-physics controls rendered at these new dimensions/geometry/FPS.
Use the old frozen measurement unchanged; reject failures rather than retune it.
Also retain native-VAE first-image roundtrips as an encoder/decoder diagnostic,
separate from generated first-frame preservation. No VAE diagnostic is a generated
video or a guarantee about subsequent dynamics.

## Prespecified gates and denominators

1. Initial objects: both expected color components unique, complete and visible;
   each center error<=0.25input radius(8px), equivalent radius ratio0.75..1.25,
   mask IoU>=0.60against the authored input mask. Pixel MAE is diagnostic only.
2. Initial setup: all four fiducials detected; each position error<=3px. Report
   object preservation and apparatus/fiducial preservation separately.
3. Coverage: frame-wise both-object usable-component fraction>=0.95, no missing
   run>2frames; unique components must be untruncated and area within0.75..1.25
   of input component area. No gap filling or interpolation. This is instrument
   availability, not verified tracking accuracy on generated images.
4. Physics applicability: use the unchanged prior measurement's camera, area,
   release, event and fit gates. Static hovering can have high coverage and still
   be unmeasurable. Only eligible cases get the directed relation score. Missing
   measurements remain N/A, never physical failure or success by default.

Expansion requires all four outputs pass initial-object, initial-setup and
coverage gates, plus at least one admissible initial-fall measurement per setup.
Otherwise stop expansion and report the exact failing gate/cases. No threshold,
seed, initial-image, model or guidance tuning after viewing production outputs.
Four clips are2setups with2seed repeats; frames are not independent samples.
Descriptive feasibility only: no significance, ranking, new human labels or claim
that procedural gate success establishes real-image physics accuracy.
