# Recovery and real-I2V observation/depth diagnostic

Previous observation_gate/tracking_validation data are development only and immutable.
Root cause: the previous gate rejects unsupported proposed coordinates but never
moves them or retrieves a target after it reappears elsewhere. A separate derived
recovery candidate can search current RGB against immutable initial templates.
No future GT, depth, network visibility or confidence enters that search.

Fixed recovery rule: prior primary template aperture .5, NCC>=.9, competing initial
identity margin>=.15; spatial runner-up NCC gap>=.08 outside one initial radius;
two consecutive consistent candidates (distance<=one initial radius) to confirm a
new location after loss. Raw tracker coordinates remain separate. If an identical
initial appearance pair loses identity observability, latch unknown as before.
No template updates from potentially mistaken self-predictions. Existing raw
coordinates may be used only when independently supported and uniquely matched.
This is a fixed-scale restricted RGB retrieval baseline, not XMem reproduction.

New controls: original10 conditions + long_gap_reappearance + lookalike_distractor,
4seeds 9101(development),19101/29101/39101(heldout), 33frames384x256. Novel textures
per seed/object; same-appearance twins remain identical. Occlusion width varies
by new seed; long-gap return time varies; exact duplicated lookalike distractor
appears after initialization. Videos must be pixel-unique and not repeat old70.
Freeze source/input/model receipts before inference; no heldout tuning. Compare
native CoTracker3, previous independent gate and separate recovered coordinates.
Centers are primary, exclude frame0. Count correct accepted(PCK3px), coverage,
false-visible, wrong-other, unsupported unknown identity, and event recovery delay.
Reappearance events: previously invisible>=2frames then identifiable visible>=2;
an event recovers only after a correct accepted observation. Keep failed events
in denominator. Production needs per-condition>=90% correct retention, >=99%
joint accepted precision and no unsupported ambiguous-identity claims. Sequence,
not frame, is experimental unit; n=3seed replicates per heldout condition.

RealI2V uses all4 existing LTX generations,97frames512x768. Before any new tracker
output is displayed, inspect raw RGB frame0 and prespecified frames12,24,...,96
with full intervening RGB available. Independent AI visual reference records
visible/absent/uncertain and identifiable/unknown separately, approximate image
center + uncertainty where meaningful. It is AI annotation, NOT human ground
truth. No tracker, gate, segmentation output or depth is used to create reference.
An object which deforms, merges or changes identity may be non-localizable or
unknown; do not force coordinates. Sampled-reference results are diagnostic only,
not dense97frame calibrated accuracy. Human CSV must remain pending until supplied.
No reuse of reference coordinates for model initialization except allowed frame0.
For center references report distance to reference with its uncertainty; distinguish
surface query tracking from full-object center under deformation.

Depth: official Depth Anything V2 Metric Hypersim Small, code
a561b849ebae10a6f5ef49e26c83cbbcd36c71bf, HF revision
3bc65d4e14a6786a61acec16453c50e12bf5f338, checkpoint SHA256
b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545.
Native FP32 infer_image, input_size518,max_depth20, no per-frame GT alignment.
Use one development-frame scalar only as an explicitly labeled sensitivity
diagnostic if evaluating metric-scale drift; primary raw metric errors remain.
Test controlled sensor GT and static-camera I2V background temporal variation.
Real I2V has no depth/camera calibration GT: finite depth, temporal consistency
or algebraic reprojection alone cannot establish physical scale or world motion.
Any assumed intrinsics/static extrinsics are diagnostic hypotheses. Do not score
physical relations, gravity, or imply independent 3D accuracy validation.
