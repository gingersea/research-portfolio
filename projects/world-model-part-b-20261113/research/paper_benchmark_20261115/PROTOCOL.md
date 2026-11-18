# Human-annotated paper benchmark: first-query TAP-Vid-DAVIS

All30 official TAP-Vid-DAVIS videos, no score-based exclusion or parameter search.
The published annotations are human-checked with optical-flow interpolation
assistance, not independent manual clicks on every frame. Frozen point IDs and
occlusion labels are ground truth for point tracking, not object-center/material
physics or metric3D truth. Official first-visible query convention; input RGB
256x256; normalized target XY multiplied by256, matching the official reader.
All post-query frames are evaluated; query/before-query frames excluded as the
official first protocol specifies. No frame truncation or synthetic replacement.

Official dataset zip generation1671476530685857,
MD5a7e786a1a5d26ba941b7c10bed21c20e,1668710491bytes. Official source revision
ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b. Preserve exact original source; execute
its unmodified metric and first-query function bodies with their NumPy dependency,
without installing TensorFlow-only readers. Record extraction hashes and verify
against hand-constructed fixtures. Preserve resize method and original arrays.

Primary native baseline is pinned official CoTracker3 offline FP32,6iterations,
fixed6x6support grid, backward_tracking=False. Run each evaluation query separately
so query results do not depend on other annotated query positions. Query frame is
native initialized; excluded from evaluation. Keep every failed attempt and its
video in accounting. No future GT coordinates/visibility enter model inference.

Primary endpoints: official occlusion accuracy(OA), average visible point accuracy
over1,2,4,8,16px(APD/average_pts_within_thresh), average Jaccard(AJ). Aggregate by
equal video mean over30, not by pooling frames. Supplementary descriptive event
recovery: >=2occluded frames then >=2visible frames after query, first declared
visible position within4px in that visible run; retain failures, report delays.
Also false-visible claims and accepted within4px coverage. Sequence is experiment
unit; point/frame observations are dependent, no frame-level confidence claims.

The prior two-object recovery gate requires object radii and competing initial
identities; DAVIS point IDs may lie on the same object and supply no such labels.
Do not invent object radii/IDs or report that gate as directly validated here.
Public benchmark usage is not proof that it was unseen by pretrained model
developers. Frozen release replication, not an unseen private generalization test.

Any estimated-depth/TAPIP3D extension uses the same first queries and video subset
selected BEFORE scores, records depth/camera assumptions, and reports only2D
metrics when3D GT is absent. No relation/gravity/physical score is enabled.
