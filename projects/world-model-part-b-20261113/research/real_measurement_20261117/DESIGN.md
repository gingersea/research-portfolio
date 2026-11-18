# Real-domain measurement priority (2026-11-17)

User authorization: choose the next direction and continue; optimize the
measurement instrument, not generators. Existing failures/results stay frozen.

Priority1 is external point measurement and false acceptance with coverage.
Priority2 is a separately redesigned physical-judge calibration. Depth/3D and
new generator expansion wait for evidence that 2D correspondence is useful.
This does not claim that surface-point labels validate object-center physics.

## Question and candidate data

Can the frozen native CoTracker3 and the already-frozen DAVIS confidence gate
provide accurate accepted correspondences on previously untested real videos,
especially when points leave view and return? EgoPoints (WACV2025) is the first
candidate: expert manual point/visibility/out-of-view labels and specific ReID
evaluation. Availability, annotation sparsity and source-license boundaries
must be inspected before selecting a sample. No outcome-based substitutions.

Read the original dataloader/scorer; unlabeled frames must NEVER be scored as
occluded. Participant/source-video grouping must prevent split leakage. ReID
correctness and false in-view/visible claims must be distinguished.

The first comparison will include native claims and v*c>0.995 frozen in the
November15 DAVIS experiment. An optional reciprocal-query check will be decided
and frozen after understanding data/model interfaces, before viewing new scores.
It tests correspondence self-consistency using only RGB and original queries,
not future truth. No training or search on evaluation outcomes.

## Deliverables and stop condition

Produce a frozen sample/denominator protocol, an actually executed external
baseline comparison, per-video failures and precision/coverage/ReID outputs,
reproducible source/runtime/input receipts, different-family review with honest
coverage, and a Chinese report plus local handoff. If the original data cannot
be obtained within reasonable attempts, preserve attempts and move to the next
documented non-Google source; do not invent labels or spend the round only on
downloads. Conclude this round when the predeclared comparison and its audit
are complete, irrespective of whether a scientific improvement is found.

No paid API, no Google/gingersea connection, no1081 changes or publication.
