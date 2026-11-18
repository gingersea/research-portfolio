# Fixed supplementary policies and scientific limits

Before any model score, prespecify both native visibility>0.9 and the prior study's
conservative visibility*confidence>0.9. Show all30 video metrics for both; never
select a threshold or discard a video based on results. Network confidence remains
a network score, not independent identity evidence; the HUMAN trajectory/occlusion
labels are the external evaluator. All output coordinates are identical between
these policies. No manual annotation or AI judgement replaces published labels.

No new two-object recovery gate is forced onto arbitrary point tracks. This run
establishes the paper-data path and point/visibility baseline first, using official
OA/APD/AJ and post-query denominators. Apparent failures of object-center measures
in generated videos are not automatically failures of point tracking on DAVIS.

For a bounded real3D pipeline diagnostic, use the first3PStudio examples named in
the official TAPVid-3D notebook: basketball_5,softball_25,boxes_22. Fix the prefix to
64frames and the first8official queries whose query time is within that prefix.
Keep all selected points/frames, including occlusions, without score-based changes.
These3D labels are derived from calibrated multiview geometry, NOT described as
human pixel-by-pixel depth annotations. Use exact supplied intrinsics/static
camera, estimated DA-V2 depth for query3D and subsequent input; never give future
GT3D to inference. Compare CoTracker+same estimated depth and fullTAPIP3D, querywise.
Report raw metric3D errors and2D reprojection/occlusion diagnostics, NOT full
TAPVid-3D benchmark scores or proof of human annotated3D/physical accuracy. Freeze
selection before fetching labels/scoring. This auxiliary pilot is separate from
the complete30-video human-labeled DAVIS benchmark.
