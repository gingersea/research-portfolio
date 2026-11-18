# Retained attempts and remaining limitations

## Acquisition and tooling

- Original author tar is59.48GB; a32MiB range took58.90s. Full acquisition was
  replaced before new inference/scores by a bounded prefix pilot. Original
  protocol and V2/V3 acquisition amendments remain, with exact sample freeze.
- Author Dropbox archive-preview tree returned415. Its public folder metadata
  remains in sources/dropbox.html and decoded prefetch binary files.
- Detached signed download URLs returned403; altered signed query parameters
  also failed. Following the original public redirect chain succeeded.
- A direct no-proxy request to www.dropbox.com timed out connecting. No tunnel
  or global proxy/DNS settings were modified. No Google service was contacted.
- The first parallel32MiB downloader was terminated as our own job before any
  predictions, after several minutes with no new complete part. The earlier
  complete32MiB probe remains; failed source versions are kept as *_v1/v2/v3.py.
  Acquisition changed to streamed8MiB ranges with unique original-link requests,
  per-range Content-Range/ETag validation, retained partial attempts and progress.
- V3 allows stopping once the exact same first12 complete distinct-source cases
  have been identified, avoiding unused prefix bytes. No performance outcome
  enters this stopping rule. Extra completed and partial blocks are retained.
- LSP basedpyright is unavailable, previously declined; no installation or LSP
  pass claimed. Software evidence consists of actual pytest, API smoke and replay.

## Measurement and interpretation

- Unknown vis_valids frames cannot be scored as invisible. Sparse OOV→return
  labels do not establish dense absence/recovery intervals.
- Reciprocal verification may accept the wrong identity, as the explicit test
  demonstrates. It leaves all coordinates unchanged; precision improvements
  through rejection are not tracking/re-identification improvements.
- Source-video de-duplication does not make clips from one participant fully
  independent. Bootstrap uses participant groups and reports scope/valid draws.
- Archive-prefix and length-restricted selection is not a uniform sample of the
  full benchmark, nor a direct test of this project's four generated I2V clips.
- Two local different-family reviewers produced unsupported allegations. Their
  raw output and rulings remain; no broad independent approval is claimed.
- The two-identical-frame CoTracker smoke error was0.11349px. This checks the
  two-frame API and coordinate convention only, not real correspondence accuracy.
- Existing authored human point labels are not human physical-consistency labels
  of our generator outputs. Old Part A/B failed gates remain unfulfilled.
- Native source/model hashes are inherited from frozen assets. No training,
  model replacement, test-set threshold search, remote push or publication.

Actual per-case execution failures, if any, remain in predictions/*.json and
all scientific denominators in analysis/; this document does not imply success.

## Observed execution correction and scientific limits

- First12-case native run:7success,5CUDA OOM. All first records remain. Fresh
  identical1083-frame/default200 one-query inference fit in27.68GB, implicating
  repeated-call cache/allocation lifecycle rather than an unavoidable single-query
  size limit. Chunk64 was only a diagnostic probe, not the selected production fix.
- A pre-score amendment added only empty_cache before each full native query.
  All12 uniform reruns succeeded; all arrays of all7 prior successes are exactly
  equal, including finite/missing patterns and acceptance gates. Source/parameter
  bindings and differences remain; this is an execution correction, not a tracking
  accuracy improvement or selective replacement of failed cases.
- First-attempt scores retain all12 denominators in analysis_first_attempt/;
  main analysis/ uses predictions_memory/ explicitly. Label scoring started only
  after the resource variant and comparison were fixed. No threshold was tuned.
- Reciprocal empirical precision95.83% is accompanied by a90.20%–100% descriptive
  interval,2completely abstained clips and3sub95% nonempty clips. Recovery fell
  from12/16 to4/16; this is not reliable identity recovery or production validation.
- All-zero false-visible bootstrap quantiles are degenerate;0/21 is an observed
  count only. STATISTICAL_NOTE.md states why no population-zero safety claim follows.
- The experiment-design workflow source Scientific Agent Skills is cited in the
  report, with current arXiv metadata verified. It is not an accuracy guarantee.
