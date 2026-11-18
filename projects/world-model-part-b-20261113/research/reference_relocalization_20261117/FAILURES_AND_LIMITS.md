# Preserved failures and scientific limits

## Asset and transport history

- Initial raw-GitHub/SharePoint requests failed with SSL EOF errors. Original
  attempts remain in `sources/receipt.json`. The successful official GitHub API
  files are in `source_api/`, and the SHA-pinned full archive has a separate
  `implementation_assets.json` with every original file hash.
- The original PointSt3R public weight link anonymously returned an expired-link
  page. Exact HTML is in `sharepoint_direct/folder.html`. No dynamic/visibility
  PointSt3R checkpoint was loaded, no expired-link workaround was attempted,
  and no author was contacted. This is a public MASt3R static-baseline experiment.
- Official Naver MASt3R acquisition succeeded. `checkpoint.json` preserves
  URL/ETag, precise bytes, per-range hashes, retries and whole PTH SHA256. The
  publisher did not provide a separately verified PTH checksum in this study.
  The HF safetensors checksum describes a different file and is not reused as
  an asserted checksum for this PTH.
- Initial12-way EgoPoints extension was slow. It was stopped after checking its
  own PID/command; all completed blocks and interrupted partials remain. A36-way
  continuation caused repeated connection resets and SSL EOF failures, including
  six exhausted attempts for a range. Its process was stopped; errors and
  partials remain in `acquisition_v2/` and the shared parts directory.
- `resume_small_ranges.py` resumes the same bytes with8 concurrent1MiB requests,
  checking206, exact Content-Range, ETag and length. It retains completed pieces
  and provenance. Statistical selection rules, participant exclusions and the
  maximum first3GiB window were not changed. Final selection/actual range count
  is recorded in `selection.json`, not inferred from the original target.
- Direct host/content-redirect probes failed; reusing an issued content URL also
  failed an integrity assertion on its second range request. These approaches
  were not used for scientific inputs. Probe receipts are retained. No global
  proxy,1081 tunnel, daemon or DNS configuration was changed.
- A web-tool attempt to open the primary paper/repository pages returned a
  network error. Method descriptions here rely on the preserved official code,
  model card/release metadata and primary sources obtained earlier; no fresh
  full-paper reading is claimed for the failed request.

## Implementation and reviewer history

- Six meaningful descriptor tests were run red before implementation/fixes and
  then green. The last-pixel fractional query fixture exposed an overstrict
  initial boundary check; the corrected version uses border clamping for
  in-image fractional positions. This happened before any new validation
  prediction and is covered by a literal expected-coordinate fixture.
- A real four-arm CLI fixture first rejected unsupported relocation arguments.
  The implemented relocation preserves original truth SHA256 and the fixture
  now verifies failure denominators, sparse unknown labels, absent-point false
  acceptance and strict return scoring. The prior analysis source is preserved
  as `analyze_before_relocation.py`; development numbers were unchanged.
- Original-author matching was checked on real features, then expanded to all
  development cases:12 clips,23 pairs,227 forward queries, all exact; repeated
  model predictions matched the previous development arrays. This checks
  implementation fidelity, not identity correctness or physical validity.
- Local GLM and Qwen independent implementations failed at actual runtime.
  Their reviews also contained false source claims and unsupported objections.
  A separate GLM writing/numerical check confused total accepted point-times
  with accepted return events. No independent comprehensive review pass is
  claimed. Raw requests, model digests, responses, usage, latency, failed code
  and individual rulings are preserved in the review directories and
  `REVIEW_RULINGS.md`.
- Upstream PyTorch autocast deprecation warnings and the pure-PyTorch RoPE
  fallback are preserved. The optional compiled RoPE extension was not built.
  No GPU execution failure is hidden by those warnings.
- `basedpyright` and `biome` LSP servers are unavailable and installation was
  previously declined. No LSP pass is claimed. Actual project-wide pytest
  result is130 passed in8.83s, with its raw log and receipt.
- `roma` distribution metadata is absent in the reused environment. The actual
  released model import/inference path succeeded without it; runtime metadata
  records null rather than inventing an installed version.

## Interpretation and remaining work

Final bounded new validation:3 clips/3 new participants(P10,P30,P35),1051frames,
27queries,42 known visible coordinate-valid point-times and12 known OOV labels;
there are no known occluded negative labels. Both native and descriptor inference
completed on all3 planned clips. Native raw coordinates are correct28/42 versus
MASt3R21/42, with zero MASt3R-only correct locations and7 native-only correct
locations. Native accepted precision is24/29 versus MASt3R-cycle21/41, including
19 wrong visible locations and1 OOV false acceptance for the latter. On12
annotated returns, native raw coordinates are correct8, MASt3R7; accepted
recoveries are6 versus7. The single added accepted recovery was already located
correctly by native; no new return coordinate is recovered, and one is lost.
This supports retaining the native baseline and declining a static MASt3R
replacement on the present evidence, not a general claim against PointSt3R.

The tool session associated with acquisition returned exit143 when polled after
completion. Its full3GiB archive, final selection and final source log were
already present; the independent full-file hash/prefix/source/input audit passed
and all new inference/analysis steps completed. The exit143 is retained rather
than reported as a clean0; its cause was not independently established. The
verified file evidence is the acquisition acceptance basis.

1. The12 prior clips/10participants are development only. The new selection is
   participant-disjoint within this project and deliberately return-enriched,
   not a representative random sample or proof of pretraining-data isolation.
   Fewer than6 participants within the fixed window remain an actual small pilot;
   criteria are not relaxed to improve the result. Each planned failure retains
   its full human-label denominator.
2. Sparse `vis_valids=0` frames are unknown, not occluded negatives. First-frame
   queries are excluded from scoring. Correctness is strict Euclidean error<4
   after separate x/y scaling to256, not4 original-video pixels. No dense
   identity correctness claim follows from sparse landmark agreement.
3. Development MASt3R-cycle accepts51 wrong visible-point locations. It recovers
   15/16 accepted returns versus native12/16, but both methods already have15/16
   correct return coordinates. All three added recoveries accept coordinates
   native already localized. All visible point-times show10 MASt3R-only correct
   locations versus35 native-only correct locations. Increased acceptance is
   not newly recovered coordinates.
4. The raw MASt3R arm forces matches, including on all21 known absent development
   point-times. Its lack of a calibrated visibility head is intentionally exposed,
   not silently repaired by a descriptor confidence. Reciprocity can still be
   consistent at a wrong location and does not prove identity.
5. Participant-bootstrap intervals are descriptive and fragile with few groups.
   Degenerate intervals are suppressed; zero observed false visibility is not
   a population safety guarantee. No significance, causality or official
   full-dataset leaderboard claim is made.
6. Current RGB storage uses the earlier shared OpenCV linear512x288 preparation,
   while the author example uses PIL LANCZOS and another default height. This
   is a controlled adapter comparison, not an exact official benchmark rerun.
7. Tracking false acceptance and a physical-violation false-positive rate are
   different endpoints. The present study does not close real-domain physical
   false-positive control, Part A SI-unit fitting, Part B judge calibration,
   reliable smallest-target measurement or independent labels for generated I2V.
   None of those missing endpoints is assigned a fabricated score.
8. The bundle actually replays copied-source/cache scoring and paired diagnostics.
   That check is not a fresh GPU inference rerun in an isolated environment.
   Old development RGB inputs are not recopied into the bundle; new validation
   inputs, source prefix and model assets are preserved with explicit hashes.

The next scientific priority is explicit visibility/identity evidence together
with actual coordinate gains, followed by a full 3D comparison with defensible
camera/depth inputs. A first-person moving camera must not be silently treated
as static, and merely rejecting more tracks cannot establish that gain.
