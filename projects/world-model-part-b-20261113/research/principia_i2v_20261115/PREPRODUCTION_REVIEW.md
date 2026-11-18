# Independent preproduction review

Scope: read-only review of protocol, input preparation, frozen image-only measurement, installed native Diffusers pipeline, and local model configuration. No production outputs were inspected, no thresholds were tuned, no study implementation was changed. This report is the sole review artifact. Model downloads were incomplete at inspection, so this is not a runtime or full weight-integrity approval.

## Verified native conditioning and API

Installed `pipeline_ltx_image2video.py`: lines 541–557 VAE-encode the supplied image and populate the first latent time slice; line 832 gives conditioned tokens zero timestep; lines 878–882 exclude that slice from scheduler updates; lines 906–937 decode the complete latent video with the VAE. This is native first-image conditioning, not an output-frame pixel paste. Actual generated first-frame preservation still must be measured: the VAE is lossy and the local 0.9.5 config has `decoder_causal=false`, so a single-image VAE roundtrip does not establish generated-video first-frame identity.

No material static API/config incompatibility found. Explicit `LTXImageToVideoPipeline.from_pretrained(...)` is needed because local model_index.json names LTXPipeline. Pass `frame_rate=24` explicitly: native call default is 25 and frame_rate affects temporal positional scaling, independently of exported MP4 FPS. Width768/height512 satisfy divisibility32; frames97 satisfies 8k+1 with configured temporal compression8. `decode_timestep=0.0` is supported. The installed VAE supports LTXVideo095DownBlock3D; the local text encoder d_model4096 matches transformer caption_channels4096. These checks do not replace an actual complete-checkpoint load and inference smoke.

## Material measurement findings and required boundaries

1. Frozen measure.py lines40–58 reduces four fiducials to their mean position. Symmetric zoom/rotation or opposing marker movement can evade its camera check. First-frame individual fiducial agreement alone does not close this temporal blind spot. Before interpreting physics, retain individual spatially matched marker trajectories as a separate fixed applicability diagnostic. Do not modify the frozen metric.
2. The initial-setup gate validates fiducials, not the floor. Frozen contact_y always equals authored floor_y-radius (line62); a changed floor or changed effective ball radius can invalidate the assumed contact geometry. Independently assess floor-line preservation and actual contact against the authored geometry before calling a clip an admissible initial fall.
3. Frozen contact detection (line69) is a four-frame plateau satisfying y>=contact_y-2, with no upper bound. It can select late settling after a bounce or a plateau below the expected floor; it is not a direct first-impact detector. Monotonic initial descent/no earlier rebound and observed contact geometry need a separate prespecified applicability check. Otherwise report the frozen score only as an exploratory plateau-based relation. Failed applicability means N/A, not physical FAIL. This frozen instrument also rejects ordinary continuing bounces without a plateau; its scope is narrower than general falling-ball physics.
4. Coverage must use all97 expected frames: >=93 both-object-usable frames, no >2-frame run where either object is unusable. Decode truncation must not shorten the denominator. Coverage counts component availability, not identity/centroid accuracy. The frozen metric additionally demands every frame detectable and area(max)/area(min)<=1.25; therefore coverage PASS does not imply physical measurability.
5. Report attempted production denominator4, per-setup denominator2, including API/decode failures. Keep initial-object, fiducial/setup, coverage, applicability, relation availability and conclusive verdict separate. `borderline_relation` returns ABSTAIN but still has a fitted relation score, so relation availability must not be defined as verdict!=ABSTAIN. Only applicable measurable clips contribute to a conditional relation summary; always display its denominator alongside all4 attempts.

Parent has stated that additional wrapper diagnostics will be frozen before production. Their implementation and calibration have not been reviewed here; this report does not claim those gaps are already closed.

## Directly reproduced control evidence

Scenario: replay each original lossless control with unchanged frozen measure at the new geometry/FPS; check frozen source hash against heldout_v2 manifest. Invocation from /home/room115:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/AI_Research/shared/physics_methods_20261114/env_vpt/bin/python - <<'CHECK'
import importlib.util,json,hashlib
from pathlib import Path
p=Path('/data/AI_Research/projects/world-model-part-b-20261113/research')
src=p/'principia_gravity_20261114/measure.py'
spec=importlib.util.spec_from_file_location('frozen',src); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
manifest=json.loads((p/'principia_gravity_20261114/source_snapshots/heldout_v2/manifest.json').read_text())
print(json.dumps({'scenario':'frozen-source-integrity','pass':hashlib.sha256(src.read_bytes()).hexdigest()==manifest['source_sha256']['measure.py']}))
for case in ['A','B']:
 d=p/'principia_i2v_20261115/inputs'/case
 setup=json.loads((d/'setup.json').read_text());setup.update(fps=24,frames=97)
 for condition in ['equal','unequal']:
  r=m.measure_video(d/f'instrument_{condition}.mkv',setup)
  print(json.dumps({'scenario':case+'_'+condition,'expected':'PASS' if condition=='equal' else 'FAIL', 'observed':r['verdict'],'reason':r['reason'],'score':r.get('score'),'frames':r['decoded_frames']}))
CHECK
```

Captured stdout from direct execution (exit0):

```jsonl
{"scenario": "frozen-source-integrity", "pass": true}
{"scenario": "A_equal", "expected": "PASS", "observed": "PASS", "reason": "measured_relation", "score": 0.967848888884054, "frames": 97}
{"scenario": "A_unequal", "expected": "FAIL", "observed": "FAIL", "reason": "measured_relation", "score": 0.4421103890842226, "frames": 97}
{"scenario": "B_equal", "expected": "PASS", "observed": "PASS", "reason": "measured_relation", "score": 0.9678488888840541, "frames": 97}
{"scenario": "B_unequal", "expected": "FAIL", "observed": "FAIL", "reason": "measured_relation", "score": 0.4020748801387357, "frames": 97}
```

Binary observables: source hash equality=true and expected==observed for all four controls. Source/video paths are explicit in invocation. This checks synthetic instrument behavior only, not accuracy on model-generated images, natural imagery, false-positive rates or physical validity of production outputs.

Static source inspection invocation: `sed -n '502,567p' <pipeline>; sed -n '756,945p' <pipeline>` plus reading local model_index, transformer/config, vae/config and text_encoder/config under `/data/AI_Research/shared/principia_i2v_20261115/ltx_095`. Binary observables: image enters encoded first latent=true; preserved first latent participates in complete VAE decode=true; pixel first-frame paste in native path=false; planned geometry satisfies source shape constraints=true. Exact inspected pipeline path and hashes follow.

## Inspected source hashes

- native_pipeline: `/data/AI_Research/shared/physics_methods_20261114/env_vpt/lib/python3.12/site-packages/diffusers/pipelines/ltx/pipeline_ltx_image2video.py` SHA256 `3b09e577af5a802e8b089244ef688b21f5db5d75cecb02b61cbb15b359fde163`

- frozen_measure: `/data/AI_Research/projects/world-model-part-b-20261113/research/principia_gravity_20261114/measure.py` SHA256 `2d21c0843919139fc5c7835842dd9568d96c080f6991dd744bf1f9bace686fa5`

- protocol_at_review: `/data/AI_Research/projects/world-model-part-b-20261113/research/principia_i2v_20261115/PROTOCOL.md` SHA256 `d30dae129c930bccd39a1a1ec1f715fcbc77665c6da5d977bd6b4d44c05481a5`

Attempt-directory discovery (`omo-agent-toolkit ulw-loop status --json`) failed with EACCES spawning /home/room115/.local/bin/omo-ulw-loop. Parent explicitly assigned this sole report path; parent should include it in the main run evidence bundle. No external/network requests were made. Paper claims were not independently reverified in this scoped review.
