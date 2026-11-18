# Physics-consistency literature and method trials

## Current state (supersedes historical retrieval notes below)

FINAL: six production clips completed and audited (486frames). Both generation
sessions exited0, no remaining generation/download job. Native arrays, six all81
contact sheets and two comparisons retained. Unblinded inspection recorded in
REPORT.zh.md; no physical improvement rate claimed. Public-label reanalysis and
PNG/PDF/SVG chart complete. Eight base weights and both official source exports
freshly verified. Prior studies preservation replay passed (771PartB checksums,
269closeout and18stage0 checksums). Read final report and handoff receipt for the
completed state; remaining bullets below document intermediate history.

- Primary method sections read for eight papers: VideoPhy(v2), VideoPhy2(v1),
  PhyGenBench(v1), Physics-IQ(v1), Physics-IQ Verified(v1), Principia(v1),
  VPT(v1), LaMo(v1). Incomplete PDFs are not reading evidence; use primary HTML.
- Selective official VPT/VideoPhy2 exports completed and Git-blob verified.
  `source/{vpt,videophy}/SOURCE_MANIFEST.json` is in the external cache.
- VPT released weights complete and SHA256 verified. Published VideoPhy2 test
  annotations complete:3,397rows/600captions/197actions. All joint labels reproduced;
  unequal-coverage and382-common-prompt tables saved. Three `[nan]` rule fields
  explicitly missing. No learned evaluator has run.
- Overlay env_vpt reuses parent Torch without modifying it: torch2.9.0+cu128,
  transformers4.57.6, diffusers0.36.0, peft0.18.0, HFhub0.36.0, tokenizers0.22.2.
- Three one-step smokes passed; initial missing-autocast and imageio plugin failures
  resolved with CUDA BF16 autocast and system ffmpeg. Original upstream definitions
  unchanged. Smokes are engineering evidence, not generation quality results.
- Frozen `TRIAL_PROTOCOL.md`: six native81-frame videos, two prompts/seed42,
  matched RGB noise, three CFG arms. Production running:GPU1VPT session59703,
  GPU0base session35664. Logs:vpt_run.log/base_run.log. Do not duplicate jobs.
- Next: validate six outputs, inspect contact sheets, preserve provenance and
  write report. No new human labels, no calibrated physical ranking. Old gate failed.
- Full large-asset clone/PDF/HF-XET jobs listed below were canceled. VideoPhy2
  evaluator14.3GB download deferred after slow transfer, partial chunks retained.
  Do not leave generators/downloads running when finishing this task.

## Historical discovery notes (not live sessions)

Latest user request: read how physical-consistency papers perform their work,
investigate the methods, and actually try relevant methods. The prior closeout
and action-state pilot are complete; do not expand that toy pilot as a substitute.

Baseline working tree: /data/AI_Research/projects/world-model-part-b-20261113,
branch research/closeout-action-world-20261114, HEAD bf87025. Original published
main remains ac86e9b; original Part B snapshots are immutable. No new source or
experiment output from this request is committed yet.

Scope: bounded methods survey (roughly6-8 directly relevant primary papers,
not a systematic60-reference review), inspect full methods and official code,
then run at least one feasible official/released method with retained raw outputs,
strict provenance and honest distinctions between paper replication, adaptation,
public labeled evaluator checks, and unlabeled diagnostic scores. No paid API.

Skills: research-lookup evidence framework and scientific-critical-thinking.
Use built-in web/primary sources rather than the skill's paid Parallel pipeline
under the existing zero-paid-API constraint. Git-master loaded for git operations.
Do not install LSP; user previously declined. Do not upgrade the shared ML env.

Current live environment differs from the old snapshot: Python3.12.3,
torch2.9.0+cu128, transformers5.20.0, huggingface_hub1.33.0. Both RTX5090s
have32GB and are idle. More than3TB free on /data. Use a new isolated environment
for legacy evaluator/VPT compatibility if needed, preserving the shared environment.

## Candidate methods, not yet executed

- VideoPhy1/2: human-labeled semantic adherence and physical commonsense;
  VideoPhy2 also labels rules followed/violated/not-groundable. Official evaluator
  videophysics/videophy_2_auto is mPLUG-Owl-video/LLaMA based (not Qwen).
  Official inference uses32frames, greedy decoding and scores1..5. Need inspect
  custom source/requirements, pin weights/code, and assess on official public test
  labels before interpreting own unlabeled clips. Dataset: videophysics/videophy2_test.
- Principia: paired-object calibration-independent relations; real first-frame
  conditioning and SAM3 tracking. SAM3 is gated; do not imply this works directly
  on the old single-object T2V set or silently replace first-frame conditioning.
- VPT (arXiv2607.04653): existing Wan1.3B backbone + public adapter
  zhengzhou/VPT/pytorch_lora_weights.safetensors. Official code uses video/flow/role
  joint latents, decoupled timesteps and guidance. Need actual source before
  attempting; loading LoRA into an unmodified pipeline may not reproduce VPT.
  Candidate paired pilot with fixed prompts/seeds and matched settings.
- LaMo (arXiv2605.23878): CogVideoX latent motion prior + LoRA; weights not
  included in repo and links need checking. Training from scratch not assumed feasible.
- PhyGenBench: staged event/temporal/full-video checks, often paid VLM dependencies.
- Physics-IQ/Verified: conditioned real physical experiment continuation/reference
  metrics and improved masks/prompts; cannot pretend free T2V matches I2V setup.

## Retrieval state and exact running tool sessions

External scratch/cache (outside repository):
/data/AI_Research/shared/physics_methods_20261114/{upstream,papers}

Principia cloned successfully, exact SHA893e003804f0994ea6467b5a756ab657b82d4df9.
README and src/principia/{invariance_score.py,direction_consistency.py,scenarios/*.py}
are available. Other clones currently running (do not duplicate without checking):
videophy session94954; LaMo16758; VPT69354; physics-iq76427; PhyGenBench74422.
Their pack fetches were still pending after a few minutes; raw GitHub curl reads
DO work. They may simply contain large assets. Public URLs only, no tokens printed.

Raw README curls sessions94260,96728,50067,6021 finished successfully. README
facts above came from them and primary web sources. Prefer targeted reads now.

HF public API metadata lookup running: exec session57347, for
videophysics/videophy_2_auto and zhengzhou/VPT (files, sizes, SHA, gated status).
Poll before choosing model downloads.

PDF curl sessions (max180s+one retry, write only to external papers cache):
videophy2.pdf39681 (2503.06800); principia.pdf79326 (2609.04200);
VPT.pdf86734 (2607.04653); LaMo.pdf78785 (2605.23878);
physics_iq_verified.pdf54006 (2606.18943); videophy1.pdf33354 (2406.03520).
pdftotext is installed. Check completion and PDF validity before claiming full-text read.

## Progress ledger

1. Source discovery and resource audit: done. No new method execution yet.
2. Full-paper/code extraction and pinned evidence matrix: in progress.
3. Choose and freeze bounded executable trial(s): pending code/asset inspection.
4. Actually run, validate and aggregate; distinguish human test labels from own
   unlabeled video diagnostics: pending.
5. Research report with citations, method fidelity/deviations, raw evidence and
   next decision: pending. Do not end with only a plan or background jobs.
