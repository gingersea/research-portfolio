# Frozen pilot before production sampling

This document was written after successful 128x128, five-frame, one-step software
smokes, before any production sampling. The smoke outputs are not quality results.
The first failed smokes exposed missing CUDA autocast and a missing imageio ffmpeg
plugin. The runner now uses BF16 CUDA autocast and system ffmpeg. Original upstream
inference definitions remain unchanged and are checked against Git blob hashes.

## Fixed generation design

- Two authored prompts fixed before viewing production outputs: a rubber ball
  released above a floor, and a blue billiard ball rolling into a stationary red
  ball. Exact texts are in `run_vpt.py` and each per-arm `protocol.json`.
- One seed, 42. Identical BF16 RGB noise values, drawn once on CPU from FP32
  Gaussian noise, injected into the first prepare_latents call for each video.
  VPT auxiliary streams use the official independent seeds 44 and 45.
- Three arms per prompt: original Wan ordinary CFG 2.2, original Wan ordinary
  CFG 3.2, released VPT inner CFG 2.2 with flow/seg guidance both zero. The VPT
  expression is joint + 2.2*(joint-unconditional), equivalent to text CFG 3.2.
  The original 2.2 arm is a guidance diagnostic, not the matched comparator.
- Native 480x832, 81 frames, 16 FPS; 50 FlowMatch Euler steps, shift 5.0 in all
  arms. Shared official negative prompt. BF16 weights, CUDA autocast, CPU model
  offload, VAE tiling. Two GPUs execute independent arms, not distributed training.
- Six production videos total; no output-dependent prompt, seed or sample changes.
  Save every output as MP4 plus lossless uint8 frame arrays and SHA256 manifests.

## Method fidelity

VPT official repo `Tom-zgt/VPT` at
`3097bcd0267203a8192567d6eb2a9831b8015945`;
released adapter `zhengzhou/VPT` at
`edacaa9709af842a017361d4140fffac3b74fecc`;
weights SHA256 `fe9b57aded3514f819627d0fb25f4702f75fa0e28d74ac2bc5521fdc15fcd848`.
Use the unchanged official dual-stream transformer, triple timestep embedding,
loader and custom pipeline. The entry point extracts the inference AST definitions
to avoid training-only dependencies and expands the base transformer configuration
before the official loader replaces the full patch/projection/conditioning weights.
All 11 released full tensors must equal the loaded, dtype-cast tensors. All 480
LoRA tensors go through the official loader. This is an adapted inference trial.
No training, early checkpoint, or cross-step auto-guidance is reproduced.

This compares the released VPT bundle to original Wan under matched text strength;
it does not isolate role maps, flow, independent-time training or loss annealing.
The prompts do not establish controlled paired-object experimental assumptions.
Do not report a Principia or Physics-IQ benchmark score on these videos.

## Evaluation and stopping

Validate all six clips' decoded frame count/resolution/FPS, initial noise equality,
source/weight hashes and absence of corrupted frames. Extract native-frame contact
sheets for an explicitly unblinded qualitative inspection. Keep engineering success
separate from physical improvement. There is no new human annotation or calibrated
automatic physical judge in this pilot; no win rate, leaderboard or improvement
claim is warranted. The old Part B failed calibration gate remains failed.

Separately recompute VideoPhy2's SA>=4 AND PC>=4 joint threshold on its pinned
public human test CSV; include unequal-coverage and shared-prompt tables. Those
external human labels never apply to the six new clips. Large evaluator weights
were deferred after slow download; do not imply any learned evaluator was run.
