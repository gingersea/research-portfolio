# Verified runtime — 2026-11-06

Working tree: `.worktrees/video-bench`, branch `feat/video-bench`, base commit
`8f2abb7c66e2157eecd77d7df7a174fbb47a9194`.

## Hardware and dependencies

- Two NVIDIA RTX 5090 GPUs, 32 GB each; CUDA matrix multiplication verified.
- Python 3.12, PyTorch 2.9.0+cu128, CUDA runtime 12.8.
- diffusers 0.41.0, transformers 5.20.0, accelerate 1.13.0.
- numpy 2.5.3, scipy 1.18.1, opencv-python-headless 4.13.0.92.
- imageio 2.37.3, PyAV 17.0.1, safetensors 0.8.0,
  huggingface-hub 1.33.0, pytest 9.0.3.
- Existing `pytest.ini` disables the unrelated zarr pytest plugin.

## Local weights and inference

Local checkpoint directories:

```text
benchmark/physcheck/models_cache/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/
benchmark/physcheck/models_cache/zai-org/CogVideoX-5b/
```

The previous Wan download contained two truncated text-encoder shards. They
were resumed with range-checked HTTPS downloads, then verified against the
upstream Hugging Face SHA256 digests. The complete manifests, including
upstream revision IDs, are retained under `benchmark/results/raw/`.
Fresh downloads can use `hf download REPO --revision REVISION --local-dir PATH`;
use the revision in the manifest and verify its recorded hashes before inference.

Inference uses one GPU per process with model CPU offload, bf16 model weights,
VAE tiling, and a float32 Wan VAE. Seeds use a CPU torch generator. Model
registry defaults remain 50 denoising steps and guidance scale 5.0;
Wan uses 81 frames at 832×480/16 fps, CogVideoX 49 frames at 720×480/8 fps.
All settings, package versions, generation time, video hash, base commit and
actual source hash are written to the per-video JSON sidecar.

The Wan smoke test generated 33 frames at 832×480 in 62.69 seconds, with a
decodable MP4 and matching metadata. It is separate from production samples.

```bash
PYTHONPATH=benchmark python3 -m pytest -q
PYTHONPATH=benchmark python3 -m physcheck.detect.validate
PYTHONPATH=benchmark HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
  python3 -m physcheck.detect.run_part_a --models wan2.1_1.3b cogvideox_5b
PYTHONPATH=benchmark python3 -m physcheck.report.aggregate
PYTHONPATH=benchmark python3 benchmark/verify_part_a.py
```

For independent GPU jobs, set `PHYSCHECK_GPU_ID` and a distinct
`PHYSCHECK_RESULTS_DIR` per process, then run the combined command above to
merge/reanalyse cached videos. Concurrent processes must not write the same
results JSONL. A completed, request-matching, hash-matching video is reused;
incomplete output is regenerated. Valid metadata describing a different
request raises instead of silently reusing the wrong experiment.

## Measurement protocol at the halfway milestone

- The fixed Part A target is a red ball. Red segmentation avoids first-frame
  subtraction ghosts and excludes pendulum strings. Non-red/absent targets
  and unexplained disappearance inside the image are detection failures.
- Pixel scale is conditional on the prompt's stated 0.1 m ball diameter.
  Elongation/motion blur and substantial apparent-size changes invalidate
  scale calibration; they are not assigned a precise gravity estimate.
- The physical clock is the generated video's nominal FPS. SI estimates are
  conditional on this clock and scale convention, not independently measured
  real-world dimensions of a generated scene.
- Fit thresholds remain R² >= 0.6 and RMS residual <= 3 px. Projectile fits
  check both axes. Pendulum estimates require >=0.9 fitted cycles and relative
  standard error <=25%. Horizontal motion compares quadratic and continuous
  move-then-stop fits to avoid dilution by a stationary tail.
- Fit standard errors describe conditional regression uncertainty only; they
  do not include camera, scale, segmentation, or generation uncertainty.
- Failure records stay in the denominator. Detection/calibration/fit failure
  does not by itself establish a physical violation, and fit success does not
  establish physical correctness. No speed error is invented for prompts
  that do not state a numerical initial speed.

Known deferred minor: malformed legacy JSON sidecars require manual repair;
new sidecars use atomic writes. Part B sampling/judging and final paper claims
are outside the accepted halfway milestone.
