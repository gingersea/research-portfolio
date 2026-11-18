"""Video generation pipeline for physcheck.

Loads open-weight T2V models via diffusers, generates videos from prompts,
and writes each video with a metadata sidecar for full traceability.
"""

from __future__ import annotations

import json
import hashlib
import time
from functools import lru_cache
from importlib.metadata import version
import subprocess
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from physcheck.config import Config
from physcheck.generate.models import MODEL_REGISTRY


def _get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _pipeline_cls(registry_entry: dict):
    from diffusers import CogVideoXPipeline, WanPipeline

    cls_name = registry_entry["pipeline_cls"]
    return {"WanPipeline": WanPipeline, "CogVideoXPipeline": CogVideoXPipeline}[cls_name]


def source_hash() -> str:
    """Identify the actual source, including uncommitted implementation changes."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if any(p in ("artifacts", "models_cache", "__pycache__") for p in path.relative_to(root).parts):
            continue
        if path.suffix in (".py", ".json"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _load_pipeline(model_key: str, models_dir: str, gpu_id: int):
    entry = MODEL_REGISTRY[model_key]
    local = Path(models_dir) / entry["hf_repo"]
    source = str(local) if (local / "model_index.json").exists() else entry["hf_repo"]
    pipe = _pipeline_cls(entry).from_pretrained(
        source, dtype=getattr(torch, entry["torch_dtype"]), cache_dir=models_dir,
        local_files_only=source == str(local),
    )
    if hasattr(pipe, "vae"):
        # Wan's reference inference uses float32 VAE decoding.
        if model_key.startswith("wan"):
            pipe.vae.to(dtype=torch.float32)
        pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipe


def generate_video(
    model_key: str, prompt: str, seed: int, cfg: Config, prompt_id: str | None = None,
    *, num_frames: int | None = None,
) -> dict:
    """Generate one video and write mp4 + metadata sidecar.

    Returns a metadata dict with keys: video_path, metadata.
    The metadata records model, prompt_id, seed, fps, frames, duration_s,
    resolution, and the git commit of the generating code.
    """
    entry = MODEL_REGISTRY[model_key]
    out_dir = cfg.artifacts_dir / "videos" / model_key / (prompt_id or "adhoc")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"seed_{seed}.mp4"
    meta_path = out_dir / f"seed_{seed}.json"
    request = {
        "model": model_key, "prompt_id": prompt_id, "prompt": prompt, "seed": seed,
        "fps": entry["fps"], "frames": num_frames if num_frames is not None else entry["num_frames"],
        "resolution": [entry["width"], entry["height"]],
        "num_inference_steps": entry.get("num_inference_steps", 50),
        "guidance_scale": entry.get("guidance_scale", 5.0),
    }
    if video_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if any(metadata.get(k) != v for k, v in request.items()):
            raise ValueError(f"cached generation request mismatch: {meta_path}")
        if metadata.get("video_sha256") == file_hash(video_path):
            return {"video_path": str(video_path), "metadata": metadata}

    start = time.monotonic()
    pipe = _load_pipeline(model_key, str(cfg.models_cache_dir), cfg.gpu_id)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    frames = pipe(
        prompt=prompt,
        num_frames=request["frames"],
        width=entry["width"],
        height=entry["height"],
        generator=generator,
        num_inference_steps=request["num_inference_steps"],
        guidance_scale=request["guidance_scale"],
    ).frames[0]
    arr = np.stack([np.asarray(frame) for frame in frames])
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    if arr.shape != (request["frames"], entry["height"], entry["width"], 3):
        raise ValueError(f"unexpected generated frame shape: {arr.shape}")
    temporary = out_dir / f"seed_{seed}.partial.mp4"
    iio.imwrite(temporary, arr, fps=entry["fps"], codec="libx264")
    temporary.replace(video_path)
    metadata = {
        **request,
        "duration_s": len(arr) / entry["fps"],
        "commit": _get_commit(),
        "source_sha256": source_hash(),
        "video_sha256": file_hash(video_path),
        "generation_seconds": time.monotonic() - start,
        "hf_repo": entry["hf_repo"],
        "versions": {name: version(name) for name in ("torch", "diffusers", "transformers", "imageio")},
    }
    temporary_meta = meta_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_meta.replace(meta_path)
    return {"video_path": str(video_path), "metadata": metadata}


def generate_all(
    prompt_set_key: str,
    cfg: Config,
    models: list[str],
    seeds: list[int] | None = None,
) -> list[dict]:
    """Generate videos for every prompt in a prompt set, model, and seed.

    Idempotent: skips (video, sidecar) pairs that already exist.
    prompt_set_key is one of {"part_a", "part_b", "calibration"}.
    """
    import importlib.resources as pkg_resources

    prompt_file = {
        "part_a": "part_a.json",
        "part_b": "part_b.json",
        "calibration": "calibration.json",
    }[prompt_set_key]
    prompt_data = json.loads(
        pkg_resources.files("physcheck.prompts").joinpath(prompt_file).read_text()
    )
    prompts = prompt_data["prompts"]
    seeds = seeds if seeds is not None else cfg.seed_grid
    results = []
    for entry in prompts:
        n_seeds = entry.get("n_seeds", len(seeds))
        use_seeds = seeds[:n_seeds]
        for model_key in models:
            for seed in use_seeds:
                result = generate_video(
                    model_key, entry["text"], seed, cfg, prompt_id=entry["id"]
                )
                result["prompt_kind"] = entry["kind"]
                results.append(result)
                print(
                    f"[generate] {model_key} {entry['id']} seed={seed} -> "
                    f"{Path(result['video_path']).name}"
                )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_key", choices=MODEL_REGISTRY.keys())
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load()
    result = generate_video(args.model_key, args.prompt, args.seed, cfg, num_frames=args.num_frames)
    print(json.dumps(result["metadata"], indent=2))
