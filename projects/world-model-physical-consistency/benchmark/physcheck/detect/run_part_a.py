"""Part A production driver.

For each model × Part A prompt × seed: generate the video, track the
object, fit the physics parameters, and write one JSONL line per video.
Failures (track/fit) are recorded as data, never silently skipped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from physcheck.config import Config
from physcheck.detect.fit import G_TRUE, compute_pixels_per_meter, fit_trajectory
from physcheck.detect.tracker import track_object
from physcheck.detect.validate import validate_detector
from physcheck.generate.pipeline import generate_video, source_hash, _get_commit


def _expected_value(kind: str, calib: dict) -> dict:
    """Physically expected parameter values per kind."""
    if kind == "free_fall":
        return {"g": G_TRUE}
    if kind == "projectile":
        return {"g": G_TRUE, "v0x": None}  # v0x has no stated ground truth
    if kind == "pendulum":
        L = float(calib.get("pendulum_length_m", 1.0))
        return {"T": 2 * math.pi * math.sqrt(L / G_TRUE)}
    if kind == "horizontal_motion":
        return {"v0": None, "decel": None}  # no stated ground truth
    raise ValueError(f"unknown kind {kind}")


def main(model_keys: list[str], cfg: Config) -> None:
    import importlib.resources as pkg_resources

    prompt_data = json.loads(
        pkg_resources.files("physcheck.prompts").joinpath("part_a.json").read_text()
    )
    results_dir = Path(cfg.results_dir) / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "part_a.jsonl"

    validation = validate_detector(cfg)
    if not validation["passed"]:
        raise RuntimeError(f"detector validation failed: {validation['failures']}")
    (results_dir / "detector_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    records = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            old = json.loads(line)
            records[(old["model"], old["prompt_id"], old["seed"])] = old
    analysis_hash = source_hash()
    for model in model_keys:
        for prompt in prompt_data["prompts"]:
            prompt_id = prompt["id"]
            expected = _expected_value(prompt["kind"], prompt["calibration"])
            for seed in cfg.seed_grid[:prompt["n_seeds"]]:
                record = {
                    "model": model, "prompt_id": prompt_id, "kind": prompt["kind"], "seed": seed,
                    "video_path": None, "generation_status": "pending", "track_status": "skipped",
                    "fit_status": "skipped", "params": {}, "params_std": None, "rel_errs": {},
                    "commit": _get_commit(), "analysis_source_sha256": analysis_hash,
                }
                stage = "generation"
                try:
                    gen = generate_video(model, prompt["text"], seed, cfg, prompt_id=prompt_id)
                    meta = gen["metadata"]
                    record.update({
                        "video_path": gen["video_path"], "generation_status": "ok",
                        "generation_commit": meta.get("commit"),
                        "generation_source_sha256": meta.get("source_sha256"),
                        "video_sha256": meta.get("video_sha256"), "fps": meta["fps"],
                    })
                    stage = "track"
                    track = track_object(gen["video_path"], fps=meta["fps"], cfg=cfg)
                    track_path = results_dir / "tracks" / model / prompt_id / f"seed_{seed}.json"
                    track_path.parent.mkdir(parents=True, exist_ok=True)
                    track_path.write_text(json.dumps(track, indent=2, allow_nan=False) + "\n")
                    record["track_artifact"] = (
                        str(track_path.relative_to(cfg.repo_root))
                        if track_path.is_relative_to(cfg.repo_root) else str(track_path.resolve())
                    )
                    record["track_status"] = track["status"]
                    if track["status"] != "ok":
                        record["track_reason"] = track.get("reason")
                    else:
                        stage = "fit"
                        ppm = compute_pixels_per_meter(track, prompt["calibration"])
                        record["pixels_per_meter"] = ppm
                        fit = fit_trajectory(prompt["kind"], track["trajectory_px"], track["times_s"], ppm, prompt["calibration"])
                        record["fit_status"] = fit["status"]
                        if fit["status"] == "ok":
                            record["params"] = fit["params"]
                            record["params_std"] = fit["params_std"]
                            record["rel_errs"] = {
                                key: abs(fit["params"][key] - truth) / abs(truth)
                                for key, truth in expected.items()
                                if truth is not None and key in fit["params"]
                            }
                        else:
                            record["fit_reason"] = fit.get("reason")
                except Exception as exc:
                    record[f"{stage}_status"] = "error"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                records[(model, prompt_id, seed)] = record
                temporary = out_path.with_suffix(".jsonl.tmp")
                temporary.write_text("".join(json.dumps(records[key], allow_nan=False) + "\n" for key in sorted(records)))
                temporary.replace(out_path)
                print(f"[part_a] {model} {prompt_id} seed={seed} generation={record['generation_status']} track={record['track_status']} fit={record['fit_status']}", flush=True)
    print(f"wrote {len(records)} unique lines to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["wan2.1_1.3b"])
    args = parser.parse_args()
    cfg = Config.load()
    main(args.models, cfg)
