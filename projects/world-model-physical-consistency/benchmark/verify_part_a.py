"""Audit actual Part A artifacts and derive the halfway tables from them.

Run from the worktree root: PYTHONPATH=benchmark python3 benchmark/verify_part_a.py
Requires the full two-model, four-prompt, three-seed production matrix. Fails
on missing rows, stale analyses, missing artifacts, or metadata/hash mismatch.
"""
from collections import Counter
import importlib.resources
import json
from pathlib import Path
import shutil

import av

from physcheck.config import Config
from physcheck.generate.pipeline import file_hash, source_hash
from physcheck.report.aggregate import aggregate_part_a

cfg = Config.load()
raw = cfg.results_dir / "raw"
rows = [json.loads(line) for line in (raw / "part_a.jsonl").read_text().splitlines()]
prompts = json.loads(importlib.resources.files("physcheck.prompts").joinpath("part_a.json").read_text())["prompts"]
models = ["wan2.1_1.3b", "cogvideox_5b"]
expected = {(m, p["id"], s) for m in models for p in prompts for s in (0, 1, 2)}
actual = {(r["model"], r["prompt_id"], r["seed"]) for r in rows}
assert actual == expected and len(rows) == len(expected), (len(rows), expected - actual)
verified = []
for row in rows:
    assert row["generation_status"] == "ok", row
    video = Path(row["video_path"])
    sidecar = video.with_suffix(".json")
    meta = json.loads(sidecar.read_text())
    assert meta["video_sha256"] == row["video_sha256"] == file_hash(video)
    assert row["analysis_source_sha256"] == source_hash(), "stale analysis source"
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        size = [stream.width, stream.height]
        n = sum(1 for _ in container.decode(video=0))
    assert fps == meta["fps"] == row["fps"]
    assert n == meta["frames"] and size == meta["resolution"]
    assert Path(cfg.repo_root / row["track_artifact"]).exists()
    archived = raw / "generation" / row["model"] / row["prompt_id"] / sidecar.name
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sidecar, archived)
    verified.append({"model": row["model"], "prompt_id": row["prompt_id"], "seed": row["seed"],
                     "video_sha256": meta["video_sha256"], "decoded_frames": n, "fps": fps,
                     "resolution": size, "generation_seconds": meta["generation_seconds"]})

validation = json.loads((raw / "detector_validation.json").read_text())
assert validation["passed"] and len(validation["results"]) == 6
summary = aggregate_part_a(rows)
for model in models:
    selected = [r for r in rows if r["model"] == model]
    summary["by_model"][model]["fit_status_counts"] = dict(Counter(r["fit_status"] for r in selected))
    summary["by_model"][model]["track_status_counts"] = dict(Counter(r["track_status"] for r in selected))
    summary["by_model"][model]["generation_seconds_total"] = sum(v["generation_seconds"] for v in verified if v["model"] == model)
audit = {"passed": True, "expected_videos": len(expected), "verified_videos": len(verified),
         "analysis_source_sha256": source_hash(), "videos": verified}
(raw / "part_a_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
(raw / "part_a_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

table = ["# Part A halfway results (artifact-derived)", "", "Generated from `benchmark/results/raw/part_a.jsonl`; these are measurement-coverage statistics, not a physical-correctness leaderboard.", "", "| Model | Generated | Tracked | Fitted | Mean g relative error (fitted only) | Mean T relative error (fitted only) |", "|---|---:|---:|---:|---:|---:|"]
for model in models:
    stat = summary["by_model"][model]
    n = stat["n_videos"]
    fmt = lambda value: "N/A" if value is None else f"{value:.3%}"
    table.append(f"| {model} | {n} | {round(n*stat['track_success_rate'])} | {round(n*stat['fit_success_rate'])} | {fmt(stat['g_rel_err_mean'])} | {fmt(stat['T_rel_err_mean'])} |")
table += ["", "## Per-video measurements", "", "| Model | Prompt | Seed | Track | Fit | Parameters | Reason if unavailable |", "|---|---|---:|---|---|---|---|"]
for r in rows:
    parameters = "; ".join(f"{k}={v:.6g}" for k,v in r["params"].items()) or "N/A"
    reason = r.get("error", r.get("fit_reason", r.get("track_reason", "")))
    table.append(f"| {r['model']} | {r['prompt_id']} | {r['seed']} | {r['track_status']} | {r['fit_status']} | {parameters} | {reason} |")
table += ["", "## Synthetic validation", "", "| Scene | Parameter | Truth | Estimate | Relative error |", "|---|---|---:|---:|---:|"]
for r in validation["results"]:
    table.append(f"| {r['kind']} | {r['param']} | {r['true']:.6g} | {r['estimated']:.6g} | {r['rel_err']:.3%} |")
table += ["", "Errors are conditional on the nominal FPS and the prompt-stated 0.1 m ball diameter. Rejected fits and missing measurements are not zero error. Regression standard errors do not include calibration/camera uncertainty.", ""]
(cfg.repo_root / "report/part_a_halfway_results.md").write_text("\n".join(table))
print(json.dumps({"audit": {k:v for k,v in audit.items() if k != "videos"}, "summary": summary}, indent=2))
print(f"PART_A_AUDIT_PASS: {len(verified)}/{len(expected)} unique videos decoded; hashes, FPS, frames, source and validation verified")
