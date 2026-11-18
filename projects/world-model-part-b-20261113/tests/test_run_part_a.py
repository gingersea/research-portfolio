"""Production retries must preserve other models and retain failures as data."""
import json
import pytest

from physcheck.config import Config
from physcheck.detect import run_part_a, synthetic


@pytest.mark.parametrize("external_results", [False, True])
def test_driver_is_incremental_idempotent_and_preserves_other_models(tmp_path, monkeypatch, external_results):
    root = tmp_path / "repo"
    results = tmp_path / "external" if external_results else root / "results"
    cfg = Config(root, root / "artifacts", root / "models", results, seed_grid=[0])
    raw = cfg.results_dir / "raw"
    raw.mkdir()
    path = raw / "part_a.jsonl"
    previous = {"model": "previous", "prompt_id": "fall_a1", "seed": 0, "fit_status": "ok"}
    path.write_text(json.dumps(previous) + "\n")
    calls = []

    def generate(model, text, seed, config, prompt_id=None):
        calls.append(prompt_id)
        # A completed video's result must be durable before starting the next.
        if prompt_id != "fall_a1":
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert any(r.get("model") == "test" and r["prompt_id"] == "fall_a1" for r in records)
        if prompt_id == "proj_a2":
            raise RuntimeError("injected inference failure")
        kind = {"fall_a1": "free_fall", "pend_a3": "pendulum", "horiz_a4": "horizontal_motion"}[prompt_id]
        scene = synthetic.render_scene(kind, cfg)
        return {"video_path": scene["video_path"], "metadata": {
            "model": model, "prompt_id": prompt_id, "seed": seed, "fps": 30,
            "commit": "fixture", "source_sha256": "fixture-source",
        }}

    # Neural generation is the only replaced boundary; tracking/fitting/IO are real.
    monkeypatch.setattr(run_part_a, "generate_video", generate, raising=False)
    run_part_a.main(["test"], cfg)
    run_part_a.main(["test"], cfg)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 5
    assert previous in records
    failed = next(r for r in records if r.get("prompt_id") == "proj_a2")
    assert failed["generation_status"] == "error"
    assert failed["fit_status"] == "skipped"
    assert "injected inference failure" in failed["error"]
    successful = next(r for r in records if r["model"] == "test" and r["prompt_id"] == "fall_a1")
    assert successful["fit_status"] == "ok"
    assert successful["params"]["g"] > 9
    assert successful["track_artifact"]
    assert successful["generation_source_sha256"] == "fixture-source"
