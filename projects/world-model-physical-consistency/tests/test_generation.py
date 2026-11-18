"""Exercise real video/metadata IO while replacing expensive neural inference."""
import json
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest

from physcheck.config import Config
from physcheck.generate import pipeline


@pytest.fixture
def generation(tmp_path, monkeypatch):
    cfg = Config(tmp_path, tmp_path / "artifacts", tmp_path / "models", tmp_path / "results")
    calls = []

    class TinyPipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(("load", str(source)))
            return cls()

        def to(self, *args, **kwargs):
            return self

        def enable_model_cpu_offload(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            calls.append(("generate", kwargs))
            frames = np.zeros((kwargs["num_frames"], kwargs["height"], kwargs["width"], 3), dtype=np.uint8)
            frames[..., 0] = 255
            return SimpleNamespace(frames=[frames])

    monkeypatch.setattr(pipeline, "_pipeline_cls", lambda entry: TinyPipeline)
    monkeypatch.setitem(pipeline.MODEL_REGISTRY, "tiny", {
        "hf_repo": "fixture/tiny", "torch_dtype": "float32", "num_frames": 9,
        "width": 32, "height": 32, "fps": 16.0,
    })
    return cfg, calls


def test_generation_writes_decodable_video_and_resumes_only_matching_request(generation):
    cfg, calls = generation
    result = pipeline.generate_video("tiny", "red ball", 0, cfg, "p")
    assert iio.imread(result["video_path"]).shape == (9, 32, 32, 3)
    assert result["metadata"]["fps"] == 16
    pipeline.generate_video("tiny", "red ball", 0, cfg, "p")
    assert len([c for c in calls if c[0] == "generate"]) == 1
    with pytest.raises(ValueError, match="mismatch"):
        pipeline.generate_video("tiny", "different prompt", 0, cfg, "p")


def test_incomplete_artifact_is_repaired_and_num_frames_honored(generation):
    cfg, calls = generation
    out = cfg.artifacts_dir / "videos/tiny/p"
    out.mkdir(parents=True)
    (out / "seed_0.mp4").write_bytes(b"interrupted encoding")
    result = pipeline.generate_video("tiny", "red ball", 0, cfg, "p", num_frames=5)
    assert len(iio.imread(result["video_path"])) == 5
    assert result["metadata"]["frames"] == 5
    assert len(result["metadata"]["video_sha256"]) == 64


def test_generation_uses_existing_local_checkpoint(generation):
    cfg, calls = generation
    local = cfg.models_cache_dir / "fixture/tiny"
    local.mkdir(parents=True)
    (local / "model_index.json").write_text("{}")
    pipeline.generate_video("tiny", "red ball", 0, cfg, "p")
    assert ("load", str(local)) in calls
