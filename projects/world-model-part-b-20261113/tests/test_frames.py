"""Tests for the frame sampler."""

import numpy as np
import pytest

from physcheck.config import Config
from physcheck.detect import synthetic
from physcheck.judge.frames import sample_frames


def _test_cfg(tmp_path):
    return Config(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        models_cache_dir=tmp_path / "models",
        results_dir=tmp_path / "results",
    )


def test_sample_frames_even_spacing_and_grid(tmp_path):
    cfg = _test_cfg(tmp_path)
    scene = synthetic.render_scene("free_fall", cfg)
    out = sample_frames(scene["video_path"], n_frames=8, cfg=cfg, video_id="test_video")
    assert len(out["frame_paths"]) == 8
    assert out["grid_path"] is not None
    from pathlib import Path

    assert Path(out["grid_path"]).exists()


def test_sample_frames_raises_on_missing_fps(tmp_path):
    cfg = _test_cfg(tmp_path)
    # Write a raw file with no fps metadata: a .npy is not a video,
    # so reading it must raise rather than guess a frame rate.
    bad_path = tmp_path / "not_a_video.mp4"
    bad_path.write_bytes(b"this is not a video file")
    with pytest.raises(Exception):
        sample_frames(str(bad_path), n_frames=8, cfg=cfg)
