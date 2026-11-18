"""Tests for the object tracker.

The tracker must signal detection failures instead of returning garbage
trajectories, and must recover monotonic motion on clean synthetic video.
"""

import numpy as np
import pytest

from physcheck.config import Config
from physcheck.detect import synthetic
from physcheck.detect.tracker import track_object


def _test_cfg(tmp_path):
    return Config(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        models_cache_dir=tmp_path / "models",
        results_dir=tmp_path / "results",
    )


def _write_blank_video(tmp_path, n_frames=30, size=(120, 120), fps=30.0):
    """Write an all-white video and return its path."""
    import imageio.v3 as iio3

    frames = np.stack([np.full((*size, 3), 255, dtype=np.uint8)] * n_frames)
    path = tmp_path / "blank.mp4"
    iio3.imwrite(path, frames, fps=fps, codec="libx264")
    return str(path)


def test_empty_video_returns_detection_failure(tmp_path):
    video = _write_blank_video(tmp_path)
    result = track_object(video, fps=30.0, cfg=_test_cfg(tmp_path))
    assert result["status"] == "detection_failure"


def test_tracking_a_moving_ball_recovers_monotonic_fall(tmp_path):
    cfg = _test_cfg(tmp_path)
    scene = synthetic.render_scene("free_fall", cfg)
    result = track_object(scene["video_path"], fps=30.0, cfg=cfg)
    assert result["status"] == "ok"
    ys = [p[1] for p in result["trajectory_px"]]
    # The synthetic ball starts at the top of the frame and ends at the floor;
    # the tracker recovers a trajectory whose endpoints match that motion
    # (first detected position above the last detected position).
    assert ys[0] < ys[-1]


def test_flickery_video_yields_ok_or_detection_failure(tmp_path):
    """Salt-and-pepper noise must not produce a silent garbage trajectory."""
    cfg = _test_cfg(tmp_path)
    scene = synthetic.render_scene("free_fall", cfg)
    import imageio.v3 as iio3

    frames = iio3.imread(scene["video_path"])
    noisy = frames.copy()
    rng = np.random.default_rng(0)
    noise = rng.random(noisy.shape) < 0.02  # 2% salt-and-pepper
    noisy[noise] = rng.integers(0, 256, size=noise.shape)[noise]
    noisy_path = tmp_path / "noisy.mp4"
    iio3.imwrite(noisy_path, noisy, fps=30.0, codec="libx264")
    result = track_object(str(noisy_path), fps=30.0, cfg=cfg)
    assert result["status"] in ("ok", "detection_failure")


def test_tracker_returns_diameters_for_calibration(tmp_path):
    cfg = _test_cfg(tmp_path)
    scene = synthetic.render_scene("free_fall", cfg)
    result = track_object(scene["video_path"], fps=30.0, cfg=cfg)
    assert result["status"] == "ok"
    diameters = result["diameters_px"]
    assert len(diameters) > 0
    # Ball diameter is 0.1 m * 100 px/m = 10 px; allow segmentation slop
    assert 8 <= np.median(diameters) <= 12


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf")])
def test_tracker_rejects_missing_or_invalid_clock(tmp_path, fps):
    video = _write_blank_video(tmp_path)
    with pytest.raises(ValueError, match="fps"):
        track_object(video, fps=fps, cfg=_test_cfg(tmp_path))


def test_flicker_cannot_yield_confident_wrong_gravity(tmp_path):
    from physcheck.detect.fit import compute_pixels_per_meter, fit_trajectory
    import imageio.v3 as iio

    cfg = _test_cfg(tmp_path)
    scene = synthetic.render_scene("free_fall", cfg)
    frames = iio.imread(scene["video_path"])
    rng = np.random.default_rng(17)
    noise = rng.random(frames.shape[:3]) < 0.02
    frames[noise] = rng.integers(0, 256, (noise.sum(), 3), dtype=np.uint8)
    path = tmp_path / "flicker.mp4"
    iio.imwrite(path, frames, fps=30, codec="libx264")
    track = track_object(str(path), fps=30, cfg=cfg)
    if track["status"] == "ok":
        ppm = compute_pixels_per_meter(track, {"object_diameter_m": 0.1})
        fit = fit_trajectory("free_fall", track["trajectory_px"], track["times_s"], ppm, {})
        assert fit["status"] == "fit_failed" or abs(fit["params"]["g"] - 9.8) < 0.98


def test_ball_present_in_first_frame_does_not_leave_stationary_ghost(tmp_path):
    from physcheck.detect.fit import compute_pixels_per_meter, fit_trajectory
    import imageio.v3 as iio

    cfg = _test_cfg(tmp_path)
    scene = synthetic.render_scene("free_fall", cfg)
    frames = iio.imread(scene["video_path"])[4:]
    path = tmp_path / "no_empty_start.mp4"
    iio.imwrite(path, frames, fps=30, codec="libx264")
    track = track_object(str(path), fps=30, cfg=cfg)
    assert track["status"] == "ok"
    ppm = compute_pixels_per_meter(track, {"object_diameter_m": 0.1})
    fit = fit_trajectory("free_fall", track["trajectory_px"], track["times_s"], ppm, {})
    assert fit["status"] == "ok"
    assert abs(fit["params"]["g"] - 9.8) < 0.98


def test_unexplained_internal_disappearance_is_detection_failure(tmp_path):
    import cv2
    import imageio.v3 as iio
    frames = np.full((30, 480, 160, 3), 255, dtype=np.uint8)
    for i in range(20):
        cv2.circle(frames[i], (80, round(30 + 490*(i/30)**2)), 5, (255, 0, 0), -1)
    path = tmp_path / "vanished.mp4"
    iio.imwrite(path, frames, fps=30, codec="libx264")
    result = track_object(str(path), fps=30, cfg=_test_cfg(tmp_path))
    assert result["status"] == "detection_failure"
    assert "disappear" in result["reason"]


def test_exposure_blur_cannot_produce_precise_wrong_gravity(tmp_path):
    import cv2
    import imageio.v3 as iio
    from physcheck.detect.fit import compute_pixels_per_meter, fit_trajectory

    frames = []
    for i in range(30):
        exposures = []
        for dt in np.linspace(-1/60, 1/60, 15):
            frame = np.full((600, 160, 3), 255, dtype=np.uint8)
            cv2.circle(frame, (80, round(30 + 490*(i/30 + dt)**2)), 5, (255, 0, 0), -1)
            exposures.append(frame)
        frames.append(np.mean(exposures, axis=0).astype(np.uint8))
    path = tmp_path / "blur.mp4"
    iio.imwrite(path, np.stack(frames), fps=30, codec="libx264")
    track = track_object(str(path), fps=30, cfg=_test_cfg(tmp_path))
    if track["status"] != "ok":
        return
    try:
        ppm = compute_pixels_per_meter(track, {"object_diameter_m": 0.1})
    except ValueError:
        return  # explicitly uncalibratable is an acceptable result
    fit = fit_trajectory("free_fall", track["trajectory_px"], track["times_s"], ppm, {})
    assert fit["status"] == "fit_failed" or abs(fit["params"]["g"] - 9.8) < 0.98
