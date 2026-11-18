"""Tests for physics fitting and the detector validation gate."""

import math

import numpy as np
import pytest

from physcheck.config import Config
from physcheck.detect import synthetic
from physcheck.detect.fit import compute_pixels_per_meter, fit_trajectory
from physcheck.detect.validate import validate_detector


def _test_cfg(tmp_path):
    return Config(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        models_cache_dir=tmp_path / "models",
        results_dir=tmp_path / "results",
    )


def _fall_trajectory(fps=30, duration=1.5, g=9.8, ppm=100.0, start_y_px=60.0):
    """Synthetic free-fall trajectory in pixel space (y grows downward)."""
    n = int(fps * duration)
    t = np.arange(n) / fps
    # screen y = start + 0.5*g*t^2 * ppm (px)
    ys = start_y_px + 0.5 * g * t**2 * ppm
    xs = np.full(n, 240.0)
    return list(zip(xs, ys)), list(t)


def test_free_fall_recovers_g():
    traj, times = _fall_trajectory()
    result = fit_trajectory("free_fall", traj, times, pixels_per_meter=100.0, calib={})
    assert result["status"] == "ok"
    assert abs(result["params"]["g"] - 9.8) < 0.5


def test_pendulum_recovers_period():
    # x(t) = L*sin(theta0*cos(w t)) in pixel space, sampled at 30 fps
    fps, L, g = 30, 1.0, 9.8
    ppm = 100.0
    theta0 = math.radians(20)
    n = 90
    t = np.arange(n) / fps
    xs = 240 + L * ppm * np.sin(theta0 * np.cos(math.sqrt(g / L) * t))
    ys = 60 + L * ppm * np.cos(theta0 * np.cos(math.sqrt(g / L) * t))
    traj = list(zip(xs, ys))
    result = fit_trajectory("pendulum", traj, list(t), pixels_per_meter=ppm, calib={})
    assert result["status"] == "ok"
    T_true = 2 * math.pi * math.sqrt(L / g)
    assert abs(result["params"]["T"] - T_true) / T_true < 0.10


def test_noisy_trajectory_returns_fit_failed():
    rng = np.random.default_rng(42)
    traj = list(zip(rng.random(60) * 480, rng.random(60) * 480))
    times = list(np.arange(60) / 30.0)
    result = fit_trajectory("free_fall", traj, times, pixels_per_meter=100.0, calib={})
    assert result["status"] == "fit_failed"


def test_projectile_rejects_nonballistic_horizontal_motion():
    t = np.arange(30) / 30
    xs = 100 + 50 * np.sin(t * 30)
    ys = 30 + 490 * t**2
    result = fit_trajectory("projectile", list(zip(xs, ys)), list(t), 100, {})
    assert result["status"] == "fit_failed"


def test_pendulum_requires_observed_oscillation():
    t = np.arange(81) / 16
    result = fit_trajectory("pendulum", list(zip(100 + 16*t, np.full(81, 100))), list(t), 100, {})
    assert result["status"] == "fit_failed"


def test_horizontal_motion_recovers_parameters_with_stationary_tail():
    t = np.arange(81) / 16
    moving_time = np.minimum(t, 1.0)
    xs = 30 + 100 * (0.4*moving_time - 0.2*moving_time**2)
    result = fit_trajectory("horizontal_motion", list(zip(xs, np.full(81, 50))), list(t), 100, {})
    assert result["status"] == "ok"
    assert result["params"]["v0"] == pytest.approx(0.4, rel=0.05)
    assert result["params"]["decel"] == pytest.approx(0.4, rel=0.05)


def test_validation_gate_passes_on_clean_synthetic(tmp_path):
    cfg = _test_cfg(tmp_path)
    report = validate_detector(cfg)
    assert report["passed"] is True, report.get("failures")


def test_validation_gate_reports_rel_errors(tmp_path):
    cfg = _test_cfg(tmp_path)
    report = validate_detector(cfg)
    assert all("rel_err" in r for r in report["results"])
