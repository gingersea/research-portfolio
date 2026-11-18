"""Fixtures target leakage boundaries, reject-all, state, and geometry."""
import numpy as np

from common import choose_threshold, counts, split_cases, two_frame_confirmation
from geometry import backproject, sample_depth


def test_split_is_video_disjoint_and_order_independent():
    names = [f'clip{i}' for i in range(30)]
    result = split_cases(names)
    assert result == split_cases(names[::-1])
    assert len(result['calibration']) == 10 and len(result['validation']) == 20
    assert not set(result['calibration']) & set(result['validation'])


def test_confirmation_respects_query_and_reappearance():
    raw = np.array([[1, 1, 0, 1, 1, 0, 1, 0, 1, 1]], dtype=bool)
    result = two_frame_confirmation(raw, np.array([1.]))
    assert result.tolist() == [[False, True, False, False, True, False, False, False, False, True]]


def test_wrong_location_and_occlusion_are_disjoint_errors_and_query_excluded():
    query = np.array([[0, 0, 0]])
    gt = np.zeros((1, 4, 2))
    xy = np.array([[[99, 0], [0, 0], [4, 0], [0, 0]]])
    row = counts(query, gt, np.array([[0, 0, 0, 1]], bool), xy, np.ones((1, 4), bool))
    assert row == dict(positive=2, negative=1, accepted=3, correct=1, false_visible=1, wrong_location=1)


def test_calibration_cannot_win_by_rejecting_almost_everything():
    curve = [dict(threshold=.9, precision=.94, retention=.5, covered_videos=10, correct=50),
             dict(threshold=.99, precision=1., retention=.01, covered_videos=1, correct=1)]
    result = choose_threshold(curve)
    assert result['threshold'] == .9 and not result['target_met']


def test_feasible_threshold_maximizes_correct_not_precision():
    curve = [dict(threshold=.9, precision=.95, retention=.7, covered_videos=10, correct=70),
             dict(threshold=.99, precision=.99, retention=.2, covered_videos=5, correct=20)]
    assert choose_threshold(curve)['threshold'] == .9
    assert choose_threshold([])['threshold'] == 1.


def test_geometry_oracles_isolate_ray_and_depth_errors():
    k = np.array([[100., 0, 10], [0, 100., 10], [0, 0, 1]])
    point = np.array([[[10., 10.]]])
    true = backproject(point, np.array([[2.]]), k)
    wrong_depth = backproject(point, np.array([[3.]]), k)
    wrong_ray = backproject(point + np.array([5., 0]), np.array([[2.]]), k)
    assert np.allclose(np.linalg.norm(wrong_depth - true), 1.)
    assert np.allclose(np.linalg.norm(wrong_ray - true), .1)
    assert np.allclose(backproject(point, wrong_depth[..., 2], k), wrong_depth)


def test_depth_sample_uses_nearest_and_explicit_boundary_clamp():
    depth = np.array([[[1., 2.], [3., 4.]]])
    xy = np.array([[[.6, .6], [-1., 3.]]])
    assert np.array_equal(sample_depth(depth, xy), [[4., 3.]])
