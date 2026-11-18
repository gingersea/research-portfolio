"""Undefined depth lifts must stay missing, never vanish from denominators."""
import numpy as np

from physcheck.relations.evaluation import geometry_metrics
from physcheck.relations.pipeline import aggregate_rows


def test_one_invalid_world_point_makes_complete_epe_unavailable():
    gt = np.zeros((1, 2, 3))
    xyz = gt.copy()
    xyz[0, 1] = np.nan
    row = geometry_metrics(np.zeros((1, 2, 2)), xyz, np.zeros((1, 2, 2)), gt, np.ones((1, 2), bool))
    assert row['raw_epe3d_m'] is None
    assert row['finite_epe3d_m'] == 0 and row['valid_3d_fraction'] == .5
    assert row['visible_point_times'] == 2 and row['invalid_3d_point_times'] == 1


def test_incomplete_video_is_not_dropped_from_macro_score():
    rows = [{'id': str(i), 'domain': 'pstudio', 'status': 'ok', 'evaluation': {'co': {
        'raw_epe3d_m': error, 'visible_point_times': 10, 'invalid_3d_point_times': i}}}
        for i, error in enumerate([.2, None])]
    result = aggregate_rows(rows)['by_domain']['pstudio']['video_mean_metrics']['co']
    assert result['videos'] == 2 and result['raw_epe3d_m'] is None
    assert result['metric_valid_videos']['raw_epe3d_m'] == 1
    assert result['count_totals']['visible_point_times'] == 20
