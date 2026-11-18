"""Missing scientific quantities cannot be rendered as zero error or valid ranking."""
from physcheck.report.render import render_part_a, render_part_b, render_prompt_fits, render_categories


def test_unmeasurable_part_a_stays_na():
    data = {'by_model': {'m': {'n_videos': 3, 'track_success_rate': 1., 'fit_success_rate': 0.,
                             'g_rel_err_mean': None, 'T_rel_err_mean': None}}}
    text = render_part_a(data)
    assert '| m | 3 | 3/3 | 0/3 | N/A | N/A |' in text


def test_invalid_part_b_does_not_render_diagnostic_rate_as_valid():
    data = {'valid': False, 'diagnostic_only': {'violation_rate_by_model': {'m': .5},
             'per_video': [{'model': 'm'}, {'model': 'm'}]}}
    text = render_part_b(data)
    assert 'N/A (invalid calibration)' in text
    assert '| m | 2 | 0.5000 |' not in text


def test_per_prompt_missing_physics_and_invalid_category_stay_na():
    a = {'by_model': {'m': {'per_prompt': {'fall': {'n': 3, 'fit_ok': 0, 'mean_params': {}}}}}}
    assert '| m | fall | 3 | 0/3 | N/A | N/A | N/A | N/A |' in render_prompt_fits(a)
    b = {'valid': False, 'diagnostic_only': {'violations_by_category': {'m': {'rigid_body': 4}}}}
    assert '| m | rigid_body | N/A (invalid calibration) |' in render_categories(b)
