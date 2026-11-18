import numpy as np

from metrics import lift, score_case


def example():
    xy = np.array([[[10., 10.], [30., 10.]]]*3)
    visible = np.array([[True, True], [False, True], [False, True]])
    return xy, visible, np.ones((3, 2), dtype=bool)


def test_invisible_coordinate_is_not_visible_evidence():
    xy, visible, known = example()
    metrics = score_case(xy, np.ones_like(visible), xy, visible, known)
    assert metrics['false_visible_count'] == 2
    assert metrics['false_visible_rate'] == 1
    assert metrics['visible_precision'] == .5


def test_correct_abstention_keeps_visibility_and_location_distinct():
    xy, visible, known = example()
    metrics = score_case(xy, visible, xy, visible, known)
    assert metrics['visible_precision'] == metrics['visible_recall'] == 1
    assert metrics['pck3_visible'] == 1
    assert metrics['false_visible_count'] == 0


def test_query_frame_does_not_inflate_scores():
    xy, visible, known = example()
    prediction = xy.copy()
    prediction[0] = 999
    metrics = score_case(prediction, visible, xy, visible, known)
    assert metrics['evaluated_samples'] == 4
    assert metrics['visible_gt_count'] == 2
    assert metrics['pck3_visible'] == 1


def test_wrong_other_identity_is_counted_during_absence():
    xy, visible, known = example()
    prediction = xy.copy()
    prediction[1:, 0] = prediction[1:, 1]
    metrics = score_case(prediction, np.ones_like(visible), xy, visible, known)
    assert metrics['wrong_other_count'] == 2
    assert metrics['wrong_other_claim_fraction'] == .5
    assert metrics['id_association_transitions'] == 1


def test_ambiguous_history_is_reported_as_unsupported():
    xy, visible, known = example()
    known[2] = False
    metrics = score_case(xy, np.ones_like(visible), xy, visible, known)
    assert metrics['unsupported_identity_count'] == 2
    assert metrics['ambiguous_sample_count'] == 2
    assert metrics['unsupported_identity_fraction'] == 1


def test_missing_location_is_a_pck_failure():
    xy, visible, known = example()
    prediction = xy.copy()
    prediction[1, 1] = np.nan
    metrics = score_case(prediction, visible, xy, visible, known)
    assert metrics['pck3_visible'] == .5
    assert metrics['visible_missing_location_count'] == 1


def test_lift_restores_world_camera_translation():
    K = np.array([[500., 0, 384], [0, 500, 256], [0, 0, 1]])
    xy = np.array([[[434., 281.]], [[409., 281.]]])
    result = lift(xy, np.array([[2.], [2.]]), K, np.array([[1., 0, 0], [1.1, 0, 0]]))
    np.testing.assert_allclose(result, [[[1.2, .1, 2]], [[1.2, .1, 2]]])


def test_depth_scaling_does_not_change_projection():
    K = np.array([[500., 0, 384], [0, 500, 256], [0, 0, 1]])
    points = lift(np.array([[[434., 281.]]]), np.array([[4.]]), K, np.zeros((1, 3)))
    np.testing.assert_allclose(points, [[[.4, .2, 4]]])
