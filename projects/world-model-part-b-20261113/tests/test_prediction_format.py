import numpy as np

from physcheck.relations.contracts import co_prediction


def test_cached_co_formats_preserve_coordinates_and_visibility():
    xy = np.array([[[2., 3.]]])
    visible = np.array([[False]])
    for data in [{'xy': xy, 'visible': visible}, {'co_xy': xy, 'co_visible': visible}]:
        coords, claim = co_prediction(data)
        assert np.array_equal(coords, xy) and np.array_equal(claim, visible)
