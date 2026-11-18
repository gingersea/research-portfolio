"""Initial queries are allowed inputs; future targets never enter inference."""
import numpy as np
import pytest

from physcheck.relations.backends import validate_queries
from physcheck.relations.pipeline import run


def test_public_query_accepts_later_initialization_frames():
    q = np.array([[0, 1, 2], [4, 2, 3]], np.float32)
    assert np.array_equal(validate_queries(q, (5, 8, 9, 3)), q)


@pytest.mark.parametrize('query', [[[5, 1, 2]], [[-1, 1, 2]], [[.5, 1, 2]], [[0, 8, 2]], [[0, 1, 9]], [[0, np.nan, 2]]])
def test_public_query_rejects_bad_time_or_coordinate(query):
    with pytest.raises(ValueError):
        validate_queries(np.array(query), (5, 8, 9, 3))


def test_public_initialization_failure_is_recorded_not_replaced_with_cache(tmp_path, monkeypatch):
    import physcheck.relations.backends as backends

    def unavailable(*args):
        raise RuntimeError('public model unavailable')

    monkeypatch.setattr(backends, 'NativeModels', unavailable)
    cases = [{'id': domain, 'domain': domain, 'files': {}} for domain in ['davis', 'pstudio']]
    result = run({'cases': cases}, tmp_path / 'run', fresh_public=True)
    assert result['total_cases'] == result['errors'] == 2
    assert 'public model unavailable' in (tmp_path / 'run/davis/error.json').read_text()
