"""Integration boundaries that can silently corrupt a measurement pipeline."""
import numpy as np
import pytest

from physcheck.relations.contracts import lift_points, relation_decision, verify_cache
from physcheck.relations.pipeline import aggregate_rows, run
from physcheck.relations.backends import source_entries


def test_tap_file_count_is_not_a_source_file_list():
    separate = [{'file': 'model.py', 'sha256': 'expected'}]
    assert source_entries('tap', {'files': 129}, separate) == separate
    assert source_entries('co', {'source_files': separate}, []) == separate
    assert source_entries('depth', {'files': separate}, []) == separate


def test_cache_of_other_input_rejected():
    with pytest.raises(ValueError, match='input'):
        verify_cache({'status': 'ok', 'input_sha256': 'other', 'prediction_sha256': 'p'}, 'wanted', 'p')


def test_cache_mismatched_prediction_rejected():
    with pytest.raises(ValueError, match='prediction'):
        verify_cache({'status': 'ok', 'input_sha256': 'i', 'prediction_sha256': 'old'}, 'i', 'new')


def test_outside_points_do_not_become_valid_through_clamping():
    xy = np.array([[[1., 1.], [-1., 0.], [3., 0.]]])
    xyz, valid = lift_points(xy, np.full((1, 3, 3), 2.), np.eye(3))
    assert valid.tolist() == [[True, False, False]]
    assert np.allclose(xyz[0, 0], [2, 2, 2])
    assert np.isnan(xyz[0, 1:]).all()


def test_failed_measurement_assumptions_force_abstention():
    assert relation_decision({'verdict': 'PASS', 'score': .99}, False)['verdict'] == 'ABSTAIN'
    assert relation_decision({'verdict': 'PASS'}, True)['verdict'] == 'ABSTAIN'
    assert relation_decision({'verdict': 'FAIL', 'score': .2}, True)['verdict'] == 'FAIL'


def test_aggregation_keeps_errors_separate_from_physical_failures():
    rows = [{'id': 'a', 'domain': 'control', 'status': 'ok', 'relation': {'verdict': 'PASS'}},
            {'id': 'b', 'domain': 'control', 'status': 'error', 'error': 'bad cache'},
            {'id': 'c', 'domain': 'i2v', 'status': 'ok', 'relation': {'verdict': 'ABSTAIN'}}]
    result = aggregate_rows(rows)
    assert result['total_cases'] == 3 and result['errors'] == 1
    assert result['by_domain']['control']['verdicts'] == {'PASS': 1}
    assert result['by_domain']['i2v']['verdicts'] == {'ABSTAIN': 1}


def test_existing_output_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        run({'cases': []}, tmp_path)


def test_per_case_errors_stay_in_complete_run(tmp_path):
    cases = [{'id': name, 'domain': 'davis', 'files': {'input': {'path': str(tmp_path / 'missing'), 'sha256': 'missing'}}}
             for name in ['first', 'second']]
    summary = run({'cases': cases}, tmp_path / 'run')
    assert summary['total_cases'] == 2 and summary['errors'] == 2
    assert (tmp_path / 'run/first/error.json').exists()
    assert (tmp_path / 'run/second/error.json').exists()


def test_model_initialization_failure_records_all_cases_without_cache_fallback(tmp_path, monkeypatch):
    import physcheck.relations.backends as backends

    def unavailable(*args):
        raise RuntimeError('model not available')

    monkeypatch.setattr(backends, 'NativeModels', unavailable)
    cases = [{'id': name, 'domain': 'i2v', 'files': {}} for name in ['one', 'two']]
    summary = run({'cases': cases}, tmp_path / 'fresh', fresh_i2v=True)
    assert summary['total_cases'] == summary['errors'] == 2
    assert 'model not available' in (tmp_path / 'fresh/one/error.json').read_text()
    assert not (tmp_path / 'fresh/one/fresh_inference.json').exists()
