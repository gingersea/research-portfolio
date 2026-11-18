"""Wrong visual evidence must never turn into a valid independent label."""
import copy
import json

import pytest

from physcheck.relations.contracts import sha
from physcheck.relations.visual import index_opinions, validate_opinion


@pytest.fixture
def evidence(tmp_path):
    (tmp_path / 'sheet.png').write_bytes(b'fixture image bytes')
    payload = {'opaque_case_id': 'v_fixture', 'frames': 3, 'sheets': [
        {'path': 'sheet.png', 'sha256': sha(tmp_path / 'sheet.png'), 'frame_indices': [0, 1, 2]}]}
    path = tmp_path / 'visual_judge_payload.json'
    path.write_text(json.dumps(payload))
    opinion = {'opaque_case_id': 'v_fixture', 'payload_sha256': sha(path),
        'viewed_sheet_sha256': [sha(tmp_path / 'sheet.png')], 'verdict': 'ABSTAIN',
        'reason': 'No visible contact before the end.',
        'events': {key: None for key in ['left_release_frame_range', 'right_release_frame_range',
                                        'left_contact_frame_range', 'right_contact_frame_range']},
        'assumptions': {'synchronized_release': 'unclear', 'static_camera': 'supported', 'both_falls_observed': 'violated'},
        'provenance': {'kind': 'context_aware_ai', 'judge_id': 'fixture', 'prior_context_seen': True}}
    return path, opinion


def test_matching_visual_evidence_is_diagnostic_not_independent(evidence):
    path, opinion = evidence
    result = validate_opinion(opinion, path)
    assert result['status'] == 'diagnostic' and result['verdict'] == 'ABSTAIN'
    assert not result['independent_validation'] and not result['ground_truth_label']


def test_duplicate_or_foreign_case_is_rejected(evidence):
    _, opinion = evidence
    with pytest.raises(ValueError, match='duplicate'):
        index_opinions([opinion, opinion], {'v_fixture'})
    with pytest.raises(ValueError, match='unknown'):
        index_opinions([opinion], {'another'})
    assert index_opinions([], {'v_fixture'}) == {}


def test_missing_viewed_sheet_is_rejected(evidence):
    path, opinion = evidence
    opinion['viewed_sheet_sha256'] = []
    with pytest.raises(ValueError, match='sheet'):
        validate_opinion(opinion, path)


def test_changed_payload_or_image_is_rejected(evidence):
    path, opinion = evidence
    bad = copy.deepcopy(opinion)
    bad['payload_sha256'] = 'other'
    with pytest.raises(ValueError, match='payload'):
        validate_opinion(bad, path)
    (path.parent / 'sheet.png').write_bytes(b'modified')
    with pytest.raises(ValueError, match='image'):
        validate_opinion(opinion, path)


def test_conclusive_opinion_requires_observable_events(evidence):
    path, opinion = evidence
    opinion['verdict'] = 'FAIL'
    with pytest.raises(ValueError, match='evidence'):
        validate_opinion(opinion, path)


def test_claimed_isolation_alone_does_not_prove_independence(evidence):
    path, opinion = evidence
    opinion['provenance'] = {'kind': 'isolated_ai', 'judge_id': 'unverified', 'prior_context_seen': False}
    assert not validate_opinion(opinion, path)['independent_validation']
