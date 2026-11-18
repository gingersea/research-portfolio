"""Regression checks for false positives and unsupported conclusive judgments."""
import copy
import hashlib

import cv2
import numpy as np
import pytest

from judge_gate import validate, validate_assignments, validate_receipt_hashes
from measure import colored_component


def valid_result():
    return {'case_id': 'example', 'verdict': 'PASS', 'reason': 'Both falls observed.', 'events': {key: [0, 1] for key in ['left_release_frame_range', 'right_release_frame_range', 'left_contact_frame_range', 'right_contact_frame_range']}, 'assumptions': {key: 'supported' for key in ['synchronized_release', 'static_camera', 'both_falls_observed']}, 'sheets_viewed': 1}


def test_background_never_becomes_object():
    assert colored_component(np.full((64, 96, 3), 246, np.uint8), [225, 35, 45]) is None


def test_two_matching_objects_are_ambiguous():
    image = np.full((64, 96, 3), 246, np.uint8)
    cv2.circle(image, (20, 30), 8, (225, 35, 45), -1)
    cv2.circle(image, (70, 30), 8, (225, 35, 45), -1)
    assert colored_component(image, [225, 35, 45]) is None


def test_same_material_other_color_is_not_matching_object():
    image = np.full((64, 96, 3), 246, np.uint8)
    cv2.circle(image, (20, 30), 8, (35, 65, 225), -1)
    assert colored_component(image, [225, 35, 45]) is None


@pytest.mark.parametrize('mutation', ['null_event', 'bad_identity', 'outside_frame', 'missing_sheet', 'unsupported_assumption', 'extra_field'])
def test_unsupported_judgments_are_rejected(mutation):
    result = copy.deepcopy(valid_result())
    if mutation == 'null_event':
        result['events']['left_contact_frame_range'] = None
    elif mutation == 'bad_identity':
        result['case_id'] = 'another'
    elif mutation == 'outside_frame':
        result['events']['left_release_frame_range'] = [0, 4]
    elif mutation == 'missing_sheet':
        result['sheets_viewed'] = 0
    elif mutation == 'unsupported_assumption':
        result['assumptions']['static_camera'] = 'unclear'
    else:
        result['private_label'] = 'PASS'
    with pytest.raises(ValueError):
        validate(result, {'case_id': 'example', 'frame_count': 4, 'sheets': [{}]})


def test_explicit_abstention_allows_unobserved_events():
    result = valid_result()
    result['verdict'] = 'ABSTAIN'
    result['events']['left_contact_frame_range'] = None
    result['assumptions']['both_falls_observed'] = 'unclear'
    validate(result, {'case_id': 'example', 'frame_count': 4, 'sheets': [{}]})


def test_repeated_case_cannot_impersonate_twelve_assignments():
    with pytest.raises(ValueError):
        validate_assignments([{'case_id': 'same'}] * 12, {'same': {'scene_seed': 81000, 'condition': 'equal_gravity'}})


def test_changed_first_judgment_is_rejected_despite_valid_schema():
    original = b'original first judgment'
    payload = b'payload'
    receipt = {'judgment_sha256': hashlib.sha256(original).hexdigest(), 'input_payload_sha256': hashlib.sha256(payload).hexdigest()}
    case = {'payload_sha256': receipt['input_payload_sha256']}
    with pytest.raises(ValueError):
        validate_receipt_hashes(b'rewritten explanation', payload, case, receipt)


def test_altered_payload_cannot_reuse_receipt():
    judgment = b'first judgment'
    receipt = {'judgment_sha256': hashlib.sha256(judgment).hexdigest(), 'input_payload_sha256': hashlib.sha256(b'original').hexdigest()}
    with pytest.raises(ValueError):
        validate_receipt_hashes(judgment, b'new payload', {'payload_sha256': hashlib.sha256(b'new payload').hexdigest()}, receipt)
