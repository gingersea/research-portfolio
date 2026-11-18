"""Meaningful failure controls for preservation, availability and contact validity."""
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate import assess_frames, contact_wrapper
from prepare_inputs import draw

STUDY = Path(__file__).resolve().parent


def initial_fixture():
    setup = json.loads((STUDY / 'inputs/A/setup.json').read_text())
    frame, masks = draw([point[1] for point in setup['initial_centers']])
    return setup, frame, masks


def test_known_initial_state_passes_preservation_and_availability():
    setup, frame, masks = initial_fixture()
    result = assess_frames(np.repeat(frame[None], 97, axis=0), setup, masks)
    assert result['initial_objects_passed'] and result['initial_setup_passed']
    assert result['coverage_passed'] and result['individual_fiducial_camera_passed']


def test_one_missing_decoded_frame_uses_full_expected_denominator():
    setup, frame, masks = initial_fixture()
    result = assess_frames(np.repeat(frame[None], 96, axis=0), setup, masks)
    assert result['usable_both_frames'] == 96 and result['expected_frames'] == 97
    assert not result['coverage_passed']


def test_visibility_is_not_inferred_across_a_gap():
    setup, frame, masks = initial_fixture()
    pixels = np.repeat(frame[None], 97, axis=0)
    pixels[4:8] = 242
    result = assess_frames(pixels, setup, masks)
    assert result['longest_missing_run'] == 4
    assert not result['coverage_passed']


def test_symmetric_camera_zoom_cannot_hide_in_centroid_average():
    setup, frame, masks = initial_fixture()
    transform = cv2.getRotationMatrix2D((384, 256), 0, 1.02)
    changed = cv2.warpAffine(frame, transform, (768, 512), borderValue=(242, 242, 242))
    pixels = np.repeat(frame[None], 97, axis=0)
    pixels[1:] = changed
    result = assess_frames(pixels, setup, masks)
    assert not result['individual_fiducial_camera_passed']


def test_moved_floor_is_not_initial_setup_preservation():
    setup, frame, masks = initial_fixture()
    frame[462:467] = 242
    cv2.line(frame, (20, 474), (748, 474), (70, 70, 70), 2)
    result = assess_frames(np.repeat(frame[None], 97, axis=0), setup, masks)
    assert result['initial_objects_passed']
    assert not result['initial_setup_passed']


def test_plateau_below_floor_is_not_valid_contact():
    setup, _, _ = initial_fixture()
    ys = [200, 300, 442, 442, 442, 442]
    measurement = {'score': 1., 'balls': [{'contact_frame': 2}] * 2, 'tracks': [[{'center': [224, y]} for y in ys]] * 2}
    passed, reason = contact_wrapper(measurement, setup)
    assert not passed and reason == 'plateau_not_at_floor'


def test_late_settling_after_rebound_is_not_initial_fall_time():
    setup, _, _ = initial_fixture()
    ys = [200, 300, 280, 432, 432, 432, 432]
    measurement = {'score': 1., 'balls': [{'contact_frame': 3}] * 2, 'tracks': [[{'center': [224, y]} for y in ys]] * 2}
    passed, reason = contact_wrapper(measurement, setup)
    assert not passed and reason == 'earlier_upward_motion_or_rebound'
