"""Known-geometry controls; no generated-video labels are assumed."""
import cv2
import numpy as np

from candidate import marker_positions, track_observed

COLORS = [[225, 35, 45], [35, 65, 225]]
MARKERS = [[20, 20], [300, 20], [20, 220], [300, 220]]
CENTERS = [[90, 90], [230, 90]]


def draw(centers=CENTERS, colors=COLORS, visible=(True, True), markers=True):
    image = np.full((240, 320, 3), 242, dtype=np.uint8)
    if markers:
        for x, y in MARKERS:
            cv2.rectangle(image, (x-10, y-10), (x+10, y+10), (25, 180, 35), -1)
    for center, color, present in zip(centers, colors, visible):
        if present:
            cv2.circle(image, tuple(center), 10, color, -1)
    return image


def setup():
    mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.circle(mask, (90, 90), 10, 1, -1)
    return {'colors_rgb_left_right': COLORS, 'initial_centers': CENTERS, 'radius': 10,
            'fiducial_centers': MARKERS, 'fiducial_half_width': 10,
            'expected_areas': [int(mask.sum())]*2}


def test_extra_green_artifact_is_not_missing_authored_marker():
    image = draw()
    cv2.circle(image, (160, 150), 8, (25, 180, 35), -1)
    positions = marker_positions(image, np.array(MARKERS))
    assert positions is not None
    assert np.max(np.linalg.norm(positions-np.array(MARKERS), axis=1)) < .01


def test_missing_marker_still_abstains():
    image = draw()
    image[5:35, 5:35] = 242
    assert marker_positions(image, np.array(MARKERS)) is None


def test_moving_marker_does_not_pass_camera_threshold():
    image = draw()
    image[5:35, 5:35] = 242
    cv2.rectangle(image, (15, 10), (35, 30), (25, 180, 35), -1)
    positions = marker_positions(image, np.array(MARKERS))
    assert positions is None or np.max(np.linalg.norm(positions-np.array(MARKERS), axis=1)) > 1.5


def test_hue_changes_do_not_erase_an_observed_object():
    frames = [draw(), draw(colors=[[35, 65, 225], [35, 65, 225]]), draw(colors=[[130, 130, 130], [130, 130, 130]])]
    tracks = track_observed(frames, setup(), frames[0])
    assert all(all(item['eligible'] for item in row) for row in tracks)
    assert all(np.linalg.norm(np.array(item['center'])-CENTERS[j]) < .1 for row in tracks for j, item in enumerate(row))


def test_missing_pixels_are_not_filled_by_association_memory():
    frames = [draw(), draw(visible=(False, True)), draw()]
    tracks = track_observed(frames, setup(), frames[0])
    assert not tracks[1][0]['observed']
    assert tracks[1][0]['center'] is None


def test_long_disappearance_terminates_identity_before_reappearance():
    frames = [draw()] + [draw(visible=(False, True))]*3 + [draw()]
    tracks = track_observed(frames, setup(), frames[0])
    assert not tracks[-1][0]['observed']


def test_ambiguous_overlap_does_not_claim_two_visible_objects():
    frames = [draw(), draw(centers=[[150, 90], [150, 90]])]
    tracks = track_observed(frames, setup(), frames[0])
    assert not all(item['observed'] for item in tracks[-1])


def test_near_background_object_stays_unavailable():
    frames = [draw(), draw(colors=[[238, 238, 238], COLORS[1]])]
    tracks = track_observed(frames, setup(), frames[0])
    assert not tracks[-1][0]['observed']


def test_area_change_is_separate_from_object_observation():
    image = draw(visible=(False, True))
    cv2.circle(image, (90, 90), 6, COLORS[0], -1)
    tracks = track_observed([draw(), image], setup(), draw())
    assert tracks[-1][0]['observed']
    assert not tracks[-1][0]['eligible']
