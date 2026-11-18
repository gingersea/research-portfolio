"""Independent known-path controls first; exploratory reanalysis second."""
import argparse
from collections import Counter
import csv
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from candidate import marker_positions, track_observed
from diagnose import load_detector, sha

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'principia_i2v_20261115'
COLORS = [[225, 35, 45], [35, 65, 225]]
MARKERS = [[40, 32], [728, 32], [40, 488], [728, 488]]


def control_frames(seed, condition):
    rng = np.random.default_rng(seed)
    offset = rng.integers(-12, 13, size=(2, 2))
    initial = np.array([[224, 160], [544, 160]]) + offset
    radius = 32
    mask = np.zeros((512, 768), dtype=np.uint8)
    cv2.circle(mask, (224, 160), radius, 1, -1)
    setup = {'initial_centers': initial.tolist(), 'fiducial_centers': MARKERS,
             'fiducial_half_width': 10, 'radius': radius, 'expected_areas': [int(mask.sum())]*2,
             'colors_rgb_left_right': COLORS}
    frames, truth = [], []
    for t in range(17):
        centers = initial + np.array([[0, t*2], [0, t*2]])
        colors = COLORS
        if condition == 'hue_change' and t >= 4:
            colors = [[35, 190, 200], [190, 35, 180]]
        if condition == 'grayscale' and t >= 4:
            colors = [[130, 130, 130], [165, 165, 165]]
        if condition == 'same_hue' and t >= 4:
            colors = [COLORS[1], COLORS[1]]
        if condition == 'overlap':
            centers = np.array([[224+20*min(t, 8), 160], [544-20*min(t, 8), 160]])
        visible = [not (condition == 'absence' and 6 <= t <= 9), True]
        frame = np.full((512, 768, 3), 242, dtype=np.uint8)
        frame[465:] = 225
        cv2.line(frame, (20, 464), (748, 464), (70, 70, 70), 2)
        for x, y in MARKERS:
            cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), (25, 180, 35), -1)
        for center, color, present in zip(centers, colors, visible):
            if present:
                cv2.circle(frame, tuple(center), radius, color, -1)
        if condition == 'extra_green' and t >= 4:
            cv2.circle(frame, (384, 300), 12, (25, 180, 35), -1)
        if condition == 'near_clutter' and t >= 5:
            cv2.circle(frame, (int(centers[0, 0]+100), int(centers[0, 1])), 16, (150, 50, 180), -1)
        frames.append(frame)
        truth.append({'centers': centers.tolist(), 'visible': visible,
                      'identity_must_abstain': [condition == 'near_clutter' and t >= 5 or condition == 'overlap' and t >= 7]*2})
    # Overlap setup queries are fixed to the actual first frame.
    setup['initial_centers'] = truth[0]['centers']
    return np.array(frames), setup, truth


def controls():
    folder = ROOT / 'controls_v1'
    folder.mkdir(exist_ok=False)
    baseline = load_detector()
    rows = []
    conditions = ['clean', 'hue_change', 'grayscale', 'same_hue', 'absence', 'extra_green', 'near_clutter', 'overlap']
    for seed in range(710, 718):
        for condition in conditions:
            pixels, setup, truth = control_frames(seed, condition)
            tracks = track_observed(pixels, setup, pixels[0])
            identifier = f'{condition}_{seed}'
            target = folder / f'{identifier}.npz'
            np.savez_compressed(target, frames=pixels)
            (folder / f'{identifier}_truth.json').write_text(json.dumps({'setup': setup, 'frames': truth}, indent=2) + '\n')
            errors, false_observations, forbidden_identity = [], 0, 0
            both = old_both = 0
            for frame_index, (frame, predicted, actual) in enumerate(zip(pixels, tracks, truth)):
                both += all(item['eligible'] for item in predicted)
                old_items = [baseline.component(frame, color) for color in COLORS]
                old_both += all(item is not None and item['complete'] and .75 <= item['area']/setup['expected_areas'][i] <= 1.25 for i, item in enumerate(old_items))
                for i, item in enumerate(predicted):
                    if item['observed']:
                        if not actual['visible'][i]:
                            false_observations += 1
                        elif actual['identity_must_abstain'][i]:
                            # Only object0 is affected by near_clutter; both by overlap.
                            if condition != 'near_clutter' or i == 0:
                                forbidden_identity += 1
                        else:
                            errors.append(float(np.linalg.norm(np.array(item['center'])-actual['centers'][i])))
            easy = condition in ['clean', 'hue_change', 'grayscale', 'same_hue', 'extra_green']
            marker_pass = all(marker_positions(frame, np.array(MARKERS)) is not None for frame in pixels)
            maximum = max(errors) if errors else None
            passed = false_observations == 0 and forbidden_identity == 0 and maximum is not None and maximum <= 1 and (not easy or both == 17) and marker_pass
            rows.append({'control': identifier, 'condition': condition, 'seed': seed, 'frames': 17,
                         'both_eligible': both, 'original_both_eligible': old_both,
                         'observed_position_samples': len(errors), 'max_error_px': maximum,
                         'false_observations_during_absence': false_observations,
                         'observations_after_ambiguous_identity': forbidden_identity,
                         'all_markers_matched': marker_pass, 'passed': passed,
                         'array_sha256': sha(target)})
    result = {'controls': len(rows), 'frames': len(rows)*17, 'seeds': list(range(710, 718)),
              'passed': all(row['passed'] for row in rows), 'failed_controls': [row['control'] for row in rows if not row['passed']],
              'candidate_sha256': sha(ROOT / 'candidate.py'), 'protocol_sha256': sha(ROOT / 'CORRECTION_PROTOCOL.md'),
              'cases': rows, 'scope': 'Procedural known-path measurement checks, not generated-video labels or general tracking accuracy.'}
    (folder / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))


def audit():
    calibration = json.loads((ROOT / 'controls_v1/results.json').read_text())
    assert calibration['passed'], 'Controls failed; do not promote/reanalyse production automatically'
    assert sha(ROOT / 'candidate.py') == calibration['candidate_sha256']
    assert sha(ROOT / 'CORRECTION_PROTOCOL.md') == calibration['protocol_sha256']
    folder = ROOT / 'reanalysis_v1'
    folder.mkdir(exist_ok=False)
    setups = json.loads((OLD / 'input_manifest.json').read_text())
    old_gate = json.loads((OLD / 'pilot_gate.json').read_text())
    summaries, frame_rows = [], []
    fig, axes = plt.subplots(4, 5, figsize=(12, 7))
    for setup in setups:
        masks = np.load(OLD / 'inputs' / setup['initial_id'] / 'masks.npz', allow_pickle=False)
        setup['expected_areas'] = [int(masks['left'].sum()), int(masks['right'].sum())]
        initial = np.array(Image.open(OLD / setup['image']).convert('RGB'))
        for seed in [42, 43]:
            identifier = f'{setup["initial_id"]}_seed{seed}'
            pixels = np.load(OLD / 'generation/primary' / setup['initial_id'] / f'{identifier}_frames.npz', allow_pickle=False)['frames']
            tracks = track_observed(pixels, setup, initial)
            original = next(row for row in old_gate['cases'] if row['initial_id'] == setup['initial_id'] and row['seed'] == seed)
            markers = [marker_positions(frame, np.array(setup['fiducial_centers'])) for frame in pixels]
            marker_errors = [float(np.max(np.linalg.norm(item-markers[0], axis=1))) for item in markers if item is not None] if markers[0] is not None else []
            both = [all(item['eligible'] for item in row) for row in tracks]
            longest = maximum = 0
            for okay in both:
                longest = 0 if okay else longest+1
                maximum = max(maximum, longest)
            row_index = len(summaries)
            summaries.append({'clip': identifier, 'original_eligible_frames': original['usable_both_frames'],
                              'candidate_eligible_frames': sum(both), 'expected_frames': 97,
                              'candidate_fraction': sum(both)/97, 'candidate_longest_gap': maximum,
                              'candidate_coverage_passed': sum(both)/97 >= .95 and maximum <= 2,
                              'candidate_all_markers_matched': all(item is not None for item in markers),
                              'candidate_max_individual_marker_drift_px': max(marker_errors) if marker_errors else None,
                              'candidate_camera_passed': len(marker_errors) == 97 and max(marker_errors) <= 1.5,
                              'object_reason_counts': [dict(Counter(row[i]['reason'] for row in tracks)) for i in range(2)]})
            for frame_index, row in enumerate(tracks):
                for ball, item in enumerate(row):
                    frame_rows.append({'clip': identifier, 'frame': frame_index, 'object': ball, **item})
            (folder / f'{identifier}_tracks.json').write_text(json.dumps(tracks, indent=2) + '\n')
            for column, frame_index in enumerate([0, 1, 24, 60, 96]):
                axis = axes[row_index, column]
                axis.imshow(pixels[frame_index])
                for ball, item in enumerate(tracks[frame_index]):
                    if item['observed']:
                        x, y = item['center']
                        axis.plot(x, y, '+', color='lime' if item['eligible'] else 'orange', markersize=8)
                        axis.text(x+5, y, f'ID{ball}', color='black', fontsize=7)
                axis.set_title(f'{identifier} f{frame_index}', fontsize=8)
                axis.axis('off')
    fig.suptitle('Exploratory observed-component prototype; green=area eligible, orange=area/truncation rejected')
    fig.tight_layout()
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(folder / f'observation_overlay.{extension}', dpi=180)
    plt.close(fig)
    result = {'cases': summaries, 'original_expansion_gate_unchanged': False,
              'scope': 'The original expand_generation remains false; this is exploratory instrument availability, not physical scoring.',
              'candidate_sha256': sha(ROOT / 'candidate.py')}
    # Store the actual immutable original decision explicitly.
    result['original_expansion_gate_unchanged'] = old_gate['expand_generation'] is False
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (folder / 'frame_observations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    before = json.loads((ROOT / 'original_before.sha256.json').read_text())
    assert all(sha(OLD / name) == digest for name, digest in before.items())
    (folder / 'original_integrity.json').write_text(json.dumps({'passed': True, 'unchanged_files': len(before)}, indent=2) + '\n')
    (folder / 'correction.patch').write_text(''.join(difflib.unified_diff(
        (ROOT / 'candidate_before.py').read_text().splitlines(keepends=True),
        (ROOT / 'candidate.py').read_text().splitlines(keepends=True), fromfile='candidate_before.py', tofile='candidate.py')))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['controls', 'reanalysis'], required=True)
    args = parser.parse_args()
    controls() if args.phase == 'controls' else audit()
