"""Read-only detector audit: distinguish absent cues, hue/area rejection and extra markers."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'principia_i2v_20261115'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_detector():
    spec = importlib.util.spec_from_file_location('old_i2v_detector', OLD / 'evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def green_candidates(frame):
    red, green, blue = [frame[..., index].astype(float) for index in range(3)]
    binary = np.uint8((green > 1.6 * red) & (green > 1.6 * blue) & (green > 100))
    count, _, stats, centers = cv2.connectedComponentsWithStats(binary)
    ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 30]
    return centers[ids], stats[ids, cv2.CC_STAT_AREA]


def main():
    output = ROOT / 'diagnosis'
    output.mkdir(exist_ok=False)
    # Snapshot original artifacts before any candidate correction is implemented.
    before = {str(path.relative_to(OLD)): sha(path) for path in sorted(OLD.rglob('*'))
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    (ROOT / 'original_before.sha256.json').write_text(json.dumps(before, indent=2) + '\n')
    detector = load_detector()
    setups = json.loads((OLD / 'input_manifest.json').read_text())
    gates = json.loads((OLD / 'pilot_gate.json').read_text())
    rows, summary = [], []
    for setup in setups:
        masks = np.load(OLD / 'inputs' / setup['initial_id'] / 'masks.npz', allow_pickle=False)
        masks = [masks['left'], masks['right']]
        references = np.array(setup['fiducial_centers'])
        for seed in [42, 43]:
            identifier = f'{setup["initial_id"]}_seed{seed}'
            pixels = np.load(OLD / 'generation/primary' / setup['initial_id'] / f'{identifier}_frames.npz', allow_pickle=False)['frames']
            old_gate = next(row for row in gates['cases'] if row['initial_id'] == setup['initial_id'] and row['seed'] == seed)
            valid = []
            marker_rows, object_status = [], [[], []]
            for index, frame in enumerate(pixels):
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                centers, areas = green_candidates(frame)
                distances = np.linalg.norm(references[:, None] - centers[None], axis=2)
                match_error = None
                if len(centers) >= 4:
                    r, c = linear_sum_assignment(distances)
                    match_error = float(distances[r, c].max())
                marker_rows.append({'green_components': len(centers), 'four_reference_max_error_px': match_error})
                usable = []
                for ball, color in enumerate(setup['colors_rgb_left_right']):
                    target = int(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0, 0, 0])
                    hue = np.abs(hsv[..., 0].astype(float) - target)
                    hue = np.minimum(hue, 180-hue)
                    gates_mask = (hue <= 12) & (hsv[..., 1] >= 110) & (hsv[..., 2] >= 90)
                    count, _, stats, component_centers = cv2.connectedComponentsWithStats(np.uint8(gates_mask))
                    ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 40]
                    found = detector.component(frame, color)
                    status = 'zero_color_components' if not ids else ('multiple_color_components' if len(ids) > 1 else (
                        'truncated' if not found['complete'] else (
                            'area_out_of_range' if not .75 <= found['area'] / masks[ball].sum() <= 1.25 else 'usable')))
                    okay = status == 'usable'
                    usable.append(okay)
                    object_status[ball].append(status)
                    rows.append({'clip': identifier, 'frame': index, 'object': ball, 'status': status,
                                 'color_component_count': len(ids),
                                 'largest_color_component_area': int(max(stats[ids, cv2.CC_STAT_AREA])) if ids else 0,
                                 'selected_area_ratio': found['area'] / masks[ball].sum() if found is not None else None,
                                 'green_components': len(centers), 'four_reference_max_error_px': match_error,
                                 'all_authored_markers_within_1_5px': match_error is not None and match_error <= 1.5})
                valid.append(all(usable))
            assert valid == old_gate['both_usable_by_frame'], 'Audit must reproduce every old object-availability decision'
            summary.append({'clip': identifier, 'reproduced_both_usable': sum(valid),
                            'object_status_counts': [dict(Counter(statuses)) for statuses in object_status],
                            'green_component_count_histogram': dict(Counter(row['green_components'] for row in marker_rows)),
                            'all_four_authored_markers_present_frames': sum(row['four_reference_max_error_px'] is not None and row['four_reference_max_error_px'] <= 1.5 for row in marker_rows),
                            'old_marker_reject_frames': sum(row['green_components'] != 4 for row in marker_rows),
                            'extra_green_with_authored_markers_present_frames': sum(row['green_components'] > 4 and row['four_reference_max_error_px'] is not None and row['four_reference_max_error_px'] <= 1.5 for row in marker_rows),
                            'first_both_unusable_frame': next((i for i, okay in enumerate(valid) if not okay), None)})
    with (output / 'frame_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
