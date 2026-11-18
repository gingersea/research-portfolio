"""Initial-state, instrument-coverage and physics-applicability gates, fixed in advance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    target = int(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0, 0, 0])
    distance = np.abs(hsv[:, :, 0].astype(float) - target)
    distance = np.minimum(distance, 180 - distance)
    binary = np.uint8((distance <= 12) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 90))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 40]
    if len(candidates) != 1:
        return None
    i = candidates[0]
    x, y, width, height, area = stats[i]
    complete = x > 0 and y > 0 and x + width < frame.shape[1] and y + height < frame.shape[0]
    return {'center': centers[i], 'area': int(area), 'mask': labels == i, 'complete': bool(complete)}


def fiducials(frame):
    red, green, blue = [frame[:, :, index].astype(float) for index in range(3)]
    mask = np.uint8((green > 1.6 * red) & (green > 1.6 * blue) & (green > 100))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    selected = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 30]
    return centers[selected] if len(selected) == 4 else None


def matched_positions(reference, observed):
    if observed is None:
        return None, None
    distances = np.linalg.norm(np.asarray(reference)[:, None] - observed[None], axis=2)
    rows, columns = linear_sum_assignment(distances)
    ordered = np.empty_like(observed)
    ordered[rows] = observed[columns]
    return ordered, np.linalg.norm(ordered - reference, axis=1)


def floor_line(frame, expected_y):
    start = max(0, int(expected_y - .1 * frame.shape[0]))
    end = min(frame.shape[0], int(expected_y + .1 * frame.shape[0]) + 1)
    strip = frame[start:end, int(.05 * frame.shape[1]):int(.95 * frame.shape[1])].astype(float)
    near_gray = np.ptp(strip, axis=2) <= 20
    dark = strip.max(axis=2) <= 150
    candidates = np.flatnonzero((near_gray & dark).mean(axis=1) >= .70)
    if not len(candidates):
        return None
    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
    groups = [group for group in groups if len(group) <= 7]
    if len(groups) != 1:
        return None
    return float(np.median(groups[0]) + start)


def longest_false_run(values):
    maximum = current = 0
    for value in values:
        current = 0 if value else current + 1
        maximum = max(maximum, current)
    return maximum


def assess_frames(pixels, setup, input_masks, expected_frames=97):
    if not len(pixels):
        return {'initial_objects_passed': False, 'initial_setup_passed': False, 'coverage_passed': False, 'usable_both_frames': 0, 'expected_frames': expected_frames, 'usable_both_fraction': 0., 'longest_missing_run': expected_frames}
    initial_objects = [component(pixels[0], color) for color in setup['colors_rgb_left_right']]
    object_metrics = []
    for index, observed in enumerate(initial_objects):
        if observed is None:
            object_metrics.append({'object': index, 'passed': False, 'reason': 'missing_or_ambiguous'})
            continue
        truth = input_masks[index].astype(bool)
        intersection = int(np.logical_and(observed['mask'], truth).sum())
        union = int(np.logical_or(observed['mask'], truth).sum())
        iou = intersection / union if union else 0
        error = float(np.linalg.norm(observed['center'] - setup['initial_centers'][index]))
        radius_ratio = float(np.sqrt(observed['area'] / truth.sum()))
        passed = observed['complete'] and error <= .25 * setup['radius'] and .75 <= radius_ratio <= 1.25 and iou >= .60
        object_metrics.append({'object': index, 'passed': bool(passed), 'center_error_px': error, 'mask_iou': iou, 'equivalent_radius_ratio': radius_ratio, 'complete': observed['complete']})
    first_markers, marker_errors = matched_positions(np.array(setup['fiducial_centers']), fiducials(pixels[0]))
    initial_floor = floor_line(pixels[0], setup['floor_y'])
    setup_passed = marker_errors is not None and float(marker_errors.max()) <= 3 and initial_floor is not None and abs(initial_floor - setup['floor_y']) <= 3
    valid, individual_drift, floors, tracks = [], [], [], [[], []]
    marker_missing = 0
    for frame in pixels[:expected_frames]:
        observed = [component(frame, color) for color in setup['colors_rgb_left_right']]
        usable = True
        for index, item in enumerate(observed):
            tracks[index].append(item['center'].tolist() if item is not None else None)
            usable = usable and item is not None and item['complete'] and .75 <= item['area'] / input_masks[index].sum() <= 1.25
        valid.append(bool(usable))
        current = fiducials(frame)
        if current is None or first_markers is None:
            marker_missing += 1
        else:
            _, drift = matched_positions(first_markers, current)
            individual_drift.append(float(drift.max()))
        floors.append(floor_line(frame, setup['floor_y']))
    valid += [False] * max(0, expected_frames - len(valid))
    max_drift = max(individual_drift) if individual_drift else None
    floor_errors = [abs(value - setup['floor_y']) for value in floors if value is not None]
    all_floor_observed = len(floor_errors) == expected_frames
    camera_passed = marker_missing == 0 and len(pixels) == expected_frames and max_drift is not None and max_drift <= 1.5
    floor_passed = all_floor_observed and max(floor_errors) <= 3
    coverage = sum(valid) / expected_frames
    missing_run = longest_false_run(valid)
    return {'initial_objects_passed': all(row['passed'] for row in object_metrics), 'initial_object_metrics': object_metrics, 'initial_setup_passed': bool(setup_passed), 'initial_fiducial_max_error_px': float(marker_errors.max()) if marker_errors is not None else None, 'initial_floor_y': initial_floor, 'usable_both_frames': sum(valid), 'expected_frames': expected_frames, 'usable_both_fraction': coverage, 'longest_missing_run': missing_run, 'coverage_passed': coverage >= .95 and missing_run <= 2 and len(pixels) == expected_frames, 'individual_fiducial_camera_passed': bool(camera_passed), 'fiducial_unavailable_frames': marker_missing, 'maximum_individual_fiducial_drift_px': max_drift, 'floor_observability_passed': bool(floor_passed), 'floor_unavailable_frames': expected_frames - len(floor_errors), 'maximum_floor_error_px': max(floor_errors) if floor_errors else None, 'both_usable_by_frame': valid, 'centers_by_frame': tracks}


def contact_wrapper(measurement, setup):
    if 'score' not in measurement or len(measurement.get('balls', [])) != 2:
        return False, 'frozen_measurement_unavailable'
    target = setup['floor_y'] - setup['radius']
    for index, ball in enumerate(measurement['balls']):
        track = measurement['tracks'][index]
        contact = ball['contact_frame']
        if len(track) < contact + 4 or any(point is None for point in track[:contact + 4]):
            return False, 'incomplete_contact_evidence'
        ys = np.array([point['center'][1] for point in track])
        if np.max(np.abs(ys[contact:contact + 4] - target)) > 2:
            return False, 'plateau_not_at_floor'
        if np.any(np.diff(ys[:contact + 1]) < -1):
            return False, 'earlier_upward_motion_or_rebound'
    return True, 'initial_contact_observed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse to replace gate results')
    source = STUDY.parent / 'principia_gravity_20261114/measure.py'
    old = json.loads((STUDY.parent / 'principia_gravity_20261114/source_snapshots/heldout_v2/manifest.json').read_text())
    assert sha(source) == old['source_sha256']['measure.py']
    spec = importlib.util.spec_from_file_location('frozen_gravity_measure', source)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    rows = []
    inspection = STUDY / 'inspection'
    inspection.mkdir(exist_ok=True)
    paired_noise = {}
    setups = json.loads((STUDY / 'input_manifest.json').read_text())
    assert len(setups) == 2 and {row['initial_id'] for row in setups} == {'A', 'B'}
    for setup in setups:
        masks = np.load(STUDY / 'inputs' / setup['initial_id'] / 'masks.npz', allow_pickle=False)
        input_masks = [masks['left'], masks['right']]
        original = np.array(Image.open(STUDY / setup['image']).convert('RGB'))
        for seed in [42, 43]:
            identifier = f'{setup["initial_id"]}_seed{seed}'
            receipt_path = args.generation / setup['initial_id'] / f'{identifier}.json'
            if not receipt_path.exists():
                rows.append({'initial_id': setup['initial_id'], 'seed': seed, 'status': 'missing_attempt', 'initial_objects_passed': False, 'initial_setup_passed': False, 'coverage_passed': False, 'physics_relation_available': False})
                continue
            receipt = json.loads(receipt_path.read_text())
            if receipt['status'] != 'ok':
                rows.append({'initial_id': setup['initial_id'], 'seed': seed, 'status': receipt['status'], 'initial_objects_passed': False, 'initial_setup_passed': False, 'coverage_passed': False, 'physics_relation_available': False})
                continue
            array = STUDY / receipt['array']
            video = STUDY / receipt['video']
            assert sha(array) == receipt['array_sha256'] and sha(video) == receipt['video_sha256']
            assert not receipt['first_frame_replaced'] and not receipt['conditioning']['observer_mutated_image_or_latents']
            pixels = np.load(array, allow_pickle=False)['frames']
            assert pixels.dtype == np.uint8 and pixels.shape == (97, 512, 768, 3)
            cap = cv2.VideoCapture(str(video))
            decoded = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                assert frame.shape == (512, 768, 3)
                decoded += 1
            cap.release()
            assert decoded == 97 and abs(fps - 24) < 1e-6
            metrics = assess_frames(pixels, setup, input_masks)
            # Run the frozen tracker on a lossless video derived from the stored
            # native uint8 array, not on lossy MP4 for the primary instrument.
            raw_video = inspection / f'{identifier}_native.mkv'
            if not raw_video.exists():
                import subprocess
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '768x512', '-r', '24', '-i', 'pipe:0', '-an', '-c:v', 'ffv1', '-threads', '1', str(raw_video)], input=pixels.tobytes(), check=True)
            measured = frozen.measure_video(raw_video, {**setup, 'frames': 97, 'fps': 24})
            contact_ok, contact_reason = contact_wrapper(measured, setup)
            applicable = metrics['initial_objects_passed'] and metrics['initial_setup_passed'] and metrics['individual_fiducial_camera_passed'] and metrics['floor_observability_passed'] and contact_ok
            available = bool(applicable and 'score' in measured)
            measurement_path = inspection / f'{identifier}_frozen_measurement.json'
            measurement_path.write_text(json.dumps(measured, indent=2) + '\n')
            row = {'initial_id': setup['initial_id'], 'seed': seed, 'status': 'ok', 'decoded_frames': decoded, 'metadata_fps': fps, **metrics, 'first_frame_pixel_mae_diagnostic': float(np.abs(pixels[0].astype(float) - original).mean()), 'frozen_measurement_reason': measured['reason'], 'frozen_measurement_raw_score': measured.get('score'), 'contact_wrapper_reason': contact_reason, 'physics_relation_available': available, 'physics_score': measured['score'] if available else None, 'physics_verdict': measured['verdict'] if available else 'N/A', 'frame_array_sha256': receipt['array_sha256'], 'mp4_sha256': receipt['video_sha256'], 'conditioning': receipt['conditioning']}
            rows.append(row)
            paired_noise.setdefault(seed, []).append(receipt['conditioning']['unconditioned_noise_sha256'])
            sheet = Image.new('RGB', (6 * 256, 17 * 188), 'white')
            draw = ImageDraw.Draw(sheet)
            for index, frame in enumerate(pixels):
                x, y = (index % 6) * 256, (index // 6) * 188
                sheet.paste(Image.fromarray(frame).resize((256, 171)), (x, y))
                draw.text((x + 4, y + 172), f'{identifier} f{index} t{index/24:.3f}', fill='black')
            sheet.save(inspection / f'{identifier}_all97.jpg', quality=95)
    paired = all(len(values) == 2 and len(set(values)) == 1 for values in paired_noise.values()) and set(paired_noise) == {42, 43}
    initial = all(row['initial_objects_passed'] and row['initial_setup_passed'] for row in rows)
    coverage = all(row['coverage_passed'] for row in rows)
    per_setup = {identifier: sum(row['physics_relation_available'] for row in rows if row['initial_id'] == identifier) for identifier in ['A', 'B']}
    assert len({(row['initial_id'], row['seed']) for row in rows}) == 4
    expanded = len(rows) == 4 and initial and coverage and paired and all(value >= 1 for value in per_setup.values())
    result = {'planned_clips': 4, 'successful_generation_clips': sum(row['status'] == 'ok' for row in rows), 'initial_all_passed': initial, 'coverage_all_passed': coverage, 'physics_available_by_setup': per_setup, 'paired_seed_unconditioned_noise_verified': paired, 'expand_generation': expanded, 'cases': rows, 'scope': 'Procedural input/native I2V feasibility, not a real-data benchmark or human-labeled tracking accuracy.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
