"""Image-only gravity measurement; no imports/access to procedural ground truth."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def colored_component(frame, rgb):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    target = int(cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0, 0])
    difference = np.minimum(np.abs(hsv[:, :, 0].astype(float) - target), 180 - np.abs(hsv[:, :, 0].astype(float) - target))
    mask = np.uint8((difference <= 12) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 90))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 40]
    if len(candidates) != 1:
        return None
    index = candidates[0]
    return {'center': centers[index].tolist(), 'area': int(stats[index, cv2.CC_STAT_AREA])}


def measure_video(path, setup):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {'verdict': 'ABSTAIN', 'reason': 'decode_failure'}
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracks = [[], []]
    markers = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape != (setup['height'], setup['width'], 3):
            raise ValueError('Unexpected frame resolution')
        for ball, color in enumerate(setup['colors_rgb_left_right']):
            tracks[ball].append(colored_component(frame, color))
        green = (frame[:, :, 1].astype(float) > 1.6 * frame[:, :, 0]) & (frame[:, :, 1].astype(float) > 1.6 * frame[:, :, 2]) & (frame[:, :, 1] > 100)
        ys, xs = np.where(green)
        markers.append([float(xs.mean()), float(ys.mean())] if len(xs) >= 100 else None)
    cap.release()
    base = {'decoded_frames': len(markers), 'fps': fps, 'tracks': tracks}

    def abstain(reason, **extra):
        return {**base, 'verdict': 'ABSTAIN', 'reason': reason, **extra}

    if len(markers) != setup['frames'] or not np.isclose(fps, setup['fps']):
        return abstain('decode_count_or_fps')
    if any(marker is None for marker in markers):
        return abstain('fiducial_missing')
    drift = float(np.max(np.linalg.norm(np.array(markers) - markers[0], axis=1)))
    if drift > 1.5:
        return abstain('camera_motion', marker_drift_px=drift)
    if any(any(point is None for point in track) for track in tracks):
        return abstain('object_missing_or_ambiguous')
    balls = []
    contact_y = setup['floor_y'] - setup['radius']
    for track in tracks:
        positions = np.array([point['center'] for point in track])
        areas = np.array([point['area'] for point in track])
        if float(areas.max() / areas.min()) > 1.25:
            return abstain('unstable_area')
        y = positions[:, 1]
        candidates = [index for index in range(1, len(y) - 3) if y[index] >= contact_y - 2 and np.ptp(y[index:index + 4]) <= .6]
        if not candidates:
            return abstain('no_observed_contact')
        contact = candidates[0]
        moving = np.flatnonzero((y > y[0] + 2) & (np.arange(len(y)) < contact))
        if len(moving) < 8:
            return abstain('insufficient_fall_observations')
        coefficients = np.polyfit(moving, y[moving], 2)
        residual = float(np.sqrt(np.mean((np.polyval(coefficients, moving) - y[moving])**2)))
        if coefficients[0] <= 0 or residual > 1:
            return abstain('nonquadratic_or_wrong_direction', fit_rms_px=residual)
        release = float(-coefficients[1] / (2 * coefficients[0]))
        if release < -1.5 or release > contact - 8:
            return abstain('release_not_identifiable', release_frame=release)
        height = float(contact_y - y[0])
        flight = (contact - release) / fps
        if height <= 20 or flight <= 0:
            return abstain('invalid_height_or_time')
        balls.append({'contact_frame': contact, 'release_frame': release, 'height_px': height, 'flight_seconds': flight, 'fit_rms_px': residual})
    if abs(balls[0]['release_frame'] - balls[1]['release_frame']) > 1.5:
        return abstain('asynchronous_release', balls=balls)
    height_ratio = balls[0]['height_px'] / balls[1]['height_px']
    time_ratio_squared = (balls[0]['flight_seconds'] / balls[1]['flight_seconds'])**2
    score = 1 - abs(height_ratio - time_ratio_squared) / (height_ratio + time_ratio_squared)
    verdict = 'PASS' if score >= .90 else 'FAIL' if score <= .85 else 'ABSTAIN'
    return {**base, 'verdict': verdict, 'reason': 'measured_relation' if verdict != 'ABSTAIN' else 'borderline_relation', 'balls': balls, 'height_ratio': height_ratio, 'time_ratio_squared': time_ratio_squared, 'q': height_ratio / time_ratio_squared, 'score': score, 'marker_drift_px': drift}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'heldout'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to replace existing measurements')
    index = json.loads((args.data / f'{args.split}_public_index.json').read_text())
    rows = []
    for case in index:
        setup = json.loads((args.data / case['setup_path']).read_text())
        for codec, entry in case['videos'].items():
            rows.append({'case_id': case['case_id'], 'codec': codec, **measure_video(args.data / entry['path'], setup)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'split': args.split, 'measured_clips': len(rows), 'verdict_counts': {verdict: sum(row['verdict'] == verdict for row in rows) for verdict in ['PASS', 'FAIL', 'ABSTAIN']}}))


if __name__ == '__main__':
    main()
