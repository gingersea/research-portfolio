"""Post-measurement validation against private analytic truth; no threshold tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


def interval(successes, total):
    return [float(beta.ppf(.025, successes, total - successes + 1)) if successes else 0., float(beta.ppf(.975, successes + 1, total - successes)) if successes < total else 1.]


def assess(rows, truth):
    expected = {row['case_id']: row for row in truth}
    assert len(expected) == len(truth)
    assert len(rows) == len(truth) * 2
    assert len({(row['case_id'], row['codec']) for row in rows}) == len(rows)
    cells, paired, gates = [], [], {}
    for codec in ['ffv1', 'h264']:
        for condition in sorted({row['condition'] for row in truth}):
            subset = [row for row in rows if row['codec'] == codec and expected[row['case_id']]['condition'] == condition]
            assert len(subset) == len(truth) // 6
            correct = sum(row['verdict'] == expected[row['case_id']]['expected'] for row in subset)
            cells.append({'codec': codec, 'condition': condition, 'n_scenes': len(subset), 'expected': expected[subset[0]['case_id']]['expected'], 'correct': correct, 'correct_fraction': correct / len(subset), 'clopper_pearson_95ci': interval(correct, len(subset)), 'observed_counts': {verdict: sum(row['verdict'] == verdict for row in subset) for verdict in ['PASS', 'FAIL', 'ABSTAIN']}, 'reasons': {reason: sum(row['reason'] == reason for row in subset) for reason in sorted({row['reason'] for row in subset})}})
        admissible = [row for row in rows if row['codec'] == codec and expected[row['case_id']]['condition'] in ['equal_gravity', 'unequal_gravity']]
        missing = [row['case_id'] for row in admissible if 'balls' not in row or 'score' not in row]
        errors = []
        for row in admissible:
            target = expected[row['case_id']]
            if 'balls' not in row or 'score' not in row:
                continue
            contact = max(abs(ball['contact_frame'] - true) for ball, true in zip(row['balls'], target['contact_frames_ceil']))
            release = max(abs(ball['release_frame'] - true) for ball, true in zip(row['balls'], target['release_frames']))
            height = max(abs(ball['height_px'] - true) for ball, true in zip(row['balls'], target['heights_px']))
            duration = max(abs(ball['flight_seconds'] * row['fps'] - (contact_time - start)) for ball, contact_time, start in zip(row['balls'], target['contact_frames_continuous'], target['release_frames']))
            error = {'case_id': row['case_id'], 'codec': codec, 'scene_seed': target['scene_seed'], 'condition': target['condition'], 'score': row['score'], 'native_unordered_score_on_same_measurements': 1 - abs(min(ball['height_px'] for ball in row['balls']) / max(ball['height_px'] for ball in row['balls']) - (min(ball['flight_seconds'] for ball in row['balls']) / max(ball['flight_seconds'] for ball in row['balls']))**2) / (min(ball['height_px'] for ball in row['balls']) / max(ball['height_px'] for ball in row['balls']) + (min(ball['flight_seconds'] for ball in row['balls']) / max(ball['flight_seconds'] for ball in row['balls']))**2), 'contact_error_frames': contact, 'release_error_frames': release, 'height_error_px': height, 'flight_error_frames': duration, 'score_error': abs(row['score'] - target['symmetric_score'])}
            errors.append(error)
            paired.append(error)
        maximums = {key: max((row[key] for row in errors), default=None) for key in ['contact_error_frames', 'release_error_frames', 'height_error_px', 'flight_error_frames', 'score_error']}
        limits = {'contact_error_frames': 1.5, 'release_error_frames': 1., 'height_error_px': 2., 'flight_error_frames': 1.5, 'score_error': .06}
        cell_passed = all(row['correct'] == row['n_scenes'] for row in cells if row['codec'] == codec)
        numeric_passed = not missing and all(maximums[key] is not None and maximums[key] <= limit for key, limit in limits.items())
        gates[codec] = {'classification_passed': cell_passed, 'numerical_passed': numeric_passed, 'admissible_assigned': len(admissible), 'admissible_measured': len(errors), 'missing': missing, 'maximum_errors': maximums, 'limits': limits, 'passed': cell_passed and numeric_passed}
    return {'passed': all(gate['passed'] for gate in gates.values()), 'gates_by_codec': gates, 'condition_cells': cells, 'paired_errors': paired, 'scope': 'Conditional procedural measurement validation. Codecs are technical repeats; frames are not independent samples.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not replace gate evidence')
    result = assess(json.loads(args.measurements.read_text()), json.loads(args.truth.read_text()))
    result['input_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.measurements, args.truth]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['paired_errors', 'condition_cells']}, indent=2))


if __name__ == '__main__':
    main()
