"""Validate first visual judgments, scope receipts, and the frozen calibration gate."""
import argparse
import hashlib
import json
from pathlib import Path

from assess_measurement import interval


def validate_assignments(manifest, truth):
    selected = {identifier: row for identifier, row in truth.items() if row['scene_seed'] in [81000, 81001]}
    identifiers = [row['case_id'] for row in manifest]
    conditions = {'equal_gravity', 'unequal_gravity', 'release_delay', 'occlusion', 'camera_pan', 'truncated'}
    expected_pairs = {(seed, condition) for seed in [81000, 81001] for condition in conditions}
    if len(identifiers) != 12 or len(set(identifiers)) != 12 or set(identifiers) != set(selected):
        raise ValueError('Assignments must be the exact twelve unique prespecified cases')
    if {(row['scene_seed'], row['condition']) for row in selected.values()} != expected_pairs:
        raise ValueError('Assignments do not cover the two scenes and six conditions')


def validate_receipt_hashes(judgment_bytes, payload_bytes, case, receipt):
    if hashlib.sha256(judgment_bytes).hexdigest() != receipt['judgment_sha256']:
        raise ValueError('Judgment changed after first-result receipt')
    if hashlib.sha256(payload_bytes).hexdigest() != receipt['input_payload_sha256'] or receipt['input_payload_sha256'] != case['payload_sha256']:
        raise ValueError('Payload hash does not match first-result receipt')


def validate(result, payload):
    fields = {'case_id', 'verdict', 'reason', 'events', 'assumptions', 'sheets_viewed'}
    if set(result) != fields or result['case_id'] != payload['case_id']:
        raise ValueError('Wrong schema or case identity')
    if result['verdict'] not in ['PASS', 'FAIL', 'ABSTAIN'] or not isinstance(result['reason'], str) or not result['reason'].strip():
        raise ValueError('Invalid verdict or missing explanation')
    event_names = {'left_release_frame_range', 'right_release_frame_range', 'left_contact_frame_range', 'right_contact_frame_range'}
    if not isinstance(result['events'], dict) or set(result['events']) != event_names:
        raise ValueError('Wrong event schema')
    for value in result['events'].values():
        if value is None:
            if result['verdict'] != 'ABSTAIN':
                raise ValueError('Conclusive verdict missing event evidence')
            continue
        if not isinstance(value, list) or len(value) != 2 or not all(type(item) is int for item in value) or not 0 <= value[0] <= value[1] < payload['frame_count']:
            raise ValueError('Invalid observed frame range')
    if set(result['assumptions']) != {'synchronized_release', 'static_camera', 'both_falls_observed'}:
        raise ValueError('Wrong assumption schema')
    if not all(value in ['supported', 'violated', 'unclear'] for value in result['assumptions'].values()):
        raise ValueError('Invalid assumption category')
    if result['verdict'] != 'ABSTAIN' and set(result['assumptions'].values()) != {'supported'}:
        raise ValueError('Conclusive verdict without admissible observations')
    if type(result['sheets_viewed']) is not int or result['sheets_viewed'] != len(payload['sheets']):
        raise ValueError('Incomplete frame-sheet coverage')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not replace first-result gate')
    manifest = json.loads((args.study / 'judge_input_manifest.json').read_text())
    truth = {row['case_id']: row for row in json.loads((args.study / 'data/private/heldout_truth.json').read_text())}
    receipts = json.loads(args.receipts.read_text())
    rows, errors = [], []
    validate_assignments(manifest, truth)
    if len({row['session_id'] for row in receipts['receipts'].values()}) != 12:
        raise ValueError('Judgment sessions are not twelve distinct sessions')
    for case in manifest:
        identifier = case['case_id']
        payload = json.loads((Path(case['public_input_dir']) / 'payload.json').read_text())
        target = truth[identifier]
        try:
            judgment_bytes = (args.study / 'judgments' / f'{identifier}.json').read_bytes()
            payload_bytes = (Path(case['public_input_dir']) / 'payload.json').read_bytes()
            result = json.loads(judgment_bytes)
            validate(result, payload)
            receipt = receipts['receipts'][identifier]
            validate_receipt_hashes(judgment_bytes, payload_bytes, case, receipt)
            if not receipt['all_assigned_images_viewed'] or receipt['private_or_outside_input_access'] or not receipt['fork_none_verified']:
                raise ValueError('Unverified or contaminated first judgment')
            if receipt['models_observed'] != ['gpt-6-astra'] or receipt['efforts_observed'] != ['high'] or receipt['actual_role'] != 'lazycodex-qa-executor':
                raise ValueError('Unexpected actual judge model or role')
            input_dir = Path(case['public_input_dir']).resolve()
            if input_dir != (args.study / 'judge_inputs' / identifier).resolve():
                raise ValueError('Unexpected input directory')
            if hashlib.sha256((input_dir / 'rubric.md').read_bytes()).hexdigest() != case['rubric_sha256']:
                raise ValueError('Rubric hash changed')
            observed_hashes = []
            frame_indices = []
            for sheet in payload['sheets']:
                path = (input_dir / sheet['path']).resolve()
                if not path.is_relative_to(input_dir):
                    raise ValueError('Sheet path outside assigned input')
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != sheet['sha256'] or actual not in receipt['actual_image_sha256']:
                    raise ValueError('Input image mismatch or not actually viewed')
                observed_hashes.append(actual)
                frame_indices.extend(sheet['frame_indices'])
            if observed_hashes != receipt['expected_image_sha256'] or frame_indices != list(range(payload['frame_count'])):
                raise ValueError('Input frame coverage changed')
            rows.append({'case_id': identifier, 'scene_seed': target['scene_seed'], 'condition': target['condition'], 'expected': target['expected'], 'observed': result['verdict'], 'correct': result['verdict'] == target['expected'], 'reason': result['reason']})
        except (ValueError, KeyError, FileNotFoundError) as error:
            errors.append({'case_id': identifier, 'error': str(error)})
    cells = []
    for condition in sorted({row['condition'] for row in truth.values()}):
        subset = [row for row in rows if row['condition'] == condition]
        correct = sum(row['correct'] for row in subset)
        cells.append({'condition': condition, 'assigned': 2, 'validated': len(subset), 'correct': correct, 'correct_fraction': correct / 2, 'clopper_pearson_95ci': interval(correct, 2), 'observed_counts': {verdict: sum(row['observed'] == verdict for row in subset) for verdict in ['PASS', 'FAIL', 'ABSTAIN']}})
    false_positives = sum(row['observed'] == 'FAIL' for row in rows if row['expected'] == 'PASS')
    obvious_violations_missed = sum(row['observed'] != 'FAIL' for row in rows if row['expected'] == 'FAIL')
    passed = not errors and len(rows) == 12 and all(row['correct'] for row in rows) and all(row['validated'] == row['correct'] == row['assigned'] == 2 for row in cells)
    measurement = json.loads((args.study / 'heldout_gate_v2.json').read_text())
    result = {'judge_passed': passed, 'measurement_passed': measurement['passed'], 'expanded_generation_authorized_by_gates': passed and measurement['passed'], 'validated_cases': len(rows), 'schema_or_provenance_errors': errors, 'false_positive_fail_verdicts_on_positive_cases': false_positives, 'positive_assigned': 2, 'obvious_violations_not_flagged': obvious_violations_missed, 'obvious_violation_assigned': 2, 'cells': cells, 'cases': rows, 'scope': 'Small procedural, static-frame VLM calibration; no human labels, real-data validation, general physical accuracy or generator leaderboard.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['cells', 'cases']}, indent=2))


if __name__ == '__main__':
    main()
