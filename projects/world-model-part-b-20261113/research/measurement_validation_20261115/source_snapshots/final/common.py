"""Small, explicit scoring helpers; no label-dependent inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / 'paper_benchmark_20261115'
THRESHOLDS = [0., .5, .7, .8, .9, .95, .97, .98, .99, .995, .999, .9999]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def split_cases(names):
    ordered = sorted(names, key=lambda name: hashlib.sha256(
        ('measurement-validation-v1:' + name).encode()).hexdigest())
    return {'calibration': ordered[:10], 'validation': ordered[10:]}


def load_case(case):
    inputs = np.load(PAPER / case['input'])
    truth = np.load(PAPER / case['truth'])
    path = PAPER / 'predictions' / (case['case'] + '.npz')
    receipt = json.loads(path.with_suffix('.json').read_text())
    assert sha(path) == receipt['prediction_sha256']
    assert sha(PAPER / case['input']) == case['input_sha256']
    assert sha(PAPER / case['truth']) == case['truth_sha256']
    assert receipt['status'] == 'ok' and not receipt['future_truth_read']
    pred = np.load(path)
    return inputs['queries_tyx'], truth['target_points'], truth['occluded'], {
        key: pred[key].copy() for key in pred.files}


def two_frame_confirmation(raw, query_times):
    """[N,T] bool; post-loss declaration requires two high-score observations."""
    result = np.zeros_like(raw, dtype=bool)
    for point, query in enumerate(query_times.astype(int)):
        active = bool(raw[point, query])
        previous = active
        result[point, query] = active
        for t in range(query + 1, raw.shape[1]):
            current = bool(raw[point, t])
            active = current and (active or previous)
            result[point, t] = active
            previous = current
    return result


def counts(query, gt, occluded, xy, declared):
    evaluation = np.arange(occluded.shape[1])[None, :] > query[:, 0, None]
    positive = evaluation & ~occluded
    negative = evaluation & occluded
    error = np.linalg.norm(xy - gt, axis=-1)
    accepted = evaluation & declared
    correct = accepted & positive & (error < 4)
    return {'positive': int(positive.sum()), 'negative': int(negative.sum()),
            'accepted': int(accepted.sum()), 'correct': int(correct.sum()),
            'false_visible': int((accepted & negative).sum()),
            'wrong_location': int((accepted & positive & (error >= 4)).sum())}


def pooled(rows):
    result = {key: sum(row[key] for row in rows) for key in
              ['positive', 'negative', 'accepted', 'correct', 'false_visible', 'wrong_location']}
    result.update(precision=result['correct'] / result['accepted'] if result['accepted'] else None,
                  retention=result['correct'] / result['positive'] if result['positive'] else None,
                  false_visible_rate=result['false_visible'] / result['negative'] if result['negative'] else None)
    return result


def choose_threshold(curve):
    eligible = [r for r in curve if r['retention'] >= .2 and r['covered_videos'] >= 5]
    feasible = [r for r in eligible if r['precision'] is not None and r['precision'] >= .95]
    if feasible:
        selected = max(feasible, key=lambda r: (r['correct'], -r['threshold']))
    elif eligible:
        selected = max(eligible, key=lambda r: (r['precision'], r['correct'], -r['threshold']))
    else:
        return {'threshold': 1., 'target_met': False, 'reason': 'no coverage-eligible threshold'}
    return {**selected, 'target_met': bool(feasible),
            'reason': 'target satisfied on calibration' if feasible else 'precision target infeasible at required coverage'}
