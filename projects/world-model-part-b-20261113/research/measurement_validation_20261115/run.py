"""Prepare -> calibrate -> evaluate; calibration opens only development labels."""
import argparse
from datetime import datetime, timezone
import json
import platform
import shutil
import sys

import numpy as np

from common import (PAPER, ROOT, THRESHOLDS, choose_threshold, counts, load_case,
                    pooled, save, sha, split_cases, two_frame_confirmation)

sys.path.insert(0, str(PAPER))
from official_numpy import compute_tapvid_metrics
from assess import diagnostics

METRICS = ['average_jaccard', 'average_pts_within_thresh', 'occlusion_accuracy']
POLICIES = ['native', 'vc09', 'calibrated', 'calibrated_confirmation']


def manifest():
    return json.loads((PAPER / 'data_manifest.json').read_text())['cases']


def prepare():
    split = split_cases([c['case'] for c in manifest()])
    snapshot = ROOT / 'source_snapshots' / 'before_analysis'
    snapshot.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in ROOT.iterdir():
        if path.suffix in ['.py', '.md']:
            shutil.copy2(path, snapshot / path.name)
            sources[path.name] = sha(path)
    dependencies = {str(PAPER / name): sha(PAPER / name) for name in
                    ['data_manifest.json', 'official_numpy.py', 'assess.py', 'pilot3d_manifest.json',
                     'REPORT.zh.md', 'results/summary.json', 'pilot3d_results/summary.json']}
    dependencies[str(ROOT.parents[1] / 'README.md')] = sha(ROOT.parents[1] / 'README.md')
    save(ROOT / 'split.json', split)
    save(snapshot / 'receipt.json', {'utc': datetime.now(timezone.utc).isoformat(), 'sources': sources,
         'dependencies': dependencies, 'split_sha256': sha(ROOT / 'split.json'),
         'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
         'command': sys.argv, 'previous_baseline_exposure_all30': True})
    print(json.dumps(split))


def assert_frozen_sources():
    receipt = json.loads((ROOT / 'source_snapshots/before_analysis/receipt.json').read_text())
    for name, digest in receipt['sources'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in receipt['dependencies'].items():
        assert sha(name) == digest, name
    assert sha(ROOT / 'split.json') == receipt['split_sha256']


def calibrate():
    assert_frozen_sources()
    split = json.loads((ROOT / 'split.json').read_text())
    cases = [c for c in manifest() if c['case'] in split['calibration']]
    curve = []
    for threshold in THRESHOLDS:
        rows = []
        for case in cases:
            query, gt, occ, pred = load_case(case)
            declared = (pred['visibility'] * pred['confidence']).T > threshold
            row = counts(query, gt, occ, pred['xy'].transpose(1, 0, 2), declared)
            rows.append({'case': case['case'], **row})
        curve.append({'threshold': threshold, **pooled(rows),
                      'covered_videos': sum(r['accepted'] >= .1 * r['positive'] and r['positive'] > 0 for r in rows),
                      'videos': rows})
    selected = choose_threshold(curve)
    save(ROOT / 'calibration.json', {'utc': datetime.now(timezone.utc).isoformat(),
         'fit_label_cases': [c['case'] for c in cases], 'validation_labels_used': False,
         'split_sha256': sha(ROOT / 'split.json'), 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
         'source_sha256': sha(ROOT / 'run.py'), 'curve': curve, 'selected': selected})
    print(json.dumps({k: v for k, v in selected.items() if k != 'videos'}))


def summarize(rows):
    return {**pooled(rows), **{key: float(np.mean([r[key] for r in rows])) for key in METRICS},
            'events': sum(r['event_count'] for r in rows),
            'recovered_events': sum(r['recovered_event_count'] for r in rows)}


def bootstrap(selected):
    native = [r for r in selected if r['policy'] == 'native']
    names = [r['case'] for r in native]
    indices = np.random.default_rng(20261115).integers(0, len(names), (10000, len(names)))
    lookup = {(r['policy'], r['case']): r for r in selected}
    results = []
    for policy in POLICIES:
        rows = [lookup[policy, name] for name in names]
        differences = {}
        for metric in [*METRICS, 'precision', 'retention', 'false_visible_rate']:
            def sample(values):
                if metric in METRICS:
                    return np.array([r[metric] for r in values])[indices].mean(axis=1)
                numerator, denominator = {'precision': ('correct', 'accepted'),
                     'retention': ('correct', 'positive'), 'false_visible_rate': ('false_visible', 'negative')}[metric]
                n = np.array([r[numerator] for r in values])[indices].sum(axis=1)
                d = np.array([r[denominator] for r in values])[indices].sum(axis=1)
                return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)
            delta = sample(rows) - sample(native)
            finite = delta[np.isfinite(delta)]
            differences[metric] = {'delta': summarize(rows)[metric] - summarize(native)[metric],
                'ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                'valid_resamples': len(finite)}
        results.append({'policy': policy, 'versus': 'native', 'differences': differences})
    return results


def evaluate():
    assert_frozen_sources()
    fit_path = ROOT / 'calibration.json'
    fit_hash = sha(fit_path)
    fit = json.loads(fit_path.read_text())
    split = json.loads((ROOT / 'split.json').read_text())
    assert not fit['validation_labels_used']
    assert set(fit['fit_label_cases']) == set(split['calibration'])
    threshold = fit['selected']['threshold']
    rows, events = [], []
    for case in manifest():
        query, gt, occ, pred = load_case(case)
        xy = pred['xy'].transpose(1, 0, 2)
        score = (pred['visibility'] * pred['confidence']).T
        assert np.isfinite(score).all() and np.isfinite(xy).all()
        declarations = {'native': pred['visible'].T, 'vc09': score > .9,
            'calibrated': score > threshold,
            'calibrated_confirmation': two_frame_confirmation(score > threshold, query[:, 0])}
        group = 'calibration' if case['case'] in split['calibration'] else 'validation'
        for policy, declared in declarations.items():
            official = compute_tapvid_metrics(query[None], occ[None], gt[None],
                                             (~declared)[None], xy[None], 'first')
            diagnostic = diagnostics(query, gt, occ, xy, declared)
            for event in diagnostic['events']:
                point, start, end = event['point'], event['return_frame'], event['visible_run_end'] + 1
                error = np.linalg.norm(xy[point, start:end] - gt[point, start:end], axis=-1)
                causes = 'recovered' if event['recovered'] else (
                    'no_accurate_position' if not (error < 4).any() else 'accurate_position_rejected')
                events.append({'case': case['case'], 'split': group, 'policy': policy,
                               **event, 'failure_class': causes})
            rows.append({'case': case['case'], 'split': group, 'policy': policy,
                 **{key: float(official[key][0]) for key in METRICS},
                 **counts(query, gt, occ, xy, declared),
                 'event_count': len(diagnostic['events']),
                 'recovered_event_count': sum(e['recovered'] for e in diagnostic['events'])})
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    summary = [{'split': group, 'policy': policy, 'videos': len(selected), **summarize(selected)}
               for group in ['calibration', 'validation'] for policy in POLICIES
               if (selected := [r for r in rows if r['split'] == group and r['policy'] == policy])]
    save(out / 'per_video.json', rows)
    save(out / 'events.json', events)
    save(out / 'summary.json', summary)
    save(out / 'paired_bootstrap.json', bootstrap([r for r in rows if r['split'] == 'validation']))
    save(out / 'receipt.json', {'calibration_sha256_before': fit_hash, 'calibration_sha256_after': sha(fit_path),
         'utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
         'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'sources_frozen': True,
         'split_sha256': sha(ROOT / 'split.json'), 'resampling_unit': 'video',
         'resamples': 10000, 'seed': 20261115, 'prior_baseline_exposure': True,
         'inference_rerun': False})
    assert fit_hash == sha(fit_path)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'calibrate', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare, 'calibrate': calibrate, 'evaluate': evaluate}[args.stage]()
