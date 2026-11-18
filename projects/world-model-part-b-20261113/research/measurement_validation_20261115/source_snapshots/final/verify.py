"""Artifact audit and local-copy CPU reproduction, never overwriting outputs."""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from common import PAPER, ROOT, save, sha
from run import assert_frozen_sources, METRICS


def audit():
    assert_frozen_sources()
    prior = json.loads((PAPER / 'results/per_video.json').read_text())
    rows = json.loads((ROOT / 'results/per_video.json').read_text())
    lookup = {(r['case'], r['policy']): r for r in prior}
    for row in rows:
        assert row['accepted'] == row['correct'] + row['false_visible'] + row['wrong_location']
        base = lookup[row['case'], 'native']
        assert row['average_pts_within_thresh'] == base['average_pts_within_thresh']
        if row['policy'] in ['native', 'vc09']:
            previous = lookup[row['case'], 'native' if row['policy'] == 'native' else 'conservative_vc09']
            for key in METRICS:
                assert row[key] == previous[key], (row['case'], key)
            for current, old in [('accepted', 'accepted'), ('correct', 'correct_accepted4'), ('false_visible', 'false_visible')]:
                assert row[current] == previous[old]
    events = json.loads((ROOT / 'results/events.json').read_text())
    for policy in ['native', 'vc09', 'calibrated', 'calibrated_confirmation']:
        assert sum(e['policy'] == policy for e in events) == 363
    old3 = json.loads((PAPER / 'pilot3d_results/per_case.json').read_text())
    new3 = json.loads((ROOT / 'geometry_results/per_video.json').read_text())
    for row in new3:
        method = 'cotracker_estimated_depth' if row['method'] == 'co' else 'tapip3d_estimated_depth'
        base = next(r for r in old3 if r['method'] == method and r['case'] == row['case'])
        assert row['raw_epe_m'] == base['raw_epe3d_m']
        assert row['visible_post_query'] == base['visible_gt']
        assert row['oracle_gt_xy_out_of_bounds'] == 0
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'test_measurement.py')],
                           text=True, capture_output=True)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    bound = {}
    for case in json.loads((PAPER / 'data_manifest.json').read_text())['cases']:
        for path in [PAPER / case['input'], PAPER / case['truth'],
                     PAPER / 'predictions' / (case['case'] + '.npz'),
                     PAPER / 'predictions' / (case['case'] + '.json')]:
            bound[str(path)] = sha(path)
    for case in json.loads((PAPER / 'pilot3d_manifest.json').read_text())['cases']:
        for path in [PAPER / case['input'], PAPER / case['truth'],
                     PAPER / 'pilot3d_predictions' / (case['case'] + '.npz')]:
            bound[str(path)] = sha(path)
    save(ROOT / 'verification.json', {'status': 'passed', 'per_video_rows': len(rows),
         'unchanged_previous_metrics_rows': 60, 'events_per_policy': 363,
         'unchanged_previous_3d_rows': 6, 'tests_stdout': tests.stdout, 'tests_stderr': tests.stderr,
         'original_frozen_analytic_sources_unchanged': True, 'read_only_inputs': bound,
         'results': {str(p.relative_to(ROOT)): sha(p) for folder in ['results', 'geometry_results', 'description']
                     for p in (ROOT / folder).iterdir() if p.suffix in ['.json', '.csv']},
         'static_lsp': 'not run; basedpyright unavailable and user previously declined installation'})
    print(json.dumps({'audit': 'passed', 'tests': tests.stdout.strip(), 'bound_inputs': len(bound)}))


def freeze():
    output = ROOT / 'source_snapshots' / 'final'
    output.mkdir(exist_ok=False)
    files = {}
    diff = []
    before = ROOT / 'source_snapshots' / 'before_analysis'
    for path in sorted(ROOT.iterdir()):
        if path.suffix not in ['.py', '.md']:
            continue
        shutil.copy2(path, output / path.name)
        files[path.name] = sha(path)
        old = before / path.name
        oldlines = old.read_text().splitlines(keepends=True) if old.exists() else []
        diff.extend(difflib.unified_diff(oldlines, path.read_text().splitlines(keepends=True),
                                       fromfile='before/' + path.name if old.exists() else '/dev/null',
                                       tofile='final/' + path.name))
    (output / 'exact_changes_since_analysis_start.patch').write_text(''.join(diff))
    # This study began as a new directory. Preserve its complete added-source patch.
    full = []
    for name in sorted(files):
        full.extend(difflib.unified_diff([], (ROOT / name).read_text().splitlines(keepends=True),
                                        fromfile='/dev/null', tofile='new_study/' + name))
    (output / 'exact_new_study.patch').write_text(''.join(full))
    save(output / 'receipt.json', {'sources': files, 'before_stage': 'before_analysis',
         'new_directory_baseline': 'empty; prior studies never edited',
         'late_sources': 'description, audit, packaging and report only; original analytical source hashes unchanged'})
    print(json.dumps({'snapshot': str(output), 'sources': len(files)}))


def reproduce(bundle):
    project = bundle / 'reproduction_run'
    study = project / 'research' / ROOT.name
    study.mkdir(parents=True, exist_ok=False)
    for path in (bundle / 'research' / ROOT.name).iterdir():
        if path.suffix in ['.py', '.md']:
            shutil.copy2(path, study / path.name)
    shutil.copy2(bundle / 'README.original.md', project / 'README.md')
    (project / 'research' / PAPER.name).symlink_to(bundle / 'research' / PAPER.name, target_is_directory=True)
    commands = [['run.py', 'prepare'], ['run.py', 'calibrate'], ['run.py', 'evaluate'],
                ['geometry.py'], ['describe.py']]
    logs = []
    for command in commands:
        executed = [sys.executable, str(study / command[0]), *command[1:]]
        result = subprocess.run(executed, capture_output=True, text=True)
        logs.append({'command': executed, 'exit': result.returncode,
                     'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            save(bundle / 'REPRODUCTION_FAILED.json', logs)
            raise RuntimeError(result.stderr)
    reference = bundle / 'research' / ROOT.name
    compared = []
    for folder in ['results', 'geometry_results', 'description']:
        for path in (reference / folder).iterdir():
            if path.suffix not in ['.json', '.csv'] or path.name == 'receipt.json':
                continue
            copied = study / folder / path.name
            assert sha(path) == sha(copied), path
            compared.append(str(path.relative_to(reference)))
    assert json.loads((reference / 'split.json').read_text()) == json.loads((study / 'split.json').read_text())
    fits = [json.loads((d / 'calibration.json').read_text()) for d in [reference, study]]
    assert fits[0]['selected'] == fits[1]['selected'] and fits[0]['curve'] == fits[1]['curve']
    save(bundle / 'REPRODUCTION.json', {'status': 'passed', 'commands': logs,
         'data_tables_byte_identical': compared, 'calibration_curve_selection_and_split_equal': True,
         'numerical_inputs': 'copied original frozen predictions and labels inside bundle',
         'neural_inference_rerun': False, 'environment': sys.executable,
         'limitation': 'base Python/NumPy/Matplotlib environment external; no clean-machine installation test'})
    print(json.dumps({'reproduction': 'passed', 'byte_identical_tables': len(compared)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'freeze', 'reproduce'])
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    if args.stage == 'reproduce':
        assert args.bundle is not None
        reproduce(args.bundle.resolve())
    else:
        {'audit': audit, 'freeze': freeze}[args.stage]()
