"""Versioned local handoff with actual copied-source/cache and report replay."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OLD = Path('/data/AI_Research/shared/handoff/world_model_integrated_matrix_20261115_v1_20261115T143741Z')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), str(target)
    else:
        shutil.copy2(source, target)
        assert sha(source) == sha(target)


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = Path('/data/AI_Research/shared/handoff') / ('world_model_proposal_completion_20261117_v1_' + stamp)
    destination.mkdir()
    # Baseline recovered from an already-frozen pre-study handoff, never guessed.
    baseline = ROOT / 'source_snapshots/from_frozen_20261115'
    baseline.mkdir(exist_ok=False)
    old_hashes = json.loads((OLD / 'MANIFEST.json').read_text())['copied_files']
    changed = ['benchmark/physcheck/relations/' + name for name in ['__main__.py', 'backends.py', 'contracts.py', 'pipeline.py']]
    diffs = []
    for name in changed:
        source = OLD / name
        assert sha(source) == old_hashes[name]['sha256']
        copy_file(source, baseline / Path(name).name)
        diffs.extend(difflib.unified_diff(source.read_text().splitlines(True), (PROJECT / name).read_text().splitlines(True),
                                        fromfile='frozen_20261115/' + name, tofile='current/' + name))
    for before, after in [(ROOT / 'source_snapshots/before/README.md', PROJECT / 'README.md'),
                          (ROOT / 'source_snapshots/relations_README_before.md', PROJECT / 'benchmark/physcheck/relations/README.md')]:
        diffs.extend(difflib.unified_diff(before.read_text().splitlines(True), after.read_text().splitlines(True),
                                        fromfile=str(before.relative_to(PROJECT)), tofile=str(after.relative_to(PROJECT))))
    created = ['benchmark/physcheck/relations/evaluation.py', 'benchmark/physcheck/judge/local_vlm.py',
               'benchmark/physcheck/report/render.py'] + ['tests/' + n for n in
               ['test_relations_public.py', 'test_geometry_completeness.py', 'test_local_vlm.py', 'test_report_render.py', 'test_prediction_format.py']]
    for name in created:
        diffs.extend(difflib.unified_diff([], (PROJECT / name).read_text().splitlines(True), fromfile='/dev/null', tofile=name))
    (ROOT / 'SOURCE_CHANGES.diff').write_text(''.join(diffs))
    save(ROOT / 'source_change_manifest.json', {'baseline': str(OLD), 'baseline_hashes_verified': True,
         'modified_source': {name: {'before': sha(OLD/name), 'after': sha(PROJECT/name)} for name in changed},
         'created_source': {name: sha(PROJECT/name) for name in created},
         'diff_sha256': sha(ROOT / 'SOURCE_CHANGES.diff')})

    ignore = shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc')
    for name in ['benchmark/physcheck', 'tests', 'docs', 'report']:
        shutil.copytree(PROJECT / name, destination / name, ignore=ignore)
    shutil.copytree(ROOT, destination / 'research' / ROOT.name, ignore=ignore)
    for name in ['README.md', 'pytest.ini', 'requirements-analysis.txt', 'environment-verified.json', 'LICENSE']:
        source = PROJECT / name
        if source.is_file():
            copy_file(source, destination / name)
    # Only compact historical source/summary dependencies; large histories remain external.
    compact = ['benchmark/results/raw/part_a_summary.json', 'benchmark/results/raw/part_a.jsonl',
               'benchmark/results/raw/detector_validation.json',
               'research/part_b_20261113T114040Z/run/results/leaderboard/part_b_leaderboard.json']
    for name in compact:
        copy_file(PROJECT / name, destination / name)
    for path in (PROJECT / 'research').glob('*/REPORT*'):
        if path.is_file():
            copy_file(path, destination / path.relative_to(PROJECT))
    copy_file(Path('/data/AI_Research/AGENTS.md'), destination / 'WORKSTATION_RULES_SNAPSHOT.md')

    public = json.loads((ROOT / 'public_manifest.json').read_text())
    replay = {'project': str(destination), 'cases': [], 'sources': {}}
    # Public tracking has no physical judgments; do not reuse unrelated old GPT calibration.
    metric = public['sources']['official']
    metric_target = destination / 'assets/official_numpy.py'
    copy_file(Path(metric['path']), metric_target)
    replay['sources']['official'] = {'path': str(metric_target), 'sha256': sha(metric_target)}
    expected = {}
    originals = {}
    for subdir in ['public_davis_fresh', 'public_full_rescored']:
        for case in json.loads((ROOT / subdir / 'cases.json').read_text()):
            assert case['status'] == 'ok'
            expected[case['id']] = case['evaluation']
            originals[case['id']] = ROOT / subdir / case['id'] / 'tracks.npz'
    for case in public['cases']:
        item = {key: value for key, value in case.items() if key != 'files'}
        item['files'] = {}
        inference_dir = ROOT / ('public_davis_fresh' if case['domain']=='davis' else 'public_pstudio_full') / case['id']
        sources = {name: Path(case['files'][name]['path']) for name in ['input', 'truth']}
        sources.update(prediction=inference_dir/'fresh_predictions.npz', prediction_receipt=inference_dir/'fresh_inference.json')
        for name, path in sources.items():
            target = destination / 'assets' / case['id'] / (name + path.suffix)
            copy_file(path, target)
            item['files'][name] = {'path': str(target), 'sha256': sha(target)}
        replay['cases'].append(item)
    save(destination / 'copied_public_manifest.json', replay)
    command = [sys.executable, '-m', 'physcheck.relations', '--manifest', str(destination/'copied_public_manifest.json'),
               '--output', str(destination/'replay_from_copy')]
    env = dict(os.environ, PYTHONPATH=str(destination/'benchmark'))
    result = subprocess.run(command, cwd=destination, env=env, capture_output=True, text=True)
    save(destination / 'COPY_REPLAY_COMMAND.json', {'command': command, 'cwd': str(destination),
         'PYTHONPATH': env['PYTHONPATH'], 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    assert result.returncode == 0, result.stderr
    rows = json.loads((destination/'replay_from_copy/cases.json').read_text())
    assert len(rows)==33 and all(r['status']=='ok' for r in rows)
    for row in rows:
        assert row['evaluation'] == expected[row['id']], row['id']
        current = np.load(destination/'replay_from_copy'/row['id']/'tracks.npz')
        prior = np.load(originals[row['id']])
        assert sorted(current.files) == sorted(prior.files)
        for key in current.files:
            assert np.array_equal(current[key], prior[key], equal_nan=True), (row['id'],key)
    # Regenerate from copied renderer, builder and copied JSON, byte for byte.
    builder = destination / 'research' / ROOT.name / 'build_reports.py'
    generated = destination / 'regenerated_reports'
    result = subprocess.run([sys.executable, str(builder), '--output', str(generated)], cwd=destination,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    reports = {'report.md': 'report/report.md', 'results_summary.md':'report/results_summary.md',
               'PROPOSAL_RESULTS_2026-11-17.zh.md':'docs/PROPOSAL_RESULTS_2026-11-17.zh.md'}
    for name, relative in reports.items():
        assert (generated/name).read_bytes() == (destination/relative).read_bytes(), name
    save(destination / 'COPY_REPLAY.json', {'cases': len(rows), 'all_evaluations_identical': True,
         'all_normalized_arrays_identical_including_missing_positions': True, 'regenerated_reports_byte_identical': list(reports),
         'fresh_neural_inference_rerun': False, 'public_has_no_physical_judge_calibration': True,
         'scope': 'Actual copied-source/copied-input/cache replay in existing environment; not a clean-machine model install'})
    with zipfile.ZipFile(destination/'source_bundle.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['benchmark/physcheck', 'tests']:
            for path in (destination/name).rglob('*.py'):
                archive.write(path, path.relative_to(destination))
        for path in (destination/'research'/ROOT.name).glob('*.py'):
            archive.write(path, path.relative_to(destination))
    files = {str(p.relative_to(destination)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in destination.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    save(destination/'MANIFEST.json', {'copied_files': files, 'model_weights_external': str(ROOT/'vlm_assets.json'),
         'public_upload_performed': False, 'raw_video_redistribution_rights_not_asserted': True})
    for name, info in files.items():
        assert sha(destination/name) == info['sha256']
    save(ROOT/'handoff_receipt.json', {'path': str(destination), 'files': len(files),
         'bytes': sum(i['bytes'] for i in files.values()), 'manifest_sha256': sha(destination/'MANIFEST.json'),
         'all_handoff_hashes_verified': True, 'copy_replay33_passed': True,
         'report_regeneration_byte_identical': True, 'no_new_publication': True})
    print((ROOT/'handoff_receipt.json').read_text())


if __name__ == '__main__':
    main()
