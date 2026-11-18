"""Verify unchanged sources, exercise a negative input, and replay copied code."""
from datetime import datetime, timezone
import difflib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha, verify_file
from physcheck.relations.visual import validate_opinion


def audit():
    pre = json.loads((ROOT / 'pre_run_receipt.json').read_text())
    for relative, digest in pre['code_sha256'].items():
        assert sha(PROJECT / relative) == digest
    for relative in ['README.md', 'benchmark/physcheck/relations/backends.py', 'benchmark/physcheck/relations/contracts.py']:
        assert sha(PROJECT / relative) == sha(ROOT / 'source_snapshots/before' / relative)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    entries = {e['path']: e for c in manifest['cases'] for e in c['files'].values()}
    entries.update({e['path']: e for e in manifest['sources'].values()})
    for entry in entries.values():
        verify_file(entry)
    opinions = json.loads((ROOT / 'visual_opinions.json').read_text())
    for opinion in opinions:
        folder = ROOT / 'visual_inputs' / opinion['opaque_case_id']
        folder.mkdir(parents=True, exist_ok=False)
        source_case = next(c for c in json.loads((ROOT / 'with_visual325/cases.json').read_text())
                           if c.get('visual_judge', {}).get('payload_sha256') == opinion['payload_sha256'])
        source = ROOT / 'with_visual325' / source_case['id'] / 'blind_visual_input'
        for path in source.iterdir():
            shutil.copy2(path, folder / path.name)
            assert sha(path) == sha(folder / path.name)
        assert validate_opinion(opinion, folder / 'visual_judge_payload.json')['status'] == 'diagnostic'
    tests = json.loads((ROOT / 'tests_receipt.json').read_text())
    assert tests['exit'] == 0 and '98 passed' in tests['stdout']
    summary = json.loads((ROOT / 'analysis/summary.json').read_text())
    assert summary['numeric_channels_unchanged_after_visual_import'] == 325
    save(ROOT / 'verification.json', {'status': 'passed', 'bound_existing_input_files': len(entries),
         'existing_files_sha256': {name: entry['sha256'] for name, entry in entries.items()},
         'raw_measurement_and_model_backend_unchanged': True, 'original_README_unchanged': True,
         'production_sources_equal_pre_run': True, 'tests_receipt_sha256': sha(ROOT / 'tests_receipt.json'),
         'analysis_summary_sha256': sha(ROOT / 'analysis/summary.json'),
         'diagnostic_opinions': 4, 'independent_i2v_visual_validation_pending': 4,
         'LSP': 'unavailable; no installation or pass claimed', 'review': 'author self-review'})


def fault():
    opinions = json.loads((ROOT / 'visual_opinions.json').read_text())
    opinions[0]['payload_sha256'] = 'intentionally_invalid_fixture_checksum'
    save(ROOT / 'fault_wrong_payload.json', opinions)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(PROJECT / 'benchmark')
    command = [sys.executable, '-m', 'physcheck.relations', '--manifest', str(ROOT / 'manifest.json'),
        '--domain', 'i2v', '--output', str(ROOT / 'fault_i2v4'),
        '--visual-judgments', str(ROOT / 'fault_wrong_payload.json')]
    result = subprocess.run(command, cwd=PROJECT, env=env, capture_output=True, text=True)
    save(ROOT / 'fault_command.json', {'command': command, 'exit': result.returncode,
         'stdout': result.stdout, 'stderr': result.stderr, 'purpose': 'intentional engineering negative input; not scientific sample'})
    summary = json.loads((ROOT / 'fault_i2v4/summary.json').read_text())
    assert result.returncode == 1 and summary['total_cases'] == 4 and summary['errors'] == 1
    cases = json.loads((ROOT / 'fault_i2v4/cases.json').read_text())
    assert cases[0]['status'] == 'error' and 'payload checksum' in cases[0]['error']
    assert all(c['status'] == 'ok' and c['relation']['verdict'] == 'ABSTAIN' for c in cases[1:])
    save(ROOT / 'fault_verification.json', {'status': 'passed', 'injected_wrong_hash_cases': 1,
         'expected_cli_exit': 1, 'actual_cli_exit': result.returncode, 'all_four_inputs_retained': True,
         'bad_input_not_counted_as_physical_FAIL': True, 'remaining_three_cases_completed': True})


def freeze():
    final = ROOT / 'source_snapshots/final'
    final.mkdir(exist_ok=False)
    paths = [*(PROJECT / 'benchmark/physcheck/relations').glob('*.py'),
         PROJECT / 'benchmark/physcheck/relations/README.md', PROJECT / 'tests/test_relations_visual.py',
         *ROOT.glob('*.py'), *ROOT.glob('*.md')]
    diffs, hashes = [], {}
    for path in paths:
        relative = path.relative_to(PROJECT)
        target = final / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha(path)
        before = ROOT / 'source_snapshots/before' / relative
        old = before.read_text().splitlines(keepends=True) if before.exists() else []
        diffs.extend(difflib.unified_diff(old, path.read_text().splitlines(keepends=True),
            fromfile='before/' + str(relative) if before.exists() else '/dev/null', tofile='after/' + str(relative)))
    (final / 'exact_changes.patch').write_text(''.join(diffs))
    save(final / 'receipt.json', {'sources_sha256': hashes,
         'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'opencv-python', 'pytest']},
         'scope': 'visual ingestion/control label scoring/accounting only; no numerical-model optimization'})


def package():
    bundle = Path('/data/AI_Research/shared/handoff') / ('world_model_integrated_matrix_20261115_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    bundle.mkdir()
    copied = bundle / 'research' / ROOT.name
    shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    shutil.copytree(PROJECT / 'benchmark/physcheck/relations', bundle / 'benchmark/physcheck/relations',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (bundle / 'tests').mkdir()
    shutil.copy2(PROJECT / 'tests/test_relations_visual.py', bundle / 'tests/test_relations_visual.py')
    files = {}
    for path in bundle.rglob('*'):
        if path.is_file():
            relative = path.relative_to(bundle)
            original = PROJECT / relative
            assert sha(path) == sha(original)
            files[str(relative)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    save(bundle / 'MANIFEST.json', {'copied_files': files, 'all_copy_hashes_verified': True,
         'external_dependencies': 'hashed original video/label/cache/frozen-source paths and existing Python environment',
         'neural_inference_reused': True, 'new_raw_videos': False})
    env = dict(os.environ)
    env['PYTHONPATH'] = str(bundle / 'benchmark')
    command = [sys.executable, '-m', 'physcheck.relations', '--manifest', str(copied / 'manifest.json'),
         '--output', str(bundle / 'replay_from_copy'), '--visual-judgments', str(copied / 'visual_opinions.json')]
    result = subprocess.run(command, cwd=bundle, env=env, capture_output=True, text=True)
    save(bundle / 'COPY_REPLAY_COMMAND.json', {'command': command, 'exit': result.returncode,
         'stdout': result.stdout, 'stderr': result.stderr})
    assert result.returncode == 0, result.stderr
    original = copied / 'with_visual325'
    for name in ['cases.json', 'summary.json']:
        assert (original / name).read_bytes() == (bundle / 'replay_from_copy' / name).read_bytes(), name
    save(bundle / 'COPY_REPLAY.json', {'status': 'passed', 'cases': 325,
         'cases_and_summary_byte_identical': True, 'source_executed_from_copy': True,
         'numeric_inference_inputs': 'original existing files, verified by frozen hashes',
         'neural_inference_rerun': False, 'clean_machine_environment_tested': False})
    save(ROOT / 'handoff_receipt.json', {'path': str(bundle), 'copied_files': len(files),
         'copied_bytes': sum(f['bytes'] for f in files.values()), 'all_copy_hashes_verified': True,
         'copy_replay325_passed': True})
    print(json.dumps({'handoff': str(bundle), 'copy_files': len(files), 'copy_replay325': 'passed'}))


if __name__ == '__main__':
    audit()
    fault()
    freeze()
    package()
