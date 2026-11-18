"""Declare all existing inputs and immutable dependencies; no score selection."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha


def entry(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def main():
    study = PROJECT / 'research'
    gravity = study / 'principia_gravity_20261114'
    i2v = study / 'principia_i2v_20261115'
    identity = study / 'identity_recovery_20261115'
    paper = study / 'paper_benchmark_20261115'
    cases = []
    index = {c['case_id']: c for c in json.loads((gravity / 'data/heldout_public_index.json').read_text())}
    for judge in json.loads((gravity / 'judge_input_manifest.json').read_text()):
        case = index[judge['case_id']]
        cases.append({'id': case['case_id'], 'domain': 'control', 'files': {
             'input': entry(gravity / 'data' / case['videos']['ffv1']['path']),
             'setup': entry(gravity / 'data' / case['setup_path']),
             'judgment': entry(gravity / 'judgments' / (case['case_id'] + '.json'))}})
    for domain, name in [('davis', 'data_manifest.json'), ('pstudio', 'pilot3d_manifest.json')]:
        for case in json.loads((paper / name).read_text())['cases']:
            folder = 'predictions' if domain == 'davis' else 'pilot3d_predictions'
            cases.append({'id': domain + '_' + case['case'], 'domain': domain,
                 'frames': case['frames'], 'files': {'input': entry(paper / case['input']),
                 'truth': entry(paper / case['truth']),
                 'prediction': entry(paper / folder / (case['case'] + '.npz')),
                 'prediction_receipt': entry(paper / folder / (case['case'] + '.json'))}})
    setups = {c['initial_id']: c for c in json.loads((i2v / 'input_manifest.json').read_text())}
    for case in json.loads((identity / 'real_manifest.json').read_text()):
        files = {'input': entry(case['source']),
            'masks': entry(i2v / 'inputs' / case['initial'] / 'masks.npz'),
            'lossless_video': entry(i2v / 'inspection' / f'{case["initial"]}_seed{case["seed"]}_native.mkv')}
        for name, folder in [('co', 'real_predictions/cotracker'), ('tap', 'real_predictions/tapip3d'), ('depth', 'estimated_depth_native_attention')]:
            files[name] = entry(identity / folder / (case['clip'] + '.npz'))
            files[name + '_receipt'] = entry(identity / folder / (case['clip'] + '.json'))
        cases.append({'id': 'i2v_' + case['clip'], 'domain': 'i2v', 'files': files,
                      'setup': setups[case['initial']], 'fps': 24, 'frames': 97})
    manifest = {'project': str(PROJECT), 'cases': cases, 'selection': 'all30DAVIS+all3PStudio+all4I2V+exact12prior_control_judge_assignments',
        'sources': {'gravity': entry(gravity / 'measure.py'), 'i2v': entry(i2v / 'evaluate.py'),
                    'official': entry(paper / 'official_numpy.py')},
        'judge_calibration': {'script': entry(gravity / 'judge_gate.py'), 'study': str(gravity),
                              'receipts': entry(gravity / 'judge_session_receipts.json')}}
    assert len(cases) == 49
    save(ROOT / 'manifest.json', manifest)
    before = ROOT / 'source_snapshots' / 'before_run'
    before.mkdir(parents=True, exist_ok=False)
    sources = [* (PROJECT / 'benchmark/physcheck/relations').glob('*.py'),
               PROJECT / 'tests/test_relations.py', Path(__file__),
               PROJECT / 'README.md', PROJECT / 'docs/superpowers/specs/2026-11-15-integrated-measurement-framework.md',
               PROJECT / 'docs/superpowers/plans/2026-11-15-integrated-measurement-framework.md']
    hashes = {}
    for path in sources:
        relative = path.relative_to(PROJECT)
        destination = before / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = sha(path)
    save(before / 'receipt.json', {'sources': hashes, 'manifest_sha256': sha(ROOT / 'manifest.json'),
         'new_package_before_baseline': 'empty directory; existing studies/README unchanged'})
    print(json.dumps({'cases': len(cases), 'manifest': str(ROOT / 'manifest.json')}))


if __name__ == '__main__':
    main()
