"""Expand the existing public control index without selecting by labels/scores."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha


def entry(path):
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def main():
    prior = PROJECT / 'research/integrated_framework_20261115/manifest.json'
    manifest = json.loads(prior.read_text())
    gravity = PROJECT / 'research/principia_gravity_20261114'
    judges = {c['case_id'] for c in json.loads((gravity / 'judge_input_manifest.json').read_text())}
    controls = []
    for case in json.loads((gravity / 'data/heldout_public_index.json').read_text()):
        for codec in ['ffv1', 'h264']:
            files = {'input': entry(gravity / 'data' / case['videos'][codec]['path']),
                     'setup': entry(gravity / 'data' / case['setup_path']),
                     'truth': entry(gravity / 'data/private/heldout_truth.json')}
            assert files['input']['sha256'] == case['videos'][codec]['sha256']
            if codec == 'ffv1' and case['case_id'] in judges:
                files['judgment'] = entry(gravity / 'judgments' / (case['case_id'] + '.json'))
            controls.append({'id': case['case_id'] + '_' + codec, 'domain': 'control',
                'reference_key': case['case_id'], 'codec': codec, 'files': files})
    manifest['cases'] = controls + [c for c in manifest['cases'] if c['domain'] != 'control']
    manifest['selection'] = 'all288 existing control video encodings + all30DAVIS + all3PStudio + all4I2V'
    assert len(controls) == 288 and len(manifest['cases']) == 325
    save(ROOT / 'manifest.json', manifest)
    save(ROOT / 'pre_run_receipt.json', {'prior_manifest_sha256': sha(prior),
         'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'manifest_sha256': sha(ROOT / 'manifest.json'),
         'code_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in (PROJECT / 'benchmark/physcheck/relations').glob('*.py')},
         'label_based_selection': False, 'new_generation': False, 'optimization': False})
    print(json.dumps({'cases': 325, 'control_videos': 288, 'source': str(prior)}))


if __name__ == '__main__':
    main()
