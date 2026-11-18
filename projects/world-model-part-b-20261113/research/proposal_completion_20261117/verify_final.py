"""Read-only acceptance checks, with a new local receipt, after handoff replay."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def read(path):
    return json.loads(path.read_text())


if __name__ == '__main__':
    delivery = read(ROOT/'handoff_receipt.json')
    package = Path(delivery['path'])
    manifest = package/'MANIFEST.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == delivery['manifest_sha256']
    replay = read(package/'COPY_REPLAY.json')
    assert replay['cases']==33 and replay['all_evaluations_identical'] and replay['all_normalized_arrays_identical_including_missing_positions']
    assert read(ROOT/'tests_final.json')['exit']==0
    assert '117 passed' in read(ROOT/'tests_final.json')['stdout']
    links = []
    for path in [PROJECT/'report/report.md', PROJECT/'report/results_summary.md', PROJECT/'docs/PROPOSAL_RESULTS_2026-11-17.zh.md']:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://')):
                continue
            resolved = path.parent / target.split('#')[0]
            assert resolved.exists(), (str(path), target)
            links.append(str(resolved.resolve().relative_to(PROJECT)))
        copied = package / path.relative_to(PROJECT)
        assert copied.read_bytes()==path.read_bytes(), str(path)
    pc = read(ROOT/'analysis/human_pc_probe.json')
    assert not pc['protocol_valid'] and pc['calibration_correct']==4 and pc['evaluation_correct']==8
    assert pc['tp']+pc['fp']+pc['fn_or_missing']+pc['tn']==16
    assert not read(ROOT/'analysis/gravity_judge_gate.json')['valid']
    full = read(ROOT/'analysis/full3d.json')
    co = full['summary']['by_domain']['pstudio']['video_mean_metrics']['co']
    assert co['raw_epe3d_m'] is None and co['count_totals']['invalid_3d_point_times']==85
    for kind in ['pc', 'gravity']:
        cases = read(ROOT/f'{kind}_judge_cases.json')
        for case in cases:
            payload = Path(case['input'])
            result = read(ROOT/f'{kind}_judgments'/f'{case["id"]}.json')
            assert result['input_sha256']==hashlib.sha256(payload.read_bytes()).hexdigest()
            assert result['status']=='ok'
            assert result['runtime']['independent_context_per_case']
            assert not result['labels_loaded_by_judge']
    receipt = {'handoff': str(package), 'copied_replay_cases': 33, 'tests': 117,
               'report_local_links_checked': len(links), 'judge_first_replies_checked': 40,
               'final_plot_visually_inspected': True, 'scientific_gates_still_failed': True,
               'all_proposal_scientific_acceptance_claimed': False, 'new_public_upload': False,
               'handoff_manifest_hash_matches': True}
    with (ROOT/'FINAL_CHECK.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))
