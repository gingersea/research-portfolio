"""All planned cases; validation gates remain invalid when first replies fail."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha


def interval(k, n):
    return [float(beta.ppf(.025, k, n-k+1)) if k else 0., float(beta.ppf(.975, k+1, n-k)) if k < n else 1.]


def main():
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=False)
    public = json.loads((ROOT / 'public_davis_fresh/cases.json').read_text())
    assert len(public) == 30 and all(r['status'] == 'ok' for r in public)
    deltas = []
    old = PROJECT / 'research/paper_benchmark_20261115'
    for row in public:
        name = row['id'].removeprefix('davis_')
        prior = np.load(old / 'predictions' / (name + '.npz'))
        current = np.load(ROOT / 'public_davis_fresh' / row['id'] / 'fresh_predictions.npz')
        deltas.append({'case': name, 'max_abs_xy_change': float(np.max(np.abs(current['co_xy'] - prior['xy']))),
                       'visibility_differences': int((current['co_visible'] != prior['visible']).sum())})
    save(out / 'davis_replication.json', {'videos': 30, 'queries': 650, 'fresh_inference': True,
         'max_xy_change': max(r['max_abs_xy_change'] for r in deltas), 'cases': deltas,
         'summary': json.loads((ROOT / 'public_davis_fresh/summary.json').read_text())})
    full = json.loads((ROOT / 'public_full_rescored/cases.json').read_text())
    assert len(full) == 3 and all(r['status'] == 'ok' for r in full)
    paired = []
    for row in full:
        name = row['id'].removeprefix('pstudio_')
        inp = np.load(ROOT / 'pstudio_full' / name / 'input.npz')
        truth = np.load(ROOT / 'pstudio_full' / name / 'truth.npz')
        pred = np.load(ROOT / 'public_pstudio_full' / row['id'] / 'fresh_predictions.npz')
        pos = truth['visible'] & (np.arange(150)[:, None] > inp['queries_tyx'][None, :, 0])
        common = pos & np.isfinite(pred['co_xyz']).all(axis=-1) & np.isfinite(pred['tap_xyz']).all(axis=-1)
        paired.append({'case': name, 'all_visible': int(pos.sum()), 'common_valid': int(common.sum()),
             **{method + '_common_epe_m': float(np.linalg.norm(pred[method + '_xyz'] - truth['tracks_xyz'], axis=-1)[common].mean()) for method in ['co', 'tap']},
             'measurements': row['evaluation']})
    save(out / 'full3d.json', {'frames_each': 150, 'queries_each': 8, 'cases': paired,
         'summary': json.loads((ROOT / 'public_full_rescored/summary.json').read_text()),
         'common_subset_is_diagnostic_not_full_coverage_claim': True})
    mapping = json.loads((ROOT / 'gravity_private_mapping.json').read_text())
    truth = {r['case_id']: r for r in json.loads((PROJECT / 'research/principia_gravity_20261114/data/private/heldout_truth.json').read_text())}
    gravity = []
    for case in mapping:
        result = json.loads((ROOT / 'gravity_judgments' / (case['id'] + '.json')).read_text())
        target = truth[case['original_case_id']]
        observed = result.get('judgment', {}).get('verdict')
        gravity.append({'id': case['id'], 'expected': target['expected'], 'observed': observed,
                        'status': result['status'], 'condition': target['condition'], 'correct': observed == target['expected']})
    valid = len(gravity) == 12 and all(r['correct'] for r in gravity)
    i2v = [json.loads((ROOT / 'gravity_judgments' / f'i_{n:03d}.json').read_text()) for n in range(1,5)]
    save(out / 'gravity_judge_gate.json', {'valid': valid, 'calibration_cases': 12,
         'correct': sum(r['correct'] for r in gravity), 'cases': gravity,
         'obvious_violations_missed': sum(r['expected']=='FAIL' and r['observed']!='FAIL' for r in gravity),
         'false_positive_FAIL': sum(r['expected']=='PASS' and r['observed']=='FAIL' for r in gravity),
         'i2v_first_replies': i2v, 'i2v_usable_physical_judgments': valid,
         'independent_Qwen_context': True, 'human_labels_created': False})
    plan = json.loads((ROOT / 'human_probe_plan.json').read_text())['rows']
    pc = []
    for target in plan:
        record = json.loads((ROOT / 'pc_judgments' / (target['opaque_id'] + '.json')).read_text())
        score = record.get('judgment', {}).get('pc_score')
        consistent = score >= 4 if score is not None else None
        pc.append({'id': target['opaque_id'], 'split': target['split'], 'human_pc': target['human_pc'],
                   'human_consistent': target['consistent_reference'], 'predicted_pc': score,
                   'predicted_consistent': consistent, 'correct': consistent == target['consistent_reference'] if consistent is not None else False,
                   'status': record['status']})
    calibration = [r for r in pc if r['split']=='calibration']
    evaluation = [r for r in pc if r['split']=='evaluation']
    caption = {r['opaque_id']: r['caption'] for r in plan}
    clusters = sorted({caption[r['id']] for r in evaluation})
    grouped = [[r for r in evaluation if caption[r['id']] == key] for key in clusters]
    ccounts = np.array([sum(r['correct'] for r in group) for group in grouped])
    sizes = np.array([len(group) for group in grouped])
    draws = np.random.default_rng(20261117).integers(0, len(clusters), (10000, len(clusters)))
    bootstrap = ccounts[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    pc_valid = len(calibration)==8 and all(r['correct'] and r['status']=='ok' for r in calibration)
    correct = sum(r['correct'] for r in evaluation)
    tp = sum(r['human_consistent'] and r['predicted_consistent'] is True for r in evaluation)
    fp = sum(not r['human_consistent'] and r['predicted_consistent'] is True for r in evaluation)
    fn = sum(r['human_consistent'] and r['predicted_consistent'] is not True for r in evaluation)
    tn = sum(not r['human_consistent'] and r['predicted_consistent'] is False for r in evaluation)
    result = {'protocol_valid': pc_valid, 'calibration_correct': sum(r['correct'] for r in calibration),
         'calibration_planned': 8, 'evaluation_planned': 16, 'evaluation_judged': sum(r['status']=='ok' for r in evaluation),
         'evaluation_correct': correct, 'diagnostic_accuracy': correct/16,
         'caption_clusters': len(clusters), 'caption_cluster_bootstrap95ci': np.quantile(bootstrap,[.025,.975]).tolist(),
         'ci_scope': 'descriptive caption-cluster resampling,10000,seed20261117; not new-domain/model population guarantee',
         'tp': tp, 'fp': fp, 'fn_or_missing': fn, 'tn': tn,
         'precision_consistent': tp/(tp+fp) if tp+fp else None, 'recall_consistent': tp/(tp+fn),
         'score3_uncertain_count': sum(r['predicted_pc']==3 for r in evaluation),
         'unavailable_or_failed': sum(r['status']!='ok' for r in pc), 'cases': pc,
         'scope': 'fixed balanced external-human-label probe; not model-generator prevalence or a replication of author AutoEval',
         'new_calibration_does_not_validate_old_PartB': True}
    save(out / 'human_pc_probe.json', result)
    save(out / 'receipt.json', {'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
         'initial_replies_used': True, 'rejudged_to_pass': False, 'samples_removed_or_replaced': False})
    print(json.dumps({'gravity_calibration_correct': sum(r['correct'] for r in gravity), 'gravity_valid': valid,
                      'human_PC_calibration': result['calibration_correct'], 'human_PC_valid': pc_valid,
                      'human_PC_test_correct': correct, 'human_PC_test_n':16}))


if __name__ == '__main__':
    main()
