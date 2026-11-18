"""Fixed scene-based control accounting and opinion-channel noninterference."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha


def ci(correct, total):
    return [float(beta.ppf(.025, correct, total - correct + 1)) if correct else 0.,
            float(beta.ppf(.975, correct + 1, total - correct)) if correct < total else 1.]


def main():
    baseline = json.loads((ROOT / 'baseline325/cases.json').read_text())
    imported = json.loads((ROOT / 'with_visual325/cases.json').read_text())
    assert len(baseline) == len(imported) == 325
    assert all(r['status'] == 'ok' for r in baseline + imported)
    reference = {r['id']: r for r in baseline}
    for row in imported:
        prior = reference[row['id']]
        for field in ['relation', 'evaluation', 'measurement', 'observation']:
            assert row.get(field) == prior.get(field), (row['id'], field)
        assert row['stages']['tracking']['artifact_sha256'] == prior['stages']['tracking']['artifact_sha256']
    controls = [r for r in imported if r['domain'] == 'control']
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    cases = {c['id']: c for c in manifest['cases']}
    records = []
    for row in controls:
        c = cases[row['id']]
        records.append({'id': row['id'], 'reference_key': c['reference_key'], 'codec': c['codec'],
             **row['reference'], 'observed': row['relation']['verdict'],
             'correct': row['relation']['verdict'] == row['reference']['expected'],
             'score': row['relation']['score'], 'instrument_reason': row['measurement']['reason'],
             'split': 'calibration_scene' if row['reference']['scene_seed'] in [81000, 81001] else 'noncalibration'})
    primary = [r for r in records if r['split'] == 'noncalibration']
    assert len(primary) == 264 and len({r['scene_seed'] for r in primary}) == 22
    cells = []
    for codec in ['ffv1', 'h264']:
        for condition in sorted({r['condition'] for r in records}):
            selected = [r for r in primary if r['codec'] == codec and r['condition'] == condition]
            assert len(selected) == len({r['scene_seed'] for r in selected}) == 22
            correct = sum(r['correct'] for r in selected)
            cells.append({'codec': codec, 'condition': condition, 'scenes': 22, 'correct': correct,
                 'ci95_conditional': ci(correct, 22), 'expected': selected[0]['expected'],
                 'verdicts': dict(Counter(r['observed'] for r in selected))})
    scene_rows = []
    for seed in sorted({r['scene_seed'] for r in primary}):
        selected = [r for r in primary if r['scene_seed'] == seed]
        assert len(selected) == 12
        scene_rows.append({'scene_seed': seed, 'all_conditions_and_codecs_correct': all(r['correct'] for r in selected)})
    disagreements = []
    for key in sorted({r['reference_key'] for r in records}):
        pair = [r for r in records if r['reference_key'] == key]
        assert len(pair) == 2
        if pair[0]['observed'] != pair[1]['observed']:
            disagreements.append(pair)
    old = json.loads((PROJECT / 'research/principia_gravity_20261114/heldout_measurements_v2.json').read_text())
    old_index = {(r['case_id'], r['codec']): r for r in old}
    regression = []
    for row in controls:
        case = cases[row['id']]
        prior = old_index[case['reference_key'], case['codec']]
        current = row['measurement']
        assert current['verdict'] == prior['verdict'] and current['reason'] == prior['reason']
        diff = abs(current['score'] - prior['score']) if 'score' in current else None
        if diff is not None:
            assert np.isclose(current['score'], prior['score'], rtol=0, atol=1e-10)
        regression.append({'id': row['id'], 'verdict_equal': True, 'score_absolute_difference': diff})
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=False)
    summary = {'total_framework_inputs': 325, 'framework_errors': 0,
         'controls': 288, 'scenes': 24, 'codecs_are_technical_repeats': True,
         'control_verdicts': dict(Counter(r['observed'] for r in records)),
         'noncalibration_scenes': 22, 'noncalibration_inputs': 264,
         'noncalibration_all_correct_scenes': sum(r['all_conditions_and_codecs_correct'] for r in scene_rows),
         'all_correct_scene_ci95': ci(sum(r['all_conditions_and_codecs_correct'] for r in scene_rows), 22),
         'codec_disagreements': len(disagreements), 'codec_pairs': 144,
         'false_FAIL_on_equal_gravity': sum(r['expected'] == 'PASS' and r['observed'] == 'FAIL' for r in primary),
         'unequal_gravity_not_FAIL': sum(r['expected'] == 'FAIL' and r['observed'] != 'FAIL' for r in primary),
         'inadmissible_scored_conclusively': sum(r['expected'] == 'ABSTAIN' and r['observed'] != 'ABSTAIN' for r in primary),
         'visual_opinions_imported': sum(r.get('visual_judge', {}).get('status') == 'diagnostic' for r in imported),
         'i2v_independent_visual_pending': 4,
         'numeric_channels_unchanged_after_visual_import': 325,
         'prior_verdicts_reproduced': len(regression),
         'max_prior_score_difference': max(r['score_absolute_difference'] or 0 for r in regression),
         'new_unseen_generalization_claim': False}
    for name, data in [('summary.json', summary), ('per_control.json', records), ('condition_cells.json', cells),
                       ('per_scene.json', scene_rows), ('codec_disagreements.json', disagreements), ('prior_regression.json', regression)]:
        save(out / name, data)
    with (out / 'per_control.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save(out / 'receipt.json', {'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'source_sha256': sha(Path(__file__)),
         'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'manifest.json', ROOT / 'visual_opinions.json',
                       ROOT / 'baseline325/cases.json', ROOT / 'with_visual325/cases.json']},
         'interval': 'per-codec/condition exact Clopper-Pearson, descriptive unadjusted95%; scene-level n22',
         'previous_data_exposure_disclosed': True, 'future_truth_used_only_in_evaluation': True})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
