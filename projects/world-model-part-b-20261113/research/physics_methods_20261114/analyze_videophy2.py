"""Recompute the published human-label protocol, retaining coverage denominators."""
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent


def main():
    manifest = json.loads((STUDY / 'dataset_manifest.json').read_text())
    path = Path(manifest['local_dir']) / 'videophy2_test.csv'
    expected = next(r['sha256'] for r in manifest['files'] if r['file'].endswith('.csv'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    data = pd.read_csv(path)
    assert not data.video_url.duplicated().any()
    assert data.sa.between(1, 5).all() and data.pc.between(1, 5).all()
    assert data.is_hard.isin([0, 1]).all()
    joint = data.sa.ge(4) & data.pc.ge(4)
    assert (joint.astype(int) == data.joint).all()
    rows = []
    for subset, frame in [('all', data), ('hard', data[data.is_hard.eq(1)])]:
        for model, part in frame.groupby('model_name'):
            rows.append({'subset': subset, 'model': model, 'n': len(part), 'prompts': part.caption.nunique(), 'actions': part.action.nunique(), 'sa_ge4_n': int(part.sa.ge(4).sum()), 'pc_ge4_n': int(part.pc.ge(4).sum()), 'joint_n': int(part.joint.sum()), 'sa_ge4_pct': float(100 * part.sa.ge(4).mean()), 'pc_ge4_pct': float(100 * part.pc.ge(4).mean()), 'joint_pct': float(100 * part.joint.mean()), 'pc_minus_joint_pp': float(100 * (part.pc.ge(4).mean() - part.joint.mean()))})
    table = pd.DataFrame(rows)
    table.to_csv(STUDY / 'videophy2_human_recomputed.csv', index=False)
    # The released file has unequal model coverage. Restrict the six non-Sora
    # models to the exact same caption set for a separate descriptive table.
    models = sorted(set(data.model_name) - {'sora'})
    common = set.intersection(*(set(data.loc[data.model_name.eq(model), 'caption']) for model in models))
    matched = data[data.model_name.isin(models) & data.caption.isin(common)]
    assert not matched.duplicated(['model_name', 'caption']).any()
    matched_rows = []
    for model, part in matched.groupby('model_name'):
        matched_rows.append({'model': model, 'n': len(part), 'sa_ge4_pct': 100 * part.sa.ge(4).mean(), 'pc_ge4_pct': 100 * part.pc.ge(4).mean(), 'joint_pct': 100 * part.joint.mean()})
    pd.DataFrame(matched_rows).to_csv(STUDY / 'videophy2_human_common_prompts.csv', index=False)
    rules = {}
    for column in ['physics_rules_followed', 'physics_rules_unfollowed', 'physics_rules_cannot_be_determined', 'human_violated_rules']:
        # The upstream table contains three literal [nan] placeholders. Preserve
        # their row IDs as missing rule fields; do not invent a physical rule.
        missing_rows = data.index[data[column].eq('[nan]')].tolist()
        parsed = [ast.literal_eval(value) for value in data.loc[~data.index.isin(missing_rows), column]]
        assert all(isinstance(value, list) and all(isinstance(item, str) for item in value) for value in parsed)
        rules[column] = {'rule_mentions': sum(map(len, parsed)), 'videos_with_any': sum(bool(value) for value in parsed), 'missing_field_rows': missing_rows, 'valid_field_rows': len(parsed)}
    candidates = sum(rules[column]['rule_mentions'] for column in ['physics_rules_followed', 'physics_rules_unfollowed', 'physics_rules_cannot_be_determined'])
    result = {'dataset': manifest, 'scope': 'Reanalysis of external published human labels, not new local-model evaluation or new human labeling.', 'rows': len(data), 'prompts': data.caption.nunique(), 'actions': data.action.nunique(), 'hard_rows': int(data.is_hard.sum()), 'joint_disagreements': int((joint.astype(int) != data.joint).sum()), 'rules': rules, 'candidate_rule_mentions': candidates, 'candidate_rule_undetermined_fraction': rules['physics_rules_cannot_be_determined']['rule_mentions'] / candidates, 'common_prompt_count': len(common), 'common_prompt_models': models, 'coverage_caveat': 'The public test file is a subset; sample sizes and prompts differ by model. The all-row table does not reproduce the full paper data or a balanced new-model leaderboard. No individual annotator votes or inter-rater reliability are available in this table.'}
    (STUDY / 'videophy2_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'dataset'}, indent=2))


if __name__ == '__main__':
    main()
