"""Export actual per-scene measurements, denominators and a bounded score figure."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    gate = json.loads((STUDY / 'heldout_gate_v2.json').read_text())
    errors = gate['paired_errors']
    write_csv(STUDY / 'heldout_per_scene_errors.csv', errors)
    cells = [{key: (json.dumps(value) if isinstance(value, (dict, list)) else value) for key, value in row.items()} for row in gate['condition_cells']]
    write_csv(STUDY / 'measurement_condition_counts.csv', cells)
    diagnostics = []
    for codec in ['ffv1', 'h264']:
        violations = [row for row in errors if row['codec'] == codec and row['condition'] == 'unequal_gravity']
        count = sum(row['native_unordered_score_on_same_measurements'] >= .9 for row in violations)
        diagnostics.append({'codec': codec, 'known_violations': len(violations), 'native_unordered_scores_ge_local_pass_threshold': count, 'directed_scores_ge_local_pass_threshold': sum(row['score'] >= .9 for row in violations), 'limit': 'Local0.90 threshold diagnostic; not a published Principia binary decision rule.'})
    h264 = [row for row in errors if row['codec'] == 'h264']
    control = sorted([row for row in h264 if row['condition'] == 'equal_gravity'], key=lambda row: row['scene_seed'])
    violation = sorted([row for row in h264 if row['condition'] == 'unequal_gravity'], key=lambda row: row['scene_seed'])
    assert len(control) == len(violation) == 24
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for left, right in zip(control, violation):
        axes[0].plot([0, 1], [left['score'], right['score']], color='0.75', lw=.6, zorder=0)
    axes[0].scatter(np.zeros(24), [row['score'] for row in control], s=18, color='#2878a5', label='24 scene pairs')
    axes[0].scatter(np.ones(24), [row['score'] for row in violation], s=18, color='#b35a35')
    axes[0].set(xticks=[0, 1], xticklabels=['Equal gravity', 'Factor-4 violation'], ylabel='Identity-preserving relation score', title='H264 held-out procedural controls')
    for i, row in enumerate(violation):
        axes[1].plot([0, 1], [row['score'], row['native_unordered_score_on_same_measurements']], color='0.75', lw=.6, zorder=0)
    axes[1].scatter(np.zeros(24), [row['score'] for row in violation], s=18, color='#2878a5')
    axes[1].scatter(np.ones(24), [row['native_unordered_score_on_same_measurements'] for row in violation], s=18, color='#b35a35')
    axes[1].set(xticks=[0, 1], xticklabels=['Directed identity', 'Native min/max'], ylabel='Relation score', title='Same 24 known violations, different pairing')
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-.3, 1.3)
        ax.axhline(.90, color='0.4', ls='--', lw=.8)
        ax.axhline(.85, color='0.6', ls=':', lw=.8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('Lines connect the same scene. Dashed0.90/dotted0.85 are local calibration thresholds; no generator scores.', fontsize=9)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(STUDY / f'gravity_calibration.{extension}', dpi=180)
    plt.close(fig)
    result = {'native_pairing_diagnostic': diagnostics, 'scene_success_ci_24_of_24': gate['condition_cells'][0]['clopper_pearson_95ci'], 'note': 'Finite procedural sample; error bounds are maxima over all assigned admissible cases. Codec repeats and conditions are dependent within scene.'}
    (STUDY / 'summary_diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
