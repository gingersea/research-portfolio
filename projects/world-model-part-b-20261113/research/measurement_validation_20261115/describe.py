"""Descriptive heterogeneity, video-cluster uncertainty and publication artifacts."""
from collections import Counter
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, save

COLORS = ['#2878a4', '#ca7a28', '#23835b', '#8a598e']
POLICIES = ['native', 'vc09', 'calibrated', 'calibrated_confirmation']
LABELS = ['Native', 'v*c > 0.9', 'v*c > 0.995', '0.995 + 2-frame confirmation']


def export(fig, folder, name):
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(folder / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    out = ROOT / 'description'
    out.mkdir(exist_ok=False)
    rows = json.loads((ROOT / 'results/per_video.json').read_text())
    events = json.loads((ROOT / 'results/events.json').read_text())
    selected = [r for r in rows if r['split'] == 'validation']
    lookup = {(r['policy'], r['case']): r for r in selected}
    names = [r['case'] for r in selected if r['policy'] == 'native']
    draws = np.random.default_rng(20261115).integers(0, len(names), (10000, len(names)))
    uncertainty = []
    for policy in POLICIES:
        rr = [lookup[policy, n] for n in names]
        c, a, p = (np.array([r[key] for r in rr]) for key in ['correct', 'accepted', 'positive'])
        precision = c[draws].sum(axis=1) / a[draws].sum(axis=1)
        retention = c[draws].sum(axis=1) / p[draws].sum(axis=1)
        uncertainty.append({'policy': policy, 'precision_ci95': np.quantile(precision, [.025, .975]).tolist(),
              'retention_ci95': np.quantile(retention, [.025, .975]).tolist(),
              'videos_with_acceptances': int((a > 0).sum()),
              'videos_precision_ge95': int(((a > 0) & (c >= .95 * a)).sum()),
              'per_video_precision_range': [min(r['correct'] / r['accepted'] for r in rr if r['accepted']),
                                            max(r['correct'] / r['accepted'] for r in rr if r['accepted'])]})
    save(out / 'absolute_uncertainty.json', uncertainty)
    failures = []
    for group in ['calibration', 'validation', 'all']:
        for policy in POLICIES:
            ee = [e for e in events if e['policy'] == policy and (group == 'all' or e['split'] == group)]
            failures.append({'split': group, 'policy': policy, 'events': len(ee),
                             **dict(Counter(e['failure_class'] for e in ee))})
    save(out / 'event_diagnosis.json', failures)
    with (out / 'per_video.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    names.sort(key=lambda n: lookup['native', n]['correct'] / lookup['native', n]['accepted'])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, field, ylabel in zip(axes, ['accepted', 'positive'], ['Accepted precision <4px (%)', 'Correct visible-point retention (%)']):
        for policy, label, color in zip(['native', 'calibrated'], [LABELS[0], LABELS[2]], [COLORS[0], COLORS[2]]):
            value = [100 * lookup[policy, n]['correct'] / lookup[policy, n][field] for n in names]
            ax.plot(range(20), value, 'o-', lw=.8, color=color, label=label)
        ax.set_ylim(0, 102)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
    axes[0].axhline(95, ls=':', color='#777777', label='95% calibration target')
    axes[0].set_title('Higher precision costs coverage; 20 validation videos shown individually')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[1].set_xticks(range(20), names, rotation=55, ha='right')
    export(fig, out, 'precision_coverage')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(4)
    for cause, label, color in [('recovered', 'Recovered (<4px and accepted)', '#23835b'),
         ('accurate_position_rejected', 'Some <4px positions, all rejected', '#ca7a28'),
         ('no_accurate_position', 'No <4px position in visible run', '#7e8791')]:
        heights = [next(r for r in failures if r['split'] == 'validation' and r['policy'] == policy).get(cause, 0) for policy in POLICIES]
        ax.bar(range(4), heights, bottom=bottom, color=color, label=label)
        bottom += heights
    ax.set_xticks(range(4), LABELS, fontsize=9)
    ax.set_ylim(0, 310)
    ax.set_ylabel('Reappearance events (count; n=283)')
    ax.set_title('Confirmation rejects more recoveries; coordinates are unchanged')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), fontsize=9)
    export(fig, out, 'return_diagnosis')

    geo = json.loads((ROOT / 'geometry_results/per_video.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    variants = ['raw', 'oracle_true_z', 'oracle_true_ray', 'oracle_xy_da_depth']
    for ax, method, title in zip(axes, ['co', 'tap'], ['CoTracker + DA-V2', 'TAPIP3D + DA-V2']):
        rr = [r for r in geo if r['method'] == method]
        for i, variant in enumerate(variants):
            values = np.array([r[variant + '_epe_m'] * 100 for r in rr])
            ax.bar(i, values.mean(), color=COLORS[i], alpha=.4)
            ax.scatter(i + np.array([-.1, 0, .1]), values, s=35, marker='o', color=COLORS[i])
        ax.set_xticks(range(4), ['Raw', 'True Z\n(predicted ray)', 'True ray\n(predicted Z)', 'True ray\n(DA at true XY)'], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 42)
    axes[0].set_ylabel('Visible-point 3D error (cm)')
    fig.suptitle('GT-assisted diagnostics only: true depth reduces error; true 2D alone does little', fontsize=11)
    fig.text(.5, -.02, 'Bars: equal-video means; dots: all 3 geometry-labeled clips. Oracles are not deployable methods.', ha='center', fontsize=9)
    export(fig, out, 'geometry_oracles')
    print(json.dumps({'uncertainty': uncertainty, 'event_diagnosis': failures}, indent=2))


if __name__ == '__main__':
    main()
