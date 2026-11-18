"""Descriptive plots; no missing geometry is silently treated as zero."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main(folder):
    geometry = json.loads((ROOT / 'analysis/full3d.json').read_text())
    pc = json.loads((ROOT / 'analysis/human_pc_probe.json').read_text())
    gravity = json.loads((ROOT / 'analysis/gravity_judge_gate.json').read_text())
    folder.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(3)
    co = [c['measurements']['co']['raw_epe3d_m'] for c in geometry['cases']]
    tap = [c['measurements']['tap']['raw_epe3d_m'] for c in geometry['cases']]
    for i, (a, b) in enumerate(zip(co, tap)):
        if a is not None:
            axes[0].plot([i-.12, i+.12], [a*100, b*100], color='0.7', zorder=1)
        else:
            axes[0].text(i, 9, 'N/A\n85/559 missing', ha='center', fontsize=9, color='#185a9d')
    axes[0].scatter(x-.12, [np.nan if v is None else v*100 for v in co], label='CoTracker + depth', color='#185a9d', s=55)
    axes[0].scatter(x+.12, np.array(tap)*100, label='TAPIP3D', color='#c86718', marker='s', s=45)
    axes[0].set(xticks=x, xticklabels=[c['case'] for c in geometry['cases']], ylabel='Full GT-visible 3D EPE (cm)',
                title='Selected full-length geometry (3 videos)\n150 frames, same 8 queries per video', ylim=(0, 125), xlim=(-.45, 2.45))
    axes[0].legend(frameon=False, loc='upper right')
    matrix = np.array([[pc['tn'], pc['fp']], [pc['fn_or_missing'], pc['tp']]])
    axes[1].imshow(matrix, cmap='Blues', vmin=0, vmax=pc['evaluation_planned']//2)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=22,
                         color='white' if matrix[i,j] > 4 else 'black')
    axes[1].set(xticks=[0,1], yticks=[0,1], xticklabels=['Inconsistent', 'Consistent'],
                yticklabels=['Inconsistent', 'Consistent'], xlabel='Qwen prediction', ylabel='Author human PC class',
                title=f'Diagnostic test only (16 videos)\nPC calibration {pc["calibration_correct"]}/8; gravity {gravity["correct"]}/12: both INVALID')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(folder / f'summary.{suffix}', dpi=180)
    plt.close(fig)
    (folder / 'source_data.json').write_text(json.dumps({'geometry': geometry['cases'], 'pc_test_confusion': matrix.tolist(),
           'uncertainty': 'Individual selected videos; no error bars or population inference. PC results are diagnostic after failed calibration.'}, indent=2))
    print(str(folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    main(parser.parse_args().output)
