"""Generate descriptive per-clip plots; no accuracy claims."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args()
rows = json.loads((args.results/'clips.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, model in zip(axes, ['cogvideox_5b', 'wan2.1_1.3b']):
    selected = sorted([r for r in rows if r['model']==model], key=lambda r:(r['prompt_id'],r['seed']))
    for i, r in enumerate(selected):
        a, b = r['color_nonempty_fraction'], r['learned_nonempty_fraction']
        ax.plot([i, i], [a,b], color='0.65', linewidth=1)
    ax.scatter(range(12), [r['color_nonempty_fraction'] for r in selected], label='Color threshold', color='#0072B2', marker='o', s=26)
    ax.scatter(range(12), [r['learned_nonempty_fraction'] for r in selected], label='Frozen TinyUNet', color='#D55E00', marker='x', s=30)
    ax.set_xticks(range(12), [r['prompt_id'].split('_')[0]+str(r['seed']) for r in selected], rotation=60, fontsize=8)
    ax.set_title(f'{model} (12 clips)')
    ax.set_xlabel('Prompt and seed; paired observations')
    ax.set_ylim(-.03,1.03)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Fraction of frames with nonempty mask')
axes[1].legend(fontsize=8, loc='lower right')
fig.suptitle('Unlabeled transfer screening — output presence, not accuracy', fontsize=11)
fig.tight_layout()
for ext in ['png','pdf','svg']:
    fig.savefig(args.results/f'transfer_screening.{ext}', dpi=200)
