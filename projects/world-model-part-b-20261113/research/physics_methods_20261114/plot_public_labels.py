"""Plot descriptive external human-label rates on the identical prompt subset."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(STUDY / 'videophy2_human_common_prompts.csv')
    assert len(data) == 6 and data.n.eq(382).all()
    fig, ax = plt.subplots(figsize=(8, 5.3), layout='constrained')
    y = np.arange(len(data))
    ax.barh(y + .18, data.pc_ge4_pct, .32, label='Physical commonsense >= 4', color='#527da8')
    ax.barh(y - .18, data.joint_pct, .32, label='Semantic >= 4 AND physical >= 4', color='#bd724b')
    for offset, column in [(.18, 'pc_ge4_pct'), (-.18, 'joint_pct')]:
        for position, value in zip(y, data[column]):
            ax.text(value + 1, position + offset, f'{value:.1f}', va='center', fontsize=9)
    ax.set(yticks=y, yticklabels=data.model, xlim=(0, 100), xlabel='Published human-labeled pass fraction (%)', title='VideoPhy2: identical 382 prompts per model')
    ax.set_title('VideoPhy2: identical 382 prompts per model', pad=54)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    fig.supxlabel('External test-label reanalysis; these are not scores for the six new generated videos.', fontsize=9)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(STUDY / f'videophy2_common_prompts.{suffix}', dpi=180)
    plt.close(fig)
    print('Saved PNG/PDF/SVG; six published models x 382 identical prompts.')


if __name__ == '__main__':
    main()
