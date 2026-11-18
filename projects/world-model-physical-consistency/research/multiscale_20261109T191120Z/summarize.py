"""Summaries of fixed-protocol results, including adverse outcomes."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,required=True)
args=parser.parse_args()
root=args.results
rows=json.loads((root/'test/summary.json').read_text())
effects=json.loads((root/'test/effects.json').read_text())


def select(size,strategy,method):
    return [r for r in rows if r['patch_size']==size and r['strategy']==strategy and
            (r['method']==method or r['method'].startswith(method+'_'))]


def wilson(p,n):
    z=NormalDist().inv_cdf(.975)
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0,center-radius),min(1,center+radius)


fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True,sharey='row')
for column,strategy in enumerate(['native','warp96','letterbox96']):
    for i,size in enumerate([24,48,96]):
        for offset,method,color in [(-.1,'single_scale','#0072B2'),(.1,'multiscale','#D55E00')]:
            subset=select(size,strategy,method)
            for rownum,metric in enumerate(['positive_iou','negative_fpr']):
                values=[r[metric] for r in subset]
                axes[rownum,column].scatter([i+offset-.025,i+offset,i+offset+.025],values,s=28,
                    color=color,label=method if i==0 else None)
                axes[rownum,column].plot([i+offset-.06,i+offset+.06],[np.mean(values)]*2,color=color,linewidth=2)
    axes[0,column].set_title(strategy)
    for rownum in range(2):
        ax=axes[rownum,column]
        ax.set_ylim(-.035,1.035)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xticks(range(3),[24,48,96])
    axes[1,column].set_xlabel('Scene patch size (px)')
axes[0,0].set_ylabel('Positive mask IoU (256 scenes / seed)')
axes[1,0].set_ylabel('Negative frame FPR (64 scenes / seed)')
axes[0,0].legend(fontsize=8)
fig.suptitle('Synthetic held-out test: dots = training seeds; bars = mean of 3 seeds',fontsize=11)
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(root/f'multiscale_comparison.{ext}',dpi=220)

table=['# Fixed held-out synthetic test','',
       'IoU: 256 positive scenes per condition; negative FPR: 64 negative scenes. Training seeds are repeated fits on the same data, not additional independent test scenes.','',
       '| Patch | Input | Method | Positive IoU | Positive detection | Centroid error (px) | Negative FP / 64 | FPR Wilson 95% CI |',
       '|---:|---|---|---:|---:|---:|---:|---|']
for row in rows:
    lo,hi=wilson(row['negative_fpr'],64)
    table.append(f"| {row['patch_size']} | {row['strategy']} | {row['method']} | {row['positive_iou']:.4f} | {row['positive_detection_rate']:.4f} | {row['positive_centroid_error_px']:.3f} | {round(row['negative_fpr']*64)} | [{lo:.4f}, {hi:.4f}] |")
table+=['','## Paired multiscale minus single-scale contrasts','',
        '| Patch | Input | Primary? | Mean IoU difference [conditional 95% CI] | Differences by training seed 7,17,27 |',
        '|---:|---|---|---|---|']
for row in effects:
    lo,hi=row['ci95_conditional_on_three_training_seeds']
    values=', '.join(f'{v:.4f}' for v in row['per_seed_mean_differences'])
    table.append(f"| {row['patch_size']} | {row['strategy']} | {row['primary']} | {row['mean_iou_difference']:.4f} [{lo:.4f}, {hi:.4f}] | {values} |")
table+=['','Confidence intervals resample paired scenes conditional on the 3 trained seeds; they do not represent uncertainty across all possible training runs. No P values or multiplicity-adjusted claims are made.']
(root/'TABLES.md').write_text('\n'.join(table)+'\n')
grouped=[]
for size in [24,48,96]:
    for strategy in ['native','warp96','letterbox96']:
        for arm in ['single_scale','multiscale']:
            subset=select(size,strategy,arm)
            grouped.append({'patch_size':size,'strategy':strategy,'arm':arm,'n_training_seeds':3,
                **{metric:{'mean':float(np.mean([r[metric] for r in subset])),
                           'min':float(min(r[metric] for r in subset)),
                           'max':float(max(r[metric] for r in subset))}
                   for metric in ['positive_iou','positive_detection_rate','negative_fpr']}})
(root/'grouped_summary.json').write_text(json.dumps(grouped,indent=2)+'\n')
