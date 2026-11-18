"""Produce plots and tables directly from stress-test result records."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,required=True)
args=parser.parse_args()
root=args.results
summary=json.loads((root/'summary.json').read_text())
rows=json.loads((root/'per_scene.json').read_text())
contrasts=json.loads((root/'paired_contrasts.json').read_text())
rng=np.random.default_rng(11)
fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
strategies=['native','warp96','letterbox96']
for ax,size in zip(axes,[24,48,96]):
    for offset,method,color in [(-.13,'color','#0072B2'),(.13,'learned','#D55E00')]:
        for i,strategy in enumerate(strategies):
            selected=[r for r in rows if r['patch_size']==size and r['method']==method and r['strategy']==strategy]
            values=np.array([r['iou'] for r in selected])
            row=next(r for r in summary if r['patch_size']==size and r['method']==method and r['strategy']==strategy)
            lo,hi=row['iou']['ci95'];mean=row['iou']['mean']
            ax.scatter(i+offset+rng.uniform(-.045,.045,len(values)),values,s=5,alpha=.10,color=color,rasterized=True)
            ax.errorbar(i+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color=color,label=method if i==0 else None)
    ax.set_title(f'Scene patch {size} px (256 scenes)')
    ax.set_xticks(range(3),['Native\n192×112','Warp\n96×96','Letterbox\n96×96'])
    ax.set_ylim(-.03,1.04)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Mask IoU at original scene resolution')
axes[0].legend(fontsize=8)
fig.suptitle('Synthetic scale stress: points = scenes; bars = bootstrap 95% CI',fontsize=11)
fig.tight_layout()
for ext in ['png','pdf','svg']:
    fig.savefig(root/f'scale_stress.{ext}',dpi=220)
lines=['# Synthetic stress results','', '| Scene patch (px) | Input | Method | Mean IoU [95% CI] | Mean centroid error (px) | Nonempty fraction |', '|---:|---|---|---|---:|---:|']
for r in summary:
    s=r['iou'];lo,hi=s['ci95']
    lines.append(f"| {r['patch_size']} | {r['strategy']} | {r['method']} | {s['mean']:.4f} [{lo:.4f}, {hi:.4f}] | {r['centroid_error_px']['mean']:.4f} | {r['nonempty']['mean']:.4f} |")
lines+=['','## Paired IoU contrasts','','| Scene patch | Method | Contrast | Mean difference [95% CI] |','|---:|---|---|---|']
for r in contrasts:
    lo,hi=r['ci95']
    lines.append(f"| {r['patch_size']} | {r['method']} | {r['contrast']} | {r['mean_iou_difference']:.4f} [{lo:.4f}, {hi:.4f}] |")
(root/'TABLES.md').write_text('\n'.join(lines)+'\n')
