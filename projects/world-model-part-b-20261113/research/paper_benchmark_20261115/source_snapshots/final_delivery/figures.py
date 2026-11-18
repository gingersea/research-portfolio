"""Per-video human-benchmark variation and separate geometry-pilot figures."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def save(fig,path):
    for extension in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+extension),dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    output=ROOT/'figures';output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    videos=json.loads((ROOT/'results/per_video.json').read_text());summary=json.loads((ROOT/'results/summary.json').read_text())
    names=[r['case'] for r in videos if r['policy']=='native'];x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(15,5))
    for policy,offset,color in [('native',-.12,'#2777aa'),('conservative_vc09',.12,'#c86b2a')]:
        scores=[next(r for r in videos if r['case']==name and r['policy']==policy)['average_jaccard']*100 for name in names]
        ax.scatter(x+offset,scores,color=color,s=25,label=policy)
    for i,name in enumerate(names):
        pair=[r['average_jaccard']*100 for r in videos if r['case']==name]
        ax.plot([i-.12,i+.12],pair,color='#999999',lw=.6,zorder=0)
    ax.set_xticks(x,names,rotation=65,ha='right',fontsize=8);ax.set_ylim(0,100)
    ax.set_ylabel('Official Average Jaccard (%)');ax.set_title('All30 human-annotated TAP-Vid-DAVIS videos; independent first queries')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.5),ncol=2);fig.tight_layout();save(fig,output/'davis_per_video')
    metric_names=['average_jaccard','average_pts_within_thresh','occlusion_accuracy']
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(3)
    for row,offset,color in zip(summary,[-.17,.17],['#2777aa','#c86b2a']):
        ax.bar(x+offset,[row[m]*100 for m in metric_names],.3,color=color,label=row['policy'])
    ax.set_xticks(x,['AJ','APD avg (1/2/4/8/16px)','Occlusion accuracy']);ax.set_ylim(0,100)
    ax.set_ylabel('Equal-video mean (%)');ax.set_title('Published human labels, native coordinates unchanged between policies')
    ax.legend(loc='lower right');fig.tight_layout();save(fig,output/'davis_official_metrics')
    events=json.loads((ROOT/'results/reappearance_events.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4.5))
    max_delay=max(e['delay_frames'] for e in events if e['recovered'])
    d=np.arange(max_delay+1)
    for policy,color in [('native','#2777aa'),('conservative_vc09','#c86b2a')]:
        ev=[e for e in events if e['policy']==policy]
        times=np.array([e['delay_frames'] if e['recovered'] else np.inf for e in ev])
        ax.step(d,[(times<=n).mean()*100 for n in d],where='post',label=f'{policy}: {np.isfinite(times).sum()}/{len(times)}',color=color)
    ax.set_ylim(0,100);ax.set_xlabel('Frames after human-labeled reappearance')
    ax.set_ylabel('Fraction of ALL events recovered (%)');ax.set_title('Failed events remain in denominator; dependent events, descriptive only')
    ax.legend(loc='lower right');fig.tight_layout();save(fig,output/'reappearance')
    pilot=json.loads((ROOT/'pilot3d_results/per_case.json').read_text())
    names=[r['case'] for r in pilot if r['method']=='cotracker_estimated_depth'];x=np.arange(3)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for method,offset,color in [('cotracker_estimated_depth',-.17,'#2777aa'),('tapip3d_estimated_depth',.17,'#c86b2a')]:
        values=[next(r for r in pilot if r['case']==name and r['method']==method)['raw_epe3d_m']*100 for name in names]
        ax.bar(x+offset,values,.3,color=color,label=method)
    ax.set_xticks(x,names);ax.set_ylabel('Raw metric3D mean error (cm)')
    ax.set_ylim(bottom=0);ax.set_title('PStudio pilot ONLY: 3clips x64frames, first8 eligible queries; no GT scale fit')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15));fig.tight_layout();save(fig,output/'pstudio_raw3d')
    with (ROOT/'results/per_video.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(videos[0]));writer.writeheader();writer.writerows(videos)
    # Descriptive video-cluster bootstrap for the fixed auxiliary-policy contrast.
    rng=np.random.default_rng(20261115)
    native=np.array([[r[m] for m in metric_names] for r in videos if r['policy']=='native'])
    conservative=np.array([[r[m] for m in metric_names] for r in videos if r['policy']=='conservative_vc09'])
    delta=(conservative-native)*100
    bootstrap=delta[rng.integers(0,30,(10000,30))].mean(axis=1)
    contrast={'unit':'video, paired descriptive bootstrap; not an I2V/population causal inference','replicates':10000,'seed':20261115,
              'metrics':{name:{'mean_change_percentage_points':float(delta[:,i].mean()),
                               'percentile95_interval':np.percentile(bootstrap[:,i],[2.5,97.5]).tolist()} for i,name in enumerate(metric_names)}}
    (ROOT/'results/fixed_policy_contrast.json').write_text(json.dumps(contrast,indent=2)+'\n')
    print(json.dumps({'figures':sorted(p.name for p in output.iterdir()),'contrast':contrast},indent=2))


if __name__=='__main__':main()
