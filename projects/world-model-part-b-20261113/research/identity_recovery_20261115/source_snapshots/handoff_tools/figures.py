"""Control recovery, failed cross-domain coverage and depth-drift diagnostics."""
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
V2=ROOT.parent/'identity_recovery_multiscale_20261115'


def save(fig,path):
    for suffix in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+suffix),dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    output=ROOT/'figures';output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    summary=json.loads((V2/'control_results/summary.json').read_text())
    heldout=[r for r in summary if r['split']=='heldout']
    conditions=[r['condition'] for r in heldout if r['method']=='raw' and r['condition']!='ALL']
    fig,ax=plt.subplots(figsize=(13,5.2));x=np.arange(len(conditions))
    for offset,method,color in [(-.25,'raw','#b6b6b6'),(0,'previous_gate','#377ba8'),(.25,'recovery','#c36d34')]:
        values=[next(r for r in heldout if r['condition']==c and r['method']==method)['retained_correct']*100 for c in conditions]
        ax.bar(x+offset,values,.23,label=method,color=color)
    ax.axhline(90,color='black',ls='--',lw=.8);ax.set_ylim(0,105)
    ax.set_xticks(x,[c.replace('_','\n') for c in conditions],rotation=40,ha='right')
    ax.set_ylabel('Correct accepted / all identifiable visible (%)')
    ax.set_title('Recovery V2: 36 new heldout sequences, 3 seeds/condition; centers, query frame excluded')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.58),ncol=3);fig.tight_layout()
    save(fig,output/'controlled_recovery')
    canvas=Image.new('RGB',(5*384,2*280+30),'white');draw=ImageDraw.Draw(canvas)
    name='identity_replacement_191201';data=np.load(V2/'data'/name/'input.npz');truth=np.load(V2/'data'/name/'truth.npz')
    raw=np.load(V2/'predictions/cotracker/primary/heldout'/f'{name}.npz')
    recovery=np.load(V2/'control_results'/f'{name}.npz')
    for row,method in enumerate(['raw','recovery']):
        xy=raw['xy'][:,[0,5]] if method=='raw' else recovery['xy']
        visible=raw['official_visible'][:,[0,5]] if method=='raw' else recovery['accepted']
        for col,t in enumerate([0,15,16,17,24]):
            tile=Image.fromarray(data['video'][t]);td=ImageDraw.Draw(tile)
            for obj,q in enumerate([0,5]):
                if truth['visible'][t,q]:
                    u,v=truth['xy'][t,q];td.ellipse((u-6,v-6,u+6,v+6),outline='white',width=2)
                u,v=xy[t,obj];color='#1adf61' if visible[t,obj] else '#ee951f'
                td.line((u-5,v-5,u+5,v+5),fill=color,width=2);td.line((u-5,v+5,u+5,v-5),fill=color,width=2)
                td.text((u+6,v-10),str(obj),fill=color)
            canvas.paste(tile,(col*384,row*280));draw.text((col*384+4,row*280+258),f'{method} frame{t}',fill='black')
    draw.text((4,562),'White ring: visible GT center. X: raw or separately recovered coordinate; green=claimed, orange=abstained.',fill='black')
    canvas.save(output/'return_example.png')
    depths=json.loads((ROOT/'real_depth_results/depth_controls.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4.5))
    groups=['clean','occlusion','camera_motion','identity_replacement']
    for offset,region,color in [(-.12,'objects','#c36d34'),(.12,'background','#377ba8')]:
        for i,c in enumerate(groups):
            values=[r[region]['AbsRel']*100 for r in depths if r['condition']==c]
            ax.scatter(i+offset+np.linspace(-.035,.035,len(values)),values,color=color)
            ax.plot([i+offset-.07,i+offset+.07],[np.mean(values)]*2,color=color,lw=3)
        ax.plot([],[],marker='o',color=color,label=region)
    ax.set_xticks(range(4),[c.replace('_','\n') for c in groups]);ax.set_ylim(0,100)
    ax.set_ylabel('Raw metric-depth AbsRel (%)');ax.set_title('Official DA-V2 Hypersim Small: 12 controls; no GT scale fitting')
    ax.legend();fig.tight_layout();save(fig,output/'depth_control_error')
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True)
    static=json.loads((ROOT/'real_depth_results/depth_real_static.json').read_text())
    for ax,row in zip(axes.ravel(),static):
        ratios=np.load(ROOT/'real_depth_results'/f'{row["clip"]}_static_depth.npz')['ratios']
        for i in range(4):ax.plot(np.arange(97)/24,ratios[:,i],lw=1,label=f'patch{i+1}')
        ax.axhline(1,color='black',ls='--',lw=.7)
        ax.set_title(f'{row["clip"]}: max change {row["background_relative_change_max"]*100:.1f}%')
        ax.set_xlabel('Time (s)');ax.set_ylabel('Estimated depth / same patch at t0')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Real I2V static-background sensitivity, 4 fixed 5x5 patches; no depth ground truth')
    fig.tight_layout();save(fig,output/'real_depth_drift')
    scores=json.loads((ROOT/'real_depth_results/reference_scores.json').read_text())
    with (ROOT/'real_depth_results/reference_scores.csv').open('w',newline='') as f:
        columns=[k for k in scores[0] if k!='errors_px'];w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(scores)
    appearance=[]
    for case in json.loads((ROOT/'real_manifest.json').read_text()):
        inputs=np.load(case['source']);rgb=inputs[inputs.files[0]]
        gray=cv2.cvtColor(rgb[0],cv2.COLOR_RGB2GRAY).astype(np.float32)
        for obj,center in enumerate(case['initial_centers']):
            patch=cv2.getRectSubPix(gray,(33,33),tuple(map(float,center)))
            appearance.append({'clip':case['clip'],'object':obj,'initial_gray_patch_std':float(patch.std()),
                               'gate_current_patch_std_minimum':1.,'reference_is_uniform_or_low_texture':bool(patch.std()<1)})
    (ROOT/'real_depth_results/initial_appearance_diagnostic.json').write_text(json.dumps(appearance,indent=2)+'\n')
    print(json.dumps({'figures':sorted(p.name for p in output.iterdir()),'initial_appearance':appearance},indent=2))


if __name__=='__main__':main()
