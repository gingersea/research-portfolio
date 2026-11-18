"""Descriptive figures and all-query auxiliary results; no threshold selection."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from evaluate import lift_scene, metrics

ROOT = Path(__file__).resolve().parent
CONDITIONS = ['clean','textureless','hue_change','same_hue','occlusion','disappearance',
              'camera_motion','crossing','identity_replacement','ambiguous_twins']
MODELS = ['cotracker','tapip3d']
LABELS = {'cotracker':'CoTracker3 + sensor depth','tapip3d':'TAPIP3D (sensor RGB-D)'}
COLORS = {'cotracker':'#2473a8','tapip3d':'#ce6b25'}


def save(fig,path):
    for suffix in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+suffix),dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures',default='figures')
    args = parser.parse_args()
    folder = ROOT/args.figures
    folder.mkdir(exist_ok=False)
    summary = json.loads((ROOT/'comparison/summary.json').read_text())
    rows = json.loads((ROOT/'comparison/case_metrics.json').read_text())
    heldout = [r for r in rows if r['split']=='heldout']
    lookup = {(r['condition'],r['model'],r['mode']):r for r in summary if r['split']=='heldout'}
    if not (ROOT/'comparison/summary.csv').exists():
        with (ROOT/'comparison/summary.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(14,6),sharey=True)
    x = np.arange(len(CONDITIONS))
    for ax,method in zip(axes,MODELS):
        for offset,mode,color in [(-.25,'raw','#b0b0b0'),(0,'primary',COLORS[method]),(.25,'coverage','#71ad87')]:
            values = [lookup[c,method,mode]['retained_correct']*100 for c in CONDITIONS]
            ax.bar(x+offset,values,.23,color=color,label=mode)
        ax.axhline(90,color='black',ls='--',lw=.8,label='90% coverage criterion')
        ax.set_xticks(x,[c.replace('_','\n') for c in CONDITIONS],rotation=40,ha='right')
        ax.set_ylim(0,105);ax.set_title(LABELS[method])
        ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.57),ncol=2)
        if ax is axes[0]:
            ax.set_ylabel('Correct accepted / visible identifiable GT (%)')
    fig.suptitle('30 heldout sequences: 3 seeds per condition; query frame excluded; centers only')
    fig.tight_layout();save(fig,folder/'coverage')
    fig,axes = plt.subplots(1,2,figsize=(14,5))
    for ax,mode,title in zip(axes,['raw','common_primary'],['All identifiable visible GT','Same jointly accepted observations']):
        for method,offset in zip(MODELS,[-.14,.14]):
            for i,c in enumerate(CONDITIONS):
                data = [r['counts'] for r in heldout if r['condition']==c and r['model']==method and r['mode']==mode]
                vals = [(d['epe3d_sum']/d['positive'] if mode=='raw' else d['accepted_epe3d_sum']/d['true_visible'])*100
                        for d in data if (d['positive'] if mode=='raw' else d['true_visible'])>0]
                if not vals:continue
                ax.scatter(i+offset+np.linspace(-.04,.04,len(vals)),vals,s=22,color=COLORS[method],alpha=.7)
                ax.plot([i+offset-.08,i+offset+.08],[np.mean(vals)]*2,color=COLORS[method],lw=3)
            ax.plot([],[],color=COLORS[method],marker='o',label=LABELS[method])
        ax.set_xticks(x,[c.replace('_','\n') for c in CONDITIONS],rotation=40,ha='right')
        ax.set_yscale('log');ax.set_ylabel('World 3D mean error (cm, log scale)')
        ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Dots: sequences; short lines: unweighted sequence means; no frame-level error bars; twins common=N/A')
    fig.tight_layout();save(fig,folder/'paired_3d')
    canvas = Image.new('RGB',(5*384,4*285+45),'white');draw=ImageDraw.Draw(canvas)
    for row,(condition,method) in enumerate((c,m) for c in ['identity_replacement','hue_change'] for m in MODELS):
        name=f'{condition}_1810'
        video=np.load(ROOT/'data'/name/'input.npz')['video']
        truth=np.load(ROOT/'data'/name/'truth.npz')
        pred=np.load(ROOT/'predictions'/method/'primary/heldout'/f'{name}.npz')
        mask=np.load(ROOT/'comparison'/f'{method}_{name}_masks.npz')['primary']
        for col,t in enumerate([0,12,18,24,32]):
            tile=Image.fromarray(video[t]);td=ImageDraw.Draw(tile)
            for obj,q in enumerate([0,5]):
                if truth['visible'][t,q] and truth['identifiable'][t,q]:
                    u,v=truth['xy'][t,q];td.ellipse((u-5,v-5,u+5,v+5),outline='white',width=2)
                u,v=map(float,pred['xy'][t,q])
                color='#1be064' if mask[t,obj] else '#f79824'
                td.line((u-5,v-5,u+5,v+5),fill=color,width=2);td.line((u-5,v+5,u+5,v-5),fill=color,width=2)
                td.text((u+5,v-12),str(obj),fill=color)
            canvas.paste(tile,(col*384,row*285))
            draw.text((col*384+3,row*285+258),f'{condition} / {method} / frame {t}',fill='black')
    draw.text((5,4*285+5),'White ring: identifiable visible GT. X: raw prediction; green=accepted by primary gate + native visibility; orange=rejected.',fill='black')
    canvas.save(folder/'failure_overlays.png')
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    auxiliary=[]
    for case in manifest['cases']:
        truth=np.load(ROOT/case['truth']);sensor=np.load(ROOT/case['rgbd_input'])
        for method in MODELS:
            pred=np.load(ROOT/'predictions'/method/'primary'/case['split']/f'{case["case"]}.npz')
            result=metrics.score_case(pred['xy'],pred['official_visible'],truth['xy'],truth['visible'],truth['identifiable'])
            xyz=pred['xyz_world'] if method=='tapip3d' else lift_scene(pred['xy'],sensor)
            valid=truth['visible']&truth['identifiable'];valid[0]=False
            e3=np.linalg.norm(xyz-truth['xyz_world'],axis=-1)
            auxiliary.append({'case':case['case'],'condition':case['condition'],'split':case['split'],'model':method,
                              'raw_all10':result,'positive3d':int(valid.sum()),'epe3d_sum':float(e3[valid].sum())})
    aux_path = ROOT/'comparison/auxiliary_all10.json'
    if aux_path.exists():
        assert json.loads(aux_path.read_text()) == auxiliary
    else:
        aux_path.write_text(json.dumps(auxiliary,indent=2)+'\n')
    print(json.dumps({'figures':sorted(p.name for p in folder.iterdir()),'auxiliary_sequences':len(auxiliary)},indent=2))


if __name__=='__main__':main()
