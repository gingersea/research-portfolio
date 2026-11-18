"""Deterministic development examples: worst accepted error and gained return."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha,verify_file

if __name__=='__main__':
    out=ROOT/'development_examples_v2';out.mkdir(exist_ok=False)
    manifest=json.loads((PROJECT/'research/real_measurement_20261117/manifest.json').read_text())
    bad=[];gains=[]
    for case in manifest['cases']:
        truth=dict(np.load(verify_file(case['files']['truth']),allow_pickle=False))
        descriptor=dict(np.load(ROOT/'predictions_development'/(case['id']+'.npz'),allow_pickle=False))
        native=dict(np.load(PROJECT/'research/real_measurement_20261117/predictions_memory'/(case['id']+'.npz'),allow_pickle=False))
        known=truth['visibility_valid'].copy();known[0]=False
        positive=known&truth['visible']&truth['coordinate_valid']
        error=np.linalg.norm((descriptor['xy']-truth['xy'])*[.5,256/288],axis=-1)
        native_error=np.linalg.norm((native['xy']-truth['xy'])*[.5,256/288],axis=-1)
        for t,p in np.argwhere(positive&descriptor['reciprocal']&(error>=4)):
            bad.append({'case':case['id'],'time':int(t),'point':int(p),'normalized_error':float(error[t,p]),'kind':'largest accepted wrong location'})
        for p in range(positive.shape[1]):
            pending=False
            for t in range(1,len(positive)):
                if known[t,p] and truth['out_of_view'][t,p]:pending=True
                elif pending and positive[t,p]:
                    if descriptor['reciprocal'][t,p] and error[t,p]<4 and not(native['native'][t,p] and native_error[t,p]<4):
                        gains.append({'case':case['id'],'time':t,'point':p,'normalized_error':float(error[t,p]),'kind':'additional accepted return; native XY already correct'})
                    pending=False
    selected=[max(bad,key=lambda r:r['normalized_error'])]
    if gains:selected.append(gains[0])
    fig,axes=plt.subplots(len(selected),2,figsize=(11,3.8*len(selected)),squeeze=False)
    byid={c['id']:c for c in manifest['cases']}
    for row,example in enumerate(selected):
        c=byid[example['case']];data=np.load(verify_file(c['files']['input']),allow_pickle=False)
        truth=np.load(verify_file(c['files']['truth']),allow_pickle=False)
        pred=np.load(ROOT/'predictions_development'/(c['id']+'.npz'),allow_pickle=False)
        native=np.load(PROJECT/'research/real_measurement_20261117/predictions_memory'/(c['id']+'.npz'),allow_pickle=False)
        t=example['time'];p=example['point']
        for col,frame in enumerate([0,t]):
            ax=axes[row,col];ax.imshow(data['video'][frame]);ax.axis('off')
            gt=truth['xy'][frame,p];ax.scatter(*gt,s=150,facecolors='none',edgecolors='#00ff38',linewidths=2,label='Human point')
            if col:
                ax.scatter(*pred['xy'][t,p],s=70,c='#ff3060',marker='x',label='MASt3R cycle')
                ax.scatter(*native['xy'][t,p],s=60,c='#00c9ff',marker='+',label='Native CoTracker')
                ax.legend(loc='lower right',fontsize=8)
            ax.set_title(('Reference query' if col==0 else f'Target frame {t}; error={example["normalized_error"]:.1f}/256')+'\n'+example['kind'],fontsize=10)
    fig.suptitle('Development examples only; fixed selection rule, not new validation',fontsize=12)
    fig.tight_layout()
    for suffix in ['png','pdf','svg']:fig.savefig(out/('examples.'+suffix),dpi=160)
    plt.close(fig)
    save(out/'selection.json',{'selected':selected,'wrong_location_candidates':len(bad),'additional_accepted_return_candidates':len(gains),
         'selection':'largest accepted error, then first native-failed/descriptor-correct annotated return in archive order',
         'source_sha256':sha(Path(__file__)),'not_representative_examples':True})
