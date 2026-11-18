"""Show all observed OOV false accepts and the largest accepted location error."""
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
    out=ROOT/'validation_examples';out.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'manifest_validation.json').read_text());wrong=[];absent=[]
    for c in manifest['cases']:
        g=np.load(verify_file(c['files']['truth']),allow_pickle=False)
        p=np.load(ROOT/'predictions_descriptor_validation'/(c['id']+'.npz'),allow_pickle=False)
        known=g['visibility_valid'].copy();known[0]=False
        error=np.linalg.norm((p['xy']-g['xy'])*[.5,256/288],axis=-1)
        for t,i in np.argwhere(known&g['visible']&g['coordinate_valid']&p['reciprocal']&(error>=4)):
            wrong.append({'case':c['id'],'time':int(t),'point':int(i),'error':float(error[t,i]),'kind':'Worst accepted visible-location error'})
        for t,i in np.argwhere(known&~g['visible']&g['out_of_view']&p['reciprocal']):
            absent.append({'case':c['id'],'time':int(t),'point':int(i),'kind':'Known OOV point falsely accepted'})
    selected=[max(wrong,key=lambda r:r['error']),*absent]
    cases={c['id']:c for c in manifest['cases']}
    fig,axes=plt.subplots(len(selected),2,figsize=(11,3.7*len(selected)),squeeze=False)
    for row,r in enumerate(selected):
        c=cases[r['case']];data=np.load(verify_file(c['files']['input']),allow_pickle=False)
        g=np.load(verify_file(c['files']['truth']),allow_pickle=False)
        a=np.load(ROOT/'predictions_native_validation'/(c['id']+'.npz'),allow_pickle=False)
        b=np.load(ROOT/'predictions_descriptor_validation'/(c['id']+'.npz'),allow_pickle=False)
        t=r['time'];i=r['point']
        for col,frame in enumerate([0,t]):
            ax=axes[row,col];ax.imshow(data['video'][frame]);ax.axis('off')
            if col==0 or g['visible'][frame,i]:
                ax.scatter(*g['xy'][frame,i],s=150,facecolors='none',edgecolors='#00ff38',linewidths=2,label='Human point')
            else:ax.text(5,20,'Human label: OUT OF VIEW',color='white',bbox={'facecolor':'black','alpha':.75})
            if col:
                ax.scatter(*b['xy'][frame,i],s=70,c='#ff3060',marker='x',label='MASt3R cycle: accepted')
                ax.scatter(*a['xy'][frame,i],s=60,c='#00c9ff',marker='+',label=f'Co native: accepted={bool(a["native"][frame,i])}')
                ax.legend(loc='lower right',fontsize=8)
            ax.set(xlim=(-.5,511.5),ylim=(287.5,-.5),title=f'{r["kind"]}\n'+('Reference query' if col==0 else f'Target frame {t}'))
    fig.suptitle('New participant P10: diagnostic examples, no threshold tuning',fontsize=12);fig.tight_layout()
    for suffix in ['png','pdf','svg']:fig.savefig(out/('examples.'+suffix),dpi=160)
    plt.close(fig)
    save(out/'selection.json',{'rule':'largest accepted visible error and ALL observed known-OOV false accepts',
        'wrong_candidates':len(wrong),'oov_false_accepts':len(absent),'selected':selected,'source_sha256':sha(Path(__file__)),
        'no_target_truth_plotted_for_absent_points':True,'not_representative_sampling':True})
