"""Initial query placement only, without future truth or prediction overlays."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    manifest=json.loads((ROOT/'manifest.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,case in zip(axes.flat,manifest['cases'][:4]):
        inputs=np.load(case['files']['input']['path'])
        ax.imshow(inputs['video'][0])
        q=inputs['queries_tyx']
        ax.scatter(q[:,2],q[:,1],s=25,facecolors='none',edgecolors='cyan')
        for i,row in enumerate(q):
            ax.text(row[2]+3,row[1]-2,str(i),fontsize=7,color='cyan')
        ax.set_title(case['id'],fontsize=9)
        ax.axis('off')
    fig.savefig(ROOT/'initial_query_audit.png',dpi=150)
    plt.close(fig)
