"""Scale-aware lost-target retrieval with separate temporal persistence evidence."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np

spec=importlib.util.spec_from_file_location('immutable_gate',Path(__file__).resolve().parent.parent/'observation_gate_20261115/gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def recover(video,candidate_xy,initial_centers,radii):
    obs=gate.evidence(video,candidate_xy,initial_centers,radii,.5)
    base=gate.decide(obs,{'aperture':.5,'correlation':.9,'identity_margin':.15})
    gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY).astype(np.float32) for f in video]
    refs=[];banks=[]
    for center,r in zip(initial_centers,radii):
        h=max(3,round(float(r)*.5));ref=cv2.getRectSubPix(gray[0],(2*h+1,2*h+1),tuple(map(float,center)))
        refs.append(cv2.resize(ref,(25,25)))
        sizes=sorted({max(7,int(round(len(ref)*s))//2*2+1) for s in np.arange(.75,1.251,.05)})
        banks.append([(size//2,cv2.resize(ref,(size,size))) for size in sizes])
    accepted=np.zeros(candidate_xy.shape[:2],bool);xy=np.array(candidate_xy,dtype=float,copy=True)
    previous=[None,None];streak=[0,0];unknown=False;details=[];sources=[]
    identical=obs['reference_pair_similarity']>=.95
    for t,image in enumerate(gray):
        if identical and any(r['reason']=='unresolved_lookalike_history' for r in base[t]):unknown=True
        row=[];source_row=[]
        for obj in range(2):
            continuous=t==0 or accepted[t-1,obj] and np.linalg.norm(candidate_xy[t,obj]-xy[t-1,obj])<=radii[obj]
            if base[t][obj]['accepted'] and continuous and not unknown:
                accepted[t,obj]=True;previous[obj]=np.array(xy[t,obj]);streak[obj]=2
                source='raw_supported_persistence'
                row.append({'accepted':True,'source':source,'reason':'supported_continuous_identity',
                            'candidate_xy':xy[t,obj].tolist()});source_row.append(source);continue
            maps=[];best=(-2.,None,None)
            for h,reference in banks[obj]:
                scores=cv2.matchTemplate(image,reference,cv2.TM_CCOEFF_NORMED)
                y,x=np.unravel_index(scores.argmax(),scores.shape)
                maps.append((h,scores))
                if scores[y,x]>best[0]:best=(float(scores[y,x]),np.array([x+h,y+h],dtype=float),h)
            score,center,h=best
            runner=-1.
            for hh,scores in maps:
                yy,xx=np.ogrid[:scores.shape[0],:scores.shape[1]]
                away=(xx+hh-center[0])**2+(yy+hh-center[1])**2>(1.25*radii[obj])**2
                if away.any():runner=max(runner,float(scores[away].max()))
            patch=cv2.resize(cv2.getRectSubPix(image,(2*h+1,2*h+1),tuple(center)),(25,25))
            other=gate.similarity(refs[1-obj],patch)
            valid=(not unknown and score>=.9 and patch.std()>=1 and score-runner>=.08 and
                   (identical or gate.similarity(refs[obj],patch)-other>=.15))
            reason='no_unique_current_identity_evidence';source='abstain'
            if unknown:reason='unresolved_lookalike_history'
            elif score-runner<.08:reason='multiple_spatial_identity_candidates'
            if valid:
                consistent=previous[obj] is not None and np.linalg.norm(center-previous[obj])<=radii[obj]
                streak[obj]=streak[obj]+1 if consistent else 1;previous[obj]=center;xy[t,obj]=center
                if streak[obj]>=2:accepted[t,obj]=True;reason='accepted';source='global_reacquisition'
                else:reason='pending_second_current_frame'
            else:previous[obj]=None;streak[obj]=0
            row.append({'accepted':bool(accepted[t,obj]),'source':source,'reason':reason,'candidate_xy':center.tolist(),
                        'score':score,'other_identity_score':other,'spatial_runner_up':runner,'patch_half_size':h,
                        'confirmation_streak':streak[obj]});source_row.append(source)
        if accepted[t].all() and np.linalg.norm(xy[t,0]-xy[t,1])<.5*sum(radii):
            accepted[t]=False;source_row=['abstain','abstain']
            for r in row:r.update(accepted=False,source='abstain',reason='identity_candidate_collision')
        details.append(row);sources.append(source_row)
    return {'xy':xy,'accepted':accepted,'source':sources,'details':details,
            'baseline_accepted':np.array([[r['accepted'] for r in row] for row in base]),
            'reference_pair_similarity':obs['reference_pair_similarity']}
