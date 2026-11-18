"""Separate RGB retrieval candidates; immutable initial identity, explicit abstention."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np

OLD=Path(__file__).resolve().parent.parent/'observation_gate_20261115'
spec=importlib.util.spec_from_file_location('frozen_gate',OLD/'gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
CONFIG={'aperture':.5,'correlation':.9,'identity_margin':.15}


def recover(video,candidate_xy,initial_centers,radii):
    observations=gate.evidence(video,candidate_xy,initial_centers,radii,.5)
    baseline=gate.decide(observations,CONFIG)
    gray=[cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY).astype(np.float32) for frame in video]
    halves=[max(3,round(float(r)*.5)) for r in radii]
    refs=[cv2.getRectSubPix(gray[0],(2*h+1,2*h+1),tuple(map(float,c))) for c,h in zip(initial_centers,halves)]
    canonical=[cv2.resize(r,(25,25)) for r in refs]
    identical=observations['reference_pair_similarity']>=.95
    unknown=False
    accepted=np.zeros(candidate_xy.shape[:2],bool)
    output=np.array(candidate_xy,dtype=float,copy=True)
    previous=[None,None];streak=[0,0]
    sources=[];details=[]
    for t,image in enumerate(gray):
        if identical and any(r['reason']=='unresolved_lookalike_history' for r in baseline[t]):unknown=True
        source_row=[];detail_row=[]
        for obj,(reference,h) in enumerate(zip(refs,halves)):
            scores=cv2.matchTemplate(image,reference,cv2.TM_CCOEFF_NORMED)
            y,x=np.unravel_index(scores.argmax(),scores.shape)
            center=np.array([x+h,y+h],dtype=float)
            yy,xx=np.ogrid[:scores.shape[0],:scores.shape[1]]
            away=(xx-x)**2+(yy-y)**2>float(radii[obj])**2
            runner=float(scores[away].max()) if away.any() else -1.
            patch=cv2.resize(cv2.getRectSubPix(image,(2*h+1,2*h+1),tuple(center)),(25,25))
            score=gate.similarity(canonical[obj],patch)
            other=gate.similarity(canonical[1-obj],patch)
            valid=(score>=.9 and float(patch.std())>=1 and score-runner>=.08 and
                   (identical or score-other>=.15) and not unknown)
            reason='accepted';source='abstain'
            if unknown:reason='unresolved_lookalike_history'
            elif score-runner<.08:reason='multiple_spatial_identity_candidates'
            elif not valid:reason='no_unique_current_identity_evidence'
            if valid and baseline[t][obj]['accepted']:
                output[t,obj]=candidate_xy[t,obj]
                accepted[t,obj]=True;source='raw_supported'
                previous[obj]=center;streak[obj]=2
            elif valid:
                consistent=previous[obj] is not None and np.linalg.norm(center-previous[obj])<=float(radii[obj])
                streak[obj]=streak[obj]+1 if consistent else 1
                previous[obj]=center
                output[t,obj]=center
                if streak[obj]>=2:
                    accepted[t,obj]=True;source='global_reacquisition'
                else:reason='pending_second_current_frame'
            else:
                previous[obj]=None;streak[obj]=0
            source_row.append(source)
            detail_row.append({'accepted':bool(accepted[t,obj]),'source':source,'reason':reason,
                               'global_xy':center.tolist(),'score':score,'other_identity_score':other,
                               'spatial_runner_up':runner,'confirmation_streak':streak[obj]})
        if accepted[t].all() and np.linalg.norm(output[t,0]-output[t,1])<.5*sum(radii):
            accepted[t]=False
            for item in detail_row:item.update(accepted=False,reason='candidate_identity_collision',source='abstain')
            source_row=['abstain','abstain']
        sources.append(source_row);details.append(detail_row)
    return {'xy':output,'accepted':accepted,'source':sources,'details':details,
            'baseline_accepted':np.array([[r['accepted'] for r in row] for row in baseline]),
            'reference_pair_similarity':observations['reference_pair_similarity']}
