"""Independent image evidence: fixed initial appearance and current-frame matching."""
import cv2
import numpy as np


def similarity(a, b):
    a, b = a.astype(float).ravel(), b.astype(float).ravel()
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator) if denominator > 1e-8 else 0.


def evidence(video, candidate_xy, initial_centers, radii, aperture=.6):
    gray = [cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY).astype(np.float32) for frame in video]
    half = [max(3,round(float(radius)*aperture)) for radius in radii]
    references = [cv2.getRectSubPix(gray[0],(2*h+1,2*h+1),tuple(map(float,center))) for center,h in zip(initial_centers,half)]
    canonical = [cv2.resize(patch,(25,25)) for patch in references]
    pair_similarity=similarity(canonical[0],canonical[1])
    rows=[]
    for t,image in enumerate(gray):
        objects=[]
        for obj,(reference,h) in enumerate(zip(references,half)):
            xy=np.asarray(candidate_xy[t,obj])
            item={'score':0.,'cross_score':0.,'contrast':0.,'local_offset_px':None,'evidence_xy':None,'global_score':0.}
            if not np.isfinite(xy).all() or not (h <= xy[0] < image.shape[1]-h and h <= xy[1] < image.shape[0]-h):
                objects.append(item);continue
            scores=cv2.matchTemplate(image,reference,cv2.TM_CCOEFF_NORMED)
            item['global_score']=float(scores.max())
            x,y=np.rint(xy).astype(int)-h
            x0,x1=max(0,x-3),min(scores.shape[1],x+4)
            y0,y1=max(0,y-3),min(scores.shape[0],y+4)
            local=scores[y0:y1,x0:x1]
            if not local.size:
                objects.append(item);continue
            ly,lx=np.unravel_index(local.argmax(),local.shape)
            center=np.array([x0+lx+h,y0+ly+h],dtype=float)
            patch=cv2.getRectSubPix(image,(2*h+1,2*h+1),tuple(center))
            patch_canonical=cv2.resize(patch,(25,25))
            item.update(score=similarity(canonical[obj],patch_canonical),
                        cross_score=similarity(canonical[1-obj],patch_canonical),contrast=float(patch.std()),
                        local_offset_px=float(np.linalg.norm(center-xy)),evidence_xy=center.tolist())
            objects.append(item)
        rows.append(objects)
    return {'frames':rows,'reference_pair_similarity':pair_similarity,'aperture':aperture,'radii':list(map(float,radii))}


def decide(observations,config):
    same_appearance=observations['reference_pair_similarity'] >= .95
    latched=False
    output=[]
    for t,row in enumerate(observations['frames']):
        basic=[item['score'] >= config['correlation'] and item['contrast'] >= 1 and
               item['local_offset_px'] is not None and item['local_offset_px'] <= 3 for item in row]
        if same_appearance and t > 0:
            points=[item['evidence_xy'] for item in row]
            if not all(basic) or any(point is None for point in points):
                latched=True
            elif np.linalg.norm(np.array(points[0])-points[1]) <= sum(observations['radii'])+3:
                latched=True
        decisions=[]
        for obj,item in enumerate(row):
            reason='accepted'
            if not basic[obj]:reason='no_current_appearance_evidence'
            elif same_appearance and latched:reason='unresolved_lookalike_history'
            elif not same_appearance and item['score']-item['cross_score'] < config['identity_margin']:
                reason='competing_initial_identity'
            elif not same_appearance and item['global_score']-item['score'] > .08:
                reason='better_identity_evidence_elsewhere'
            decisions.append({'accepted':reason=='accepted','reason':reason,**item})
        output.append(decisions)
    return output


def observe(video,candidate_xy,initial_centers,radii,config):
    return decide(evidence(video,candidate_xy,initial_centers,radii,config['aperture']),config)
