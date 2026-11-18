"""Actual two-frame author API check before scoring new data."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]

if __name__=='__main__':
    assets=json.loads((PROJECT/'research/tracking_validation_20261115/assets.json').read_text())
    sys.path.insert(0,assets['source_directory'])
    from cotracker.predictor import CoTrackerPredictor
    manifest=json.loads((PROJECT/'research/proposal_completion_20261117/public_manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['domain']=='davis')
    data=np.load(case['files']['input']['path'])
    query=data['queries_tyx'][0]
    frame=data['video'][int(query[0])]
    pair=np.stack([frame,frame])
    torch.set_num_threads(4)
    torch.manual_seed(20261115)
    model=CoTrackerPredictor(checkpoint=assets['checkpoint'],offline=True,v2=False,window_len=60).cuda(0).eval()
    point=np.array([0,query[2],query[1]],np.float32)
    start=time.monotonic()
    with torch.inference_mode():
        xy,visible=model(video=torch.from_numpy(pair).permute(0,3,1,2).float()[None].cuda(0),
             queries=torch.from_numpy(point)[None,None].cuda(0),backward_tracking=False)
    value=xy.cpu().numpy()
    result={'synthetic_identity_pair_from_existing_DAVIS':case['id'],'source_frame':int(query[0]),
            'frame_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'native_query_txy':point.tolist(),
            'prediction_xy':value.tolist(),'visible':visible.cpu().numpy().tolist(),
            'identity_pair_pixel_error':float(np.linalg.norm(value[0,1,0]-point[1:])),
            'seconds':time.monotonic()-start,'future_truth_read':False,'scope':'API/two-identical-frame check only, not real-domain accuracy'}
    with (ROOT/'smoke_reverse.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))
