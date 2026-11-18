"""Single-query OOM reproduction with source inputs only; no label scores."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--chunk',type=int,default=200)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    assets=json.loads((PROJECT/'research/tracking_validation_20261115/assets.json').read_text())
    sys.path.insert(0,assets['source_directory'])
    from cotracker.predictor import CoTrackerPredictor
    torch.set_num_threads(4)
    torch.manual_seed(20261115)
    torch.cuda.set_device(args.gpu)
    model=CoTrackerPredictor(checkpoint=assets['checkpoint'],offline=True,v2=False,window_len=60).cuda(args.gpu).eval()
    forward=model.model.forward
    def wrapped(*a,**kw):
        return forward(*a,**kw,fmaps_chunk_size=args.chunk)
    model.model.forward=wrapped
    manifest=json.loads((ROOT/'manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['frames']==1083)
    data=np.load(case['files']['input']['path'])
    query=data['queries_tyx'][0,[0,2,1]].copy()
    folder=ROOT/'memory_probes'
    folder.mkdir(exist_ok=True)
    name=f'chunk{args.chunk}'
    record={'case':case['id'],'frames':case['frames'],'chunk':args.chunk,'fresh_process':True,'gpu':args.gpu,
            'query_index':0,'future_truth_read':False}
    started=time.monotonic()
    try:
        video=torch.from_numpy(data['video']).permute(0,3,1,2).float()[None].cuda(args.gpu)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            xy,visible=model(video=video,queries=torch.from_numpy(query)[None,None].cuda(args.gpu),backward_tracking=False)
        np.savez_compressed(folder/(name+'.npz'),xy=xy.cpu().numpy(),visible=visible.cpu().numpy())
        record.update(status='ok',peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    record['seconds']=time.monotonic()-started
    with (folder/(name+'.json')).open('x') as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k!='traceback'}))
