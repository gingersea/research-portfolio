"""One evaluation query at a time; fixed author support grid, no future GT."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fetch import sha

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--part',type=int,choices=[0,1],required=True);args=parser.parse_args()
    assets=json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
    source=Path(assets['source_directory'])
    assert sha(assets['checkpoint'])==assets['checkpoint_sha256']
    assert all(sha(source/r['file'])==r['sha256'] for r in assets['source_files'])
    sys.path.insert(0,str(source));from cotracker.predictor import CoTrackerPredictor
    torch.set_num_threads(4);torch.manual_seed(20261115);torch.cuda.set_device(args.gpu)
    device=f'cuda:{args.gpu}'
    model=CoTrackerPredictor(checkpoint=assets['checkpoint'],offline=True,v2=False,window_len=60).to(device).eval()
    captured=[];native=model.model.forward

    def observer(*a,**kw):
        out=native(*a,**kw);captured.append([v.detach().cpu().numpy().copy() for v in out[1:3]]);return out

    model.model.forward=observer
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    assert manifest['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    output=ROOT/'predictions';output.mkdir(exist_ok=True)
    for case in manifest['cases']:
        if case['index']%2!=args.part:continue
        receipt_path=output/f'{case["case"]}.json';assert not receipt_path.exists()
        assert sha(ROOT/case['input'])==case['input_sha256']
        inputs=np.load(ROOT/case['input']);rgb=inputs['video'];query=inputs['queries_tyx']
        T,N=len(rgb),len(query);xy=np.empty((T,N,2),np.float32);visible=np.empty((T,N),bool)
        raw_v=np.empty((T,N),np.float32);raw_c=np.empty((T,N),np.float32)
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        print(json.dumps({'event':'start','case':case['case'],'frames':T,'queries':N}),flush=True)
        record={'case':case['case'],'input_sha256':case['input_sha256'],'checkpoint_sha256':assets['checkpoint_sha256'],
                'source_revision':assets['source_revision'],'runner_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
                'query_mode':'first','evaluation_queries_jointly_inferred':False,'support_grid':6,'iterations':6,'dtype':'float32',
                'future_truth_read':False,'frames':T,'queries':N}
        try:
            video=torch.from_numpy(rgb).permute(0,3,1,2).float()[None].to(device)
            with torch.inference_mode():
                for i,q in enumerate(query):
                    captured.clear();single=torch.from_numpy(q[[0,2,1]].copy())[None,None].to(device)
                    predicted,declared=model(video=video,queries=single,backward_tracking=False)
                    assert len(captured)==1
                    xy[:,i]=predicted[0,:,0].cpu().numpy();visible[:,i]=declared[0,:,0].cpu().numpy()
                    raw_v[:,i]=captured[0][0][0,:,0];raw_c[:,i]=captured[0][1][0,:,0]
            assert np.isfinite(xy).all() and np.isfinite(raw_v).all() and np.isfinite(raw_c).all()
            target=output/f'{case["case"]}.npz'
            np.savez_compressed(target,xy=xy,visible=visible,visibility=raw_v,confidence=raw_c,
                                conservative_visible=(raw_v*raw_c>.9))
            record.update(status='ok',prediction_sha256=sha(target),elapsed_seconds=time.monotonic()-started,
                          model_resolution=list(model.interp_shape),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        except Exception as error:
            record.update(status='api_failure',exception=repr(error),elapsed_seconds=time.monotonic()-started)
            receipt_path.write_text(json.dumps(record,indent=2)+'\n');raise
        receipt_path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'event':'complete',**record}),flush=True)


if __name__=='__main__':main()
