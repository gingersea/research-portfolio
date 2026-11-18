"""Same querywise full CoTracker3 and frozen two-frame reverse gate on new cases."""
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
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha,verify_file
from physcheck.relations.sparse import inside,reciprocal_accept

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1);args=p.parse_args()
    manifest=json.loads((ROOT/'manifest_validation.json').read_text())
    freeze=json.loads((ROOT/'validation_freeze.json').read_text())
    assert freeze['manifest_sha256']==sha(ROOT/'manifest_validation.json')
    for file,digest in freeze['sources'].items():assert sha(Path(file))==digest
    assets=json.loads((PROJECT/'research/tracking_validation_20261115/assets.json').read_text())
    assert sha(Path(assets['checkpoint']))==assets['checkpoint_sha256']
    assert all(sha(Path(assets['source_directory'])/r['file'])==r['sha256'] for r in assets['source_files'])
    sys.path.insert(0,assets['source_directory'])
    from cotracker.predictor import CoTrackerPredictor
    torch.set_num_threads(4);torch.manual_seed(20261115);torch.cuda.set_device(args.gpu)
    device=f'cuda:{args.gpu}'
    output=ROOT/'predictions_native_validation';output.mkdir(exist_ok=False)
    model=CoTrackerPredictor(checkpoint=assets['checkpoint'],offline=True,v2=False,window_len=60).to(device).eval()
    for case in manifest['cases']:
        start=time.monotonic();record={'case':case['id'],'input_sha256':case['files']['input']['sha256'],
            'manifest_sha256':sha(ROOT/'manifest_validation.json'),'runner_sha256':sha(Path(__file__)),
            'checkpoint_sha256':assets['checkpoint_sha256'],'source_revision':assets['source_revision'],
            'future_truth_read':False,'requested_timestamps_provided':True,'evaluation_queries_jointly_inferred':False,
            'support_grid':6,'iterations':6,'feature_chunk_size':200,'empty_cache_per_full_query':True,'dtype':'float32','gpu':args.gpu}
        print(json.dumps({'event':'start','case':case['id'],'frames':case['frames'],'queries':case['queries']}),flush=True)
        try:
            data=np.load(verify_file(case['files']['input']),allow_pickle=False)
            rgb=data['video'];q=data['queries_tyx'];times=data['requested_times'];t,h,w=rgb.shape[:3];n=len(q)
            xy=np.empty((t,n,2),np.float32);native=np.empty((t,n),bool);cycle=np.zeros((t,n),bool)
            reverse=np.full((len(times),n,2),np.nan,np.float32);reverse_vis=np.zeros((len(times),n),bool)
            frames=torch.from_numpy(rgb).permute(0,3,1,2).float()[None].to(device)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for point,query in enumerate(q):
                    torch.cuda.empty_cache()
                    coords,vis=model(video=frames,queries=torch.from_numpy(query[[0,2,1]].copy())[None,None].to(device),backward_tracking=False)
                    xy[:,point]=coords[0,:,0].cpu().numpy();native[:,point]=vis[0,:,0].cpu().numpy()
                    for slot,target in enumerate(times):
                        candidate=xy[target,point]
                        if not native[target,point] or not inside(candidate,(h,w)):continue
                        query_back=np.array([0,candidate[0],candidate[1]],np.float32)
                        back,back_vis=model(video=frames[:,[int(target),0]],queries=torch.from_numpy(query_back)[None,None].to(device),backward_tracking=False)
                        reverse[slot,point]=back[0,1,0].cpu().numpy();reverse_vis[slot,point]=bool(back_vis[0,1,0].cpu())
                        cycle[target,point]=reciprocal_accept(candidate,native[target,point],reverse[slot,point],reverse_vis[slot,point],query[[2,1]],(h,w))
                    print(json.dumps({'event':'query_done','case':case['id'],'query':point+1,'queries':n}),flush=True)
            native &= inside(xy,(h,w));assert np.isfinite(xy).all()
            path=output/(case['id']+'.npz')
            np.savez_compressed(path,xy=xy,native=native,reciprocal=cycle,reverse_xy=reverse,reverse_visible=reverse_vis,requested_times=times)
            record.update(status='ok',prediction_sha256=sha(path),peak_gpu_bytes=torch.cuda.max_memory_allocated())
            del frames
        except Exception as error:
            record.update(status='inference_failed',error=repr(error),traceback=traceback.format_exc(),planned_case_retained=True)
            if 'frames' in locals():del frames
            torch.cuda.empty_cache()
        record['elapsed_seconds']=time.monotonic()-start;save(output/(case['id']+'.json'),record)
        print(json.dumps({'event':'done','case':case['id'],'status':record['status'],'seconds':record['elapsed_seconds']}),flush=True)
