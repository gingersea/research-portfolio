"""Same fixed model/inputs; release unused CUDA cache before full query calls."""
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--part',type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    assets=json.loads((PROJECT/'research/tracking_validation_20261115/assets.json').read_text())
    assert sha(Path(assets['checkpoint']))==assets['checkpoint_sha256']
    assert all(sha(Path(assets['source_directory'])/r['file'])==r['sha256'] for r in assets['source_files'])
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert manifest['amendment_sha256']==sha(ROOT/'PROTOCOL_v2.md')
    assert manifest['acquisition_stop_sha256']==sha(ROOT/'PROTOCOL_v3.md')
    frozen=json.loads((ROOT/'memory_execution_freeze.json').read_text())
    assert frozen['runner_sha256']==sha(Path(__file__))
    assert frozen['manifest_sha256']==sha(ROOT/'manifest.json')
    sys.path.insert(0,assets['source_directory'])
    from cotracker.predictor import CoTrackerPredictor
    torch.set_num_threads(4)
    torch.manual_seed(20261115)
    torch.cuda.set_device(args.gpu)
    device=f'cuda:{args.gpu}'
    model=CoTrackerPredictor(checkpoint=assets['checkpoint'],offline=True,v2=False,window_len=60).to(device).eval()
    captured=[]
    original=model.model.forward
    def observer(*a,**kw):
        out=original(*a,**kw)
        captured.append([v.detach().cpu().numpy().copy() for v in out[1:3]])
        return out
    model.model.forward=observer
    output=ROOT/'predictions_memory'
    output.mkdir(exist_ok=True)
    for index,case in enumerate(manifest['cases']):
        if index%2!=args.part:
            continue
        path=output/(case['id']+'.json')
        assert not path.exists()
        started=time.monotonic()
        record={'case':case['id'],'input_sha256':case['files']['input']['sha256'],
                'checkpoint_sha256':assets['checkpoint_sha256'],'source_revision':assets['source_revision'],
                'runner_sha256':sha(Path(__file__)),'manifest_sha256':sha(ROOT/'manifest.json'),
                'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'amendment_sha256':sha(ROOT/'PROTOCOL_v2.md'),
                'future_truth_read':False,'requested_timestamps_provided':True,'evaluation_queries_jointly_inferred':False,
                'support_grid':6,'iterations':6,'dtype':'float32','gpu':args.gpu}
        record.update(resource_change='empty_cuda_cache_before_each_full_query_only',
                      feature_chunk_size=200,first_attempts_retained=True)
        print(json.dumps({'event':'start','case':case['id'],'frames':case['frames'],'queries':case['queries']}),flush=True)
        try:
            # Only input.npz; truth and original annotation are not opened here.
            data=np.load(verify_file(case['files']['input']),allow_pickle=False)
            rgb=data['video']
            query=data['queries_tyx']
            requested=data['requested_times']
            assert (query[:,0]==0).all()
            t,h,w=rgb.shape[:3]
            n=len(query)
            xy=np.empty((t,n,2),np.float32)
            native=np.empty((t,n),bool)
            v=np.empty((t,n),np.float32)
            c=np.empty((t,n),np.float32)
            reverse=np.full((len(requested),n,2),np.nan,np.float32)
            reverse_visible=np.zeros((len(requested),n),bool)
            reciprocal=np.zeros((t,n),bool)
            frames=torch.from_numpy(rgb).permute(0,3,1,2).float()[None].to(device)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for point,q in enumerate(query):
                    captured.clear()
                    torch.cuda.empty_cache()
                    output_xy,output_vis=model(video=frames,queries=torch.from_numpy(q[[0,2,1]].copy())[None,None].to(device),backward_tracking=False)
                    assert len(captured)==1
                    xy[:,point]=output_xy[0,:,0].cpu().numpy()
                    native[:,point]=output_vis[0,:,0].cpu().numpy()
                    v[:,point],c[:,point]=captured[0][0][0,:,0],captured[0][1][0,:,0]
                    for slot,target in enumerate(requested):
                        candidate=xy[target,point]
                        if not native[target,point] or not inside(candidate,(h,w)):
                            continue
                        pair=frames[:,[int(target),0]]
                        reverse_query=np.array([0,candidate[0],candidate[1]],np.float32)
                        back,visible=model(video=pair,queries=torch.from_numpy(reverse_query)[None,None].to(device),backward_tracking=False)
                        reverse[slot,point]=back[0,1,0].cpu().numpy()
                        reverse_visible[slot,point]=bool(visible[0,1,0].cpu())
                        reciprocal[target,point]=reciprocal_accept(candidate,native[target,point],reverse[slot,point],
                              reverse_visible[slot,point],q[[2,1]],(h,w))
                    print(json.dumps({'event':'query_done','case':case['id'],'query':point+1,'queries':n}),flush=True)
            native &= inside(xy,(h,w))
            confidence=native & (v*c>.995)
            assert np.isfinite(xy).all() and np.isfinite(v).all() and np.isfinite(c).all()
            target=output/(case['id']+'.npz')
            np.savez_compressed(target,xy=xy,native=native,visibility=v,confidence=c,confidence_accepted=confidence,
                  reciprocal=reciprocal,reverse_xy=reverse,reverse_visible=reverse_visible,requested_times=requested)
            record.update(status='ok',prediction_sha256=sha(target),frames=t,queries=n,
                          peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
            del frames
        except Exception as error:
            record.update(status='inference_failed',error=repr(error),traceback=traceback.format_exc(),
                          planned_case_retained=True,elapsed_seconds=time.monotonic()-started)
            if 'frames' in locals():
                del frames
            torch.cuda.empty_cache()
        save(path,record)
        print(json.dumps({'event':'done','case':case['id'],'status':record['status'],'seconds':record['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
