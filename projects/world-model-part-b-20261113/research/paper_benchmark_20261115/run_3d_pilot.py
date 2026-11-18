"""Complete native learned pipeline on prespecified calibrated paper examples."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from fetch import sha

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4);torch.manual_seed(20261115);torch.cuda.set_device(1);device='cuda:1'
    co=json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
    tap=json.loads((ROOT.parent/'observation_gate_20261115/tapip_assets.json').read_text())
    da=json.loads((ROOT.parent/'identity_recovery_20261115/depth_assets.json').read_text())
    for assets in [co,tap,da]:assert sha(assets['checkpoint'])==assets['checkpoint_sha256']
    sys.path.insert(0,co['source_directory']);from cotracker.predictor import CoTrackerPredictor
    cotracker=CoTrackerPredictor(checkpoint=co['checkpoint'],offline=True,v2=False,window_len=60).to(device).eval()
    sys.path[:0]=[tap['source_directory'],str(Path(tap['source_directory'])/'third_party/pointops2')]
    from utils.inference_utils import load_model,inference
    original=torch.hub.load

    def pinned(repo,name,*a,**kw):
        assert repo=='facebookresearch/co-tracker' and name=='cotracker3_offline' and not a and not kw
        return CoTrackerPredictor(checkpoint=co['checkpoint'],offline=True,v2=False,window_len=60)

    torch.hub.load=pinned
    try:tapip=load_model(tap['checkpoint']).to(device).eval()
    finally:torch.hub.load=original
    weights=torch.load(tap['checkpoint'],map_location='cpu',weights_only=False)['weight']
    assert all(torch.equal(tapip.state_dict()[k].cpu(),v) for k,v in weights.items());del weights
    sys.path.insert(0,str(Path(da['source_directory'])/'metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.dinov2_layers import attention
    attention.XFORMERS_AVAILABLE=False
    depthmodel=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
    depthmodel.load_state_dict(torch.load(da['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    depthmodel=depthmodel.to(device).eval()
    output=ROOT/'pilot3d_predictions';output.mkdir(exist_ok=False)
    for case in json.loads((ROOT/'pilot3d_manifest.json').read_text())['cases']:
        assert sha(ROOT/case['input'])==case['input_sha256']
        inp=np.load(ROOT/case['input']);rgb=inp['video'];queries=inp['queries_tyx'];K=inp['intrinsics'];T=len(rgb);N=len(queries)
        record={'case':case['case'],'input_sha256':case['input_sha256'],'future_truth_read':False,
                'runner_sha256':sha(Path(__file__)),'querywise_inference':True,'iterations':6,'support_grid':6,
                'checkpoint_sha256':{name:a['checkpoint_sha256'] for name,a in [('cotracker',co),('tapip3d',tap),('depth',da)]},
                'camera':'supplied intrinsics; static identity extrinsics','depth':'estimated; no GT depths supplied'}
        started=time.monotonic();print(json.dumps({'event':'start','case':case['case']}),flush=True)
        try:
            with torch.inference_mode():
                depth=np.stack([depthmodel.infer_image(cv2.cvtColor(im,cv2.COLOR_RGB2BGR),input_size=518) for im in rgb]).astype(np.float32)
                video=torch.from_numpy(rgb).permute(0,3,1,2).float().to(device)
                co_xy=np.empty((T,N,2),np.float32);co_vis=np.empty((T,N),bool)
                xyz=np.empty((T,N,3),np.float32);tap_vis=np.empty((T,N),bool)
                intrinsics=torch.from_numpy(np.repeat(K[None],T,axis=0)).to(device)
                extrinsics=torch.eye(4,device=device)[None].repeat(T,1,1)
                d_tensor=torch.from_numpy(depth).to(device)
                for i,q in enumerate(queries):
                    nativeq=torch.from_numpy(q[[0,2,1]].copy())[None,None].to(device)
                    p,v=cotracker(video=video[None],queries=nativeq,backward_tracking=False)
                    co_xy[:,i]=p[0,:,0].cpu().numpy();co_vis[:,i]=v[0,:,0].cpu().numpy()
                    t,y,x=q;ix,iy=int(round(float(x))),int(round(float(y)))
                    assert 0<=ix<rgb.shape[2] and 0<=iy<rgb.shape[1]
                    initial=np.linalg.inv(K)@np.array([x,y,1],np.float32)*depth[int(t),iy,ix]
                    query3d=torch.from_numpy(np.r_[t,initial].astype(np.float32))[None].to(device)
                    p3,v3=inference(model=tapip,video=video/255.,depths=d_tensor,intrinsics=intrinsics,
                        extrinsics=extrinsics,query_point=query3d,num_iters=6,grid_size=6,bidrectional=False,vis_threshold=.9)
                    xyz[:,i]=p3.cpu().numpy().reshape(T,3);tap_vis[:,i]=v3.cpu().numpy().reshape(T)
                pixels=xyz@K.T;tap_xy=pixels[...,:2]/pixels[...,2:]
                rounded=np.rint(co_xy).astype(int);rounded[...,0]=np.clip(rounded[...,0],0,rgb.shape[2]-1);rounded[...,1]=np.clip(rounded[...,1],0,rgb.shape[1]-1)
                sampled=depth[np.arange(T)[:,None],rounded[...,1],rounded[...,0]]
                rays=np.concatenate([co_xy,np.ones_like(co_xy[...,:1])],axis=-1)@np.linalg.inv(K).T
                co_xyz=rays*sampled[...,None]
            assert np.isfinite(xyz).all() and np.isfinite(depth).all() and (depth>0).all()
            path=output/f'{case["case"]}.npz';np.savez_compressed(path,depth=depth,co_xy=co_xy,co_xyz=co_xyz,
                co_visible=co_vis,tap_xy=tap_xy,tap_xyz=xyz,tap_visible=tap_vis)
            record.update(status='ok',prediction_sha256=sha(path),elapsed_seconds=time.monotonic()-started)
        except Exception as error:
            record.update(status='api_failure',exception=repr(error));(output/f'{case["case"]}.json').write_text(json.dumps(record,indent=2)+'\n');raise
        (output/f'{case["case"]}.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'event':'complete',**record}),flush=True)


if __name__=='__main__':main()
