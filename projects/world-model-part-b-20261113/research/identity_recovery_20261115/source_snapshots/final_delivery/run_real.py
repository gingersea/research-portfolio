"""Native real-I2V inference; estimated depth/assumed cameras remain diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['cotracker','tapip3d'],required=True)
    parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(args.gpu);torch.manual_seed(20261115)
    device=f'cuda:{args.gpu}'
    co=json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
    assert sha(co['checkpoint'])==co['checkpoint_sha256']
    sys.path.insert(0,co['source_directory'])
    from cotracker.predictor import CoTrackerPredictor
    if args.model=='cotracker':
        model=CoTrackerPredictor(checkpoint=co['checkpoint'],offline=True,v2=False,window_len=60).to(device).eval()
        weights=co;exact_state=None
    else:
        weights=json.loads((ROOT.parent/'observation_gate_20261115/tapip_assets.json').read_text())
        source=Path(weights['source_directory'])
        native_sources=json.loads((ROOT.parent/'observation_gate_20261115/tapip_source.json').read_text())
        assert all(sha(source/r['file'])==r['sha256'] for r in native_sources['files'])
        assert sha(weights['checkpoint'])==weights['checkpoint_sha256']
        sys.path[:0]=[str(source),str(source/'third_party/pointops2')]
        from utils.inference_utils import load_model,inference
        native_hub=torch.hub.load

        def local_dependency(repo,name,*a,**kw):
            assert repo=='facebookresearch/co-tracker' and name=='cotracker3_offline' and not a and not kw
            return CoTrackerPredictor(checkpoint=co['checkpoint'],offline=True,v2=False,window_len=60)

        torch.hub.load=local_dependency
        try:model=load_model(weights['checkpoint']).to(device).eval()
        finally:torch.hub.load=native_hub
        state=torch.load(weights['checkpoint'],map_location='cpu',weights_only=False)['weight']
        assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in state.items())
        exact_state=len(state);del state
    output=ROOT/'real_predictions'/args.model;output.mkdir(parents=True,exist_ok=False)
    reference_path=ROOT/'real_i2v/ai_visual_reference.json'
    reference_hash=sha(reference_path)
    (output/'before_inference_reference.json').write_text(json.dumps({'reference_sha256':reference_hash,
        'reference_used_as_model_input':False,'source_sha256':sha(Path(__file__))},indent=2)+'\n')
    for case in json.loads((ROOT/'real_manifest.json').read_text()):
        assert sha(case['source'])==case['source_sha256']
        inputs=np.load(case['source']);rgb=inputs[inputs.files[0]]
        T,H,W=rgb.shape[:3]
        offset=32*np.array([[0,0],[-.4,0],[.4,0],[0,-.4],[0,.4]],np.float32)
        xy0=np.concatenate([np.array(c,dtype=np.float32)+offset for c in case['initial_centers']])
        query=np.concatenate([np.zeros((10,1),np.float32),xy0],axis=-1)
        start=time.monotonic();print(json.dumps({'event':'start','model':args.model,'clip':case['clip']}),flush=True)
        record={'model':args.model,'clip':case['clip'],'frames':T,'queries':10,'source_sha256':case['source_sha256'],
                'reference_sha256_before_inference':reference_hash,'reference_labels_read':False,
                'checkpoint_sha256':weights['checkpoint_sha256'],'runner_sha256':sha(Path(__file__)),
                'dtype':'float32','support_grid':6,'iterations':6}
        try:
            with torch.inference_mode():
                video=torch.from_numpy(rgb).permute(0,3,1,2).float().to(device)
                if args.model=='cotracker':
                    xy,visible=model(video=video[None],queries=torch.from_numpy(query)[None].to(device),backward_tracking=False)
                    arrays={'xy':xy[0].cpu().numpy(),'official_visible':visible[0].cpu().numpy(),'queries':query}
                else:
                    depth_path=ROOT/'estimated_depth_native_attention'/f'{case["clip"]}.npz'
                    receipt=json.loads(depth_path.with_suffix('.json').read_text())
                    assert sha(depth_path)==receipt['depth_sha256']
                    depth=np.load(depth_path)['depth']
                    K=np.array([[W,0,(W-1)/2],[0,W,(H-1)/2],[0,0,1]],np.float32)
                    ex=np.repeat(np.eye(4,dtype=np.float32)[None],T,axis=0)
                    ji=np.rint(xy0).astype(int);d0=depth[0,ji[:,1],ji[:,0]]
                    initial=np.concatenate([xy0,np.ones((10,1),np.float32)],axis=-1)@np.linalg.inv(K).T*d0[:,None]
                    query3d=np.concatenate([query[:,:1],initial],axis=-1)
                    xyz,visible=inference(model=model,video=video/255.,depths=torch.from_numpy(depth).to(device),
                        intrinsics=torch.from_numpy(np.repeat(K[None],T,axis=0)).to(device),
                        extrinsics=torch.from_numpy(ex).to(device),query_point=torch.from_numpy(query3d).to(device),
                        num_iters=6,grid_size=6,vis_threshold=.9)
                    xyz=xyz.cpu().numpy();pixels=xyz@K.T
                    arrays={'xy':pixels[...,:2]/pixels[...,2:],'official_visible':visible.cpu().numpy(),
                            'xyz_assumed_camera':xyz,'assumed_intrinsics':K,'queries':query}
                    record.update(depth_sha256=receipt['depth_sha256'],camera_estimated=False,
                        camera_assumption='static identity extrinsics; fx=fy=image width; center principal point',
                        true_metric_world_accuracy_validated=False,final_state_tensors_equal_checkpoint=exact_state)
            assert arrays['xy'].shape==(97,10,2) and np.isfinite(arrays['xy']).all()
            path=output/f'{case["clip"]}.npz';np.savez_compressed(path,**arrays)
            record.update(status='ok',prediction_sha256=sha(path),elapsed_seconds=time.monotonic()-start)
        except Exception as error:
            record.update(status='api_failure',exception=repr(error))
            (output/f'{case["clip"]}.json').write_text(json.dumps(record,indent=2)+'\n');raise
        (output/f'{case["clip"]}.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'event':'complete',**record}),flush=True)


if __name__=='__main__':main()
