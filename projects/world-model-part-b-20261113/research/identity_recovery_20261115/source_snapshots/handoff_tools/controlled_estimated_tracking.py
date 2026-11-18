"""Known cameras + estimated depth: complete learned tracking against exact GT."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4);torch.manual_seed(20261115);torch.cuda.set_device(1)
    assets=json.loads((ROOT.parent/'observation_gate_20261115/tapip_assets.json').read_text())
    co=json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
    source=Path(assets['source_directory']);sys.path[:0]=[str(source),str(source/'third_party/pointops2'),co['source_directory']]
    assert sha(assets['checkpoint'])==assets['checkpoint_sha256'] and sha(co['checkpoint'])==co['checkpoint_sha256']
    from cotracker.predictor import CoTrackerPredictor
    from utils.inference_utils import load_model,inference
    original_hub=torch.hub.load

    def pinned(repo,name,*a,**kw):
        assert repo=='facebookresearch/co-tracker' and name=='cotracker3_offline' and not a and not kw
        return CoTrackerPredictor(checkpoint=co['checkpoint'],offline=True,v2=False,window_len=60)

    torch.hub.load=pinned
    try:model=load_model(assets['checkpoint']).cuda().eval()
    finally:torch.hub.load=original_hub
    state=torch.load(assets['checkpoint'],map_location='cpu',weights_only=False)['weight']
    assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in state.items());state_count=len(state);del state
    metrics_path=ROOT.parent/'observation_gate_20261115/evaluate.py'
    sys.path.insert(0,str(metrics_path.parent))
    spec=importlib.util.spec_from_file_location('frozen_lift',metrics_path);previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    output=ROOT/'estimated_tracking';output.mkdir(exist_ok=False);scores=[]
    for case in json.loads((ROOT/'data_manifest.json').read_text())['cases']:
        depth_path=ROOT/'estimated_depth_native_attention'/f'{case["case"]}.npz'
        if not depth_path.exists():continue
        depth_receipt=json.loads(depth_path.with_suffix('.json').read_text());assert sha(depth_path)==depth_receipt['depth_sha256']
        inp=np.load(ROOT/case['input']);sensor=np.load(ROOT/case['rgbd_input']);depth=np.load(depth_path)['depth']
        rgb,queries=inp['video'],inp['queries'];K=sensor['intrinsics'];ex=sensor['extrinsics']
        ji=np.rint(queries[:,1:]).astype(int);d0=depth[0,ji[:,1],ji[:,0]]
        local=np.concatenate([queries[:,1:],np.ones((10,1),np.float32)],axis=-1)@np.linalg.inv(K).T*d0[:,None]
        initial=(np.concatenate([local,np.ones((10,1),np.float32)],axis=-1)@np.linalg.inv(ex[0]).T)[:,:3]
        query3d=np.concatenate([queries[:,:1],initial],axis=-1)
        print(json.dumps({'event':'start','case':case['case']}),flush=True)
        with torch.inference_mode():
            xyz,visible=inference(model=model,video=torch.from_numpy(rgb).permute(0,3,1,2).float().cuda()/255.,
                depths=torch.from_numpy(depth).cuda(),intrinsics=torch.from_numpy(np.repeat(K[None],len(rgb),axis=0)).cuda(),
                extrinsics=torch.from_numpy(ex).cuda(),query_point=torch.from_numpy(query3d).cuda(),num_iters=6,grid_size=6,vis_threshold=.9)
        xyz=xyz.cpu().numpy();camera=np.einsum('tij,tnj->tni',ex,np.concatenate([xyz,np.ones_like(xyz[...,:1])],axis=-1))[...,:3]
        pixel=camera@K.T;xy=pixel[...,:2]/pixel[...,2:]
        assert np.isfinite(xyz).all() and np.isfinite(xy).all()
        target=output/f'{case["case"]}.npz';np.savez_compressed(target,xyz_world=xyz,xy=xy,visible=visible.cpu().numpy())
        (output/f'{case["case"]}.json').write_text(json.dumps({'case':case['case'],'status':'ok',
            'input_sha256':case['input_sha256'],'depth_sha256':depth_receipt['depth_sha256'],
            'source_revision':assets['source_revision'],'checkpoint_sha256':assets['checkpoint_sha256'],
            'runner_sha256':sha(Path(__file__)),'prediction_sha256':sha(target),'future_target_truth_read_during_inference':False,
            'camera_source':'known controlled calibration','depth_source':'estimated DA-V2',
            'state_tensors_equal_checkpoint':state_count},indent=2)+'\n')
        # Exact future target labels are opened only after inference for assessment.
        truth=np.load(ROOT/case['truth']);positive=truth['visible'][:,[0,5]]&truth['identifiable'][:,[0,5]];positive[0]=False
        copred=np.load(ROOT/'predictions/cotracker/primary'/case['split']/f'{case["case"]}.npz')
        estimated_sensor={'depth':depth,'intrinsics':K,'extrinsics':ex}
        lifted=previous.lift_scene(copred['xy'][:,[0,5]],estimated_sensor)
        for method,coords,world in [('cotracker_estimated_depth',copred['xy'][:,[0,5]],lifted),('tapip3d_estimated_depth',xy[:,[0,5]],xyz[:,[0,5]])]:
            error2=np.linalg.norm(coords-truth['xy'][:,[0,5]],axis=-1)[positive]
            error3=np.linalg.norm(world-truth['xyz_world'][:,[0,5]],axis=-1)[positive]
            scores.append({'case':case['case'],'condition':case['condition'],'method':method,'visible':int(positive.sum()),
                           'mean_epe2d_px':float(error2.mean()),'mean_epe3d_m':float(error3.mean()),
                           'pck3_count':int((error2<=3).sum()),'pck3d5cm_count':int((error3<=.05).sum())})
        print(json.dumps({'event':'complete','case':case['case']}),flush=True)
    (output/'metrics.json').write_text(json.dumps(scores,indent=2)+'\n')
    print(json.dumps(scores,indent=2))


if __name__=='__main__':main()
