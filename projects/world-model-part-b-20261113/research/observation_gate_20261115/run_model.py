"""Unmodified official CoTracker3/TAPIP3D inference. Future truth is never loaded."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fetch_tapip import sha

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cotracker', 'tapip3d'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--split', choices=['development', 'heldout', 'all'], default='all')
    parser.add_argument('--run-name', default='primary')
    parser.add_argument('--only')
    args = parser.parse_args()
    assert args.run_name.replace('_', '').isalnum()
    torch.set_num_threads(4)
    torch.manual_seed(20261115)
    torch.cuda.set_device(args.gpu)
    device = f'cuda:{args.gpu}'
    captured = []
    compatibility = {}
    if args.model == 'cotracker':
        assets = json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
        source = Path(assets['source_directory'])
        assert all(sha(source/r['file']) == r['sha256'] for r in assets['source_files'])
        sys.path.insert(0, str(source))
        from cotracker.predictor import CoTrackerPredictor
        model = CoTrackerPredictor(checkpoint=assets['checkpoint'], offline=True, v2=False, window_len=60).to(device).eval()
        native = model.model.forward

        def observer(*a, **kw):
            out = native(*a, **kw)
            captured.append([r.detach().cpu().numpy().copy() for r in out[1:3]])
            return out

        model.model.forward = observer
    else:
        assets = json.loads((ROOT/'tapip_assets.json').read_text())
        source = Path(assets['source_directory'])
        source_receipt = json.loads((ROOT/'tapip_source.json').read_text())
        assert all(sha(source/r['file']) == r['sha256'] for r in source_receipt['files'])
        sys.path[:0] = [str(source), str(source/'third_party/pointops2')]
        from utils.inference_utils import load_model, inference
        # Author constructor fetches a mutable CoTracker hub dependency before
        # strictly loading the complete TAPIP checkpoint. Resolve that one request
        # to the already verified, pinned native CoTracker dependency instead.
        co = json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
        co_source = Path(co['source_directory'])
        assert sha(Path(co['checkpoint'])) == co['checkpoint_sha256']
        assert all(sha(co_source/r['file']) == r['sha256'] for r in co['source_files'])
        sys.path.insert(0,str(co_source))
        from cotracker.predictor import CoTrackerPredictor
        native_hub_load = torch.hub.load

        def pinned_dependency(repo, name, *a, **kw):
            assert repo == 'facebookresearch/co-tracker' and name == 'cotracker3_offline'
            assert not a and not kw
            compatibility['constructor_hub_dependency_redirected'] = True
            return CoTrackerPredictor(checkpoint=co['checkpoint'],offline=True,v2=False,window_len=60)

        torch.hub.load = pinned_dependency
        try:
            model = load_model(assets['checkpoint']).to(device).eval()
        finally:
            torch.hub.load = native_hub_load
        checkpoint_state = torch.load(assets['checkpoint'],map_location='cpu',weights_only=False)['weight']
        actual_state = model.state_dict()
        assert actual_state.keys() == checkpoint_state.keys()
        assert all(torch.equal(actual_state[k].cpu(),v) for k,v in checkpoint_state.items())
        compatibility.update(final_state_tensors_equal_checkpoint=len(checkpoint_state),
                             cotracker_dependency_revision=co['source_revision'],
                             author_source_modified=False)
        print(json.dumps({'event':'model_loaded','compatibility':compatibility}),flush=True)
        del actual_state, checkpoint_state

        def observer(module, a, out):
            captured.append(out[0].visibs.detach().cpu().numpy().copy())

        model.register_forward_hook(observer)
    assert sha(Path(assets['checkpoint'])) == assets['checkpoint_sha256']
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(ROOT/'PROTOCOL.md')
    for case in manifest['cases']:
        if args.split != 'all' and args.split != case['split']:
            continue
        if args.only and args.only != case['case']:
            continue
        folder = ROOT/'predictions'/args.model/args.run_name/case['split']
        folder.mkdir(parents=True, exist_ok=True)
        receipt = folder/f'{case["case"]}.json'
        if receipt.exists():
            raise FileExistsError('No replacement of existing attempts: '+str(receipt))
        assert sha(ROOT/case['input']) == case['input_sha256']
        inputs = np.load(ROOT/case['input'], allow_pickle=False)
        rgb, queries = inputs['video'], inputs['queries']
        captured.clear()
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps({'event': 'start', 'case': case['case'], 'model': args.model}), flush=True)
        record = {'case': case['case'], 'model': args.model, 'input_sha256': case['input_sha256'],
                  'future_truth_read': False, 'runner_sha256': sha(Path(__file__)),
                  'protocol_sha256': sha(ROOT/'PROTOCOL.md'), 'checkpoint_sha256': assets['checkpoint_sha256'],
                  'source_revision': assets['source_revision'], 'support_grid': 6, 'iterations': 6,
                  'dtype': 'float32', 'seed': 20261115, 'frames': len(rgb), 'queries': len(queries)}
        record['compatibility'] = compatibility
        try:
            video = torch.from_numpy(rgb).permute(0,3,1,2).float().to(device)
            with torch.inference_mode():
                if args.model == 'cotracker':
                    coords, declared = model(video=video[None], queries=torch.from_numpy(queries)[None].to(device), backward_tracking=False)
                    xy, visible = coords[0].cpu().numpy(), declared[0].cpu().numpy()
                    assert len(captured) == 1
                    raw_v, raw_c = [r[0,:,:len(queries)] for r in captured[0]]
                    output = dict(xy=xy, official_visible=visible, visibility=raw_v, confidence=raw_c)
                    record.update(model_resolution=list(model.interp_shape), native_query_frame_forced=True,
                                  read_only_observer='bound_forward_return')
                else:
                    assert sha(ROOT/case['rgbd_input']) == case['rgbd_sha256']
                    sensor = np.load(ROOT/case['rgbd_input'], allow_pickle=False)
                    depth = sensor['depth']
                    K = np.repeat(sensor['intrinsics'][None], len(rgb), axis=0)
                    extrinsics = sensor['extrinsics']
                    xy0 = queries[:,1:]
                    ji = np.rint(xy0).astype(int)
                    d0 = depth[0,ji[:,1],ji[:,0]]
                    rays = np.concatenate([xy0,np.ones_like(xy0[:,:1])],axis=-1)@np.linalg.inv(K[0]).T
                    local0 = rays*d0[:,None]
                    world0 = (np.concatenate([local0,np.ones_like(local0[:,:1])],axis=-1)@np.linalg.inv(extrinsics[0]).T)[:,:3]
                    query3d = np.concatenate([queries[:,:1],world0],axis=-1).astype(np.float32)
                    xyz, declared = inference(model=model, video=video/255.,
                        depths=torch.from_numpy(depth).to(device), intrinsics=torch.from_numpy(K).to(device),
                        extrinsics=torch.from_numpy(extrinsics).to(device),
                        query_point=torch.from_numpy(query3d).to(device), num_iters=6,grid_size=6,vis_threshold=.9)
                    xyz, visible = xyz.cpu().numpy(), declared.cpu().numpy()
                    assert len(captured) == 1
                    homogeneous = np.concatenate([xyz,np.ones_like(xyz[...,:1])],axis=-1)
                    local = np.einsum('tij,tnj->tni',extrinsics,homogeneous)[...,:3]
                    pixel = np.einsum('tij,tnj->tni',K,local)
                    xy = pixel[...,:2]/pixel[...,2:]
                    output = dict(xy=xy, xyz_world=xyz, official_visible=visible,
                                  visibility_logits=captured[0][0,:,:len(queries)], query3d=query3d)
                    record.update(model_resolution=list(rgb.shape[1:3]), eval_mode=model.eval_mode,
                                  rgbd_sha256=case['rgbd_sha256'], depth_source='authored_sensor_depth',
                                  camera_source='authored_extrinsics', read_only_observer='forward_hook')
            assert xy.shape == (33,10,2) and visible.shape == (33,10)
            assert np.isfinite(xy).all()
            prediction = folder/f'{case["case"]}.npz'
            np.savez_compressed(prediction, **output)
            record.update(status='ok', prediction_sha256=sha(prediction),
                          maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        except Exception as error:
            record.update(status='api_failure', exception=repr(error), elapsed_seconds=time.monotonic()-started)
            receipt.write_text(json.dumps(record,indent=2)+'\n')
            raise
        record['elapsed_seconds'] = time.monotonic()-started
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'event':'complete',**record}),flush=True)


if __name__ == '__main__':
    main()
