"""End-to-end artifact checks; independent denominator/geometry/source verification."""
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from evaluate import lift_scene
from fetch_tapip import sha

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'tracking_validation_20261115'
PROJECT = ROOT.parent.parent


def main():
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    frozen = json.loads((ROOT/'pre_inference_freeze.json').read_text())
    for name in ['gate.py','gate_config.json','gate_operating_points.json','PROTOCOL.md','data_manifest.json']:
        assert sha(ROOT/name) == frozen['files'][name]
    prior = json.loads((OLD/'source_snapshots/dataset_v2/manifest.json').read_text())
    assert all(sha(OLD/name)==digest for name,digest in prior['sha256'].items())
    assert all(sha(PROJECT/name)==digest for name,digest in prior['read_only_baseline_sha256'].items())
    co = json.loads((OLD/'assets.json').read_text())
    tap = json.loads((ROOT/'tapip_assets.json').read_text())
    tapsource = json.loads((ROOT/'tapip_source.json').read_text())
    for assets,sources in [(co,co['source_files']),(tap,tapsource['files'])]:
        assert sha(Path(assets['checkpoint'])) == assets['checkpoint_sha256']
        assert all(sha(Path(assets['source_directory'])/r['file']) == r['sha256'] for r in sources)
    old_hashes = {r['video_pixel_sha256'] for r in json.loads((OLD/'data_manifest_v2.json').read_text())['cases']}
    seen = set();prediction_count=0;maximum_gt_lift=0.;maximum_tap_projection=0.;maximum_query_error=0.
    joint_counts = {m:{'accepted':0,'correct':0,'fp':0,'ambiguous':0,'positive':0} for m in ['cotracker','tapip3d']}
    import hashlib
    for case in manifest['cases']:
        for key,digest in [('input','input_sha256'),('truth','truth_sha256'),('rgbd_input','rgbd_sha256')]:
            assert sha(ROOT/case[key]) == case[digest]
        inputs=np.load(ROOT/case['input']);truth=np.load(ROOT/case['truth']);sensor=np.load(ROOT/case['rgbd_input'])
        digest=hashlib.sha256(inputs['video'].tobytes()).hexdigest()
        assert digest==case['video_pixel_sha256'] and digest not in seen|old_hashes
        seen.add(digest)
        lifted=lift_scene(truth['xy'],sensor)
        positive=truth['visible']&truth['identifiable']
        maximum_gt_lift=max(maximum_gt_lift,float(np.linalg.norm(lifted-truth['xyz_world'],axis=-1)[positive].max()))
        for method in ['cotracker','tapip3d']:
            path=ROOT/'predictions'/method/'primary'/case['split']/case['case']
            receipt=json.loads(path.with_suffix('.json').read_text())
            assert receipt['status']=='ok' and not receipt['future_truth_read']
            assert sha(path.with_suffix('.npz'))==receipt['prediction_sha256']
            pred=np.load(path.with_suffix('.npz'));prediction_count+=1
            assert pred['xy'].shape==(33,10,2) and np.isfinite(pred['xy']).all()
            if method=='tapip3d':
                assert receipt['compatibility']['final_state_tensors_equal_checkpoint']==362
                assert not receipt['compatibility']['author_source_modified']
                xyz=pred['xyz_world'];K=sensor['intrinsics'];ex=sensor['extrinsics']
                camera=np.einsum('tij,tnj->tni',ex,np.concatenate([xyz,np.ones_like(xyz[...,:1])],axis=-1))[...,:3]
                projected=camera@K.T;projected=projected[...,:2]/projected[...,2:]
                maximum_tap_projection=max(maximum_tap_projection,float(abs(projected-pred['xy']).max()))
                maximum_query_error=max(maximum_query_error,float(abs(pred['query3d'][:,1:]-truth['xyz_world'][0]).max()))
            if case['split']=='heldout':
                mask=np.load(ROOT/'comparison'/f'{method}_{case["case"]}_masks.npz')['primary'].copy()
                mask[0]=False
                visible,known=[truth[k][:,[0,5]] for k in ['visible','identifiable']]
                positive=visible&known;positive[0]=False
                e=np.linalg.norm(pred['xy'][:,[0,5]]-truth['xy'][:,[0,5]],axis=-1)
                c=joint_counts[method]
                c['accepted']+=int(mask.sum());c['correct']+=int((mask&positive&(e<=3)).sum())
                c['fp']+=int((mask&known&~visible).sum());c['ambiguous']+=int((mask&~known).sum())
                c['positive']+=int(positive.sum())
            if method=='cotracker':
                before=ROOT/'gate_validation'/f'{method}_{case["case"]}_gate.json'
                after=ROOT/'comparison'/f'{method}_{case["case"]}_gate.json'
                assert before.read_bytes()==after.read_bytes()
    assert prediction_count==80 and len(seen)==40
    assert maximum_gt_lift<1e-5 and maximum_tap_projection<1e-4 and maximum_query_error<1e-5
    summary=json.loads((ROOT/'comparison/summary.json').read_text())
    for method,c in joint_counts.items():
        r=next(r for r in summary if r['split']=='heldout' and r['condition']=='ALL' and r['model']==method and r['mode']=='primary')
        assert (c['accepted'],c['correct'],c['fp'],c['ambiguous'],c['positive']) == (
            r['accepted'],r['correct_accepted'],r['false_visible'],r['unsupported'],r['positive'])
    record={'status':'passed','native_model_attempts':prediction_count,'sequences_per_model':40,'frames_per_model':1320,
            'heldout_sequences':30,'development_sequences':10,'heldout_evaluated_center_samples':1920,
            'unique_video_pixel_hashes':40,'overlap_with_prior':0,'unchanged_frozen_gate_protocol_and_inputs':True,
            'unchanged_prior_source_files':len(prior['sha256']),'unchanged_prior_baseline_files':len(prior['read_only_baseline_sha256']),
            'verified_author_files':{'cotracker':len(co['source_files']),'tapip3d':len(tapsource['files'])},
            'maximum_gt_sensor_lift_error_m':maximum_gt_lift,'maximum_tapip_projection_error_px':maximum_tap_projection,
            'maximum_initial_query_3d_error_m':maximum_query_error,'independent_center_counts':joint_counts,
            'full_tapip3d_inference_executed':True,'old_lifting_diagnostic_used_as_tapip':False,
            'python_executable':sys.executable,'python':sys.version,
            'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','opencv-python','matplotlib','pytest','hydra-core','python-box','pointops2']}}
    target=ROOT/'verification.json'
    assert not target.exists()
    target.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
