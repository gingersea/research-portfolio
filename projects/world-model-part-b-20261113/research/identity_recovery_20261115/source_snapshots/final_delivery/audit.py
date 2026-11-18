"""Audit immutable inputs, scientific denominators and actual native/depth runs."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent
V2=ROOT.parent/'identity_recovery_multiscale_20261115'


def main():
    counts={}
    for directory,stage in [(ROOT,'before_control_inference'),(V2,'before_inference')]:
        frozen=json.loads((directory/'source_snapshots'/stage/'manifest.json').read_text())
        for key in ['PROTOCOL.md','recovery.py','data_manifest.json']:
            assert sha(directory/key)==frozen['files'][key]
        for path,digest in frozen['read_only_dependencies'].items():assert sha(ROOT.parent/path)==digest
        m=json.loads((directory/'data_manifest.json').read_text())
        assert len(m['cases'])==48 and len({r['video_pixel_sha256'] for r in m['cases']})==48
        for case in m['cases']:
            for key,digest in [('input','input_sha256'),('truth','truth_sha256'),('rgbd_input','rgbd_sha256')]:
                assert sha(directory/case[key])==case[digest]
            receipt=json.loads((directory/'predictions/cotracker/primary'/case['split']/f'{case["case"]}.json').read_text())
            assert receipt['status']=='ok' and not receipt['future_truth_read']
            assert sha(directory/'predictions/cotracker/primary'/case['split']/f'{case["case"]}.npz')==receipt['prediction_sha256']
        counts[directory.name]=48
    depth_assets=json.loads((ROOT/'depth_assets.json').read_text())
    assert sha(depth_assets['checkpoint'])==depth_assets['checkpoint_sha256']
    assert all(sha(Path(depth_assets['source_directory'])/r['file'])==r['sha256'] for r in depth_assets['files'])
    depth_frames=0
    for path in (ROOT/'estimated_depth_native_attention').glob('*.json'):
        receipt=json.loads(path.read_text());assert receipt['status']=='ok' and not receipt['future_truth_read']
        assert sha(path.with_suffix('.npz'))==receipt['depth_sha256'];depth_frames+=receipt['frames']
    assert depth_frames==784
    ref=ROOT/'real_i2v/ai_visual_reference.json';maximum_projection=0.
    for case in json.loads((ROOT/'real_manifest.json').read_text()):
        assert sha(case['source'])==case['source_sha256']
        for model in ['cotracker','tapip3d']:
            path=ROOT/'real_predictions'/model/f'{case["clip"]}.json';receipt=json.loads(path.read_text())
            assert receipt['status']=='ok' and receipt['reference_sha256_before_inference']==sha(ref)
            assert not receipt['reference_labels_read'] and sha(path.with_suffix('.npz'))==receipt['prediction_sha256']
            pred=np.load(path.with_suffix('.npz'));assert pred['xy'].shape==(97,10,2)
            if model=='tapip3d':
                assert receipt['final_state_tensors_equal_checkpoint']==362 and not receipt['camera_estimated']
                pixels=pred['xyz_assumed_camera']@pred['assumed_intrinsics'].T
                xy=pixels[...,:2]/pixels[...,2:]
                maximum_projection=max(maximum_projection,float(abs(xy-pred['xy']).max()))
    assert maximum_projection<1e-4
    summary=json.loads((V2/'control_results/summary.json').read_text())
    r=next(r for r in summary if r['split']=='heldout' and r['condition']=='ALL' and r['method']=='recovery')
    assert (r['correct'],r['accepted'],r['positive'],r['events'],r['recovered_events'])==(1825,1825,1905,15,15)
    assert r['false_visible']==r['wrong_other']==r['unsupported']==0
    metrics=ROOT.parent/'tracking_validation_20261115/metrics.py'
    spec=importlib.util.spec_from_file_location('old_metric',metrics);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    depth_visibility=[]
    for case in json.loads((ROOT/'data_manifest.json').read_text())['cases']:
        path=ROOT/'estimated_tracking'/f'{case["case"]}.npz'
        if not path.exists():continue
        receipt=json.loads(path.with_suffix('.json').read_text());assert receipt['status']=='ok'
        assert sha(path)==receipt['prediction_sha256'] and receipt['state_tensors_equal_checkpoint']==362
        truth=np.load(ROOT/case['truth']);pred=np.load(path)
        result=old.score_case(pred['xy'][:,[0,5]],pred['visible'][:,[0,5]],truth['xy'][:,[0,5]],truth['visible'][:,[0,5]],truth['identifiable'][:,[0,5]])
        depth_visibility.append({'case':case['case'],**result})
    assert len(depth_visibility)==12
    (ROOT/'estimated_tracking/visibility_metrics.json').write_text(json.dumps(depth_visibility,indent=2)+'\n')
    record={'status':'passed','control_native_inferences':counts,'real_native_inferences':8,
            'controlled_estimated_depth_native_tapip_inferences':12,'estimated_depth_sequences':16,'estimated_depth_frames':depth_frames,
            'AI_reference_sha256':sha(ref),'AI_reference_rows':72,'human_annotations_received':False,
            'frozen_parameters_preserved':True,'old_dependencies_unchanged':True,'new_control_video_pixels_unique':96,
            'maximum_assumed_camera_projection_error_px':maximum_projection,
            'projection_is_engineering_check_not_3d_accuracy':True,'physical_scoring_enabled':False,
            'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','opencv-python','matplotlib','pytest','hydra-core','python-box']}}
    path=ROOT/'verification.json';assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
