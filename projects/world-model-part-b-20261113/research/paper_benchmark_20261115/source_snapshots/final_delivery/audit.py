"""Independent label/source/denominator audit with unmodified official functions."""
import ast
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from fetch import sha

ROOT=Path(__file__).resolve().parent


def bodies(path):
    source=path.read_text();parsed=ast.parse(source)
    return {n.name:ast.get_source_segment(source,n) for n in parsed.body if isinstance(n,ast.FunctionDef)
            and n.name in ['compute_tapvid_metrics','sample_queries_first']}


def main():
    assert bodies(ROOT/'official_evaluation_datasets.py')==bodies(ROOT/'official_numpy.py')
    frozen=json.loads((ROOT/'source_snapshots/before_davis_inference/manifest.json').read_text())
    for name in ['PROTOCOL.md','ANALYSIS_PLAN.md','run_cotracker.py','data_manifest.json','official_numpy.py']:
        assert sha(ROOT/name)==frozen['files'][name]
    for path,digest in frozen['read_only_dependencies'].items():assert sha(ROOT.parent/path)==digest
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    assert len(manifest['cases'])==30 and manifest['total_frames']==1999 and manifest['total_queries']==650
    assert len({r['video_pixel_sha256'] for r in manifest['cases']})==30
    calls=0
    for case in manifest['cases']:
        for key,digest in [('input','input_sha256'),('truth','truth_sha256')]:assert sha(ROOT/case[key])==case[digest]
        path=ROOT/'predictions'/f'{case["case"]}.json';receipt=json.loads(path.read_text())
        assert receipt['status']=='ok' and not receipt['future_truth_read'] and not receipt['evaluation_queries_jointly_inferred']
        assert sha(path.with_suffix('.npz'))==receipt['prediction_sha256'];calls+=receipt['queries']
        p=np.load(path.with_suffix('.npz'));inp=np.load(ROOT/case['input']);truth=np.load(ROOT/case['truth'])
        assert p['xy'].shape==(case['frames'],case['queries'],2) and np.isfinite(p['xy']).all()
        for i,q in enumerate(inp['queries_tyx']):
            t=int(q[0]);assert not truth['occluded'][i,t]
            assert np.linalg.norm(truth['target_points'][i,t]-q[[2,1]])<1e-3
    assert calls==650
    summary=json.loads((ROOT/'results/summary.json').read_text())
    assert summary[0]['average_pts_within_thresh']==summary[1]['average_pts_within_thresh']
    assert summary[0]['evaluated']==summary[1]['evaluated']==38231
    for row in summary:assert row['positive']+row['negative']==row['evaluated'] and row['event_count']==363
    pilot=json.loads((ROOT/'pilot3d_manifest.json').read_text())
    for case in pilot['cases']:
        assert case['frames']==64 and case['queries']==8 and case['max_query_projection_error_px']<1
        path=ROOT/'pilot3d_predictions'/f'{case["case"]}.json';receipt=json.loads(path.read_text())
        assert receipt['status']=='ok' and not receipt['future_truth_read'] and receipt['querywise_inference']
        assert sha(path.with_suffix('.npz'))==receipt['prediction_sha256']
    co=json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
    tap=json.loads((ROOT.parent/'observation_gate_20261115/tapip_assets.json').read_text())
    da=json.loads((ROOT.parent/'identity_recovery_20261115/depth_assets.json').read_text())
    tapfiles=json.loads((ROOT.parent/'observation_gate_20261115/tapip_source.json').read_text())['files']
    for a,files in [(co,co['source_files']),(tap,tapfiles),(da,da['files'])]:
        assert sha(a['checkpoint'])==a['checkpoint_sha256']
        assert all(sha(Path(a['source_directory'])/r['file'])==r['sha256'] for r in files)
    record={'status':'passed','human_benchmark_videos':30,'human_benchmark_frames':1999,'independent_query_inferences':650,
            'post_query_samples':38231,'reappearance_events':363,'human_labels':'published flow-assisted, human-checked point trajectories/occlusion',
            'official_function_bodies_unmodified':True,'input_and_truth_separated':True,'parameters_not_tuned_on_scores':True,
            'pilot3d_videos':3,'pilot3d_queries':24,'pilot3d_frames_per_video':64,
            'pilot3d_labels':'calibrated PStudio geometry, not pure manual depth annotations',
            'private_unseen_generalization_claim':False,'object_center_gate_validated':False,'generated_I2V_physics_validated':False,
            'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','mediapy','opencv-python','pytest']}}
    path=ROOT/'verification.json';assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
