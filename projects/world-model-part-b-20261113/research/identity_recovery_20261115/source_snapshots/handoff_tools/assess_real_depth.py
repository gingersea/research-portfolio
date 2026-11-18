"""AI-reference diagnostics and estimated-depth checks, without invented 3D truth."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent
V2=ROOT.parent/'identity_recovery_multiscale_20261115'
spec=importlib.util.spec_from_file_location('frozen_v2_recovery',V2/'recovery.py')
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)


def ref_score(rows,xy,claim):
    c={'samples':0,'positive':0,'negative':0,'unknown':0,'accepted':0,'correct_nominal8':0,
       'certain_within8':0,'possible_within8':0,'false_visible':0,'unsupported':0,'errors_px':[]}
    for t,obj,visibility,identity,x,y,uncertainty,note in rows:
        if t==0:continue
        c['samples']+=1
        if claim[t,obj]:c['accepted']+=1
        if identity=='unknown' or visibility=='uncertain':
            c['unknown']+=1;c['unsupported']+=int(claim[t,obj]);continue
        if visibility=='absent':
            c['negative']+=1;c['false_visible']+=int(claim[t,obj]);continue
        assert visibility=='visible' and identity=='identifiable' and x is not None
        c['positive']+=1
        error=float(np.linalg.norm(xy[t,obj]-[x,y]));c['errors_px'].append(error)
        if claim[t,obj]:
            c['correct_nominal8']+=int(error<=8)
            c['certain_within8']+=int(error+uncertainty<=8)
            c['possible_within8']+=int(max(0,error-uncertainty)<=8)
    return c


def main():
    output=ROOT/'real_depth_results';output.mkdir(exist_ok=False)
    reference_path=ROOT/'real_i2v/ai_visual_reference.json'
    reference=json.loads(reference_path.read_text());reference_hash=sha(reference_path)
    control_manifest=json.loads((ROOT/'data_manifest.json').read_text())['cases']
    depth_rows=[]
    for case in control_manifest:
        path=ROOT/'estimated_depth_native_attention'/f'{case["case"]}.npz'
        if not path.exists():continue
        depth=np.load(path)['depth'];gt=np.load(ROOT/case['rgbd_input'])['depth']
        row={'case':case['case'],'condition':case['condition'],'frames':len(depth),'raw_metric_scale':True}
        regions={'ALL':np.ones_like(gt,bool),'background':np.isclose(gt,8),
                 'objects':np.isclose(gt,3.8)|np.isclose(gt,4.4),'occluder':np.isclose(gt,2.5)}
        for name,mask in regions.items():
            if not mask.any():row[name]=None;continue
            p,g=depth[mask],gt[mask]
            row[name]={'samples':int(mask.sum()),'AbsRel':float(np.mean(abs(p-g)/g)),
                       'RMSE_m':float(np.sqrt(np.mean((p-g)**2))),
                       'median_scale_ratio':float(np.median(p/g)),
                       'delta1':float(np.mean(np.maximum(p/g,g/p)<1.25))}
        depth_rows.append(row)
    real_rows=[];reference_scores=[];reference_depth=[]
    for case in json.loads((ROOT/'real_manifest.json').read_text()):
        rgbfile=np.load(case['source']);rgb=rgbfile[rgbfile.files[0]]
        depth=np.load(ROOT/'estimated_depth_native_attention'/f'{case["clip"]}.npz')['depth']
        patch_centers=[(96,64),(384,64),(672,64),(384,400)]
        samples=np.stack([np.median(depth[:,y-2:y+3,x-2:x+3],axis=(1,2)) for x,y in patch_centers],axis=1)
        colors=np.stack([rgb[:,y-2:y+3,x-2:x+3].mean(axis=(1,2)) for x,y in patch_centers],axis=1)
        ratios=samples/samples[0]
        real_rows.append({'clip':case['clip'],'frames':len(depth),'static_background_patch_centers':patch_centers,
            'background_depth_median_m':float(np.median(samples)),
            'background_depth_CV_mean':float(np.mean(samples.std(axis=0)/samples.mean(axis=0))),
            'background_relative_change_median':float(np.median(abs(ratios-1))),
            'background_relative_change_max':float(abs(ratios-1).max()),
            'background_rgb_mae_from_initial_0_255':float(abs(colors-colors[0]).mean()),
            'depth_gt_available':False,'camera_calibration_gt_available':False})
        np.savez_compressed(output/f'{case["clip"]}_static_depth.npz',depth_samples=samples,ratios=ratios,patch_centers=patch_centers)
        for row in reference['clips'][case['clip']]:
            t,obj,vis,identity,x,y,uncertainty,note=row
            if vis=='visible' and identity=='identifiable' and x is not None:
                value=float(np.median(depth[t,int(y)-2:int(y)+3,int(x)-2:int(x)+3]))
                reference_depth.append({'clip':case['clip'],'frame':t,'object':obj,'estimated_depth_m':value,
                                        'independent_true_depth_available':False,'reference_uncertainty_px':uncertainty})
        for model in ['cotracker','tapip3d']:
            path=ROOT/'real_predictions'/model/f'{case["clip"]}.npz'
            receipt=json.loads(path.with_suffix('.json').read_text())
            assert receipt['reference_sha256_before_inference']==reference_hash and not receipt['reference_labels_read']
            assert sha(path)==receipt['prediction_sha256']
            pred=np.load(path);xy=pred['xy'][:,[0,5]];native=pred['official_visible'][:,[0,5]]
            recovered=v2.recover(rgb,xy,case['initial_centers'],case['radii'])
            np.savez_compressed(output/f'{case["clip"]}_{model}_recovery.npz',xy=recovered['xy'],accepted=recovered['accepted'],
                                 baseline_accepted=recovered['baseline_accepted'])
            (output/f'{case["clip"]}_{model}_reasons.json').write_text(json.dumps(recovered['details'],indent=2)+'\n')
            for method,(coords,claim) in {'raw':(xy,native),'previous_gate':(xy,native&recovered['baseline_accepted']),
                                        'recovery_v2':(recovered['xy'],recovered['accepted'])}.items():
                reference_scores.append({'clip':case['clip'],'model':model,'method':method,
                    'reference_type':'AI_visual_reference_not_human_GT','dense_claims_post_query':int(claim[1:].sum()),
                    **ref_score(reference['clips'][case['clip']],coords,claim)})
    for name,value in [('depth_controls.json',depth_rows),('depth_real_static.json',real_rows),
                       ('reference_scores.json',reference_scores),('reference_depth.json',reference_depth)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    receipt={'reference_sha256':reference_hash,'reference_type':reference['annotator_type'],'human_labels_available':False,
             'v2_recovery_sha256':sha(V2/'recovery.py'),'control_depth_sequences':len(depth_rows),'real_clips':len(real_rows),
             'true_3d_accuracy_validation':False,'physical_scoring_enabled':False,
             'real_metrics':'8px nominal center comparison to approximate AI reference, uncertainty bounds also recorded; not validated human accuracy.'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'receipt':receipt,'real_static_depth':real_rows,'reference_scores':reference_scores},indent=2))


if __name__=='__main__':main()
