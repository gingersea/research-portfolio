"""Fixed center endpoints, independent RGB gate, and paired learned RGB-D comparison."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np

from fetch_tapip import sha
from gate import evidence, decide

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_metrics',ROOT.parent/'tracking_validation_20261115/metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def ratio(a,b):
    return float(a/b) if b else None


def counts(xy, xyz, claim, truth):
    gtxy = truth['xy'][:,[0,5]]
    gt3d = truth['xyz_world'][:,[0,5]]
    visible, identifiable = [truth[k][:,[0,5]] for k in ['visible','identifiable']]
    evaluated = np.ones_like(visible);evaluated[0] = False
    positive = evaluated & visible & identifiable
    negative = evaluated & ~visible & identifiable
    ambiguous = evaluated & ~identifiable
    e2 = np.linalg.norm(xy-gtxy,axis=-1)
    e3 = np.linalg.norm(xyz-gt3d,axis=-1)
    correct = positive & np.isfinite(e2) & (e2 <= 3)
    accepted = claim & evaluated
    m = metrics.score_case(xy,claim,gtxy,visible,identifiable)
    values = {'samples':int(evaluated.sum()),'positive':int(positive.sum()),'negative':int(negative.sum()),
              'ambiguous':int(ambiguous.sum()),'accepted':int(accepted.sum()),
              'correct_accepted':int((accepted&correct).sum()),'correct_raw_location':int(correct.sum()),
              'true_visible':int((accepted&positive).sum()),'false_visible':int((accepted&negative).sum()),
              'unsupported':int((accepted&ambiguous).sum()),'wrong_other':m['wrong_other_count'],
              'pck3d_5cm':int((positive & (e3 <= .05)).sum()),
              'pck3d_5cm_accepted':int((accepted&positive&(e3<=.05)).sum()),
              'epe2d_sum':float(e2[positive].sum()),'epe3d_sum':float(e3[positive].sum()),
              'accepted_epe3d_sum':float(e3[positive&accepted].sum())}
    assert all(np.isfinite(v) for v in values.values())
    return values


def summarize(rows):
    result = {key:sum(r[key] for r in rows) for key in rows[0]}
    n = result
    result.update(joint_precision=ratio(n['correct_accepted'],n['accepted']),
                  retained_correct=ratio(n['correct_accepted'],n['positive']),
                  pck3_visible=ratio(n['correct_raw_location'],n['positive']),
                  visibility_precision=ratio(n['true_visible'],n['true_visible']+n['false_visible']),
                  visibility_recall=ratio(n['true_visible'],n['positive']),
                  false_visible_rate=ratio(n['false_visible'],n['negative']),
                  wrong_other_fraction=ratio(n['wrong_other'],n['accepted']),
                  unsupported_rate=ratio(n['unsupported'],n['ambiguous']),
                  pck3d_5cm_fraction=ratio(n['pck3d_5cm'],n['positive']),
                  pck3d_5cm_retained=ratio(n['pck3d_5cm_accepted'],n['positive']),
                  mean_epe2d_px=ratio(n['epe2d_sum'],n['positive']),
                  mean_epe3d_m=ratio(n['epe3d_sum'],n['positive']),
                  accepted_epe3d_m=ratio(n['accepted_epe3d_sum'],n['true_visible']))
    return result


def lift_scene(xy,sensor):
    depth,K,ex = sensor['depth'],sensor['intrinsics'],sensor['extrinsics']
    ji = np.rint(xy).astype(int)
    inside = (ji[...,0]>=0)&(ji[...,0]<depth.shape[2])&(ji[...,1]>=0)&(ji[...,1]<depth.shape[1])
    # Outside pixels count as very large error; no ground-truth target depth supplied.
    ji[...,0] = np.clip(ji[...,0],0,depth.shape[2]-1)
    ji[...,1] = np.clip(ji[...,1],0,depth.shape[1]-1)
    d = depth[np.arange(len(depth))[:,None],ji[...,1],ji[...,0]]
    rays = np.concatenate([xy,np.ones_like(xy[...,:1])],axis=-1)@np.linalg.inv(K).T
    camera = rays*d[...,None]
    homogeneous = np.concatenate([camera,np.ones_like(camera[...,:1])],axis=-1)
    world = np.einsum('tij,tnj->tni',np.linalg.inv(ex),homogeneous)[...,:3]
    world[~inside] = 1e6
    return world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=['cotracker','tapip3d'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--run-name',default='primary')
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(parents=True,exist_ok=False)
    points = json.loads((ROOT/'gate_operating_points.json').read_text())
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    rows = []
    full = {}
    for case in manifest['cases']:
        assert sha(ROOT/case['input']) == case['input_sha256']
        assert sha(ROOT/case['truth']) == case['truth_sha256']
        assert sha(ROOT/case['rgbd_input']) == case['rgbd_sha256']
        inputs = np.load(ROOT/case['input'],allow_pickle=False)
        truth = np.load(ROOT/case['truth'],allow_pickle=False)
        sensor = np.load(ROOT/case['rgbd_input'],allow_pickle=False)
        setup = json.loads((ROOT/'data'/case['case']/'setup.json').read_text())
        methods = {}
        for method in args.models:
            folder = ROOT/'predictions'/method/args.run_name/case['split']
            receipt = json.loads((folder/f'{case["case"]}.json').read_text())
            assert receipt['status'] == 'ok', 'Model/API failure must not silently disappear'
            path = folder/f'{case["case"]}.npz'
            assert sha(path) == receipt['prediction_sha256']
            pred = np.load(path,allow_pickle=False)
            xy = pred['xy'][:,[0,5]]
            xyz = pred['xyz_world'][:,[0,5]] if method == 'tapip3d' else lift_scene(xy,sensor)
            raw = pred['official_visible'][:,[0,5]].astype(bool)
            # No GT/visibility/confidence/depth is passed into the independent gate.
            obs = evidence(inputs['video'],xy,setup['initial_centers'],setup['radii'],points['primary']['aperture'])
            masks = {'raw':raw}
            gate_receipt = {'case':case['case'],'method':method,'evidence':obs,'operating_points':{}}
            for name,config in points.items():
                decisions = decide(obs,config)
                independent = np.array([[r['accepted'] for r in frame] for frame in decisions])
                masks[f'gate_only_{name}'] = independent
                masks[name] = raw & independent
                gate_receipt['operating_points'][name] = decisions
            (output/f'{method}_{case["case"]}_gate.json').write_text(json.dumps(gate_receipt,indent=2)+'\n')
            np.savez_compressed(output/f'{method}_{case["case"]}_masks.npz',**masks)
            methods[method] = {'xy':xy,'xyz':xyz,'masks':masks}
            for mode,claim in masks.items():
                c = counts(xy,xyz,claim,truth)
                rows.append({'case':case['case'],'condition':case['condition'],'split':case['split'],
                             'seed':case['seed'],'model':method,'mode':mode,'counts':c})
        if len(methods)==2:
            common = {}
            for mode in ['raw','primary','coverage']:
                shared = methods['cotracker']['masks'][mode]&methods['tapip3d']['masks'][mode]
                common[mode] = shared
                for method,values in methods.items():
                    rows.append({'case':case['case'],'condition':case['condition'],'split':case['split'],
                                 'seed':case['seed'],'model':method,'mode':'common_'+mode,
                                 'counts':counts(values['xy'],values['xyz'],shared,truth)})
        full[case['case']] = methods
    groups = defaultdict(list)
    for r in rows:
        for condition in [r['condition'],'ALL']:
            groups[(r['split'],condition,r['model'],r['mode'])].append(r['counts'])
    summary = [{'split':key[0],'condition':key[1],'model':key[2],'mode':key[3],
                'sequences':len(values),**summarize(values)} for key,values in groups.items()]
    (output/'case_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    receipt = {'models':args.models,'cases':len(manifest['cases']),'status':'all attempts included',
               'data_manifest_sha256':sha(ROOT/'data_manifest.json'),'gate_sha256':sha(ROOT/'gate.py'),
               'operating_points_sha256':sha(ROOT/'gate_operating_points.json'),
               'evaluator_sha256':sha(Path(__file__)),'frame0_excluded':True,'primary_query_indices':[0,5],
               '3d_threshold_m':.05,'lift_depth_sampling':'nearest scene depth at predicted pixel',
               'gate_changes_coordinates':False}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps([r for r in summary if r['condition']=='ALL' and r['split']=='heldout' and not r['mode'].startswith('gate_only')],indent=2))


if __name__ == '__main__':
    main()
