"""Prefix-pilot geometry labels, raw metric scale, explicit point/visibility masks."""
import json
from pathlib import Path

import numpy as np

from fetch import sha

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT/'pilot3d_results';output.mkdir(exist_ok=False);rows=[]
    for case in json.loads((ROOT/'pilot3d_manifest.json').read_text())['cases']:
        path=ROOT/'pilot3d_predictions'/f'{case["case"]}.npz';receipt=json.loads(path.with_suffix('.json').read_text())
        assert receipt['status']=='ok' and not receipt['future_truth_read'] and sha(path)==receipt['prediction_sha256']
        inp=np.load(ROOT/case['input']);truth=np.load(ROOT/case['truth']);pred=np.load(path)
        query=inp['queries_tyx'];T=case['frames'];evaluate=np.arange(T)[:,None]>query[None,:,0]
        visible=truth['visible'];positive=evaluate&visible;negative=evaluate&~visible
        for method,prefix in [('cotracker_estimated_depth','co'),('tapip3d_estimated_depth','tap')]:
            xy=pred[prefix+'_xy'];xyz=pred[prefix+'_xyz'];declared=pred[prefix+'_visible']
            e3=np.linalg.norm(xyz-truth['tracks_xyz'],axis=-1);e2=np.linalg.norm(xy-truth['tracks_xy'],axis=-1)
            inside=(xy[...,0]>=0)&(xy[...,0]<case['image_shape'][1])&(xy[...,1]>=0)&(xy[...,1]<case['image_shape'][0])
            accepted=declared&evaluate
            row={'case':case['case'],'method':method,'selected_queries':8,'frames':T,'evaluated':int(evaluate.sum()),
                'visible_gt':int(positive.sum()),'invisible_gt':int(negative.sum()),'accepted':int(accepted.sum()),
                'false_visible':int((declared&negative).sum()),'visible_prediction_outside_image':int((positive&~inside).sum()),
                'raw_epe3d_m':float(e3[positive].mean()),'epe2d_px':float(e2[positive].mean()),
                'pck3d5cm_count':int((positive&(e3<.05)).sum()),'pck2d4px_count':int((positive&(e2<4)).sum()),
                'correct_accepted2d4_count':int((accepted&positive&(e2<4)).sum()),
                'predicted_true_depth_ratio_median':float(np.median(xyz[...,2][positive]/truth['tracks_xyz'][...,2][positive])),
                'metric_scale_aligned_to_GT':False,'annotation_type':'PStudio calibrated geometry, not pure manual3D annotation'}
            rows.append(row)
    (output/'per_case.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary=[]
    for method in ['cotracker_estimated_depth','tapip3d_estimated_depth']:
        selected=[r for r in rows if r['method']==method]
        summary.append({'method':method,'sequences':3,'queries':24,'prefix_frames_per_sequence':64,
            'video_mean_raw_epe3d_m':float(np.mean([r['raw_epe3d_m'] for r in selected])),
            'video_mean_epe2d_px':float(np.mean([r['epe2d_px'] for r in selected])),
            **{k:sum(r[k] for r in selected) for k in ['evaluated','visible_gt','invisible_gt','accepted','false_visible',
                'pck3d5cm_count','pck2d4px_count','correct_accepted2d4_count','visible_prediction_outside_image']}})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'summary':summary,'cases':rows},indent=2))


if __name__=='__main__':main()
