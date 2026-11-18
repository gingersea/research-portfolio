"""Unmodified official video-mean metrics plus prespecified recovery accounting."""
import argparse
import json
from pathlib import Path

import numpy as np

from fetch import sha
from official_numpy import compute_tapvid_metrics

ROOT=Path(__file__).resolve().parent


def diagnostics(query,gt,occluded,xy,declared):
    N,T=occluded.shape
    evaluation=np.arange(T)[None,:]>query[:,0,None]
    positive=evaluation&~occluded;negative=evaluation&occluded
    error=np.linalg.norm(xy-gt,axis=-1)
    correct=positive&declared&(error<4)
    events=[]
    for point in range(N):
        for t in range(2,T-1):
            if t<=query[point,0] or not occluded[point,t-2:t].all() or occluded[point,t:t+2].any():continue
            end=t+1
            while end+1<T and not occluded[point,end+1]:end+=1
            hits=np.flatnonzero(correct[point,t:end+1])
            events.append({'point':point,'return_frame':t,'visible_run_end':end,'recovered':bool(len(hits)),
                           'delay_frames':int(hits[0]) if len(hits) else None})
    return {'evaluated':int(evaluation.sum()),'positive':int(positive.sum()),'negative':int(negative.sum()),
            'accepted':int((declared&evaluation).sum()),'correct_accepted4':int(correct.sum()),
            'false_visible':int((declared&negative).sum()),'events':events}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results');args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    rows=[];event_rows=[]
    for case in manifest['cases']:
        receipt=json.loads((ROOT/'predictions'/f'{case["case"]}.json').read_text())
        assert receipt['status']=='ok' and not receipt['future_truth_read'] and not receipt['evaluation_queries_jointly_inferred']
        path=ROOT/'predictions'/f'{case["case"]}.npz';assert sha(path)==receipt['prediction_sha256']
        assert sha(ROOT/case['truth'])==case['truth_sha256']
        inputs=np.load(ROOT/case['input']);truth=np.load(ROOT/case['truth']);pred=np.load(path)
        gt,occ=truth['target_points'],truth['occluded'];query=inputs['queries_tyx']
        xy=pred['xy'].transpose(1,0,2)
        for policy,key in [('native','visible'),('conservative_vc09','conservative_visible')]:
            declared=pred[key].T
            metrics=compute_tapvid_metrics(query[None],occ[None],gt[None],(~declared)[None],xy[None],'first')
            diagnostic=diagnostics(query,gt,occ,xy,declared)
            events=diagnostic.pop('events')
            row={'case':case['case'],'policy':policy,'frames':case['frames'],'queries':case['queries'],
                 **{k:float(v[0]) for k,v in metrics.items()},**diagnostic,
                 'event_count':len(events),'recovered_event_count':sum(e['recovered'] for e in events)}
            rows.append(row)
            event_rows.extend({'case':case['case'],'policy':policy,**e} for e in events)
    summary=[]
    metric_names=['average_jaccard','average_pts_within_thresh','occlusion_accuracy']
    for policy in ['native','conservative_vc09']:
        selected=[r for r in rows if r['policy']==policy];assert len(selected)==30
        r={'policy':policy,'videos':30,**{k:float(np.mean([v[k] for v in selected])) for k in metric_names}}
        for k in ['evaluated','positive','negative','accepted','correct_accepted4','false_visible','event_count','recovered_event_count']:
            r[k]=sum(v[k] for v in selected)
        r['accepted_precision4']=r['correct_accepted4']/r['accepted'] if r['accepted'] else None
        r['correct_retention4']=r['correct_accepted4']/r['positive']
        r['recovery_delays']=[e['delay_frames'] for e in event_rows if e['policy']==policy and e['recovered']]
        summary.append(r)
    for name,value in [('per_video.json',rows),('reappearance_events.json',event_rows),('summary.json',summary)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    (output/'receipt.json').write_text(json.dumps({'data_manifest_sha256':sha(ROOT/'data_manifest.json'),
        'scorer_sha256':sha(ROOT/'official_numpy.py'),'assessor_sha256':sha(Path(__file__)),
        'aggregation':'unweighted video mean for official metrics; pooled supplemental counts explicitly separate',
        'human_annotation_source':True,'query_mode':'first'},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
