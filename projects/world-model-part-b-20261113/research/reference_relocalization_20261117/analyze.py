"""Four frozen arms, known sparse labels, participant-cluster uncertainty."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha,verify_file
from physcheck.relations.sparse import sparse_counts

ARMS={'cotracker_native':('native','native'),'cotracker_cycle':('native','reciprocal'),
      'mast3r_raw':('descriptor','raw'),'mast3r_cycle':('descriptor','reciprocal')}
METRICS={'precision':('correct','accepted'),'retention':('correct','positive'),
         'coordinate_accuracy':('located_all_visible','positive'),'recovery':('recovered','returned_points'),
         'strict_recovery':('strict_recovered','returned_points'),'false_visible_rate':('false_visible','negative')}

def rates(counts):
    return {key:counts[a]/counts[b] if counts[b] else None for key,(a,b) in METRICS.items()}

def main(args):
    args.output.mkdir(exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    assert manifest['cases'],'no eligible holdout; cannot fabricate validation'
    rows=[];receipts={}
    for case in manifest['cases']:
        truth_record=case['files']['truth']
        if args.relocated_data:
            truth_record=dict(truth_record,path=str(args.relocated_data/case['id']/Path(truth_record['path']).name))
        truth=dict(np.load(verify_file(truth_record),allow_pickle=False))
        predictions={}
        for kind,directory in [('native',args.native),('descriptor',args.descriptor)]:
            path=directory/(case['id']+'.json')
            receipt=json.loads(path.read_text());receipts[str(path)]=sha(path)
            assert receipt['input_sha256']==case['files']['input']['sha256'] and receipt['future_truth_read'] is False
            prediction=None
            if receipt['status']=='ok':
                assert sha(path.with_suffix('.npz'))==receipt['prediction_sha256']
                prediction=dict(np.load(path.with_suffix('.npz'),allow_pickle=False))
            predictions[kind]=(prediction,receipt['status'])
        for arm,(kind,key) in ARMS.items():
            prediction,status=predictions[kind]
            counts=sparse_counts(None if prediction is None else prediction['xy'],None if prediction is None else prediction[key],truth,(288,512))
            assert counts['accepted']==sum(counts[k] for k in ['correct','false_visible','wrong_location','unverifiable_accept'])
            rows.append({'case':case['id'],'participant':case['participant'],'arm':arm,'inference_status':status,**counts,**rates(counts)})
    groups=sorted({r['participant'] for r in rows})
    draws=np.random.default_rng(20261117).integers(0,len(groups),(10000,len(groups)))
    count_keys=[key for key in rows[0] if key not in {'case','participant','arm','inference_status'} and key not in METRICS]
    summaries={};boots={}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        counts={key:sum(r[key] for r in subset) for key in count_keys}
        cluster={key:np.array([sum(r[key] for r in subset if r['participant']==p) for p in groups]) for key in count_keys}
        intervals={};boots[arm]={}
        for metric,(a,b) in METRICS.items():
            n=cluster[a][draws].sum(1);d=cluster[b][draws].sum(1)
            values=np.divide(n,d,out=np.full(len(d),np.nan),where=d>0)
            finite=values[np.isfinite(values)];boots[arm][metric]=values
            degenerate=bool(len(finite) and np.ptp(finite)==0)
            intervals[metric]={'ci95':np.quantile(finite,[.025,.975]).tolist() if len(groups)>=2 and len(finite) and not degenerate else None,
                'valid_draws':len(finite),'degenerate':degenerate,'denominator_clusters':int((cluster[b]>0).sum()),
                'interpretation':'descriptive selected-participant bootstrap; no population safety bound'}
        summaries[arm]={'counts':counts,'pooled':rates(counts),'participant_bootstrap':intervals,
                        'inference_ok':sum(r['inference_status']=='ok' for r in subset),'planned_cases':len(subset)}
    paired={}
    for arm in ARMS:
        if arm=='cotracker_native':continue
        paired[arm]={}
        for metric in METRICS:
            delta=boots[arm][metric]-boots['cotracker_native'][metric];finite=delta[np.isfinite(delta)]
            before=summaries['cotracker_native']['pooled'][metric];after=summaries[arm]['pooled'][metric]
            paired[arm][metric]={'difference':after-before if before is not None and after is not None else None,
               'ci95':np.quantile(finite,[.025,.975]).tolist() if len(groups)>=2 and len(finite) and np.ptp(finite)>0 else None,
               'valid_draws':len(finite)}
    save(args.output/'per_video.json',rows)
    with (args.output/'per_video.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(args.output/'summary.json',{'split':args.split,'clips':len(manifest['cases']),'participant_clusters':len(groups),
          'arms':summaries,'paired_vs_native':paired,'threshold_tuning':False,'population_representativeness':False,
          'method':'participant cluster bootstrap10000 seed20261117; unknown sparse labels excluded'})
    save(args.output/'receipt.json',{'manifest_sha256':sha(args.manifest),'source_sha256':sha(Path(__file__)),
          'scorer_sha256':sha(PROJECT/'benchmark/physcheck/relations/sparse.py'),'prediction_receipts':receipts,
          'all_planned_failures_retained':True,'inferential_generator_ranking':False})
    print(json.dumps({k:{'counts':v['counts'],'pooled':v['pooled']} for k,v in summaries.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--descriptor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['development','new_participant_validation'],required=True)
    p.add_argument('--relocated-data',type=Path,help='copied per-case truth directory; original SHA remains mandatory')
    main(p.parse_args())
