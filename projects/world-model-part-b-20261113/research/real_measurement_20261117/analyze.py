"""Fixed three-arm evaluation on known sparse labels; participant-cluster intervals."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha,verify_file
from physcheck.relations.sparse import sparse_counts

ARMS={'native':'native','confidence_0995':'confidence_accepted','reciprocal':'reciprocal'}
RATIOS={'precision':('correct','accepted'),'retention':('correct','positive'),
        'false_visible_rate':('false_visible','negative'),'false_oov_rate':('false_visible_oov','negative_oov'),
        'false_occluded_rate':('false_visible_occluded','negative_occluded'),
        'recovery':('recovered','returned_points'),'strict_recovery':('strict_recovered','returned_points'),
        'coordinate_accuracy':('located_all_visible','positive')}


def ratios(row):
    return {name:row[a]/row[b] if row[b] else None for name,(a,b) in RATIOS.items()}


def main(output,predictions):
    output.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    rows=[]
    receipts={}
    for case in manifest['cases']:
        truth=dict(np.load(verify_file(case['files']['truth']),allow_pickle=False))
        path=predictions/(case['id']+'.json')
        receipt=json.loads(path.read_text())
        receipts[case['id']]=sha(path)
        assert receipt['input_sha256']==case['files']['input']['sha256'] and not receipt['future_truth_read']
        prediction=None
        if receipt['status']=='ok':
            p=path.with_suffix('.npz')
            assert sha(p)==receipt['prediction_sha256']
            prediction=dict(np.load(p,allow_pickle=False))
            # Both extra arms are acceptance gates, never coordinate updates.
            assert not (prediction['confidence_accepted'] & ~prediction['native']).any()
            assert not (prediction['reciprocal'] & ~prediction['native']).any()
        for arm,key in ARMS.items():
            row=sparse_counts(None if prediction is None else prediction['xy'],
                              None if prediction is None else prediction[key],truth,(288,512))
            assert row['accepted']==row['correct']+row['wrong_location']+row['false_visible']+row['unverifiable_accept']
            rows.append({'case':case['id'],'participant':case['participant'],'source_video':case['source_video'],
                         'arm':arm,'inference_status':receipt['status'],**row,**ratios(row)})
    counts=[k for k in rows[0] if k not in ['case','participant','source_video','arm','inference_status'] and k not in RATIOS]
    groups=sorted({c['participant'] for c in manifest['cases']})
    draws=np.random.default_rng(20261117).integers(0,len(groups),(10000,len(groups)))
    summaries={}
    bootstrap_ratios={}
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm]
        totals={key:sum(r[key] for r in selected) for key in counts}
        total_rates=ratios(totals)
        grouped={key:np.array([sum(r[key] for r in selected if r['participant']==g) for g in groups]) for key in counts}
        intervals={}
        bootstrap_ratios[arm]={}
        for metric,(numerator,denominator) in RATIOS.items():
            n=grouped[numerator][draws].sum(axis=1)
            d=grouped[denominator][draws].sum(axis=1)
            values=np.divide(n,d,out=np.full(len(d),np.nan),where=d>0)
            finite=values[np.isfinite(values)]
            bootstrap_ratios[arm][metric]=values
            intervals[metric]={'ci95':np.quantile(finite,[.025,.975]).tolist() if len(groups)>=2 and len(finite) else None,
                               'valid_draws':len(finite),'planned_draws':10000}
        useful=sum(r['correct']>=5 for r in selected)
        summaries[arm]={'cases_planned':len(selected),'inference_ok':sum(r['inference_status']=='ok' for r in selected),
                        'counts':totals,'pooled':total_rates,
                        'video_macro':{k:float(np.mean([r[k] for r in selected if r[k] is not None]))
                            if any(r[k] is not None for r in selected) else None for k in RATIOS},
                        'video_macro_valid_denominators':{k:sum(r[k] is not None for r in selected) for k in RATIOS},
                        'participant_bootstrap':intervals,'clips_with_at_least5_correct':useful,
                        'empirical_gate_passed':total_rates['precision'] is not None and total_rates['precision']>=.95
                            and total_rates['retention']>=.2 and useful*2>=len(selected)}
    paired={}
    for arm in ['confidence_0995','reciprocal']:
        paired[arm]={}
        for metric in ['precision','retention','false_visible_rate','recovery']:
            delta=bootstrap_ratios[arm][metric]-bootstrap_ratios['native'][metric]
            finite=delta[np.isfinite(delta)]
            before=summaries['native']['pooled'][metric]
            after=summaries[arm]['pooled'][metric]
            paired[arm][metric]={'difference':after-before if before is not None and after is not None else None,
                'ci95':np.quantile(finite,[.025,.975]).tolist() if len(groups)>=2 and len(finite) else None,
                'valid_draws':len(finite)}
    save(output/'per_video.json',rows)
    result={'planned_clips':len(manifest['cases']),'participant_clusters':len(groups),'arms':summaries,
            'paired_cluster_differences':paired,'confidence_intervals':'Descriptive participant-cluster bootstrap,10000,seed20261117',
            'full_dataset_or_generated_physics_validity_claim':False}
    save(output/'summary.json',result)
    save(output/'receipt.json',{'source_sha256':sha(Path(__file__)),'scorer_sha256':sha(PROJECT/'benchmark/physcheck/relations/sparse.py'),
             'manifest_sha256':sha(ROOT/'manifest.json'),'prediction_directory':str(predictions),
             'predictions':receipts,'threshold_search':False,'failed_cases_removed':False})
    print(json.dumps({arm:{'counts':s['counts'],'pooled':s['pooled'],'gate':s['empirical_gate_passed']} for arm,s in summaries.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'analysis')
    parser.add_argument('--predictions',type=Path,default=ROOT/'predictions')
    args=parser.parse_args()
    main(args.output,args.predictions)
