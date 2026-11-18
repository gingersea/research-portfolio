"""New heldout recovery, explicit events and full-frame denominators."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np

from fetch_depth import sha
from recovery import recover

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('old_metrics',ROOT.parent/'tracking_validation_20261115/metrics.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def score(xy,accepted,truth):
    gt=truth['xy'][:,[0,5]];vis=truth['visible'][:,[0,5]];known=truth['identifiable'][:,[0,5]]
    evaluate=np.ones_like(vis);evaluate[0]=False
    positive=evaluate&vis&known;negative=evaluate&~vis&known;amb=evaluate&~known
    correct=positive&(np.linalg.norm(xy-gt,axis=-1)<=3)
    accepted=accepted&evaluate
    fixed=old.score_case(xy,accepted,gt,vis,known)
    return {'accepted':int(accepted.sum()),'correct':int((accepted&correct).sum()),'positive':int(positive.sum()),
            'false_visible':int((accepted&negative).sum()),'negative':int(negative.sum()),
            'unsupported':int((accepted&amb).sum()),'ambiguous':int(amb.sum()),'wrong_other':fixed['wrong_other_count']}


def events(xy,accepted,truth):
    rows=[];vis=truth['visible'][:,[0,5]];known=truth['identifiable'][:,[0,5]]
    good=accepted&vis&known&(np.linalg.norm(xy-truth['xy'][:,[0,5]],axis=-1)<=3)
    for obj in range(2):
        for t in range(2,len(vis)-1):
            if not vis[t-2:t,obj].any() and vis[t:t+2,obj].all() and known[t:t+2,obj].all():
                end=t+1
                while end+1<len(vis) and vis[end+1,obj] and known[end+1,obj]:end+=1
                hits=np.flatnonzero(good[t:end+1,obj])
                rows.append({'object':obj,'return_frame':t,'visible_run_end':end,
                             'recovered':bool(len(hits)),'delay_frames':int(hits[0]) if len(hits) else None})
    return rows


def main():
    output=ROOT/'control_results';output.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    assert sha(ROOT/'recovery.py')==manifest['recovery_sha256']
    rows=[];event_rows=[]
    for case in manifest['cases']:
        inputs=np.load(ROOT/case['input']);truth=np.load(ROOT/case['truth'])
        assert sha(ROOT/case['input'])==case['input_sha256']
        setup=json.loads((ROOT/'data'/case['case']/'setup.json').read_text())
        predpath=ROOT/'predictions/cotracker/primary'/case['split']/f'{case["case"]}.npz'
        pred=np.load(predpath);xy=pred['xy'][:,[0,5]];native=pred['official_visible'][:,[0,5]]
        result=recover(inputs['video'],xy,setup['initial_centers'],setup['radii'])
        np.savez_compressed(output/f'{case["case"]}.npz',**{k:result[k] for k in ['xy','accepted','baseline_accepted']})
        (output/f'{case["case"]}.json').write_text(json.dumps({'case':case['case'],'details':result['details'],
            'recovery_sha256':sha(ROOT/'recovery.py'),'input_sha256':case['input_sha256'],
            'prediction_sha256':sha(predpath),'raw_coordinates_preserved':True},indent=2)+'\n')
        methods={'raw':(xy,native),'previous_gate':(xy,native&result['baseline_accepted']),
                 'recovery':(result['xy'],result['accepted'])}
        for name,(coordinates,declared) in methods.items():
            rows.append({'case':case['case'],'condition':case['condition'],'split':case['split'],'method':name,
                         **score(coordinates,declared,truth)})
            for event in events(coordinates,declared,truth):
                event_rows.append({'case':case['case'],'condition':case['condition'],'split':case['split'],'method':name,**event})
    groups=defaultdict(list)
    for row in rows:
        for condition in [row['condition'],'ALL']:groups[row['split'],condition,row['method']].append(row)
    summary=[]
    for key,group in groups.items():
        c={k:sum(r[k] for r in group) for k in ['accepted','correct','positive','negative','false_visible','unsupported','ambiguous','wrong_other']}
        ev=[r for r in event_rows if r['split']==key[0] and (key[1]=='ALL' or r['condition']==key[1]) and r['method']==key[2]]
        c.update(joint_precision=c['correct']/c['accepted'] if c['accepted'] else None,
                 retained_correct=c['correct']/c['positive'] if c['positive'] else None,
                 events=len(ev),recovered_events=sum(e['recovered'] for e in ev),
                 recovery_delays=[e['delay_frames'] for e in ev if e['recovered']])
        summary.append({'split':key[0],'condition':key[1],'method':key[2],'sequences':len(group),**c})
    for name,value in [('cases.json',rows),('events.json',event_rows),('summary.json',summary)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps([r for r in summary if r['split']=='heldout' and r['condition'] in ['ALL','identity_replacement','long_gap_reappearance','lookalike_distractor']],indent=2))


if __name__=='__main__':main()
