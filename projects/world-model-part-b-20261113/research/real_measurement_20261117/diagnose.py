"""Post-hoc error attribution only; no inference, tuning or label changes."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    cases=json.loads((ROOT/'manifest.json').read_text())['cases']
    events=[]
    located_rejected=unlocated=0
    for case in cases:
        gt=np.load(case['files']['truth']['path'])
        pred=np.load(ROOT/'predictions_memory'/(case['id']+'.npz'))
        known=gt['visibility_valid'].copy()
        known[0]=False
        positive=known & gt['visible'] & gt['coordinate_valid']
        error=np.linalg.norm((pred['xy']-gt['xy'])*np.array([256/512,256/288]),axis=-1)
        correct_position=positive & (error<4)
        located_rejected+=int((correct_position & ~pred['native']).sum())
        unlocated+=int((positive & ~(error<4)).sum())
        schedule={int(t):i for i,t in enumerate(pred['requested_times'])}
        for point in range(case['queries']):
            pending=None
            for t in range(1,case['frames']):
                if known[t,point] and gt['out_of_view'][t,point]:
                    pending=t
                elif pending is not None and positive[t,point]:
                    slot=schedule[t]
                    reverse=pred['reverse_xy'][slot,point]
                    q=np.load(case['files']['input']['path'])['queries_tyx'][point,[2,1]]
                    cycle=float(np.linalg.norm((reverse-q)*np.array([256/512,256/288]))) if np.isfinite(reverse).all() else None
                    events.append({'case':case['id'],'participant':case['participant'],'point':point,'oov_frame':pending,'return_frame':t,
                        'position_correct':bool(error[t,point]<4),'native_accepted':bool(pred['native'][t,point]),
                        'confidence_accepted':bool(pred['confidence_accepted'][t,point]),'reciprocal_accepted':bool(pred['reciprocal'][t,point]),
                        'reverse_visible':bool(pred['reverse_visible'][slot,point]),'cycle_error_normalized_px':cycle})
                    pending=None
    lost=[e for e in events if e['position_correct'] and e['native_accepted'] and not e['reciprocal_accepted']]
    rows=json.loads((ROOT/'analysis/per_video.json').read_text())
    result={'post_hoc_descriptive_only':True,'new_inference_or_tuning':False,
            'visible_positions_inaccurate':unlocated,'correct_positions_rejected_by_native_visibility':located_rejected,
            'return_events':len(events),'return_participant_groups':len({e['participant'] for e in events}),
            'correct_native_returns_rejected_by_reciprocity':len(lost),
            'lost_returns_with_reverse_visibility_rejection':sum(not e['reverse_visible'] for e in lost),
            'lost_returns_with_cycle_error_at_least4':sum(e['cycle_error_normalized_px'] is not None and e['cycle_error_normalized_px']>=4 for e in lost),
            'per_arm_video_precision_at_least95':{arm:{'passed':sum(r['precision'] is not None and r['precision']>=.95 for r in rows if r['arm']==arm),
                       'nonempty':sum(r['precision'] is not None for r in rows if r['arm']==arm),'planned':len(cases)}
                 for arm in ['native','confidence_0995','reciprocal']},'events':events}
    with (ROOT/'diagnosis.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='events'}))
