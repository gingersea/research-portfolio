"""Separate real coordinate gains from accepting already-correct native positions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha,verify_file

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--descriptor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--relocated-data',type=Path)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    manifest=json.loads(args.manifest.read_text());events=[];visible_pairs=[]
    for case in manifest['cases']:
        file=case['files']['truth']
        if args.relocated_data:file=dict(file,path=str(args.relocated_data/case['id']/Path(file['path']).name))
        g=dict(np.load(verify_file(file),allow_pickle=False));known=g['visibility_valid'].copy();known[0]=False
        positive=known&g['visible']&g['coordinate_valid'];pred=[]
        for directory,key in [(args.native,'native'),(args.descriptor,'reciprocal')]:
            path=directory/(case['id']+'.json');receipt=json.loads(path.read_text())
            if receipt['status']=='ok':
                assert sha(path.with_suffix('.npz'))==receipt['prediction_sha256']
                a=np.load(path.with_suffix('.npz'),allow_pickle=False)
                error=np.linalg.norm((a['xy']-g['xy'])*[.5,256/288],axis=-1)
                pred.append((np.isfinite(error)&(error<4),a[key]))
            else:pred.append((np.zeros_like(positive),np.zeros_like(positive)))
        (native_loc,native_accept),(desc_loc,desc_accept)=pred
        visible_pairs.append({'case':case['id'],'participant':case['participant'],'positive':int(positive.sum()),
            'both_located':int((positive&native_loc&desc_loc).sum()),'native_only_located':int((positive&native_loc&~desc_loc).sum()),
            'descriptor_only_located':int((positive&~native_loc&desc_loc).sum()),'neither_located':int((positive&~native_loc&~desc_loc).sum())})
        for point in range(known.shape[1]):
            pending=False
            for t in range(1,len(known)):
                if known[t,point] and g['out_of_view'][t,point]:pending=True
                elif pending and positive[t,point]:
                    events.append({'case':case['id'],'participant':case['participant'],'time':t,'point':point,
                        'native_located':bool(native_loc[t,point]),'native_recovered':bool(native_loc[t,point]&native_accept[t,point]),
                        'descriptor_located':bool(desc_loc[t,point]),'descriptor_recovered':bool(desc_loc[t,point]&desc_accept[t,point])})
                    pending=False
    summary={'return_events':len(events),'return_participants':len({r['participant'] for r in events}),
        'native_return_coordinates_correct':sum(r['native_located'] for r in events),
        'descriptor_return_coordinates_correct':sum(r['descriptor_located'] for r in events),
        'new_return_coordinates':sum(not r['native_located'] and r['descriptor_located'] for r in events),
        'lost_return_coordinates':sum(r['native_located'] and not r['descriptor_located'] for r in events),
        'additional_accepted_returns':sum(not r['native_recovered'] and r['descriptor_recovered'] for r in events),
        'additional_accepted_returns_already_located_by_native':sum(not r['native_recovered'] and r['descriptor_recovered'] and r['native_located'] for r in events),
        'all_visible_pairing':{k:sum(r[k] for r in visible_pairs) for k in ['positive','both_located','native_only_located','descriptor_only_located','neither_located']},
        'analysis':'descriptive diagnostic; no threshold tuning; identical strict4/256 criterion',
        'source_sha256':sha(Path(__file__)),'manifest_sha256':sha(args.manifest)}
    save(args.output/'events.json',events);save(args.output/'per_video.json',visible_pairs);save(args.output/'summary.json',summary)
    print(json.dumps(summary))
