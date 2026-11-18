"""Compare all comparable first outputs, not reference-label scores."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    cases=json.loads((ROOT/'manifest.json').read_text())['cases']
    rows=[]
    for case in cases:
        first=json.loads((ROOT/'predictions'/(case['id']+'.json')).read_text())
        current=json.loads((ROOT/'predictions_memory'/(case['id']+'.json')).read_text())
        row={'case':case['id'],'first_status':first['status'],'memory_status':current['status']}
        if first['status']=='ok' and current['status']=='ok':
            a=np.load(ROOT/'predictions'/(case['id']+'.npz'))
            b=np.load(ROOT/'predictions_memory'/(case['id']+'.npz'))
            assert sorted(a.files)==sorted(b.files)
            comparison={}
            for key in a.files:
                same=np.array_equal(a[key],b[key],equal_nan=True)
                field={'bitwise_values_equal_including_missing':same,'shape':list(a[key].shape)}
                if np.issubdtype(a[key].dtype,np.floating):
                    valid=np.isfinite(a[key]) & np.isfinite(b[key])
                    field['max_abs_difference']=float(np.max(np.abs(a[key][valid]-b[key][valid]))) if valid.any() else None
                    field['finite_pattern_equal']=bool(np.array_equal(np.isfinite(a[key]),np.isfinite(b[key])))
                else:
                    field['different_values']=int((a[key]!=b[key]).sum())
                comparison[key]=field
            row['arrays']=comparison
        rows.append(row)
    first_ok=sum(r['first_status']=='ok' for r in rows)
    compared=sum('arrays' in r for r in rows)
    result={'cases':len(cases),'first_ok':first_ok,'memory_ok':sum(r['memory_status']=='ok' for r in rows),
            'compared_successes':compared,'all_first_successes_compared':compared==first_ok,
            'all_compared_values_equal':all(f['bitwise_values_equal_including_missing'] for r in rows for f in r.get('arrays',{}).values()),
            'human_reference_scores_read':False,'rows':rows}
    with (ROOT/'memory_comparison.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
