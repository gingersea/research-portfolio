"""Reference integrity and masking audit, without opening any predictions."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,verify_file,sha

if __name__=='__main__':
    manifest=json.loads((ROOT/'manifest.json').read_text())
    rows=[]
    for case in manifest['cases']:
        truth=np.load(verify_file(case['files']['truth']))
        data=np.load(verify_file(case['files']['input']))
        known=truth['visibility_valid']
        visible=truth['visible']
        out=truth['out_of_view']
        occ=truth['occluded']
        coordinate=truth['coordinate_valid']
        assert np.allclose(data['queries_tyx'][:,[2,1]],truth['xy'][0])
        assert np.isfinite(truth['xy'][coordinate]).all()
        row={'case':case['id'],'known_visible_without_coordinates':int((known&visible&~coordinate).sum()),
             'known_negative_unclassified':int((known&~visible&~(out|occ)).sum()),
             'known_negative_double_classified':int((known&~visible&out&occ).sum()),
             'known_visible_flagged_absent':int((known&visible&(out|occ)).sum()),
             'coordinate_valid_without_known_visible':int((coordinate&~(known&visible)).sum()),
             'known_label_point_times_including_queries':int(known.sum()),
             'unknown_point_times':int((~known).sum()),'requested_frames':data['requested_times'].tolist()}
        rows.append(row)
    save(ROOT/'annotation_audit.json',{'manifest_sha256':sha(ROOT/'manifest.json'),'rows':rows,'predictions_read':False})
    print(json.dumps(rows))
