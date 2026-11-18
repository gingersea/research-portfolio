"""Select image-gate parameters on already-seen prior controls only."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from gate import evidence,decide
from fetch_tapip import sha

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'tracking_validation_20261115'


def main():
    output=ROOT/'calibration'
    output.mkdir(exist_ok=False)
    manifest=json.loads((OLD/'data_manifest_v2.json').read_text())
    configurations=[{'aperture':a,'correlation':c,'identity_margin':m} for a in [.5,.65] for c in [.8,.9,.95] for m in [.05,.15]]
    counts=[{'config':cfg,'accepted_correct':0,'accepted_bad':0,'visible_identifiable':0,'accepted_total':0,'unsupported':0} for cfg in configurations]
    for case in manifest['cases']:
        inputs=np.load(OLD/case['input'],allow_pickle=False)
        truth=np.load(OLD/case['truth'],allow_pickle=False)
        predicted=np.load(OLD/'predictions/dataset_v2'/case['split']/f'{case["case"]}.npz',allow_pickle=False)
        xy=predicted['xy'][:,[0,5]]
        raw_visible=predicted['official_visible'][:,[0,5]]
        radii=320*.22/truth['xyz_world'][0,[0,5],2]
        initial=inputs['queries'][[0,5],1:]
        cache={a:evidence(inputs['video'],xy,initial,radii,a) for a in [.5,.65]}
        actual_xy=truth['xy'][:,[0,5]]
        visible=truth['visible'][:,[0,5]]
        identifiable=truth['identifiable'][:,[0,5]]
        valid=visible & identifiable
        valid[0]=False
        correct=valid & (np.linalg.norm(xy-actual_xy,axis=-1)<=3)
        unknown=~identifiable;unknown[0]=False
        for row in counts:
            cfg=row['config']
            observed=decide(cache[cfg['aperture']],cfg)
            accepted=np.array([[item['accepted'] for item in frame] for frame in observed]) & raw_visible
            accepted[0]=False
            row['accepted_correct']+=int((accepted&correct).sum())
            row['accepted_bad']+=int((accepted&~correct&~unknown).sum())
            row['accepted_total']+=int(accepted.sum())
            row['unsupported']+=int((accepted&unknown).sum())
            row['visible_identifiable']+=int(valid.sum())
    # Safety first, then retained correct observations; deterministic tie-break.
    selected=min(counts,key=lambda row:(row['accepted_bad']+row['unsupported'],-row['accepted_correct'],
                                        -row['config']['correlation'],row['config']['aperture'], -row['config']['identity_margin']))
    result={'development_source':'All30 prior controls are now development material; no unseen validation claim.',
            'selection':'min accepted_bad+unsupported, then max accepted_correct; fixed deterministic tie-break',
            'cases':counts,'selected':selected,'gate_source_sha256':sha(ROOT/'gate.py')}
    (output/'grid.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'gate_config.json').write_text(json.dumps(selected['config'],indent=2)+'\n')
    print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
