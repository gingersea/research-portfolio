"""Event denominators, reference provenance and fresh V2 input integrity."""
import json
from pathlib import Path

import numpy as np

from assess_controls import events
from assess_real_depth import ref_score
from fetch_depth import sha

ROOT=Path(__file__).resolve().parent


def test_missing_recovery_stays_in_event_denominator():
    truth={'xy':np.zeros((8,10,2)), 'visible':np.ones((8,10),bool),'identifiable':np.ones((8,10),bool)}
    truth['visible'][2:5,0]=False
    xy=np.zeros((8,2,2));claim=np.zeros((8,2),bool)
    result=events(xy,claim,truth)
    assert len(result)==1 and result[0]['return_frame']==5 and not result[0]['recovered']
    claim[7,0]=True
    assert events(xy,claim,truth)[0]['delay_frames']==2


def test_reference_unknown_and_absent_are_different():
    rows=[[1,0,'visible','identifiable',10,10,4,''],[2,0,'absent','identifiable',None,None,None,''],
          [3,0,'visible','unknown',10,10,4,'']]
    xy=np.full((4,2,2),10.);claim=np.ones((4,2),bool)
    result=ref_score(rows,xy,claim)
    assert result['positive']==1 and result['negative']==1 and result['unknown']==1
    assert result['correct_nominal8']==1 and result['false_visible']==1 and result['unsupported']==1
    abstain=ref_score(rows,xy,np.zeros_like(claim))
    assert abstain['correct_nominal8']==0 and abstain['positive']==1 and abstain['accepted']==0


def test_ai_reference_is_bound_before_real_model_predictions():
    path=ROOT/'real_i2v/ai_visual_reference.json';reference=json.loads(path.read_text())
    assert not reference['human_ground_truth'] and reference['annotator_type']=='AI_visual_reference'
    assert sum(map(len,reference['clips'].values()))==72
    for model in ['cotracker','tapip3d']:
        for clip in reference['clips']:
            receipt=json.loads((ROOT/'real_predictions'/model/f'{clip}.json').read_text())
            assert receipt['reference_sha256_before_inference']==sha(path) and not receipt['reference_labels_read']


def test_v2_pixels_and_frozen_recovery_are_new():
    prior=set()
    for directory,file in [('tracking_validation_20261115','data_manifest_v2.json'),('observation_gate_20261115','data_manifest.json'),('identity_recovery_20261115','data_manifest.json')]:
        prior|={r['video_pixel_sha256'] for r in json.loads((ROOT.parent/directory/file).read_text())['cases']}
    v2=ROOT.parent/'identity_recovery_multiscale_20261115'
    manifest=json.loads((v2/'data_manifest.json').read_text());pixels=[r['video_pixel_sha256'] for r in manifest['cases']]
    assert len(pixels)==len(set(pixels))==48 and not set(pixels)&prior
    assert sha(v2/'recovery.py')==manifest['recovery_sha256']
