"""Adversarial behavior tests: independent RGB gate and ambiguity history."""
import inspect

import numpy as np

from gate import decide, observe

CONFIG = {'aperture':.5,'correlation':.8,'identity_margin':.15}


def test_visible_texture_and_flat_occlusion():
    rng = np.random.default_rng(123)
    frame = np.zeros((90,150,3),dtype=np.uint8)+100
    frame[30:51,20:41] = rng.integers(20,240,(21,21,3),dtype=np.uint8)
    frame[30:51,100:121] = rng.integers(20,240,(21,21,3),dtype=np.uint8)
    hidden = frame.copy();hidden[25:55,15:45]=100
    video = np.stack([frame,frame,hidden])
    xy = np.tile([[30.,40.],[110.,40.]],(3,1,1))
    result = observe(video,xy,xy[0],[18,18],CONFIG)
    assert all(item['accepted'] for item in result[1])
    assert not result[2][0]['accepted']
    assert result[2][1]['accepted']
    swapped = xy.copy();swapped[1] = xy[1,::-1]
    result = observe(video,swapped,xy[0],[18,18],CONFIG)
    assert not any(item['accepted'] for item in result[1])
    assert set(inspect.signature(observe).parameters) == {'video','candidate_xy','initial_centers','radii','config'}


def test_identical_history_remains_unknown_after_occlusion():
    good = {'score':1.,'cross_score':1.,'contrast':20.,'local_offset_px':0.,'global_score':1.}
    row = [{**good,'evidence_xy':[20.,20.]},{**good,'evidence_xy':[120.,20.]}]
    lost = [{**row[0],'score':0.},row[1]]
    output = decide({'frames':[row,lost,row],'radii':[10.,10.],'reference_pair_similarity':1.},CONFIG)
    assert all(item['accepted'] for item in output[0])
    assert not any(item['accepted'] for item in output[2])
    assert output[2][0]['reason'] == 'unresolved_lookalike_history'


def test_competing_identity_and_displaced_local_match_are_rejected():
    item = {'score':.9,'cross_score':.88,'contrast':20.,'local_offset_px':0.,'global_score':.9,'evidence_xy':[10.,10.]}
    displaced = {**item,'score':1.,'cross_score':0.,'local_offset_px':4.}
    output = decide({'frames':[[item,displaced]],'radii':[10.,10.],'reference_pair_similarity':0.},CONFIG)
    assert not any(row['accepted'] for row in output[0])
