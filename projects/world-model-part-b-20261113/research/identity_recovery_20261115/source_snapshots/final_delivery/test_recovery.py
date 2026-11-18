"""Recovery must find a reappearing identity without borrowing the other identity."""
import inspect

import numpy as np

from recovery import recover


def clip():
    rng=np.random.default_rng(55)
    a=rng.integers(20,230,(21,21,3),dtype=np.uint8)
    b=rng.integers(20,230,(21,21,3),dtype=np.uint8)
    video=np.full((5,90,160,3),100,dtype=np.uint8)
    for t in [0,1]:video[t,30:51,20:41]=a
    for t in [0,1,2]:video[t,30:51,110:131]=b
    for t in [3,4]:video[t,30:51,110:131]=a
    xy=np.tile([[30.,40.],[120.,40.]],(5,1,1))
    return video,xy


def test_reacquires_original_at_new_position_after_two_frames():
    video,xy=clip();before=xy.copy()
    output=recover(video,xy,xy[0],[18.,18.])
    assert not output['accepted'][2,0] and not output['accepted'][3,0]
    assert output['accepted'][4,0]
    np.testing.assert_allclose(output['xy'][4,0],[120.,40.],atol=1)
    assert not output['accepted'][3:,1].any()
    np.testing.assert_array_equal(xy,before)
    assert output['source'][4][0]=='global_reacquisition'


def test_equal_appearance_after_loss_and_duplicate_matches_abstain():
    video,xy=clip()
    video[0:2,30:51,110:131]=video[0,30:51,20:41]
    output=recover(video,xy,xy[0],[18.,18.])
    assert not output['accepted'][3:].any()
    video,xy=clip()
    video[1:,60:81,20:41]=video[0,30:51,20:41]
    output=recover(video,xy,xy[0],[18.,18.])
    assert not output['accepted'][1:,0].any()


def test_no_truth_depth_or_tracker_score_input():
    assert set(inspect.signature(recover).parameters)=={'video','candidate_xy','initial_centers','radii'}
