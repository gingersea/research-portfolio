"""Scale-changing reappearance and temporally supported distractor handling."""
import cv2
import numpy as np

from recovery import recover


def scene():
    rng=np.random.default_rng(55)
    a=rng.integers(20,230,(21,21,3),dtype=np.uint8)
    b=rng.integers(20,230,(21,21,3),dtype=np.uint8)
    video=np.full((5,90,160,3),100,dtype=np.uint8)
    video[:2,30:51,20:41]=a
    video[:3,30:51,110:131]=b
    small=cv2.resize(a,(17,17),interpolation=cv2.INTER_LINEAR)
    video[3:,32:49,112:129]=small
    xy=np.tile([[30.,40.],[120.,40.]],(5,1,1))
    return video,xy,a


def test_scaled_return_recovers_and_does_not_steal_other_identity():
    video,xy,_=scene();output=recover(video,xy,xy[0],[18.,18.])
    assert not output['accepted'][3,0] and output['accepted'][4,0]
    np.testing.assert_allclose(output['xy'][4,0],[120.,40.],atol=2)
    assert not output['accepted'][3:,1].any()


def test_supported_persistence_survives_new_distractor():
    video,xy,a=scene();video[:]=video[0]
    video[1:,60:81,20:41]=a
    output=recover(video,xy,xy[0],[18.,18.])
    assert output['accepted'][:,0].all()
    video[2:,30:51,20:41]=100
    video[2:,60:81,110:131]=a
    output=recover(video,xy,xy[0],[18.,18.])
    assert not output['accepted'][2:,0].any()


def test_identical_occluded_pair_cannot_be_reidentified_by_template():
    video,xy,a=scene();video[:2,30:51,110:131]=a
    output=recover(video,xy,xy[0],[18.,18.])
    assert not output['accepted'][3:].any()
