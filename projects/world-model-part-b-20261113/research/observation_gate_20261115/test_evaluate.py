"""Metric denominators, privileged-depth geometry and abstention accounting."""
import numpy as np

from evaluate import counts, lift_scene, summarize


def test_abstention_does_not_hide_missing_coverage_or_create_precision():
    xy = np.zeros((3,2,2))
    xyz = np.zeros((3,2,3))
    truth = {'xy':np.zeros((3,10,2)), 'xyz_world':np.zeros((3,10,3)),
             'visible':np.ones((3,10),bool),'identifiable':np.ones((3,10),bool)}
    result = summarize([counts(xy,xyz,np.zeros((3,2),bool),truth)])
    assert result['positive'] == 4 and result['samples'] == 4
    assert result['joint_precision'] is None
    assert result['retained_correct'] == 0
    assert result['pck3_visible'] == 1


def test_false_visible_and_unsupported_both_reduce_joint_precision():
    xy = np.zeros((3,2,2));xyz = np.zeros((3,2,3))
    truth = {'xy':np.zeros((3,10,2)), 'xyz_world':np.zeros((3,10,3)),
             'visible':np.ones((3,10),bool),'identifiable':np.ones((3,10),bool)}
    truth['visible'][1,0] = False
    truth['identifiable'][2,5] = False
    result = summarize([counts(xy,xyz,np.ones((3,2),bool),truth)])
    assert result['false_visible'] == 1 and result['unsupported'] == 1
    assert result['accepted'] == 4 and result['correct_accepted'] == 2
    assert result['joint_precision'] == .5


def test_scene_depth_uses_predicted_pixel_and_respects_camera_translation():
    sensor = {'depth':np.array([[[2.,8.],[2.,8.]]],np.float32),
              'intrinsics':np.eye(3,dtype=np.float32),'extrinsics':np.eye(4,dtype=np.float32)[None]}
    sensor['extrinsics'][0,0,3] = -3
    xy = np.array([[[0.,0.],[1.,0.]]])
    world = lift_scene(xy,sensor)
    np.testing.assert_allclose(world[0],[[3,0,2],[11,0,8]])
