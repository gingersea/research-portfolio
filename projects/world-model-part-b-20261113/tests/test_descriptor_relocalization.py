"""Spatial alignment, tie handling, reciprocity and absence limitations."""
import numpy as np
import pytest
import torch


def implementation():
    from physcheck.relations.descriptor import descriptor_match
    return descriptor_match


def test_fractional_query_uses_bilinear_features_not_nearest_pixel():
    # q=(.75,.25) interpolates [.25,.75]; best target is [1,3].
    reference=torch.tensor([[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]]])
    target=torch.tensor([[[1.,0.],[0.,1.]],[[1.,3.],[-1.,0.]]])
    xy,back,accepted=implementation()(reference,target,torch.tensor([[.75,.25]]))
    np.testing.assert_array_equal(xy.cpu(),[[0,1]])


def test_tie_is_first_row_major_and_invalid_query_is_rejected():
    desc=torch.tensor([[[1.,0.],[1.,0.]],[[0.,1.],[0.,1.]]])
    xy,back,accepted=implementation()(desc,desc,torch.tensor([[1.,0.],[-1.,0.]]))
    np.testing.assert_array_equal(xy[0].cpu(),[0,0])
    assert not bool(accepted[1]) and torch.isnan(xy[1]).all()


def test_nonreciprocal_match_rejects_without_moving_candidate():
    ref=torch.tensor([[[1.,0.],[.9,.1],[0.,1.]]])
    target=torch.tensor([[[1.,0.],[0.,1.],[-1.,0.]]])
    xy,back,accepted=implementation()(ref,target,torch.tensor([[1.,0.]]))
    np.testing.assert_array_equal(xy.cpu(),[[0,0]])
    np.testing.assert_array_equal(back.cpu(),[[0,0]])
    assert not bool(accepted[0])  # normalized return error=256/3>4


def test_reciprocal_descriptor_match_does_not_establish_visibility_or_identity():
    # An absent point may have a perfect, reciprocal lookalike.
    desc=torch.eye(4).reshape(2,2,4)
    xy,back,accepted=implementation()(desc,desc,torch.tensor([[1.,1.]]))
    np.testing.assert_array_equal(xy.cpu(),[[1,1]])
    assert bool(accepted[0])  # evaluator must still count known absence as false-visible


def test_zero_or_nonfinite_descriptors_are_not_silent_matches():
    desc=torch.eye(4).reshape(2,2,4)
    zero=torch.zeros_like(desc)
    xy,back,accepted=implementation()(zero,desc,torch.tensor([[0.,0.]]))
    assert torch.isnan(xy).all() and not accepted.any()
    with pytest.raises(ValueError,match='finite'):
        implementation()(desc,desc*float('nan'),torch.tensor([[0.,0.]]))


def test_fractional_last_pixel_cell_uses_border_clamping():
    desc=torch.eye(4).reshape(2,2,4)
    xy,back,accepted=implementation()(desc,desc,torch.tensor([[1.75,1.5]]))
    np.testing.assert_array_equal(xy.cpu(),[[1,1]])
