"""Sparse unknown labels and failed inferences must stay scientifically honest."""
import numpy as np

from physcheck.relations.sparse import sparse_counts, reciprocal_accept


def example():
    # Query, unknown frame, labeled OOV, labeled return.
    xy=np.zeros((4,1,2),np.float32)
    truth={'xy':xy.copy(),'coordinate_valid':np.array([[1],[0],[0],[1]],bool),
           'visibility_valid':np.array([[1],[0],[1],[1]],bool),
           'visible':np.array([[1],[0],[0],[1]],bool),
           'out_of_view':np.array([[0],[0],[1],[0]],bool)}
    return xy,truth


def test_unlabeled_zero_is_not_an_occluded_negative():
    xy,truth=example()
    row=sparse_counts(xy,np.ones((4,1),bool),truth,(256,256))
    assert row['positive']==1 and row['negative']==1 and row['accepted']==2
    assert row['correct']==1 and row['false_visible']==1
    assert row['returned_points']==1 and row['recovered']==1 and row['strict_recovered']==0


def test_failed_inference_retains_reference_denominator_and_recovery():
    _,truth=example()
    row=sparse_counts(None,None,truth,(256,256))
    assert row['positive']==1 and row['negative']==1
    assert row['accepted']==row['correct']==row['recovered']==0
    assert row['returned_points']==1


def test_bad_location_and_unverifiable_claim_are_not_silently_removed():
    xy,truth=example()
    truth['coordinate_valid'][3]=False
    row=sparse_counts(xy,np.ones((4,1),bool),truth,(256,256))
    assert row['accepted']==2 and row['unverifiable_accept']==1 and row['correct']==0
    xy[3]=[10,0]
    truth['coordinate_valid'][3]=True
    row=sparse_counts(xy,np.ones((4,1),bool),truth,(256,256))
    assert row['wrong_location']==1 and row['correct']==0


def test_reciprocal_gate_rejects_bad_cycle_invisible_and_outside():
    candidate=np.array([[20,20],[20,20],[20,20],[256,20]],float)
    back=np.array([[0,0],[8,0],[0,0],[0,0]],float)
    reference=np.zeros((4,2))
    accepted=reciprocal_accept(candidate,np.ones(4,bool),back,np.array([1,1,0,1],bool),reference,(256,256))
    assert accepted.tolist()==[True,False,False,False]


def test_error_normalization_is_invariant_to_aspect_and_storage_resize():
    xy,truth=example()
    xy[3]=[3,0]
    row=sparse_counts(xy,np.ones((4,1),bool),truth,(256,256))
    transformed={key:value.copy() for key,value in truth.items()}
    transformed['xy']*=np.array([2,288/256])
    rescaled=sparse_counts(xy*np.array([2,288/256]),np.ones((4,1),bool),transformed,(288,512))
    assert row==rescaled and row['correct']==1


def test_a_cycle_consistent_wrong_identity_stays_a_false_acceptance():
    xy,truth=example()
    xy[3]=[30,30]
    claim=reciprocal_accept(xy[3],np.ones(1,bool),np.zeros((1,2)),np.ones(1,bool),np.zeros((1,2)),(256,256))
    assert claim[0]  # Self-consistency is not independent identity ground truth.
    declared=np.zeros((4,1),bool)
    declared[3]=claim
    row=sparse_counts(xy,declared,truth,(256,256))
    assert row['wrong_location']==1 and row['correct']==row['recovered']==0
