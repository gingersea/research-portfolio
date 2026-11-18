"""Official metric conventions and query leakage/occlusion denominator checks."""
import numpy as np

from official_numpy import compute_tapvid_metrics,sample_queries_first


def test_perfect_prediction_and_post_query_only():
    query=np.array([[[1.,10.,10.]]]);tracks=np.zeros((1,1,4,2));occ=np.array([[[True,False,False,True]]])
    pred=tracks.copy();pred[:,:,0]=100;pred[:,:,1]=100
    result=compute_tapvid_metrics(query,occ,tracks,occ,pred,'first')
    assert result['average_jaccard'][0]==1 and result['occlusion_accuracy'][0]==1


def test_declaring_invisible_visible_reduces_jaccard():
    query=np.zeros((1,1,3));tracks=np.zeros((1,1,4,2));occ=np.array([[[False,False,True,True]]])
    visible_claims=np.zeros_like(occ)
    result=compute_tapvid_metrics(query,occ,tracks,visible_claims,tracks,'first')
    assert result['average_pts_within_thresh'][0]==1
    assert np.isclose(result['average_jaccard'][0],1/3)
    assert np.isclose(result['occlusion_accuracy'][0],1/3)


def test_first_query_is_first_visible_not_hidden_location():
    occ=np.array([[True,True,False,False],[False,False,False,False]])
    tracks=np.arange(16).reshape(2,4,2).astype(float)
    result=sample_queries_first(occ,tracks,np.zeros((4,12,12,3)))
    np.testing.assert_array_equal(result['query_points'][0,:,0],[2,0])
    np.testing.assert_array_equal(result['query_points'][0,0,1:],tracks[0,2,::-1])


def test_reject_all_cannot_pass_joint_tracking_metric():
    query=np.zeros((1,1,3));tracks=np.zeros((1,1,4,2));occ=np.zeros((1,1,4),bool)
    result=compute_tapvid_metrics(query,occ,tracks,np.ones_like(occ),tracks,'first')
    assert result['average_pts_within_thresh'][0]==1
    assert result['average_jaccard'][0]==0 and result['occlusion_accuracy'][0]==0


def test_failed_event_is_retained_and_query_frames_excluded():
    from assess import diagnostics
    query=np.array([[1.,0.,0.]])
    gt=np.zeros((1,8,2));occ=np.array([[False,False,True,True,False,False,False,False]])
    claim=np.zeros((1,8),bool)
    result=diagnostics(query,gt,occ,gt,claim)
    assert result['evaluated']==6 and result['events'][0]['return_frame']==4
    assert not result['events'][0]['recovered']
    claim[0,6]=True
    assert diagnostics(query,gt,occ,gt,claim)['events'][0]['delay_frames']==2
