"""Unbound or malformed model responses cannot become calibrated judgments."""
import pytest

from physcheck.judge.local_vlm import parse_pc


def test_pc_response_is_bound_to_sampled_frames_and_id():
    text = '```json\n{"video_id":"v1","pc_score":2,"reason":"Ball changes shape.","violations":[{"frame_idx":4,"category":"rigid_body","description":"Shape changes."}]}\n```'
    assert parse_pc(text, 'v1', [0, 4, 8])['pc_score'] == 2
    with pytest.raises(ValueError):
        parse_pc(text, 'other', [0, 4, 8])
    with pytest.raises(ValueError):
        parse_pc(text, 'v1', [0, 3, 8])


@pytest.mark.parametrize('score', [0, 6, True, 2.5])
def test_pc_score_must_be_integer_one_through_five(score):
    import json
    with pytest.raises(ValueError):
        parse_pc(json.dumps({'video_id': 'v1', 'pc_score': score, 'reason': 'x', 'violations': []}), 'v1', [0])
