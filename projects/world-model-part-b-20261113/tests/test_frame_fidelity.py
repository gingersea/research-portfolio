"""Catch channel swapping and single-frame sampling failures using real video."""
import cv2
import numpy as np
import pytest

from physcheck.judge.frames import sample_frames
from physcheck.config import Config


def make_video(tmp_path, rgb, count=8):
    path = tmp_path / 'control.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 8, (64, 48))
    assert writer.isOpened()
    image = np.full((48, 64, 3), rgb[::-1], dtype=np.uint8)
    for _ in range(count):
        writer.write(image)
    writer.release()
    return path


@pytest.mark.parametrize('rgb,dominant_bgr', [((255, 0, 0), 2), ((0, 0, 255), 0)])
def test_saved_grid_preserves_red_and_blue(tmp_path, rgb, dominant_bgr):
    path = make_video(tmp_path, rgb)
    cfg = Config(tmp_path, tmp_path/'artifacts', tmp_path/'models', tmp_path/'results')
    sampled = sample_frames(str(path), cfg=cfg, video_id='control')
    grid = cv2.imread(sampled['grid_path'])
    single = cv2.imread(sampled['frame_paths'][0])
    assert int(single[24, 32, dominant_bgr]) > 220
    assert int(grid[90, 160, dominant_bgr]) > 220
    assert int(grid[90, 160, 2-dominant_bgr]) < 30


def test_single_frame_video_remains_sampleable(tmp_path):
    path = make_video(tmp_path, (255, 0, 0), count=1)
    out = sample_frames(str(path), n_frames=8)
    assert out['indices'] == [0]
    assert out['n_total'] == 1


def test_request_one_frame_selects_first_frame(tmp_path):
    path = make_video(tmp_path, (255, 0, 0))
    assert sample_frames(str(path), n_frames=1)['indices'] == [0]


@pytest.mark.parametrize('count', [0, -1])
def test_nonpositive_sample_count_is_rejected(tmp_path, count):
    path = make_video(tmp_path, (255, 0, 0))
    with pytest.raises(ValueError, match='positive'):
        sample_frames(str(path), n_frames=count)
