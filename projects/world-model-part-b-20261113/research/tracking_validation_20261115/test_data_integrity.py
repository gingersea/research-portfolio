"""Prevent repeated pixels across nominally independent control splits."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def test_heldout_videos_have_no_development_pixel_duplicates():
    manifest = json.loads((ROOT/'data_manifest_v2.json').read_text())
    groups = {'development': set(), 'heldout': set()}
    for case in manifest['cases']:
        with np.load(ROOT/case['input'], allow_pickle=False) as inputs:
            digest = hashlib.sha256(inputs['video'].tobytes()).hexdigest()
        assert digest == case['video_pixel_sha256']
        groups[case['split']].add(digest)
    assert len(groups['development']) == 10
    assert len(groups['heldout']) == 20
    assert not groups['development'] & groups['heldout']


def test_first_query_matches_projected_material_point():
    manifest = json.loads((ROOT/'data_manifest_v2.json').read_text())
    for case in manifest['cases']:
        inputs = np.load(ROOT/case['input'], allow_pickle=False)
        truth = np.load(ROOT/case['truth'], allow_pickle=False)
        np.testing.assert_allclose(inputs['queries'][:, 1:], truth['xy'][0], atol=1e-5)
        assert truth['visible'][0].all()


def test_ambiguous_twins_not_scored_as_uniquely_identifiable_history():
    truth = np.load(ROOT/'data_v2/ambiguous_twins_1910/truth.npz', allow_pickle=False)
    assert truth['identifiable'][:12].all()
    assert not truth['identifiable'][12:].any()
