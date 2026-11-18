"""Tests for result aggregation."""

import pytest

from physcheck.report.aggregate import aggregate_part_a, aggregate_part_b


def test_aggregate_part_a_handles_fit_failed():
    lines = [
        {"model": "wan", "prompt_id": "fall_a1", "kind": "free_fall", "seed": 0,
         "track_status": "ok", "fit_status": "fit_failed"},
        {"model": "wan", "prompt_id": "fall_a1", "kind": "free_fall", "seed": 1,
         "track_status": "ok", "fit_status": "ok", "rel_errs": {"g": 0.4}},
    ]
    agg = aggregate_part_a(lines)
    m = agg["by_model"]["wan"]
    assert m["fit_success_rate"] == pytest.approx(0.5)
    assert m["track_success_rate"] == pytest.approx(1.0)
    assert m["g_rel_err_mean"] == pytest.approx(0.4)


def test_aggregate_part_b_computes_rates():
    outputs = [
        {"video_id": "v1", "violations": [
            {"frame_idx": 0, "category": "gravity", "confidence": 0.9}]},
        {"video_id": "v2", "violations": []},
        {"video_id": "v3", "violations": [
            {"frame_idx": 1, "category": "collision", "confidence": 0.8}]},
        {"video_id": "v4", "violations": []},
    ]
    manifest = {
        "v1": {"model": "m1", "prompt_id": "p1", "seed": 0, "kind": "k"},
        "v2": {"model": "m1", "prompt_id": "p2", "seed": 0, "kind": "k"},
        "v3": {"model": "m2", "prompt_id": "p1", "seed": 0, "kind": "k"},
        "v4": {"model": "m2", "prompt_id": "p2", "seed": 0, "kind": "k"},
    }
    agg = aggregate_part_b(outputs, manifest)
    assert agg["violation_rate_by_model"]["m1"] == pytest.approx(0.5)
    assert agg["mean_violations_per_video"]["m2"] == pytest.approx(0.5)
    assert agg["violations_by_category"]["m1"]["gravity"] == 1
