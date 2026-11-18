"""Tests for judge protocol and calibration gate."""

import json

import pytest

from physcheck.judge.calibration import ProtocolInvalid, check_calibration
from physcheck.judge.protocol import (
    ProtocolError,
    build_judge_payload,
    parse_judge_output,
)

VALID_JSON = """{
  "video_id": "v1",
  "violations": [
    {"frame_idx": 3, "category": "gravity", "description": "ball floats", "confidence": 0.9}
  ],
  "notes": ""
}"""


def test_build_judge_payload_raises_without_video_id():
    with pytest.raises(ValueError):
        build_judge_payload({"grid_path": "x.png"})


def test_build_judge_payload_is_blind():
    payload = build_judge_payload(
        {"video_id": "partB_col_b1_0", "grid_path": "x.png", "n_frames": 8}
    )
    text = payload["prompt"].lower()
    assert "wan" not in text
    assert "cogvideox" not in text
    assert "model_name" not in text


def test_parse_judge_output_accepts_valid_json_and_fences():
    out = parse_judge_output(VALID_JSON, "v1", "claude", "test-version")
    assert out["violations"][0]["category"] == "gravity"
    fenced = "```json\n" + VALID_JSON + "\n```"
    out2 = parse_judge_output(fenced, "v1", "claude", "test-version")
    assert out2["violations"][0]["category"] == "gravity"


def test_parse_judge_output_rejects_bad_category():
    bad = VALID_JSON.replace('"gravity"', '"warp_speed"')
    with pytest.raises(ProtocolError):
        parse_judge_output(bad, "v1", "claude", "v")


def test_parse_judge_output_rejects_bad_confidence():
    bad = VALID_JSON.replace("0.9", "1.5")
    with pytest.raises(ProtocolError):
        parse_judge_output(bad, "v1", "claude", "v")


def test_calibration_invalid_if_a_control_missed():
    outputs = {
        "calib_float_person": {"violations": []},  # missed!
        "calib_floating_glass": {
            "violations": [{"category": "support", "frame_idx": 1, "confidence": 0.9}]
        },
        "calib_pass_through": {
            "violations": [{"category": "collision", "frame_idx": 2, "confidence": 0.8}]
        },
        "calib_uphill": {
            "violations": [{"category": "gravity", "frame_idx": 3, "confidence": 0.7}]
        },
    }
    with pytest.raises(ProtocolInvalid):
        check_calibration(outputs)


def test_calibration_valid_when_all_flagged():
    outputs = {
        "calib_float_person": {
            "violations": [{"category": "gravity", "frame_idx": 0, "confidence": 0.9}]
        },
        "calib_floating_glass": {
            "violations": [{"category": "support", "frame_idx": 1, "confidence": 0.9}]
        },
        "calib_pass_through": {
            "violations": [{"category": "collision", "frame_idx": 2, "confidence": 0.8}]
        },
        "calib_uphill": {
            "violations": [{"category": "gravity", "frame_idx": 3, "confidence": 0.7}]
        },
    }
    result = check_calibration(outputs)
    assert result["valid"] is True
