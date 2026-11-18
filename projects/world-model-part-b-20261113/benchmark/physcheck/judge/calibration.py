"""Calibration gate for the Part B judge.

Control prompts (calibration.json) describe scenes with *known* physical
violations. The judge must flag at least one violation of the expected kind
in every calibration video — otherwise the whole judging protocol is
declared invalid.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

from physcheck.judge.protocol import CATEGORIES


class ProtocolInvalid(Exception):
    """Raised when the calibration gate fails."""


def load_calibration_prompts() -> dict:
    data = json.loads(
        pkg_resources.files("physcheck.prompts").joinpath("calibration.json").read_text()
    )
    return {p["id"]: p for p in data["prompts"]}


def check_calibration(
    judge_outputs: dict[str, dict], calibration_prompts: dict | None = None
) -> dict:
    """Check that every calibration video was judged with a violation.

    judge_outputs: {video_id: parsed judge output dict (with "violations")}.
    Returns {"valid": bool, "failed_video_ids": [...]}.
    Raises ProtocolInvalid if invalid (names the failed videos).
    """
    prompts = calibration_prompts if calibration_prompts is not None else load_calibration_prompts()
    failed = []
    for prompt_id, prompt in prompts.items():
        expected = prompt["expected_violation_kind"]
        if prompt_id not in judge_outputs:
            failed.append(f"{prompt_id} (missing judge output)")
            continue
        violations = judge_outputs[prompt_id].get("violations", [])
        ok = any(
            v["category"] == expected or v["category"] == "other"
            for v in violations
        )
        if not ok:
            failed.append(f"{prompt_id} (no {expected} violation found)")

    result = {"valid": len(failed) == 0, "failed_video_ids": failed}
    if not result["valid"]:
        raise ProtocolInvalid(
            "calibration gate failed: " + "; ".join(failed)
        )
    return result
