"""Part B judge protocol: fixed prompt template, JSON schema, parsing.

Reproducibility contract:
- The judge prompt is a fixed template (this module's JUDGE_TEMPLATE).
- Judge output must parse into JUDGE_OUTPUT_SCHEMA.
- Judging is blind: payloads contain an opaque video_id, never model identity.
- Calibration gate lives in calibration.py.
"""

from __future__ import annotations

import json
import re

#: Fixed judge instruction template. {n_frames} is substituted at build time.
JUDGE_TEMPLATE = """You are a physics-consistency judge for generated videos.
You are shown {n_frames} annotated frames sampled evenly from a video.

Identify physical violations visible in the frames. A violation is an event
that contradicts elementary physics (gravity, support, collision/contact,
fluid behavior, rigid-body motion).

For each violation, report:
- frame_idx: the index of the frame (from the frame label) where it is visible
- category: one of "gravity", "support", "collision", "fluid", "rigid_body", "other"
- description: a one-sentence description of the violation
- confidence: a number between 0 and 1

Output JSON only, in exactly this schema:
{{
  "video_id": "<the video id you were given>",
  "violations": [
    {{"frame_idx": 0, "category": "gravity", "description": "...", "confidence": 0.9}}
  ],
  "notes": ""
}}

Do not speculate about motion between frames; judge only what is visible in
the frames. If no violation is visible, output an empty violations list.
"""

#: Categories the judge may use.
CATEGORIES = {"gravity", "support", "collision", "fluid", "rigid_body", "other"}

#: Expected keys and types of a judge output record.
JUDGE_OUTPUT_SCHEMA = {
    "video_id": str,
    "violations": list,
    "notes": str,
}


class ProtocolError(ValueError):
    """Raised when a judge output does not conform to the protocol."""


def build_judge_payload(record: dict) -> dict:
    """Build a judge payload from a record with grid_path and n_frames.

    record must contain video_id (opaque), grid_path, n_frames.
    Raises ValueError if video_id is missing (blind-but-traceable contract).
    """
    if "video_id" not in record:
        raise ValueError("judge payload requires video_id")
    if "grid_path" not in record:
        raise ValueError("judge payload requires grid_path")
    n_frames = record.get("n_frames", "N")
    prompt = JUDGE_TEMPLATE.format(n_frames=n_frames)
    return {
        "prompt": prompt,
        "schema": JUDGE_OUTPUT_SCHEMA,
        "video_id": record["video_id"],
        "grid_path": record["grid_path"],
    }


def parse_judge_output(text: str, video_id: str, judge_model: str, judge_version: str) -> dict:
    """Parse and validate a judge's textual answer into a normalized dict.

    Accepts raw JSON or fenced JSON (```json ... ```).
    Raises ProtocolError on any schema violation.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"judge output is not valid JSON: {e}")

    if not isinstance(data, dict):
        raise ProtocolError("judge output must be a JSON object")
    if "violations" not in data:
        raise ProtocolError("judge output missing 'violations'")
    if not isinstance(data["violations"], list):
        raise ProtocolError("'violations' must be a list")

    out = {
        "video_id": video_id,
        "violations": [],
        "notes": str(data.get("notes", "")),
        "judge_model": judge_model,
        "judge_version": judge_version,
    }
    for i, v in enumerate(data["violations"]):
        if not isinstance(v, dict):
            raise ProtocolError(f"violation {i} must be an object")
        cat = v.get("category")
        if cat not in CATEGORIES:
            raise ProtocolError(
                f"violation {i}: category {cat!r} not in {sorted(CATEGORIES)}"
            )
        conf = v.get("confidence")
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            raise ProtocolError(f"violation {i}: confidence {conf!r} not in [0,1]")
        idx = v.get("frame_idx")
        if not isinstance(idx, int) or idx < 0:
            raise ProtocolError(f"violation {i}: frame_idx {idx!r} invalid")
        out["violations"].append(
            {
                "frame_idx": idx,
                "category": cat,
                "description": str(v.get("description", "")),
                "confidence": float(conf),
            }
        )
    return out
