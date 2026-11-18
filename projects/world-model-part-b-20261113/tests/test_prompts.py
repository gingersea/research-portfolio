"""Prompt set validation.

The prompt JSON files (part_a.json, part_b.json, calibration.json) are the
benchmark's definition of "what gets generated". These tests pin their schema
so a malformed prompt file fails loudly instead of producing a broken run.
"""

import importlib.resources as pkg_resources
import json


def _load(name: str) -> dict:
    return json.loads(
        pkg_resources.files("physcheck.prompts").joinpath(name).read_text()
    )


def _check_prompts(data: dict, part: str):
    assert data["part"] == part, f"expected part {part}, got {data['part']}"
    prompts = data["prompts"]
    assert len(prompts) > 0
    kinds = set()
    for p in prompts:
        assert "id" in p, "every prompt needs an id"
        assert "kind" in p, f"{p['id']} needs a kind"
        assert "text" in p, f"{p['id']} needs a text"
        assert isinstance(p.get("n_seeds"), int), f"{p['id']} needs int n_seeds"
        kinds.add(p["kind"])
    return prompts, kinds


def test_part_a_schema_and_calibration():
    prompts, kinds = _check_prompts(_load("part_a.json"), "A")
    assert kinds == {"free_fall", "projectile", "pendulum", "horizontal_motion"}
    for p in prompts:
        assert "calibration" in p, f"Part A prompt {p['id']} needs calibration"
        assert "object_diameter_m" in p["calibration"]
        if p["kind"] == "pendulum":
            assert "pendulum_length_m" in p["calibration"]


def test_part_b_schema():
    prompts, kinds = _check_prompts(_load("part_b.json"), "B")
    assert kinds == {"collision", "fluid", "support", "multi_object"}
    assert len(prompts) == 8


def test_prompt_ids_unique_across_sets():
    ids = set()
    for name in ("part_a.json", "part_b.json"):
        for p in _load(name)["prompts"]:
            assert p["id"] not in ids, f"duplicate prompt id {p['id']}"
            ids.add(p["id"])
