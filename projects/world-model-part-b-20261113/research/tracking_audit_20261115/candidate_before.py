"""Baseline adapter for regression reproduction; replaced only in this new study."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('original_detector', ROOT.parent / 'principia_i2v_20261115/evaluate.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def marker_positions(frame, references):
    positions = original.fiducials(frame)
    if positions is None:
        return None
    return original.matched_positions(references, positions)[0]


def track_observed(frames, setup, initial_image):
    rows = []
    for frame in frames:
        objects = []
        for index, color in enumerate(setup['colors_rgb_left_right']):
            found = original.component(frame, color)
            observed = found is not None
            objects.append({'observed': observed, 'eligible': bool(observed and found['complete'] and .75 <= found['area']/setup['expected_areas'][index] <= 1.25),
                            'center': found['center'].tolist() if observed else None,
                            'reason': 'original_color_detection' if observed else 'original_color_missing_or_ambiguous'})
        rows.append(objects)
    return rows
