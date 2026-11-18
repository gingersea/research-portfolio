{
  "concrete_findings": [
    {
      "description": "The `descriptor_match` function in `descriptor.py` uses `align_corners=True` and `padding_mode='border'` in `grid_sample`, which matches the protocol's requirement for bilinear sampling with border clamping. This aligns with the specified method for handling out-of-bound queries.",
      "evidence": "In `descriptor.py`, the `grid_sample` call includes `align_corners=True` and `padding_mode='border'`."
    },
    {
      "description": "The `sparse_counts` function in `sparse.py` correctly excludes unknown frames from being treated as negatives, as required by the protocol. It only considers `known` frames for evaluation.",
      "evidence": "In `sparse.py`, the `sparse_counts` function defines `known = truth['visibility_valid'].astype(bool).copy()` and uses this to filter all counts, ensuring unknown frames are not treated as negatives."
    },
    {
      "description": "The `run_descriptor.py` script explicitly uses the public MASt3R baseline and does not load any PointSt3R weights or visibility heads, as required by the protocol.",
      "evidence": "In `run_descriptor.py`, the model is loaded using `AsymmetricMASt3R.from_pretrained(weights['checkpoint'])`, and the `record` dictionary explicitly states `'model':'public MASt3R static baseline, not PointSt3R fine-tuned'`."
    }
  ],
  "limitations": [
    {
      "description": "The protocol specifies that the validation set must exclude all prior 10 participants, but the `prepare_validation.py` script does not explicitly verify that the selected clips are disjoint from the prior participants' IDs. It only checks that the participant is not in `selection['excluded_participants']`, but this list is not defined in the provided code.",
      "evidence": "In `prepare_validation.py`, the script checks `assert record['participant'] not in selection['excluded_participants']`, but `selection['excluded_participants']` is not defined in the provided code."
    },
    {
      "description": "The `run_native.py` script uses `torch.inference_mode()` for inference, but the protocol does not explicitly require this. While this is a best practice, the absence of a requirement means this could be a deviation from the protocol.",
      "evidence": "In `run_native.py`, the script uses `with torch.inference_mode():` for inference, but the protocol does not explicitly require this."
    },
    {
      "description": "The `analyze.py` script uses a bootstrap with 10,000 draws, but the protocol does not specify the exact method for handling degenerate cases (e.g., all-zero resamples). This could lead to differences in the interpretation of the results.",
      "evidence": "In `analyze.py`, the script uses `draws = np.random.default_rng(20261117).integers(0, len(groups), (10000, len(groups)))` for bootstrapping, but the protocol does not specify how to handle degenerate cases."
    }
  ],
  "sampler_python": {
    "json_string": "def sample(desc, query_xy):\n    h, w, d = desc.shape\n    n = query_xy.shape[0]\n    xy = np.zeros((n, 2), dtype=np.float32)\n    for i in range(n):\n        x, y = query_xy[i]\n        x_floor = int(np.floor(x))\n        y_floor = int(np.floor(y))\n        x_frac = x - x_floor\n        y_frac = y - y_floor\n        x0 = max(0, min(x_floor, w - 1))\n        x1 = max(0, min(x_floor + 1, w - 1))\n        y0 = max(0, min(y_floor, h - 1))\n        y1 = max(0, min(y_floor + 1, h - 1))\n        weights = np.array([\n            (1 - x_frac) * (1 - y_frac),\n            x_frac * (1 - y_frac),\n            (1 - x_frac) * y_frac,\n            x_frac * y_frac\n        ])\n        indices = np.array([\n            y0 * w + x0,\n            y0 * w + x1,\n            y1 * w + x0,\n            y1 * w + x1\n        ])\n        valid = indices < (h * w)\n        indices = indices[valid]\n        weights = weights[valid]\n        if len(indices) == 0:\n            xy[i] = np.nan\n        else:\n            values = desc.reshape(-1, d)[indices]\n            weighted_sum = np.sum(values * weights[:, np.newaxis], axis=0)\n            xy[i] = weighted_sum\n    return xy"
  }
}