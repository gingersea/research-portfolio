{
  "fixtures": {
    "q=(.75,.25)": {
      "reference": "[[1,0],[0,1]]",
      "target": "[[1,0],[0,1]]",
      "query": "[[.75,.25]]",
      "expected_xy": "[[0,1]]"
    },
    "q=(1.75,1.5)": {
      "map": "2x2 one-hot identity",
      "expected_xy": "[[1,1]]"
    }
  },
  "sampler_python": "import numpy as np\n\ndef sample(desc, query_xy):\n    \"\"\"Bilinear sampler with border clamping, matching descriptor.py logic.\"\"\"\n    desc = np.asarray(desc)\n    query_xy = np.asarray(query_xy)\n    if query_xy.ndim != 2 or query_xy.shape[1] != 2:\n        raise ValueError('query_xy must be Nx2')\n    h, w, d = desc.shape\n    N = query_xy.shape[0]\n    \n    # Normalize to [-1, 1] using align_corners=True logic\n    # grid_sample uses (x - 0.5) / (W - 1) for align_corners=True\n    # We replicate this normalization.\n    x_norm = (query_xy[:, 0] - 0.5) / (w - 1)\n    y_norm = (query_xy[:, 1] - 0.5) / (h - 1)\n    grid = np.stack((x_norm, y_norm), axis=-1)\n    \n    # Use torch for grid_sample to ensure exact match with descriptor.py\n    import torch\n    desc_t = torch.from_numpy(desc).permute(2, 0, 1).float()\n    grid_t = torch.from_numpy(grid).float()\n    \n    # padding_mode='border' and align_corners=True\n    sampled = torch.nn.functional.grid_sample(\n        desc_t[None], grid_t[None], align_corners=True, padding_mode='border'\n    )[0, :, 0].T.numpy()\n    \n    return sampled",
  "findings": [
    "XY vs YX: descriptor.py uses `index%w` for x and `index//w` for y, which is correct for row-major flattening. The provided `samp.py` uses `y0_clip * dim2` (where dim2=W) for indexing, which is also correct. No discrepancy found.",
    "Pixel Centers: descriptor.py uses `align_corners=True`. The provided `samp.py` uses `x - 0.5` normalization, which is the correct implementation for `align_corners=True`. No discrepancy found.",
    "Tie Handling: descriptor.py explicitly states 'Argmax ties use the first row-major pixel'. The provided `samp.py` uses `torch.argmax`, which returns the first index in case of ties. No discrepancy found."
  ],
  "limits": {
    "max_findings": 3,
    "max_words_per_finding": 70,
    "max_total_words": 1100
  }
}