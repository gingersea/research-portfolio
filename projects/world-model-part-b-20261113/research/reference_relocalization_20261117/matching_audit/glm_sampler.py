import numpy as np

def sample(desc, query_xy):
    """Bilinear sampler with border clamping, matching descriptor.py logic."""
    desc = np.asarray(desc)
    query_xy = np.asarray(query_xy)
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError('query_xy must be Nx2')
    h, w, d = desc.shape
    N = query_xy.shape[0]
    
    # Normalize to [-1, 1] using align_corners=True logic
    # grid_sample uses (x - 0.5) / (W - 1) for align_corners=True
    # We replicate this normalization.
    x_norm = (query_xy[:, 0] - 0.5) / (w - 1)
    y_norm = (query_xy[:, 1] - 0.5) / (h - 1)
    grid = np.stack((x_norm, y_norm), axis=-1)
    
    # Use torch for grid_sample to ensure exact match with descriptor.py
    import torch
    desc_t = torch.from_numpy(desc).permute(2, 0, 1).float()
    grid_t = torch.from_numpy(grid).float()
    
    # padding_mode='border' and align_corners=True
    sampled = torch.nn.functional.grid_sample(
        desc_t[None], grid_t[None], align_corners=True, padding_mode='border'
    )[0, :, 0].T.numpy()
    
    return sampled
