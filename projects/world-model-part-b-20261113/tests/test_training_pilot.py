"""Training-pilot checks: renderer/labels and actual gradient-based learning."""
import numpy as np
import torch

from training_pilot import TinyUNet, make_dataset, segmentation_loss, mask_metrics


def test_synthetic_masks_match_geometric_labels_and_seed_is_reproducible():
    x, y, centers = make_dataset(8, seed=101, size=64)
    x2, y2, _ = make_dataset(8, seed=101, size=64)
    assert x.shape == (8, 3, 64, 64) and y.shape == (8, 1, 64, 64)
    assert torch.equal(x, x2) and torch.equal(y, y2)
    for mask, (cx, cy) in zip(y[:, 0].numpy(), centers):
        ys, xs = np.nonzero(mask)
        assert abs(xs.mean() - cx) < 0.1 and abs(ys.mean() - cy) < 0.1
        assert mask[round(cy), round(cx)] == 1
    assert mask_metrics(y, y)["mean_iou"] == 1.0
    assert mask_metrics(y, y)["centroid_error_px"] == 0.0


def test_tiny_unet_backpropagates_and_empty_masks_are_penalized():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = TinyUNet()
    x, y, _ = make_dataset(4, seed=102, size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model(x)
    loss = segmentation_loss(before, y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    after = model(x)
    assert after.shape == y.shape and not torch.equal(after, before)
    failed = mask_metrics(torch.zeros_like(y), y)
    assert failed["mean_iou"] == 0 and failed["detection_rate"] == 0
    assert failed["centroid_error_px"] > 32
