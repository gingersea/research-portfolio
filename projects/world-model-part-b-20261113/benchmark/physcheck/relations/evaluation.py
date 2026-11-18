"""Reference-only metrics preserving invalid predictions in completeness counts."""
import numpy as np


def geometry_metrics(xy, xyz, gt_xy, gt_xyz, positive):
    e3 = np.linalg.norm(xyz - gt_xyz, axis=-1)
    e2 = np.linalg.norm(xy - gt_xy, axis=-1)
    count = int(positive.sum())
    valid = positive & np.isfinite(e3)
    missing = int((positive & ~np.isfinite(e3)).sum())
    return {'raw_epe3d_m': float(e3[positive].mean()) if count and not missing else None,
            'finite_epe3d_m': float(e3[valid].mean()) if valid.any() else None,
            'valid_3d_fraction': float(valid.sum() / count) if count else None,
            'epe2d_px': float(e2[positive].mean()) if count else None,
            'visible_point_times': count, 'invalid_3d_point_times': missing}
