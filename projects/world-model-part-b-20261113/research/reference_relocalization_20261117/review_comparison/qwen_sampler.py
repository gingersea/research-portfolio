def sample(desc, query_xy):
    h, w, d = desc.shape
    n = query_xy.shape[0]
    xy = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        x, y = query_xy[i]
        x_floor = int(np.floor(x))
        y_floor = int(np.floor(y))
        x_frac = x - x_floor
        y_frac = y - y_floor
        x0 = max(0, min(x_floor, w - 1))
        x1 = max(0, min(x_floor + 1, w - 1))
        y0 = max(0, min(y_floor, h - 1))
        y1 = max(0, min(y_floor + 1, h - 1))
        weights = np.array([
            (1 - x_frac) * (1 - y_frac),
            x_frac * (1 - y_frac),
            (1 - x_frac) * y_frac,
            x_frac * y_frac
        ])
        indices = np.array([
            y0 * w + x0,
            y0 * w + x1,
            y1 * w + x0,
            y1 * w + x1
        ])
        valid = indices < (h * w)
        indices = indices[valid]
        weights = weights[valid]
        if len(indices) == 0:
            xy[i] = np.nan
        else:
            values = desc.reshape(-1, d)[indices]
            weighted_sum = np.sum(values * weights[:, np.newaxis], axis=0)
            xy[i] = weighted_sum
    return xy
