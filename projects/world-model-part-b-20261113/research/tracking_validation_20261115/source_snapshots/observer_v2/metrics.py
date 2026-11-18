"""Fixed point metrics, with query frames, visibility and identity kept separate."""
import numpy as np


def score_case(xy, declared, truth_xy, visible, identifiable):
    xy, truth_xy = np.asarray(xy), np.asarray(truth_xy)
    declared, visible, identifiable = [np.asarray(item, dtype=bool) for item in [declared, visible, identifiable]]
    assert xy.shape == truth_xy.shape and xy.shape[:-1] == declared.shape == visible.shape == identifiable.shape
    evaluate = np.ones_like(visible)
    evaluate[0] = False
    known = evaluate & identifiable
    positive = known & visible
    negative = known & ~visible
    finite = np.isfinite(xy).all(axis=-1)
    error = np.linalg.norm(np.nan_to_num(xy, nan=1e9)-truth_xy, axis=-1)
    error[~finite] = np.inf
    declared_known = declared & known
    tp = int((declared_known & visible).sum())
    fp = int((declared_known & ~visible).sum())
    ambiguous = evaluate & ~identifiable
    wrong, transitions = 0, 0
    previous = list(range(xy.shape[1]))
    for t in range(1, len(xy)):
        for query in range(xy.shape[1]):
            if not identifiable[t, query]:
                previous[query] = None
                continue
            if not declared[t, query] or not finite[t, query]:
                continue
            distances = np.linalg.norm(truth_xy[t]-xy[t, query], axis=1)
            distances[~visible[t]] = np.inf
            matches = np.flatnonzero(distances <= 5)
            if not len(matches):
                continue
            ordered = matches[np.argsort(distances[matches])]
            if len(ordered) > 1 and np.isclose(distances[ordered[0]], distances[ordered[1]], atol=1e-6):
                continue
            match = int(ordered[0])
            wrong += match != query
            if previous[query] is not None and previous[query] != match:
                transitions += 1
            previous[query] = match
    def fraction(a, b):
        return float(a/b) if b else None
    visible_finite_errors = error[positive & finite]
    declared_visible_errors = error[positive & declared & finite]
    return {'evaluated_samples': int(evaluate.sum()), 'identifiable_samples': int(known.sum()),
            'visible_gt_count': int(positive.sum()), 'invisible_gt_count': int(negative.sum()),
            'declared_identifiable_count': int(declared_known.sum()), 'true_visible_claim_count': tp,
            'false_visible_count': fp, 'false_visible_rate': fraction(fp, negative.sum()),
            'visible_precision': fraction(tp, declared_known.sum()), 'visible_recall': fraction(tp, positive.sum()),
            'pck3_visible_count': int((positive & (error <= 3)).sum()),
            'pck3_visible': fraction((positive & (error <= 3)).sum(), positive.sum()),
            'visible_missing_location_count': int((positive & ~finite).sum()),
            'visible_epe_finite_px': float(visible_finite_errors.mean()) if len(visible_finite_errors) else None,
            'declared_visible_epe_px': float(declared_visible_errors.mean()) if len(declared_visible_errors) else None,
            'wrong_other_count': int(wrong), 'wrong_other_claim_fraction': fraction(wrong, declared_known.sum()),
            'id_association_transitions': int(transitions), 'ambiguous_sample_count': int(ambiguous.sum()),
            'unsupported_identity_count': int((ambiguous & declared).sum()),
            'unsupported_identity_fraction': fraction((ambiguous & declared).sum(), ambiguous.sum())}


def lift(xy, depth, intrinsics, camera_centers):
    xy, depth = np.asarray(xy), np.asarray(depth)
    homogeneous = np.concatenate([xy, np.ones_like(xy[..., :1])], axis=-1)
    rays = homogeneous @ np.linalg.inv(intrinsics).T
    return rays*depth[..., None]+np.asarray(camera_centers)[:, None, :]
