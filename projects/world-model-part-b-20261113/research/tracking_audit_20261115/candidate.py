"""Restricted static-background observation prototype; no trajectory completion."""
import cv2
import numpy as np


def marker_positions(frame, references):
    red, green, blue = [frame[..., index].astype(float) for index in range(3)]
    binary = np.uint8((green > 1.6*red) & (green > 1.6*blue) & (green > 100))
    count, _, stats, centers = cv2.connectedComponentsWithStats(binary)
    ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 30]
    centers = centers[ids]
    distances = np.linalg.norm(np.asarray(references)[:, None]-centers[None], axis=2)
    candidates = distances <= 12
    if not np.all(candidates.sum(axis=1) == 1):
        return None
    chosen = candidates.argmax(axis=1)
    if len(set(chosen)) != len(references):
        return None
    return centers[chosen]


def track_observed(frames, setup, initial_image):
    background = np.median(initial_image, axis=1, keepdims=True)
    apparatus_mask = np.zeros(initial_image.shape[:2], dtype=np.uint8)
    padding = setup['fiducial_half_width'] + 6
    for x, y in setup['fiducial_centers']:
        cv2.rectangle(apparatus_mask, (x-padding, y-padding), (x+padding, y+padding), 1, -1)
    previous = np.array(setup['initial_centers'], dtype=float)
    missing = np.zeros(2, dtype=int)
    terminated = np.zeros(2, dtype=bool)
    rows = []
    for frame in frames:
        foreground = np.uint8(np.max(np.abs(frame.astype(float)-background), axis=2) > 20)
        foreground[apparatus_mask.astype(bool)] = 0
        count, _, stats, centers = cv2.connectedComponentsWithStats(foreground)
        ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 40]
        centers, stats = centers[ids], stats[ids]
        distances = np.linalg.norm(previous[:, None]-centers[None], axis=2)
        candidates = distances <= 4*setup['radius']
        candidates[terminated] = False
        conflicts = candidates.sum(axis=0) > 1
        objects = []
        for index in range(2):
            matches = np.flatnonzero(candidates[index])
            item = {'observed': False, 'eligible': False, 'center': None, 'area_ratio': None}
            if terminated[index]:
                item['reason'] = 'identity_terminated'
            elif len(matches) > 1 or (len(matches) == 1 and conflicts[matches[0]]):
                terminated[index] = True
                item['reason'] = 'ambiguous_identity'
            elif not len(matches):
                missing[index] += 1
                terminated[index] = missing[index] > 2
                item['reason'] = 'no_observed_component'
            else:
                selected = matches[0]
                x, y, width, height, area = stats[selected]
                previous[index] = centers[selected]
                missing[index] = 0
                complete = x > 0 and y > 0 and x+width < frame.shape[1] and y+height < frame.shape[0]
                ratio = area/setup['expected_areas'][index]
                eligible = complete and .75 <= ratio <= 1.25
                item.update(observed=True, eligible=bool(eligible), center=previous[index].tolist(), area_ratio=float(ratio),
                            reason='usable' if eligible else ('truncated' if not complete else 'area_out_of_range'))
            objects.append(item)
        rows.append(objects)
    return rows
