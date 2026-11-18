"""Object tracker: extract pixel trajectories from video.

Method: red foreground segmentation (the fixed Part A target), then
connected components with temporal continuity give the object's centroid
and pixel diameter. Frames without a detectable object are
skipped; if too few valid frames remain the tracker signals a detection
failure instead of returning garbage.
"""

from __future__ import annotations

import numpy as np
import imageio.v3 as iio3
import cv2

MIN_COMPONENT_AREA_PX = 20
MIN_VALID_FRAME_FRACTION = 0.5
MIN_VALID_FRAMES = 5


def track_object(video_path: str, fps: float, cfg) -> dict:
    """Track the red ball specified in the fixed Part A prompts.

    Returns:
        {"status": "ok", "trajectory_px": [(x, y), ...],
         "diameters_px": [...], "times_s": [...], "fps": float}
      or
        {"status": "detection_failure", "reason": str}
    """
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps metadata must be finite and positive")
    frames = iio3.imread(video_path)  # (T, H, W, 3) uint8
    if frames.ndim != 4:
        return {"status": "detection_failure", "reason": "cannot read video frames"}
    n_frames = len(frames)
    if n_frames < MIN_VALID_FRAMES:
        return {
            "status": "detection_failure",
            "reason": f"too few frames ({n_frames})",
        }

    trajectory_px: list[tuple[float, float]] = []
    diameters_px: list[float] = []
    widths_px: list[float] = []
    heights_px: list[float] = []
    times_s: list[float] = []

    prev_center = None  # temporal-continuity anchor (px)
    for i in range(n_frames):
        # A first-frame difference retains a stationary ghost when the model
        # does not obey "empty first frame". Color also excludes pendulum strings.
        red, green, blue = frames[i].astype(np.float32).transpose(2, 0, 1)
        mask = ((red > 80) & (red > 1.4 * green) & (red > 1.4 * blue)).astype(np.uint8) * 255

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n_labels <= 1:
            continue  # no foreground component this frame
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            continue
        # Candidate components above the minimum area
        candidates = [1 + k for k in range(len(areas)) if areas[k] >= MIN_COMPONENT_AREA_PX]
        if not candidates:
            continue
        if len(candidates) == 1:
            best = candidates[0]
        elif prev_center is not None:
            # Compression artifacts can leave ghosts: choose the component
            # closest to the previous frame's centroid (temporal continuity).
            best = min(
                candidates,
                key=lambda k: (centroids[k][0] - prev_center[0]) ** 2
                + (centroids[k][1] - prev_center[1]) ** 2,
            )
        else:
            # First detection: largest component
            best = max(candidates, key=lambda k: areas[k - 1])
        cx, cy = centroids[best]
        prev_center = (float(cx), float(cy))
        # Diameter from bounding box (max of width/height)
        w = stats[best, cv2.CC_STAT_WIDTH]
        h = stats[best, cv2.CC_STAT_HEIGHT]
        diameters_px.append(float(max(w, h)))
        widths_px.append(float(w))
        heights_px.append(float(h))
        trajectory_px.append((float(cx), float(cy)))
        times_s.append(i / fps)

    if len(trajectory_px) < MIN_VALID_FRAMES:
        return {
            "status": "detection_failure",
            "reason": f"object detected in only {len(trajectory_px)}/{n_frames} frames",
        }
    if len(trajectory_px) < MIN_VALID_FRAME_FRACTION * n_frames:
        return {
            "status": "detection_failure",
            "reason": (
                f"object present in {len(trajectory_px)}/{n_frames} frames "
                f"(< {MIN_VALID_FRAME_FRACTION:.0%})"
            ),
        }

    indices = [round(t * fps) for t in times_s]
    for j, following in enumerate(indices[1:] + [n_frames]):
        if following - indices[j] > max(2, round(0.15 * fps)):
            cx, cy = trajectory_px[j]
            margin = 2 * diameters_px[j]
            if min(cx, cy, frames.shape[2] - cx, frames.shape[1] - cy) > margin:
                return {"status": "detection_failure", "reason": "object disappears inside frame"}

    return {
        "status": "ok",
        "trajectory_px": trajectory_px,
        "diameters_px": diameters_px,
        "widths_px": widths_px,
        "heights_px": heights_px,
        "times_s": times_s,
        "fps": fps,
        "method": "red_component_temporal_continuity_v1",
        "frame_indices": indices,
        "n_total_frames": n_frames,
    }
