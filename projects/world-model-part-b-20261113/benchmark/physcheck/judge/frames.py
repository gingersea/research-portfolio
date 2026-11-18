"""Frame sampler for Part B judging.

Extracts evenly spaced frames from a video and composes an annotated grid
image (each cell labeled with its frame index) for the VLM judge. The
judge sees static frames only — time dynamics belong to Part A.
"""

from __future__ import annotations

import cv2
import imageio.v3 as iio3
import numpy as np

from physcheck.config import Config


def _read_fps(video_path: str) -> float:
    """Read fps from video metadata; raise if absent (never guess)."""
    meta = iio3.immeta(video_path, plugin="pyav")
    fps = meta.get("fps") if isinstance(meta, dict) else None
    if fps is None:
        # fall back to container average rate
        import av

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
    if not fps:
        raise ValueError(f"video {video_path} has no fps metadata")
    return float(fps)


def sample_frames(video_path: str, n_frames: int = 8, cfg: Config | None = None, video_id: str | None = None) -> dict:
    """Sample n_frames evenly spaced frames and compose a labeled grid.

    Returns {"frame_paths": [...], "grid_path": str, "fps": float,
             "n_total": int}.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    fps = _read_fps(video_path)
    frames = iio3.imread(video_path)  # (T, H, W, 3)
    n_total = len(frames)
    if n_total == 0:
        raise ValueError("video contains no frames")
    if n_total < n_frames:
        n_frames = n_total
    indices = [0] if n_frames == 1 else [round(i * (n_total - 1) / (n_frames - 1)) for i in range(n_frames)]
    indices = sorted(set(indices))

    out_dir = (
        (cfg.artifacts_dir / "frames" / (video_id or "anon"))
        if cfg
        else None
    )
    frame_paths = []
    for idx in indices:
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"frame_{idx:04d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
            frame_paths.append(str(path))
        else:
            frame_paths.append(f"frame_{idx:04d}.png")

    # Compose grid (2 rows for 8 frames, or ceil(n/4) rows of 4)
    n_cols = 4
    n_rows = int(np.ceil(len(indices) / n_cols))
    cell_h, cell_w = 180, 320  # scaled cell size
    grid = np.full((n_rows * (cell_h + 20), n_cols * (cell_w + 10), 3), 255, dtype=np.uint8)
    for k, idx in enumerate(indices):
        r, c = divmod(k, n_cols)
        # imageio decodes RGB; OpenCV's grid writer and labels use BGR.
        cell = cv2.cvtColor(cv2.resize(frames[idx], (cell_w, cell_h)), cv2.COLOR_RGB2BGR)
        y0, x0 = r * (cell_h + 20), c * (cell_w + 10)
        grid[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell
        label = f"frame {idx}/{n_total - 1}"
        cv2.putText(grid, label, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    grid_path = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_path = str(out_dir / "grid.png")
        cv2.imwrite(grid_path, grid)

    return {
        "frame_paths": frame_paths,
        "grid_path": grid_path,
        "fps": fps,
        "n_total": n_total,
        "indices": indices,
    }
