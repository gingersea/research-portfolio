"""Synthetic ground-truth animation renderer.

Renders simple physics scenes with exactly known parameters so that the
detector pipeline (tracker + fitter) can be validated before it touches
generated videos. All scenes use a constant pixel-to-meter scale:

    PIXELS_PER_METER = 100.0

A ball of diameter_m meters therefore spans diameter_m * 100 pixels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio
import numpy as np

PIXELS_PER_METER = 100.0

#: Physics values used by all synthetic scenes (SI units).
G = 9.8


def _ball_mask(size, center, radius_px):
    """Boolean mask of a filled circle inside an image of shape size=(H, W)."""
    H, W = size
    y, x = np.mgrid[0:H, 0:W]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return dist <= radius_px


def _draw_frame(size, center_px, radius_px):
    frame = np.full(size + (3,), 255, dtype=np.uint8)  # white background
    mask = _ball_mask(size, center_px, radius_px)
    frame[mask] = (220, 30, 30)  # solid red ball (RGB)
    return frame


def _fall_positions(kind: str, params: dict, t: np.ndarray, size, radius_px):
    """Return ball-center positions (in pixels, y down) for each time t."""
    H, W = size
    center = []
    if kind == "free_fall":
        g, h = params["g"], params["height_m"]
        x = W / 2.0
        # Empty background frames first (tracker background reference), then
        # the ball appears and falls ballistically; clip at floor impact.
        # Free-fall duration from height h: sqrt(2h/g); the scene renders
        # with a short duration so the fall is captured (see render_scene).
        for i, ti in enumerate(t):
            if i < 4:
                continue  # empty background frames
            y_m = h - 0.5 * g * (ti - t[4]) ** 2
            if y_m <= 0.0:
                break
            center.append((x, H - 60 - y_m * PIXELS_PER_METER))
    elif kind == "projectile":
        g, v0x, v0y, h = params["g"], params["v0x"], params["v0y"], params["height_m"]
        for i, ti in enumerate(t):
            if i < 4:
                continue  # empty background frames
            trel = ti - t[4]
            x_m = v0x * trel
            y_m = h - (v0y * trel + 0.5 * g * trel**2)
            if y_m <= 0.0:
                break  # clip at floor impact: whole trajectory is ballistic
            center.append((40 + x_m * PIXELS_PER_METER, H - 60 - y_m * PIXELS_PER_METER))
    elif kind == "pendulum":
        g, L, theta0 = params["g"], params["length_m"], math.radians(params["theta0_deg"])
        pivot = (W / 2.0, 60.0)
        L_px = L * PIXELS_PER_METER
        for i, ti in enumerate(t):
            if i < 4:
                continue  # empty background frames
            trel = ti - t[4]
            theta = theta0 * math.cos(math.sqrt(g / L) * trel)
            x = pivot[0] + L_px * math.sin(theta)
            y = pivot[1] + L_px * math.cos(theta)
            center.append((x, y))
    elif kind == "horizontal_motion":
        v0, decel = params["v0x"], params["decel"]
        y = H - 60 - radius_px
        for i, ti in enumerate(t):
            if i < 4:
                continue  # empty background frames
            trel = ti - t[4]
            t_stop = v0 / decel
            tc = min(trel, t_stop)
            x_m = v0 * tc - 0.5 * decel * tc**2
            center.append((40 + x_m * PIXELS_PER_METER, y))
    else:
        raise ValueError(f"unknown kind {kind}")
    return center


def render_scene(
    kind: str,
    cfg,
    fps: int = 30,
    duration_s: float = 2.0,
    size: tuple[int, int] = (480, 480),
) -> dict:
    """Render a synthetic scene and write mp4 + ground_truth.json.

    Returns {"video_path": str, "ground_truth": {"kind", "params"}}.
    """
    scenes = {
        "free_fall": {"g": G, "diameter_m": 0.1, "height_m": 2.0},
        "projectile": {"g": G, "v0x": 3.0, "v0y": 0.0, "height_m": 1.5, "diameter_m": 0.1},
        "pendulum": {"g": G, "length_m": 1.0, "theta0_deg": 20.0, "diameter_m": 0.1},
        "horizontal_motion": {"v0x": 2.0, "decel": 0.5, "diameter_m": 0.1},
    }
    if kind not in scenes:
        raise ValueError(f"unknown kind {kind}")

    params = scenes[kind]
    out_dir = Path(cfg.artifacts_dir) / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{kind}.mp4"

    n_frames = int(fps * duration_s)
    if kind == "free_fall":
        # Render only the ballistic portion: 4 empty frames + the fall.
        t_fall = math.sqrt(2 * params["height_m"] / G)
        n_frames = 4 + int(math.ceil(fps * t_fall))
    elif kind == "projectile":
        # Ballistic portion only: 4 empty frames + the flight to the floor.
        t_fall = math.sqrt(2 * params["height_m"] / G)
        n_frames = 4 + int(math.ceil(fps * t_fall))
    t = np.arange(n_frames) / fps
    radius_px = params["diameter_m"] * PIXELS_PER_METER / 2.0

    # Center positions must stay inside the frame; compute with radius taken
    # into account via scene geometry (drawing clamps implicitly).
    centers = _fall_positions(kind, params, t, size, radius_px)

    # Render all frames. For free_fall the first 4 frames are empty
    # background (centers is shorter than the full frame count); frame
    # indices must stay aligned with the physics timeline, so pad with
    # blank frames at the start.
    blank = np.full(size + (3,), 255, dtype=np.uint8)
    pad = n_frames - len(centers)
    frames = [blank.copy() for _ in range(max(pad, 0))]
    for c in centers:
        x = min(max(c[0], radius_px + 1), size[1] - radius_px - 1)
        y = min(max(c[1], radius_px + 1), size[0] - radius_px - 1)
        frames.append(_draw_frame(size, (x, y), radius_px))
    frames = frames[:n_frames]

    # Write via PyAV with lossless H.264 (crf 0) to avoid compression ghosts
    # that would break the tracker's background subtraction on plain scenes.
    import av

    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = size[1], size[0]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "0", "preset": "veryfast"}
    for f in frames:
        frame = av.VideoFrame.from_ndarray(f, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    ground_truth = {
        "kind": kind,
        "params": params,
        "fps": fps,
        "n_frames": len(centers),
    }
    (out_dir / f"{kind}_ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2)
    )
    return {"video_path": str(video_path), "ground_truth": ground_truth}


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
    from physcheck.config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--kind", default=None)
    args = parser.parse_args()
    cfg = Config.load()
    kinds = (
        ["free_fall", "projectile", "pendulum", "horizontal_motion"]
        if args.all
        else [args.kind]
    )
    for k in kinds:
        r = render_scene(k, cfg)
        print(k, "->", r["video_path"])
