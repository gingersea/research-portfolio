"""Detector validation gate.

Renders synthetic ground-truth animations, runs the full detector pipeline
(track + fit), and compares recovered physics parameters against truth.
The gate must pass before the detector is trusted on generated video.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from physcheck.detect import synthetic
from physcheck.detect.fit import G_TRUE, compute_pixels_per_meter, fit_trajectory
from physcheck.detect.tracker import track_object

#: Tolerance rules per kind: (param, truth source, max relative error).
GATE_RULES = {
    "free_fall": [("g", "truth", 0.10)],
    "projectile": [("g", "truth", 0.10), ("v0x", "truth", 0.15)],
    "pendulum": [("T", "analytic", 0.10)],
    "horizontal_motion": [("v0", "truth", 0.15), ("decel", "truth", 0.50)],
}


def _truth_value(kind: str, param: str, ground_truth: dict) -> float:
    params = ground_truth["params"]
    if param == "T":
        L = params["length_m"]
        return 2 * math.pi * math.sqrt(L / G_TRUE)
    # Parameter naming: synthetic scenes use v0x for the initial velocity
    if param == "v0" and "v0x" in params:
        return float(params["v0x"])
    if param in params:
        return float(params[param])
    raise KeyError(f"truth for {param} not available in {kind}")


def validate_detector(cfg) -> dict:
    """Run the detector on all synthetic scenes and score against truth.

    Returns {"passed": bool, "results": [...], "failures": [...]}.
    """
    results = []
    failures = []
    for kind in GATE_RULES:
        scene = synthetic.render_scene(kind, cfg)
        ground_truth = scene["ground_truth"]
        track = track_object(scene["video_path"], fps=ground_truth["fps"], cfg=cfg)
        if track["status"] != "ok":
            failures.append(f"{kind}: track failed ({track.get('reason')})")
            continue
        # Pixel scale: synthetic uses a fixed constant; recover it from the
        # tracked ball diameter for a true end-to-end test.
        try:
            ppm = compute_pixels_per_meter(
                track, {"object_diameter_m": ground_truth["params"]["diameter_m"]}
            )
        except (KeyError, ValueError) as e:
            failures.append(f"{kind}: calibration failed ({e})")
            continue
        fit = fit_trajectory(kind, track["trajectory_px"], track["times_s"], ppm, {})
        if fit["status"] != "ok":
            failures.append(f"{kind}: fit failed ({fit.get('reason')})")
            continue
        for param, src, tol in GATE_RULES[kind]:
            true = _truth_value(kind, param, ground_truth)
            est = fit["params"][param]
            rel_err = abs(est - true) / true
            results.append(
                {
                    "kind": kind,
                    "param": param,
                    "true": true,
                    "estimated": est,
                    "rel_err": rel_err,
                }
            )
            if rel_err > tol:
                failures.append(
                    f"{kind}.{param}: rel_err {rel_err:.3f} > {tol} "
                    f"(true {true:.3f}, est {est:.3f})"
                )

    passed = len(failures) == 0 and len(results) > 0
    report = {"passed": passed, "results": results, "failures": failures}
    out = Path(cfg.artifacts_dir) / "detector_validation.json"
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
    from physcheck.config import Config

    cfg = Config.load()
    report = validate_detector(cfg)
    print(json.dumps(report, indent=2))
