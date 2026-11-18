"""Physics parameter fitting from tracked trajectories.

Fits closed-form kinematics to pixel trajectories and converts to physical
units via the pixel-to-meter scale. Fit-quality thresholds (R^2 and pixel
RMS) reject trajectories that do not actually follow the assumed physics —
a "confident wrong number" is worse than a reported failure.
"""

from __future__ import annotations

import math

import numpy as np

#: Fit-quality thresholds. Fits below these are reported as fit_failed.
MIN_R2 = 0.6
MAX_RESIDUAL_RMS_PX = 3.0

#: Physical references for relative-error computation.
G_TRUE = 9.8


def compute_pixels_per_meter(track_result: dict, calib: dict) -> float:
    """Pixel-to-meter scale from the known object diameter.

    Uses the median measured pixel diameter of the tracked component
    divided by the stated physical diameter in the prompt's calibration.
    """
    diameters_px = track_result.get("diameters_px") or []
    if not diameters_px:
        raise ValueError("tracker returned no diameters; cannot calibrate")
    widths = np.asarray(track_result.get("widths_px", diameters_px))
    heights = np.asarray(track_result.get("heights_px", diameters_px))
    aspect = np.maximum(widths, heights) / np.minimum(widths, heights)
    if np.quantile(aspect, 0.8) > 1.25:
        raise ValueError("unreliable diameter calibration: elongated/blurred ball")
    if np.quantile(diameters_px, 0.9) / np.quantile(diameters_px, 0.1) > 1.35:
        raise ValueError("unreliable diameter calibration: changing apparent size")
    diameter_px = float(np.median(diameters_px))
    diameter_m = float(calib["object_diameter_m"])
    if diameter_px <= 0 or diameter_m <= 0:
        raise ValueError("invalid calibration values")
    return diameter_px / diameter_m


def _check_quality(x: np.ndarray, y: np.ndarray, y_fit: np.ndarray) -> dict | None:
    """Return a fit_failed result dict if quality thresholds are violated."""
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rms_px = float(np.sqrt(ss_res / len(y)))
    if r2 < MIN_R2 or rms_px > MAX_RESIDUAL_RMS_PX:
        return {
            "status": "fit_failed",
            "reason": f"R2={r2:.3f}, RMS_px={rms_px:.2f} (thresholds "
                      f"{MIN_R2}, {MAX_RESIDUAL_RMS_PX})",
            "params": {},
            "params_std": None,
        }
    return None


def _lstsq(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit and per-parameter standard errors."""
    A = np.column_stack([x, np.ones_like(x)])
    coef, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    n = len(y)
    dof = n - len(coef)
    if dof <= 0:
        return coef, np.full_like(coef, np.nan)
    s2 = float(np.sum((y - A @ coef) ** 2)) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    std = np.sqrt(np.diag(cov))
    return coef, std


def fit_trajectory(
    kind: str,
    trajectory_px: list[tuple[float, float]],
    times_s: list[float],
    pixels_per_meter: float,
    calib: dict,
) -> dict:
    """Fit physics parameters to a pixel trajectory.

    Returns {"status": "ok"|"fit_failed", "reason", "params", "params_std"}.
    Params per kind:
      free_fall:        {"g": float}
      projectile:       {"g": float, "v0x": float}
      pendulum:         {"T": float}
      horizontal_motion: {"v0": float, "decel": float}
    """
    if len(trajectory_px) < 5:
        return {
            "status": "fit_failed",
            "reason": "too few trajectory points",
            "params": {},
            "params_std": None,
        }
    xs = np.array([p[0] for p in trajectory_px])
    ys = np.array([p[1] for p in trajectory_px])
    t = np.array(times_s)
    ppm = pixels_per_meter

    if kind == "free_fall":
        # y(t) = y0 + v0 t + 0.5 g t^2  (screen y grows downward)
        A = np.column_stack([0.5 * t**2, t, np.ones_like(t)])
        coef, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        g_px = coef[0]  # px/s^2
        y_fit = A @ coef
        fail = _check_quality(xs, ys, y_fit)
        if fail:
            return fail
        # Standard error from residual variance
        n = len(t)
        dof = n - 3
        resid = ys - y_fit
        s2 = float(np.sum(resid**2)) / max(dof, 1)
        cov = s2 * np.linalg.inv(A.T @ A)
        g_std = float(np.sqrt(cov[0, 0])) / ppm
        return {
            "status": "ok",
            "reason": None,
            "params": {"g": float(g_px / ppm)},
            "params_std": {"g": g_std},
        }

    if kind == "projectile":
        # x(t) = x0 + v0x t ; y(t) = y0 + v0y t + 0.5 g t^2
        vx, vx_std = _lstsq(t, xs)
        fail = _check_quality(xs, xs, vx[0] * t + vx[1])
        if fail:
            return fail
        A = np.column_stack([0.5 * t**2, t, np.ones_like(t)])
        coef, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        y_fit = A @ coef
        fail = _check_quality(xs, ys, y_fit)
        if fail:
            return fail
        n = len(t)
        resid = ys - y_fit
        s2 = float(np.sum(resid**2)) / max(n - 3, 1)
        cov = s2 * np.linalg.inv(A.T @ A)
        g_std = float(np.sqrt(cov[0, 0])) / ppm
        return {
            "status": "ok",
            "reason": None,
            "params": {
                "g": float(coef[0] / ppm),
                "v0x": float(vx[0] / ppm),
            },
            "params_std": {"g": g_std, "v0x": float(vx_std[0] / ppm)},
        }

    if kind == "pendulum":
        # x(t) = x0 + A sin(w t + phi); fit frequency via linearized search.
        # Initial frequency guess from the FFT peak of the mean-subtracted
        # signal — far more robust than a fixed guess (w ~ sqrt(g/L) in rad/s
        # varies by an order of magnitude across videos).
        from scipy.optimize import curve_fit

        def model(t, x0, A, w, phi):
            return x0 + A * np.sin(w * t + phi)

        try:
            dt = np.median(np.diff(t))
            xc = xs - xs.mean()
            freqs = np.fft.rfftfreq(len(t), d=dt)
            spec = np.abs(np.fft.rfft(xc))
            # Skip DC bin
            peak = np.argmax(spec[1:]) + 1
            w0 = 2 * np.pi * freqs[peak]
            if not (0.1 < w0 < 50):
                w0 = 1.0
            A0 = max((xs.max() - xs.min()) / 2, 1.0)
            popt, pcov = curve_fit(
                model, t, xs, p0=[xs.mean(), A0, w0, 0.0], maxfev=100000
            )
        except Exception:
            return {
                "status": "fit_failed",
                "reason": "pendulum fit did not converge",
                "params": {},
                "params_std": None,
            }
        x_fit = model(t, *popt)
        fail = _check_quality(xs, xs, x_fit)
        if fail:
            return fail
        T = 2 * math.pi / abs(popt[2])
        T_std = 2 * math.pi * math.sqrt(np.diag(pcov)[2]) / popt[2] ** 2
        if not np.isfinite(T_std) or (t[-1] - t[0]) / T < 0.9 or T_std / T > 0.25:
            return {"status": "fit_failed", "reason": "period unidentifiable: insufficient oscillation coverage or uncertainty", "params": {}, "params_std": None}
        return {
            "status": "ok",
            "reason": None,
            "params": {"T": float(T)},
            "params_std": {"T": float(T_std)},
        }

    if kind == "horizontal_motion":
        # x(t) = x0 + v0 t - 0.5 decel t^2, valid only until stop
        A = np.column_stack([t, -0.5 * t**2, np.ones_like(t)])
        coef, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
        x_fit = A @ coef
        # Compare a continuous move-then-stop model with the unconstrained
        # quadratic. A stationary tail must not dilute initial speed/deceleration.
        from scipy.optimize import curve_fit

        def stopping(t, v0, decel, x0):
            tau = np.minimum(t, v0 / decel)
            return x0 + v0 * tau - 0.5 * decel * tau**2

        try:
            delta = t - t[0]
            v_guess = max(float((xs[4] - xs[0]) / (t[4] - t[0])), 1.0)
            p, covariance = curve_fit(
                stopping, delta, xs, p0=[v_guess, max(v_guess, 1.0), xs[0]],
                bounds=([0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]), maxfev=10000,
            )
            stopped_fit = stopping(delta, *p)
            if p[0] / p[1] < delta[-1] and np.sum((xs - stopped_fit)**2) < np.sum((xs - x_fit)**2):
                failure = _check_quality(xs, xs, stopped_fit)
                if failure:
                    return failure
                errors = np.sqrt(np.diag(covariance)) / ppm
                if not np.all(np.isfinite(errors)) or np.count_nonzero(delta < p[0]/p[1]) < 5:
                    return {"status": "fit_failed", "reason": "stop dynamics underresolved", "params": {}, "params_std": None}
                return {
                    "status": "ok", "reason": None,
                    "params": {"v0": float(p[0] / ppm), "decel": float(p[1] / ppm)},
                    "params_std": {"v0": float(errors[0]), "decel": float(errors[1])},
                }
        except (RuntimeError, ValueError):
            pass  # Quadratic remains applicable to motion without a stop.
        fail = _check_quality(xs, xs, x_fit)
        if fail:
            return fail
        n = len(t)
        resid = xs - x_fit
        s2 = float(np.sum(resid**2)) / max(n - 3, 1)
        cov = s2 * np.linalg.inv(A.T @ A)
        return {
            "status": "ok",
            "reason": None,
            "params": {
                "v0": float(coef[0] / ppm),
                "decel": float(coef[1] / ppm),
            },
            "params_std": {
                "v0": float(np.sqrt(cov[0, 0]) / ppm),
                "decel": float(np.sqrt(cov[1, 1]) / ppm),
            },
        }

    raise ValueError(f"unknown kind {kind}")
