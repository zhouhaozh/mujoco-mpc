"""Fit a flight prefix without contact or future observations as inputs.

This is an offline-at-cut inference method: it may revisit every observation
already received, but never reads rows after ``cut_t_s``.  Shared aerodynamic
parameters and priors must be frozen from training by the caller.  The state
is [p(m), v(m/s), b_eff(1/s)], not measured spin.  Its covariance is a local
linearized approximation and is not a calibrated confidence interval.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from . import identified_dynamics as dynamics
from .flight import initial_guess
from .identified_filter import landing_batch


DEFAULT_CONFIG = {
    "window_s": 0.0,
    "representative_points": 72,
    "minimum_samples": 10,
    "minimum_span_s": 0.040,
    "measurement_sigma_xyz": [0.006, 0.006, 0.008],
    "b_prior_sigma_per_s": 0.04149060165072289,
    "b_limit_per_s": 2.0,
    "loss_scale_sigma": 1.0,
    "max_nfev": 100,
    "max_step_s": 0.004,
    "maximum_initial_speed_m_s": 150.0,
}


def _configuration(config: dict[str, Any] | None) -> dict[str, Any]:
    out = {**DEFAULT_CONFIG}
    if config:
        unknown = set(config) - set(out)
        if unknown:
            raise ValueError(f"Unknown prefix configuration: {sorted(unknown)}")
        out.update(config)
    for key, value in out.items():
        if key == "measurement_sigma_xyz":
            continue
        if not np.isfinite(value) or value < 0 or (key != "window_s" and value == 0):
            raise ValueError(f"{key} must be finite and {'nonnegative' if key == 'window_s' else 'positive'}")
    for key in ("representative_points", "minimum_samples", "max_nfev"):
        if int(out[key]) != out[key]:
            raise ValueError(f"{key} must be an integer")
        out[key] = int(out[key])
    if out["representative_points"] < out["minimum_samples"]:
        raise ValueError("representative_points is smaller than minimum_samples")
    sigma = np.asarray(out["measurement_sigma_xyz"], dtype=float)
    if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("measurement_sigma_xyz must be three positive finite numbers")
    out["measurement_sigma_xyz"] = sigma
    return out


def _prefix(raw_times, raw_xyz, cut_t_s: float) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    times, xyz = np.asarray(raw_times, dtype=float), np.asarray(raw_xyz, dtype=float)
    if times.ndim != 1 or xyz.shape != (len(times), 3) or not np.isfinite(cut_t_s):
        raise ValueError("times N, positions N x 3, and a finite cut timestamp required")
    # Mask before examining XYZ. Changes to any future row cannot affect fit.
    candidates = np.flatnonzero(times <= cut_t_s)
    retained, previous = [], -np.inf
    nonfinite, nonincreasing = 0, 0
    for index in candidates:
        t, pos = times[index], xyz[index]
        if not np.isfinite(t) or not np.isfinite(pos).all():
            nonfinite += 1
        elif t <= previous:
            nonincreasing += 1
        else:
            retained.append(index)
            previous = float(t)
    indices = np.asarray(retained, dtype=int)
    return times[indices], xyz[indices], {
        "raw_rows": int(len(times)), "rows_at_or_before_cut": int(len(candidates)),
        "nonfinite_prefix_rows": nonfinite, "nonincreasing_prefix_rows": nonincreasing,
        "finite_monotonic_prefix_rows": int(len(indices)),
    }


def _mixed_loss(position_rows: int):
    """Robust position likelihood, Gaussian b prior (not robustified)."""
    def loss(z: np.ndarray) -> np.ndarray:
        out = np.empty((3, len(z)))
        root = np.sqrt(1 + z[:position_rows])
        out[0, :position_rows] = 2 * (root - 1)
        out[1, :position_rows] = 1 / root
        out[2, :position_rows] = -0.5 / root**3
        out[0, position_rows:] = z[position_rows:]
        out[1, position_rows:] = 1
        out[2, position_rows:] = 0
        return out
    return loss


def estimate_prefix(raw_times, raw_xyz, cut_t_s: float,
                    parameters: dict[str, Any], contact_height: float,
                    config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the flight state at the last received timestamp, using only prefix.

    ``cut_t_s`` is the requested observation cutoff, not a contact time or a
    supplied missing duration.  The resulting ``capture_t_s`` can be earlier
    because observations are discrete or missing.  ``contact_height`` is used
    only as fixed metadata; no future event enters the fit. Insufficient input
    returns a structured nonconverged result instead of invented state values.
    """
    started = perf_counter()
    cfg = _configuration(config)
    k_drag = float(parameters["k_drag"])
    if not np.isfinite(k_drag) or k_drag < 0 or not np.isfinite(contact_height):
        raise ValueError("Finite nonnegative drag and finite contact height required")
    times, xyz, counts = _prefix(raw_times, raw_xyz, float(cut_t_s))
    if len(times) and cfg["window_s"] > 0:
        included = times >= times[-1] - cfg["window_s"]
        times, xyz = times[included], xyz[included]
    span = float(np.ptp(times)) if len(times) else 0.0
    diagnostics: dict[str, Any] = {
        **counts, "fit_rows_available": int(len(times)), "fit_span_s": span,
        "window_s": float(cfg["window_s"]), "cut_t_s": float(cut_t_s),
        "uses_post_contact_observations": False,
        "b_interpretation": "effective Magnus vector; not measured spin",
        "uncertainty_interpretation": "local linearized approximation, not calibrated coverage",
    }
    if len(times) < cfg["minimum_samples"] or span < cfg["minimum_span_s"]:
        diagnostics.update(reason="insufficient_prefix", fit_latency_ms=(perf_counter()-started)*1000)
        return {"capture_t_s": float(times[-1]) if len(times) else None,
                "state": None, "covariance": None, "converged": False,
                "diagnostics": diagnostics}
    sample = np.unique(np.linspace(0, len(times)-1,
                                 min(cfg["representative_points"], len(times))).astype(int))
    fit_t, fit_p = times[sample] - times[0], xyz[sample]
    sigma = cfg["measurement_sigma_xyz"]
    x0 = np.r_[initial_guess(times, xyz), np.zeros(3)]
    speed_bound, b_bound = cfg["maximum_initial_speed_m_s"], cfg["b_limit_per_s"]
    x0[3:6] = np.clip(x0[3:6], -0.99*speed_bound, 0.99*speed_bound)
    lower = np.r_[np.full(3, -np.inf), np.full(3, -speed_bound), np.full(3, -b_bound)]
    upper = -lower
    position_rows = len(fit_t) * 3
    cached_x, cached_residual, cached_jac = None, None, None

    def evaluate(x: np.ndarray) -> None:
        nonlocal cached_x, cached_residual, cached_jac
        if cached_x is not None and np.array_equal(x, cached_x):
            return
        prediction, jac = dynamics.simulate_batch_with_jacobian(
            fit_t[None], x[None], k_drag, cfg["max_step_s"])
        prior_jac = np.zeros((3, 9))
        prior_jac[:, 6:9] = np.eye(3) / cfg["b_prior_sigma_per_s"]
        cached_residual = np.r_[((prediction[0]-fit_p)/sigma).ravel(),
                                x[6:9] / cfg["b_prior_sigma_per_s"]]
        cached_jac = np.vstack(((jac[0, :, :, :9]/sigma[None, :, None]).reshape(-1, 9), prior_jac))
        cached_x = x.copy()

    def residual(x: np.ndarray) -> np.ndarray:
        evaluate(x)
        return cached_residual.copy()

    def jacobian(x: np.ndarray) -> np.ndarray:
        evaluate(x)
        # scipy robust-loss scaling modifies its Jacobian in-place.
        return cached_jac.copy()

    fit = least_squares(residual, x0, jac=jacobian, bounds=(lower, upper),
                        loss=_mixed_loss(position_rows), f_scale=cfg["loss_scale_sigma"],
                        max_nfev=cfg["max_nfev"], x_scale="jac",
                        xtol=1e-8, ftol=1e-8, gtol=1e-7)
    evaluate(fit.x)
    initial_cov = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-10)
    state, transition = dynamics.step_with_jacobian(fit.x, span, k_drag, cfg["max_step_s"])
    covariance = transition @ initial_cov @ transition.T
    covariance = (covariance + covariance.T) / 2
    sampled_error = (cached_residual[:position_rows].reshape(-1, 3) * sigma)
    errors = np.linalg.norm(sampled_error, axis=1)
    position_jac = cached_jac[:position_rows].copy()
    sensitivity = dynamics.jacobian_diagnostics(position_jac)
    diagnostics.update({
        "reason": "fit_success" if fit.success else "optimizer_not_converged",
        "fit_start_t_s": float(times[0]), "capture_t_s": float(times[-1]),
        "sampled_rows": int(len(sample)), "nfev": int(fit.nfev),
        "optimizer_status": int(fit.status), "optimizer_message": str(fit.message),
        "cost": float(fit.cost), "optimality": float(fit.optimality),
        "sampled_position_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "sampled_position_median_m": float(np.median(errors)),
        "sampled_position_p95_m": float(np.quantile(errors, .95)),
        "robust_row_weight_min": float(np.min(1 / np.sqrt(1 +
            (cached_residual[:position_rows] / cfg["loss_scale_sigma"])**2))),
        "prior_sigma_b_per_s": float(cfg["b_prior_sigma_per_s"]),
        "b_bound_active": bool(np.any(abs(fit.x[6:]) >= .999*b_bound)),
        "sensitivity_without_prior": sensitivity,
        "fit_latency_ms": (perf_counter()-started)*1000,
    })
    valid = bool(fit.success and np.isfinite(state).all() and np.isfinite(covariance).all())
    return {"capture_t_s": float(times[-1]), "state": state, "covariance": covariance,
            "converged": valid, "initial_state": fit.x.copy(), "initial_t_s": float(times[0]),
            "diagnostics": diagnostics}


def predict_contact(estimate: dict[str, Any], parameters: dict[str, Any],
                    contact_height: float, *, max_horizon_s: float = 3.0,
                    step_s: float = 0.004) -> dict[str, Any]:
    """Predict first contact using the last prefix state only; no labels input."""
    state, capture = estimate.get("state"), estimate.get("capture_t_s")
    if state is None or capture is None or not estimate.get("converged", False):
        return {"valid": False, "reason": "prefix_not_converged"}
    state = np.asarray(state, dtype=float)
    result = landing_batch(state[None], np.array([False]), parameters, contact_height,
                           max_horizon_s=max_horizon_s, step_s=step_s)
    if not result["valid"][0]:
        return {"valid": False, "reason": str(result["reason"][0])}
    delta = float(result["time_to_contact"][0])
    at_contact = dynamics.step(state, delta, float(parameters["k_drag"]), step_s)
    at_contact[:3], at_contact[3:6] = result["position"][0], result["velocity"][0]
    return {"valid": True, "reason": str(result["reason"][0]),
            "capture_t_s": float(capture), "time_to_contact_s": delta,
            "contact_t_s": float(capture) + delta,
            "position": at_contact[:3].copy(), "v_minus": at_contact[3:6].copy(),
            "b_minus": at_contact[6:9].copy(), "state_minus": at_contact}
