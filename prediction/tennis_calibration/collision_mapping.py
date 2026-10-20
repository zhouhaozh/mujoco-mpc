"""Small conditional collision residuals from strictly pre-impact features.

Inputs are pre-only estimated velocity and *effective* Magnus vector.  No event
time, missing-tail duration, post-impact positions, or joint-fit incident labels
are accepted by the feature builder.  Coordinates rotate with incident planar
velocity, so predictions commute with proper rotations about the ground normal.

Ridge and exact RBF Gaussian-process regressors are deliberately small.  The
caller supplies weights whose total is equal for each independent collision;
several truncated prefixes of one trajectory must not count as independent
events.  Parameter normalization uses training arrays only.  ``fit_post_labels``
is explicitly an offline label helper, never a prediction input.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares

from .flight import initial_guess
from .identified_dynamics import (jacobian_diagnostics, simulate,
                                  simulate_batch_with_jacobian, step)

FORMAT_VERSION = 1
FEATURE_NAMES = ["incident_tangent_speed_m_s", "incident_normal_speed_m_s",
                 "b_tangent_per_s", "b_transverse_per_s", "b_normal_per_s"]


def _matrix(value, name, columns=None, *, allow_empty=False):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or (not allow_empty and len(a) == 0):
        raise ValueError(f"{name} must be a nonempty matrix")
    if (columns is not None and a.shape[1] != columns) or a.shape[1] == 0:
        raise ValueError(f"{name} has invalid dimensions")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def trajectory_weights(trajectory_ids):
    """Weights summing to one per independent trajectory, including variants."""
    identities = list(trajectory_ids)
    if not identities:
        raise ValueError("nonempty trajectory IDs required")
    counts = {}
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1
    return np.asarray([1. / counts[identity] for identity in identities])


def incident_frames(velocity):
    """Rows are local tangent, transverse and +z normal basis vectors."""
    v = _matrix(velocity, "incident_velocity", 3, allow_empty=True)
    planar = np.linalg.norm(v[:, :2], axis=1)
    if np.any(planar < 1e-6):
        raise ValueError("incident tangent speed below 1e-6 m/s has no defined frame")
    tangent = np.c_[v[:, :2] / planar[:, None], np.zeros(len(v))]
    normal = np.broadcast_to([0., 0., 1.], (len(v), 3)).copy()
    transverse = np.cross(normal, tangent)
    return np.stack([tangent, transverse, normal], axis=1)


def pre_features(velocity, b_minus):
    """Return five causal local features and frames.  Descending impact only."""
    v = _matrix(velocity, "incident_velocity", 3, allow_empty=True)
    b = _matrix(b_minus, "effective_b_minus", 3, allow_empty=True)
    if len(v) != len(b):
        raise ValueError("velocity and effective Magnus rows differ")
    if np.any(v[:, 2] >= 0):
        raise ValueError("incident velocity must be descending")
    frames = incident_frames(v)
    b_local = np.einsum("nij,nj->ni", frames, b)
    return np.c_[np.linalg.norm(v[:, :2], axis=1), -v[:, 2], b_local], frames


def baseline_velocity(velocity, e_n=.8339975149329221,
                      alpha_t=.5576036363231244):
    v = _matrix(velocity, "incident_velocity", 3, allow_empty=True)
    if not np.isfinite([e_n, alpha_t]).all() or e_n < 0 or alpha_t < 0:
        raise ValueError("finite nonnegative baseline parameters required")
    return v * np.array([alpha_t, alpha_t, -e_n])


def velocity_targets(velocity, b_minus, v_plus, e_n=.8339975149329221,
                     alpha_t=.5576036363231244):
    X, frames = pre_features(velocity, b_minus)
    vp = _matrix(v_plus, "offline_v_plus_label", 3, allow_empty=True)
    if len(vp) != len(X):
        raise ValueError("feature and label rows differ")
    residual = vp - baseline_velocity(velocity, e_n, alpha_t)
    return X, np.einsum("nij,nj->ni", frames, residual)


def effective_targets(velocity, b_minus, b_plus, *, prior_mode="carry"):
    """Local effective-b residual label, never a true-spin target."""
    X, frames = pre_features(velocity, b_minus)
    bp = _matrix(b_plus, "offline_effective_b_plus_label", 3, allow_empty=True)
    if len(bp) != len(X) or prior_mode not in {"carry", "zero"}:
        raise ValueError("matching b label rows and carry/zero prior required")
    prior = np.asarray(b_minus) if prior_mode == "carry" else np.zeros_like(bp)
    return X, np.einsum("nij,nj->ni", frames, bp-prior)


def post_lift_targets(velocity, b_minus, post_labels):
    """One informative post-b lift component and its conditional label sigma.

    The component is along e2 = +z cross incident horizontal tangent.  It is
    normal to the incident trajectory plane and generally better informed than
    the full three-vector.  All outputs are *offline supervision only*.  The
    caller freezes an uncertainty cutoff (for example .08 s^-1) before holdout
    evaluation.  Labels lacking a complete covariance receive sigma=inf and
    cannot be used; diagonal-only covariance is not silently substituted.
    """
    X, frames = pre_features(velocity, b_minus)
    labels = [post_labels] if isinstance(post_labels, dict) else list(post_labels)
    if len(labels) != len(X):
        raise ValueError("one offline post label required per feature row")
    bm = np.asarray(b_minus, dtype=float)
    Y = np.zeros((len(X), 1))
    sigma = np.full(len(X), np.inf)
    for i, row in enumerate(labels):
        if not row.get("available", False) or row.get("quality_reasons", []):
            continue
        bp = np.asarray(row.get("b_plus_per_s"), dtype=float)
        covariance_value = row.get("conditional_b_covariance_per_s2")
        if covariance_value is None:
            covariance_value = row.get("b_data_projected_covariance_per_s2")
        if bp.shape != (3,) or not np.isfinite(bp).all() or covariance_value is None:
            continue
        covariance = np.asarray(covariance_value, dtype=float)
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            continue
        covariance = .5*(covariance+covariance.T)
        if np.linalg.eigvalsh(covariance).min() < -1e-10:
            continue
        e2 = frames[i, 1]
        variance = float(e2@covariance@e2)
        if variance < -1e-10:
            continue
        Y[i, 0] = float(e2@(bp-bm[i]))
        sigma[i] = np.sqrt(max(variance, 0.))
    return X, Y, sigma


def predict_post_lift(model, velocity, b_minus):
    """Correct only trajectory-plane-normal effective-b, retaining others."""
    X, frames = pre_features(velocity, b_minus)
    residual = predict_model(model, X)
    if residual.shape[1] != 1:
        raise ValueError("post-lift scalar model must output one component")
    return np.asarray(b_minus, dtype=float) + residual*frames[:, 1]


def _weights(weights, count):
    w = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (count,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("strictly positive finite sample weights required")
    # Preserve absolute per-collision total, so duplicated prefix variants with
    # divided weights do not make the prior/regularization weaker.
    return w


def _normalization(X, w):
    mean = np.average(X, axis=0, weights=w)
    scale = np.sqrt(np.average((X-mean)**2, axis=0, weights=w))
    return mean, np.where(scale > 1e-8, scale, 1.)


def _design(X, degree):
    cols = [np.ones((len(X), 1)), X]
    if degree == 2:
        cols.append(np.stack([X[:, i]*X[:, j] for i in range(X.shape[1])
                              for j in range(i, X.shape[1])], axis=1))
    return np.concatenate(cols, axis=1)


def _kernel(A, B, length_scale):
    d = (A[:, None, :] - B[None, :, :]) / length_scale
    return np.exp(-.5*np.sum(d*d, axis=2))


def fit_model(X, Y, weights=None, config=None):
    """Fit one candidate using training arrays only; no holdout argument.

    Ridge config: ``kind=ridge, degree=1|2, ridge=1.``.
    GP config: ``kind=gp, length_scale=1., noise=.2, jitter=1e-10``.
    GP standard deviations are conditional residual uncertainty under this
    kernel/noise assumption; their held-out calibration must be evaluated.
    """
    X, Y = _matrix(X, "X"), _matrix(Y, "Y")
    if len(X) != len(Y):
        raise ValueError("feature and target row counts differ")
    cfg = dict(config or {})
    kind = cfg.pop("kind", "ridge")
    allowed = {"ridge": {"degree", "ridge"},
               "gp": {"length_scale", "noise", "jitter"}}
    if kind not in allowed or set(cfg)-allowed[kind]:
        raise ValueError("unsupported collision model or configuration key")
    w = _weights(weights, len(X))
    xm, xs = _normalization(X, w)
    ym, ys = _normalization(Y, w)
    x, y = (X-xm)/xs, (Y-ym)/ys
    model = {"format_version": FORMAT_VERSION, "kind": kind,
             "input_dim": int(X.shape[1]), "output_dim": int(Y.shape[1]),
             "x_mean": xm, "x_scale": xs, "y_mean": ym, "y_scale": ys,
             "normalization_source": "training_arrays_only"}
    if kind == "ridge":
        degree, ridge = int(cfg.get("degree", 1)), float(cfg.get("ridge", 1.))
        if degree not in (1, 2) or ridge <= 0 or not np.isfinite(ridge):
            raise ValueError("degree 1/2 and finite positive ridge required")
        design = _design(x, degree)
        penalty = ridge*np.eye(design.shape[1]); penalty[0, 0] = 1e-12
        system = (design.T*w)@design + penalty
        coef = np.linalg.solve(system, (design.T*w)@y)
        model.update(degree=degree, ridge=ridge, coefficients=coef)
        complexity = int(coef.size)
        condition = float(np.linalg.cond(system))
    else:
        length = np.asarray(cfg.get("length_scale", 1.), dtype=float)
        if length.ndim == 0:
            length = np.full(X.shape[1], float(length))
        noise, jitter = float(cfg.get("noise", .2)), float(cfg.get("jitter", 1e-10))
        if (length.shape != (X.shape[1],) or np.any(length <= 0)
                or not np.isfinite(length).all() or min(noise, jitter) <= 0
                or not np.isfinite([noise, jitter]).all()):
            raise ValueError("finite positive GP lengths/noise/jitter required")
        kernel = _kernel(x, x, length)
        # Higher weight means lower observation variance.  Repeating a gap
        # variant cannot add more independent-collision weight by itself.
        system = kernel + np.diag(noise**2/w+jitter)
        lower, _ = cho_factor(system, lower=True, check_finite=True)
        lower = np.tril(lower)
        coefficients = cho_solve((lower, True), y)
        model.update(length_scale=length, noise=noise, jitter=jitter,
                     train_x=x, coefficients=coefficients, cholesky=lower)
        complexity = int(coefficients.size)
        condition = float(np.linalg.cond(system))
    prediction = predict_model(model, X)
    err2 = np.sum((prediction-Y)**2, axis=1)
    report = {"kind": kind, "rows": len(X), "input_dim": X.shape[1],
              "output_dim": Y.shape[1], "coefficient_count": complexity,
              "weighted_training_vector_rmse": float(np.sqrt(np.average(err2, weights=w))),
              "linear_system_condition": condition, "config": _json_value(cfg),
              "interpretation": "Training fit only, not held-out prediction performance",
              "independent_collision_weights_required": True}
    return model, report


def predict_model(model, X, *, return_std=False):
    X = _matrix(X, "X", int(model["input_dim"]), allow_empty=True)
    x = (X-model["x_mean"])/model["x_scale"]
    if model["kind"] == "ridge":
        pred = _design(x, model["degree"])@model["coefficients"]
        std = None
    elif model["kind"] == "gp":
        kernel = _kernel(x, model["train_x"], model["length_scale"])
        pred = kernel@model["coefficients"]
        if return_std:
            solved = cho_solve((model["cholesky"], True), kernel.T)
            variance = np.maximum(1.-np.sum(kernel*solved.T, axis=1), 0.)
            std = np.sqrt(variance[:, None])*model["y_scale"]
        else:
            std = None
    else:
        raise ValueError("unknown collision model kind")
    pred = pred*model["y_scale"]+model["y_mean"]
    if not np.isfinite(pred).all():
        raise FloatingPointError("nonfinite collision residual prediction")
    return (pred, std) if return_std else pred


def predict_velocity(model, velocity, b_minus, e_n=.8339975149329221,
                     alpha_t=.5576036363231244):
    """Residual collision mapping; no hidden velocity/output clipping."""
    X, frames = pre_features(velocity, b_minus)
    local = predict_model(model, X)
    if local.shape[1] != 3:
        raise ValueError("velocity residual model must output three components")
    return baseline_velocity(velocity, e_n, alpha_t) + np.einsum("nji,nj->ni", frames, local)


def predict_bplus(model, velocity, b_minus, *, prior_mode="carry"):
    X, frames = pre_features(velocity, b_minus)
    if prior_mode not in {"carry", "zero"}:
        raise ValueError("carry/zero effective-b prior required")
    local = predict_model(model, X)
    if local.shape[1] != 3:
        raise ValueError("effective-b residual model must output three components")
    prior = np.asarray(b_minus) if prior_mode == "carry" else np.zeros((len(X), 3))
    return prior + np.einsum("nji,nj->ni", frames, local)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(model), indent=2, allow_nan=False)+"\n")
    return str(path)


def load_model(path):
    model = json.loads(Path(path).read_text())
    if model.get("format_version") != FORMAT_VERSION or model.get("kind") not in {"ridge", "gp"}:
        raise ValueError("unsupported collision checkpoint")
    arrays = ["x_mean", "x_scale", "y_mean", "y_scale", "coefficients"]
    if model["kind"] == "gp":
        arrays += ["train_x", "length_scale", "cholesky"]
    for key in arrays:
        model[key] = np.asarray(model[key], dtype=float)
        if not np.isfinite(model[key]).all():
            raise ValueError("nonfinite checkpoint array")
    for key, dimension in [("x_mean", model["input_dim"]), ("x_scale", model["input_dim"]),
                           ("y_mean", model["output_dim"]), ("y_scale", model["output_dim"])]:
        if model[key].shape != (dimension,) or (key.endswith("scale") and np.any(model[key] <= 0)):
            raise ValueError("invalid checkpoint normalization")
    if model["kind"] == "ridge":
        if model.get("degree") not in (1, 2):
            raise ValueError("invalid ridge checkpoint degree")
        width = 1+model["input_dim"]
        if model["degree"] == 2:
            width += model["input_dim"]*(model["input_dim"]+1)//2
        if model["coefficients"].shape != (width, model["output_dim"]):
            raise ValueError("invalid ridge checkpoint coefficients")
    else:
        n = len(model["train_x"])
        if (model["train_x"].shape != (n, model["input_dim"])
                or model["coefficients"].shape != (n, model["output_dim"])
                or model["cholesky"].shape != (n, n)
                or model["length_scale"].shape != (model["input_dim"],)
                or np.any(model["length_scale"] <= 0)
                or np.any(np.diag(model["cholesky"]) <= 0)):
            raise ValueError("invalid GP checkpoint shape or scale")
    return model


def fit_post_labels(times, positions, reference_bounce_t_s, k_drag, *,
                    start_after_s=.020, max_after_s=.400, b_prior_sigma=.15,
                    measurement_sigma_m=.008, max_nfev=80):
    """Offline long-post-segment effective [p+,v+,b+] label and diagnostics.

    Uses post observations only, with bounce time supplied solely as an offline
    reference.  The early 20ms contact guard and >=140ms span avoid freely
    fitting nine dimensions on a 30ms fragment.  Effective b is regularized;
    diagnostics must not be interpreted as proof of measured true spin.
    """
    t, p = np.asarray(times, dtype=float), _matrix(positions, "positions", 3)
    if t.shape != (len(p),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("finite strictly increasing times required")
    if not np.isfinite(reference_bounce_t_s) or min(b_prior_sigma, measurement_sigma_m) <= 0:
        raise ValueError("finite bounce reference and positive label priors required")
    selected = (t >= reference_bounce_t_s+start_after_s) & (t <= reference_bounce_t_s+max_after_s)
    t, p = t[selected], p[selected]
    if len(t) < 35 or np.ptp(t) < .14:
        raise ValueError("long_post_segment_coverage_insufficient")
    # Subsample for bounded fit cost, preserving both temporal endpoints.
    indices = np.unique(np.linspace(0, len(t)-1, min(64, len(t))).astype(int))
    times_fit, points_fit = t[indices], p[indices]
    dt = times_fit-times_fit[0]
    guess = np.r_[initial_guess(dt, points_fit), np.zeros(3)]
    guess[3:6] = np.clip(guess[3:6], -119., 119.)
    prior = np.zeros(3)

    def residual(state):
        pred = simulate(dt, state, k_drag)[:, :3]
        return np.r_[((pred-points_fit)/measurement_sigma_m).ravel(),
                     (state[6:9]-prior)/b_prior_sigma]

    def jacobian(state):
        _, jac = simulate_batch_with_jacobian(dt[None, :], state[None, :], k_drag)
        data = jac[0, :, :, :9].reshape(-1, 9)/measurement_sigma_m
        regularizer = np.zeros((3, 9)); regularizer[:, 6:9] = np.eye(3)/b_prior_sigma
        return np.vstack([data, regularizer])

    fit = least_squares(residual, guess, jac=jacobian,
                        bounds=(np.r_[[-100]*3, [-120]*3, [-2]*3],
                                np.r_[[100]*3, [120]*3, [2]*3]),
                        loss="soft_l1", f_scale=1.5, x_scale="jac",
                        max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    label = step(fit.x, reference_bounce_t_s-times_fit[0], k_drag)
    fitted = simulate(dt, fit.x, k_drag)[:, :3]
    err = np.linalg.norm(fitted-points_fit, axis=1)
    data_jac = jacobian(fit.x)[:-3]
    diagnostic = jacobian_diagnostics(data_jac)
    # Remove directions explainable by p/v before measuring b information.
    # Full 9-column rank alone can conceal a nearly prior-only b component.
    nuisance = data_jac[:, :6]
    b_information = data_jac[:, 6:9] - nuisance @ np.linalg.lstsq(nuisance, data_jac[:, 6:9], rcond=None)[0]
    b_singular = np.linalg.svd(b_information, compute_uv=False)
    b_rank = int(np.sum(b_singular > max(b_information.shape)*np.finfo(float).eps*max(float(b_singular[0]), 1.)))
    weakest_sigma = 1./float(b_singular[-1]) if b_rank == 3 and b_singular[-1] > 0 else None
    b_data_covariance = (np.linalg.inv(b_information.T@b_information)
                         if b_rank == 3 else None)
    # This covariance is local/conditional and prior-dependent, not a validated
    # confidence interval.  It helps report label reliability explicitly.
    covariance = np.linalg.pinv(jacobian(fit.x).T@jacobian(fit.x), rcond=1e-10)
    reasons = []
    if not fit.success:
        reasons.append("post_optimizer_not_converged")
    if float(np.sqrt(np.mean(err**2))) > .040:
        reasons.append("post_reconstruction_rmse_above_40mm")
    if np.max(np.abs(fit.x[6:9])) > 1.99:
        reasons.append("effective_magnus_at_bound")
    if label[5] <= 0:
        reasons.append("post_velocity_not_ascending_at_reference")
    b_usable = bool(not reasons and weakest_sigma is not None and weakest_sigma <= b_prior_sigma)
    return {"available": True, "converged": bool(fit.success), "nfev": int(fit.nfev),
            "reference_t_s": float(reference_bounce_t_s),
            "state_plus": label.tolist(), "v_plus_m_s": label[3:6].tolist(),
            "b_plus_per_s": label[6:9].tolist(), "quality_reasons": reasons,
            "post_start_t_s": float(times_fit[0]), "post_end_t_s": float(times_fit[-1]),
            "post_points": len(times_fit), "post_span_s": float(np.ptp(times_fit)),
            "position_rmse_m": float(np.sqrt(np.mean(err**2))),
            "jacobian_diagnostics": diagnostic,
            "b_label_usable": b_usable,
            "b_data_projected_rank": b_rank,
            "b_data_projected_singular_values": b_singular.tolist(),
            "b_data_weakest_sigma_per_s": weakest_sigma,
            "b_data_projected_covariance_per_s2": (None if b_data_covariance is None else b_data_covariance.tolist()),
            "b_information_definition": "measurement-normalized local b sensitivity projected orthogonal to p/v; usable only if weakest data sigma <= prior sigma",
            "conditional_b_sigma_per_s": np.sqrt(np.maximum(np.diag(covariance)[6:9], 0.)).tolist(),
            "conditional_b_covariance_per_s2": covariance[6:9, 6:9].tolist(),
            "b_prior_sigma_per_s": float(b_prior_sigma),
            "label_definition": "offline post-only reconstruction; never a causal feature",
            "true_spin_observed": False}
