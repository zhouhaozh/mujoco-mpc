"""Small causal output-residual candidates for isolated Ubuntu experiments.

The frozen physics/GP state, its height event and covariance are never modified.
Input construction accepts only a current snapshot, its uncorrected forecast and
already received measurements. Future reference positions appear exclusively as
supervised targets supplied to ``fit_residual_head`` by the training runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "horizon_s", "horizon_squared_s2", "state_height_m",
    "state_vx", "state_vy", "state_vz", "state_bx", "state_by", "state_bz",
    "forecast_dx", "forecast_dy", "forecast_dz",
    "estimated_contact_age_s", "actual_resume_age_s", "bounced", "hold_active",
    "history_span_s", "history_count_fraction",
    "last_position_residual_x", "last_position_residual_y", "last_position_residual_z",
    "local_vx", "local_vy", "local_vz",
    "local_minus_state_vx", "local_minus_state_vy", "local_minus_state_vz",
    "local_mad_x", "local_mad_y", "local_mad_z", "history_fit_available",
)


def causal_features(snapshot, forecast_xyz, horizon_s, history_times, history_xyz):
    """Construct a 31D vector; reject, rather than silently consume, future data.

    An 80ms trailing line fit supplies causal slope and quality features. If a
    modeled contact has been discovered, this fit uses only samples after that
    contact; it cannot average incoming and outgoing velocities. No offline
    contact, split membership or future residual is an input to this function.
    """
    state = np.asarray(snapshot["state"], dtype=float)
    forecast = np.asarray(forecast_xyz, dtype=float)
    stamp = float(snapshot["t_s"])
    horizon = float(horizon_s)
    times = np.asarray(history_times, dtype=float)
    points = np.asarray(history_xyz, dtype=float)
    if state.shape != (9,) or forecast.shape != (3,):
        raise ValueError("expected nine-dimensional state and XYZ forecast")
    if not np.isfinite(np.r_[state, forecast, stamp, horizon]).all() or horizon < 0:
        raise ValueError("nonfinite state/forecast or negative horizon")
    if times.ndim != 1 or points.shape != (len(times), 3):
        raise ValueError("history must have matching t and XYZ arrays")
    if not np.isfinite(times).all() or not np.isfinite(points).all():
        raise ValueError("history must be finite received observations")
    if np.any(np.diff(times) <= 0) or np.any(times > stamp + 1e-12):
        raise ValueError("history must be strictly ordered and cannot include future observations")
    contact = snapshot.get("height_contact_t_s")
    discovered = snapshot.get("height_discovery_t_s")
    resume = snapshot.get("resume_t_s")
    if discovered is not None and float(discovered) > stamp + 1e-12:
        raise ValueError("future height-event discovery cannot enter features")
    if contact is not None and float(contact) > stamp + 1e-12:
        raise ValueError("future modeled contact cannot enter features")
    if resume is not None and float(resume) > stamp + 1e-12:
        raise ValueError("future resume cannot enter features")
    lower = stamp - .08
    if contact is not None:
        lower = max(lower, float(contact))
    mask = times >= lower - 1e-12
    t, p = times[mask], points[mask]
    span = float(t[-1] - t[0]) if len(t) else 0.
    last = p[-1] - state[:3] if len(t) else np.zeros(3)
    slope, mad, available = np.zeros(3), np.zeros(3), False
    if len(t) >= 6 and span >= .02:
        # All rows are at or before the current capture. This is a regression
        # summary, not a smoothed measurement fed back into the state filter.
        dt = t - stamp
        design = np.column_stack((np.ones(len(t)), dt))
        coefficient = np.linalg.lstsq(design, p, rcond=None)[0]
        residual = p - design @ coefficient
        slope = coefficient[1]
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
        available = True
    difference = slope - state[3:6] if available else np.zeros(3)
    feature = np.r_[
        horizon, horizon * horizon, state[2], state[3:9], forecast - state[:3],
        -1. if contact is None else np.clip(stamp-float(contact), 0., 2.),
        -1. if resume is None else np.clip(stamp-float(resume), 0., 2.),
        bool(snapshot.get("bounced", False)), bool(snapshot.get("hold_active", False)),
        span, min(len(t), 40)/40., last, slope, difference, mad, available,
    ]
    if feature.shape != (len(FEATURE_NAMES),) or not np.isfinite(feature).all():
        raise ValueError("invalid causal feature vector")
    return feature


def residual_gate(snapshot, *, wait_s=.05, ramp_s=.05):
    """Delay output correction until 50ms after actual resume, then ramp 50ms."""
    if not np.isfinite([wait_s, ramp_s]).all() or wait_s < 0 or ramp_s <= 0:
        raise ValueError("invalid residual gate durations")
    resume = snapshot.get("resume_t_s")
    if (resume is None or snapshot.get("hold_active", False)
            or not snapshot.get("bounced", False)):
        return 0.
    age = float(snapshot["t_s"]) - float(resume)
    if not np.isfinite(age):
        raise ValueError("invalid snapshot clock")
    return float(np.clip((age-wait_s)/ramp_s, 0., 1.))


def _trace_weights(trajectory_ids, allowed_train_ids):
    ids = np.asarray([str(value) for value in trajectory_ids])
    allowed = {str(value) for value in allowed_train_ids}
    if not allowed or not len(ids):
        raise ValueError("explicit nonempty training IDs and samples required")
    observed = set(ids)
    forbidden = observed - allowed
    if forbidden:
        raise ValueError("non-training trajectory entered residual fit: " + ", ".join(sorted(forbidden)))
    unique, count = np.unique(ids, return_counts=True)
    sizes = dict(zip(unique, count))
    weights = np.asarray([1./(len(unique)*sizes[value]) for value in ids])
    return weights, sorted(observed), sorted(allowed)


@dataclass
class ResidualHead:
    kind: str
    mean: np.ndarray
    scale: np.ndarray
    target_scale: np.ndarray
    parameters: list[np.ndarray]
    metadata: dict

    def predict(self, features):
        """Ungated learned displacement in metres; apply_residual adds guards."""
        x = np.asarray(features, dtype=float)
        scalar = x.ndim == 1
        if scalar:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("invalid residual predictor features")
        normalized = np.clip((x-self.mean)/self.scale, -6., 6.)
        if self.kind == "ridge":
            result = np.column_stack((normalized, np.ones(len(x)))) @ self.parameters[0]
        elif self.kind == "mlp":
            w1, b1, w2, b2 = self.parameters
            result = np.tanh(normalized @ w1+b1) @ w2+b2
        else:
            raise ValueError("unsupported residual head")
        result *= self.target_scale
        return result[0] if scalar else result

    def save(self, path):
        payload = {"schema": 1, "kind": self.kind, "feature_names": list(FEATURE_NAMES),
                   "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                   "target_scale": self.target_scale.tolist(),
                   "parameters": [value.tolist() for value in self.parameters],
                   "metadata": self.metadata}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def predict_bounded(self, features, horizons_s, *, maximum_correction_m=.15):
        """Batch displacement cap; caller multiplies cached causal gate."""
        correction = self.predict(features)
        scalar = correction.ndim == 1
        values = correction[None, :] if scalar else correction
        horizons = np.asarray(horizons_s, dtype=float)
        if horizons.ndim == 0:
            horizons = np.full(len(values), float(horizons))
        if (horizons.shape != (len(values),) or not np.isfinite(horizons).all()
                or np.any(horizons < 0) or not np.isfinite(maximum_correction_m)
                or maximum_correction_m < 0):
            raise ValueError("invalid residual horizon/cap")
        bounds = maximum_correction_m*np.minimum(horizons/.5, 1.)
        lengths = np.linalg.norm(values, axis=1)
        gains = np.minimum(1., bounds/np.maximum(lengths, 1e-20))
        result = values*gains[:, None]
        return result[0] if scalar else result

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["schema"] != 1 or payload["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("residual feature schema mismatch")
        return cls(payload["kind"], np.asarray(payload["mean"]), np.asarray(payload["scale"]),
                   np.asarray(payload["target_scale"]),
                   [np.asarray(value) for value in payload["parameters"]], payload["metadata"])


def fit_residual_head(kind, features, residual_xyz, trajectory_ids, *, allowed_train_ids,
                      ridge_alpha=.03, hidden=12, epochs=300, seed=1931,
                      learning_rate=.008, mlp_l2=.01):
    """Train with trace-equal weights and train-only normalization.

    ``residual_xyz`` is reference minus the uncorrected forecast, in metres.
    The caller must retain only eligible training anchors (gate > 0), legal
    future targets and positive horizons. Validation data cannot be passed for
    scaling, fitting, early stopping or optimizer iteration selection. The MLP
    uses fixed-epoch deterministic full-batch Adam, one 12-unit tanh layer and
    an explicitly weighted objective, avoiding library-version weight defaults.
    """
    x, y = np.asarray(features, dtype=float), np.asarray(residual_xyz, dtype=float)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or y.shape != (len(x), 3)
            or len(trajectory_ids) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid residual training arrays")
    if kind not in ("ridge", "mlp"):
        raise ValueError("kind must be ridge or mlp")
    if ridge_alpha < 0 or mlp_l2 < 0 or hidden < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError("invalid residual optimizer configuration")
    weights, observed, allowed = _trace_weights(trajectory_ids, allowed_train_ids)
    mean = np.sum(weights[:, None]*x, axis=0)
    scale = np.sqrt(np.sum(weights[:, None]*(x-mean)**2, axis=0))
    scale[scale < 1e-5] = 1.
    z = np.clip((x-mean)/scale, -6., 6.)
    target_scale = np.maximum(np.sqrt(np.sum(weights[:, None]*y*y, axis=0)), .02)
    target = y/target_scale
    if kind == "ridge":
        design = np.column_stack((z, np.ones(len(z))))
        penalty = np.eye(design.shape[1])*ridge_alpha
        penalty[-1, -1] = 0.
        parameters = [np.linalg.solve(design.T @ (weights[:, None]*design)+penalty,
                                      design.T @ (weights[:, None]*target))]
    else:
        rng = np.random.default_rng(seed)
        parameters = [rng.normal(0., .12, (x.shape[1], hidden)), np.zeros(hidden),
                      rng.normal(0., .03, (hidden, 3)), np.zeros(3)]
        first = [np.zeros_like(value) for value in parameters]
        second = [np.zeros_like(value) for value in parameters]
        for iteration in range(1, epochs+1):
            w1, b1, w2, b2 = parameters
            layer = np.tanh(z @ w1+b1)
            output = layer @ w2+b2
            d_output = (2./3.)*weights[:, None]*(output-target)
            d_layer = (d_output @ w2.T)*(1-layer*layer)
            gradients = [z.T @ d_layer+2*mlp_l2*w1, np.sum(d_layer, axis=0),
                         layer.T @ d_output+2*mlp_l2*w2, np.sum(d_output, axis=0)]
            for j, gradient in enumerate(gradients):
                first[j] = .9*first[j]+.1*gradient
                second[j] = .999*second[j]+.001*gradient*gradient
                parameters[j] -= learning_rate*(first[j]/(1-.9**iteration))/(np.sqrt(second[j]/(1-.999**iteration))+1e-8)
    head = ResidualHead(kind, mean, scale, target_scale, parameters, {
        "observed_training_ids": observed, "allowed_training_ids": allowed,
        "training_rows": len(x), "training_trajectories": len(observed),
        "trace_equal_weight_sum": float(weights.sum()), "ridge_alpha": ridge_alpha,
        "hidden_units": hidden if kind == "mlp" else None,
        "epochs": epochs if kind == "mlp" else None, "seed": seed,
        "mlp_l2": mlp_l2, "learning_rate": learning_rate,
        "normalization_scope": "explicit_training_trajectories_only",
        "training_label": "reference_minus_uncorrected_forecast_xyz",
        "no_filter_state_or_contact_clock_changes": True,
    })
    if not np.isfinite(head.predict(x)).all():
        raise FloatingPointError("nonfinite fitted residual head")
    return head


def apply_residual(head, feature, snapshot, horizon_s, forecast_xyz, *,
                   wait_s=.05, ramp_s=.05, maximum_correction_m=.15):
    """Apply a bounded, ramped correction without changing tracker state.

    At H500 the displacement norm is <= 15cm; at H100 it is <= 3cm.
    Defaults are fixed candidate settings, not values selected on test data.
    """
    forecast = np.asarray(forecast_xyz, dtype=float)
    horizon = float(horizon_s)
    if (forecast.shape != (3,) or not np.isfinite(forecast).all()
            or not np.isfinite([horizon, maximum_correction_m]).all()
            or horizon < 0 or maximum_correction_m < 0):
        raise ValueError("invalid output correction arguments")
    gain = residual_gate(snapshot, wait_s=wait_s, ramp_s=ramp_s)
    if gain == 0. or horizon == 0.:
        return forecast.copy()
    correction = head.predict_bounded(feature, horizon, maximum_correction_m=maximum_correction_m)
    return forecast+gain*correction
