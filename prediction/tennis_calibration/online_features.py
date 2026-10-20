"""Shared causal feature schema for training and live residual inference."""
from __future__ import annotations

import numpy as np


FEATURE_SCHEMA = {
    "version": 1,
    "history_window_s": 0.08,
    "history_points": 16,
    "history_dimension": 82,
    "base_dimension": 97,
    "position_dimension": 98,
    "landing_dimension": 103,
    "base_order": ["history_features", "velocity_xyz_m_s", "residual_acceleration_xyz_m_s2",
                   "position_z_m", "posterior_std_pv", "bounced", "delay_s"],
    "position_append": ["requested_horizon_s_from_now"],
    "landing_append": ["physical_time_to_contact_from_capture_s", "landing_xy_minus_capture_xy_m",
                       "physical_contact_velocity_xyz_m_s"],
    "position_target_kind": "future_position_residual_m",
    "landing_target_kind": "landing_correction_m_m_s",
    "landing_target_order": ["dx_m", "dy_m", "dt_s"],
}


def base_features(history_result: dict, snapshot: dict, delay_s: float = 0.0) -> np.ndarray:
    """97 features at capture time; accepts single or identically shaped batches."""
    history = np.asarray(history_result["features"], dtype=float)
    velocity, residual = np.asarray(snapshot["v"], dtype=float), np.asarray(snapshot["a_res"], dtype=float)
    position = np.asarray(snapshot["p"], dtype=float)
    covariance = np.asarray(snapshot["covariance"], dtype=float)
    if history.shape[-1] != 82 or velocity.shape[-1] != 3 or residual.shape != velocity.shape:
        raise ValueError("History/velocity/residual do not match the frozen feature schema")
    leading = history.shape[:-1]
    if velocity.shape != leading + (3,) or position.shape != velocity.shape:
        raise ValueError("Feature leading dimensions must agree")
    if covariance.shape[:-2] != leading or covariance.shape[-1] != covariance.shape[-2] or covariance.shape[-1] not in (6, 9):
        raise ValueError("Expected capture-time 6-D or 9-D covariance")
    standard_deviation = np.sqrt(np.maximum(np.diagonal(covariance, axis1=-2, axis2=-1)[..., :6], 0))
    bounced = np.broadcast_to(np.asarray(snapshot["bounced"], dtype=float), leading)[..., None]
    delay = np.broadcast_to(np.asarray(delay_s, dtype=float), leading)[..., None]
    result = np.concatenate((history, velocity, residual, position[..., 2:3], standard_deviation, bounced, delay), axis=-1)
    if np.any(delay < 0) or not np.all(np.isfinite(result)):
        raise ValueError("Features and delay must be finite, with a nonnegative delay")
    return result


def position_features(base: np.ndarray, horizon_s: float | np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    if base.shape[-1] != 97:
        raise ValueError("Expected 97 base features")
    horizon = np.broadcast_to(np.asarray(horizon_s, dtype=float), base.shape[:-1])[..., None]
    result = np.concatenate((base, horizon), axis=-1)
    if np.any(horizon < 0) or not np.all(np.isfinite(result)):
        raise ValueError("Position features must be finite with nonnegative horizons")
    return result


def landing_features(base: np.ndarray, snapshot: dict, landing: dict) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    if base.shape[-1] != 97:
        raise ValueError("Expected 97 base features")
    leading = base.shape[:-1]
    contact_time = np.broadcast_to(np.asarray(landing["time_to_contact"], dtype=float), leading)[..., None]
    displacement = np.asarray(landing["position"], dtype=float)[..., :2] - np.asarray(snapshot["p"], dtype=float)[..., :2]
    velocity = np.asarray(landing["velocity"], dtype=float)
    if displacement.shape != leading + (2,) or velocity.shape != leading + (3,):
        raise ValueError("Landing features must have matching leading dimensions")
    result = np.concatenate((base, contact_time, displacement, velocity), axis=-1)
    if np.any(contact_time < 0) or not np.all(np.isfinite(result)):
        raise ValueError("Landing features require a finite nonnegative physical contact")
    return result
