"""Causal EKF with an effective Magnus vector and conservative impact recovery.

The 9-D state is [position m, velocity m/s, b_eff 1/s].  Its flight
acceleration is g - k_D |v| v + b_eff x v.  b_eff is an identifiable
acceleration coefficient under this model, not a measurement of angular speed.
The 6-D ablation follows the exact same observation and impact recovery logic.
Offline cleaning masks and contact labels are never inputs.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from . import identified_dynamics as physics
from .online_filter import OnlineFilter, DEFAULT_CONFIG as OLD_CONFIG


DEFAULT_CONFIG = {
    **OLD_CONFIG,
    "magnus_drive_sigma": 0.12,
    "initial_magnus_sigma": 0.6,
    "history_window_s": 0.24,
    "bounce_pre_min_span_s": 0.018,
    "bounce_post_min_span_s": 0.030,
    "bounce_post_min_samples": 10,
    "local_velocity_sigma_limit_m_s": 1.5,
    "local_fit_max_error_m": 0.030,
    "local_fit_min_inlier_fraction": 0.70,
    "local_fit_consistency_m_s": 1.5,
    "bounce_prior_velocity_gate": 36.0,
    "recovery_covariance_floor_m_s": 0.75,
    "recovery_position_sigma_m": 0.020,
    "recovery_max_velocity_gain": 0.60,
    "history_max_physical_speed_m_s": 120.0,
}


def _configuration(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown identified filter configuration: {sorted(unknown)}")
        result.update(config)
    sigma = np.broadcast_to(np.asarray(result["measurement_sigma_xyz"], dtype=float), (3,)).copy()
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("measurement_sigma_xyz must be positive and finite")
    result["measurement_sigma_xyz"] = sigma
    for key, value in result.items():
        if key != "measurement_sigma_xyz" and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be positive and finite")
    if result["soft_gate_threshold"] > result["gate_threshold"]:
        raise ValueError("soft_gate_threshold exceeds gate_threshold")
    if result["recovery_max_velocity_gain"] > 1 or result["local_fit_min_inlier_fraction"] > 1:
        raise ValueError("Recovery gain and inlier fraction must be <= 1")
    return result


def _padded(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    if state.shape[-1] == 9:
        return state
    if state.shape[-1] != 6:
        raise ValueError("Expected 6- or 9-D state")
    return np.concatenate((state, np.zeros((*state.shape[:-1], 3))), axis=-1)


def dynamics(state: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    return physics.rhs(_padded(state), float(parameters["k_drag"]))[..., :state.shape[-1]]


def dynamics_jacobian(state: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    return physics.rhs_jacobian(_padded(state), float(parameters["k_drag"]))[:len(state), :len(state)]


def _step(state: np.ndarray, dt: float | np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    return physics.step(_padded(state), dt, float(parameters["k_drag"]))[..., :state.shape[-1]]


def _step_with_transition(state: np.ndarray, dt: float, parameters: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    dimension = len(state)
    result, transition = physics.step_with_jacobian(_padded(state), dt, float(parameters["k_drag"]))
    return result[:dimension], transition[:dimension, :dimension]


def _impact(state: np.ndarray, parameters: dict[str, Any], contact_height: float) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    result = physics.impact_map(_padded(state), parameters)[..., :state.shape[-1]].copy()
    result[..., 2] = contact_height
    return result


def _impact_transition(state: np.ndarray, parameters: dict[str, Any], contact_height: float) -> np.ndarray:
    dimension = len(state)
    jacobian = np.empty((dimension, dimension))
    for axis in range(dimension):
        plus, minus = state.copy(), state.copy()
        epsilon = 1e-6 * max(1.0, abs(state[axis]))
        plus[axis] += epsilon
        minus[axis] -= epsilon
        jacobian[:, axis] = (_impact(plus, parameters, contact_height)
                              - _impact(minus, parameters, contact_height)) / (2 * epsilon)
    return jacobian


def _validate(states: np.ndarray, bounced: np.ndarray, horizons: np.ndarray,
              parameters: dict[str, Any], contact_height: float) -> None:
    if states.ndim != 2 or states.shape[1] not in (6, 9) or bounced.shape != (len(states),):
        raise ValueError("Expected states N x 6/9 and bounced N")
    if horizons.ndim != 1 or np.any(horizons < 0) or np.any(np.diff(horizons) < 0):
        raise ValueError("Horizons must be sorted and nonnegative")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(horizons)) or not np.isfinite(contact_height):
        raise ValueError("Nonfinite forecast input")
    if not np.isfinite(parameters["k_drag"]) or parameters["k_drag"] < 0:
        raise ValueError("Invalid drag parameter")
    _impact(np.zeros((1, 9)), parameters, contact_height)


def _contact_dt(states: np.ndarray, dt: float | np.ndarray,
                parameters: dict[str, Any], contact_height: float) -> np.ndarray:
    low, high = np.zeros(len(states)), np.broadcast_to(dt, (len(states),)).copy()
    for _ in range(14):
        middle = (low + high) / 2
        above = _step(states, middle, parameters)[:, 2] > contact_height
        low, high = np.where(above, middle, low), np.where(above, high, middle)
    return (low + high) / 2


def _crossing_step(states: np.ndarray, bounced: np.ndarray, dt: float,
                   parameters: dict[str, Any], contact_height: float) -> tuple[np.ndarray, np.ndarray]:
    start, flags = states.copy(), bounced.copy()
    immediate = ~flags & (start[:, 2] <= contact_height) & (start[:, 5] < 0)
    if np.any(immediate):
        start[immediate] = _impact(start[immediate], parameters, contact_height)
        flags[immediate] = True
    end = _step(start, dt, parameters)
    crossing = ~flags & (start[:, 2] > contact_height) & (end[:, 2] <= contact_height) & (end[:, 5] < 0)
    if np.any(crossing):
        incoming = start[crossing]
        contact_dt = _contact_dt(incoming, dt, parameters, contact_height)
        at_contact = _step(incoming, contact_dt, parameters)
        end[crossing] = _step(_impact(at_contact, parameters, contact_height), dt - contact_dt, parameters)
        flags[crossing] = True
    return end, flags


def forecast_states_batch(states: np.ndarray, bounced: np.ndarray, horizons: np.ndarray | list[float],
                          parameters: dict[str, Any], contact_height: float,
                          config: dict[str, Any] | None = None) -> np.ndarray:
    """Batched forecast, N x H x D; only the first descending impact is applied."""
    cfg = _configuration(config)
    states, bounced = np.asarray(states, dtype=float), np.asarray(bounced, dtype=bool)
    horizons = np.asarray(horizons, dtype=float)
    _validate(states, bounced, horizons, parameters, contact_height)
    output = np.empty((len(states), len(horizons), states.shape[1]))
    current, flags, previous = states.copy(), bounced.copy(), 0.0
    for index, horizon in enumerate(horizons):
        duration = float(horizon - previous)
        if duration > 0:
            steps = max(1, int(np.ceil(duration / cfg["max_step_s"])))
            for _ in range(steps):
                current, flags = _crossing_step(current, flags, duration / steps, parameters, contact_height)
        output[:, index] = current
        previous = float(horizon)
    return output


def forecast_batch(states: np.ndarray, bounced: np.ndarray, horizons: np.ndarray | list[float],
                   parameters: dict[str, Any], contact_height: float,
                   config: dict[str, Any] | None = None) -> np.ndarray:
    return forecast_states_batch(states, bounced, horizons, parameters, contact_height, config)[..., :3]


def landing_batch(states: np.ndarray, bounced: np.ndarray, parameters: dict[str, Any],
                  contact_height: float, config: dict[str, Any] | None = None,
                  max_horizon_s: float = 3.0, step_s: float = 0.004) -> dict[str, np.ndarray]:
    cfg = _configuration(config)
    states, bounced = np.asarray(states, dtype=float), np.asarray(bounced, dtype=bool)
    _validate(states, bounced, np.zeros(0), parameters, contact_height)
    if not np.isfinite(max_horizon_s) or max_horizon_s <= 0 or not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("Landing horizon and step must be positive finite")
    count = len(states)
    valid, contact_t = np.zeros(count, bool), np.full(count, np.nan)
    positions, velocities = np.full((count, 3), np.nan), np.full((count, 3), np.nan)
    reason = np.full(count, "no_contact_within_horizon", object)
    reason[bounced] = "already_bounced"
    active, current = ~bounced.copy(), states.copy()
    immediate = active & (current[:, 2] <= contact_height) & (current[:, 5] < 0)
    valid[immediate], contact_t[immediate], reason[immediate] = True, 0.0, "contact_now"
    positions[immediate], velocities[immediate] = current[immediate, :3], current[immediate, 3:6]
    positions[immediate, 2] = contact_height
    active[immediate] = False
    previous = 0.0
    while previous < max_horizon_s - 1e-12 and np.any(active):
        dt = min(step_s, cfg["max_step_s"], max_horizon_s - previous)
        index = np.flatnonzero(active)
        before, after = current[index], _step(current[index], dt, parameters)
        crossing = (before[:, 2] > contact_height) & (after[:, 2] <= contact_height) & (after[:, 5] < 0)
        if np.any(crossing):
            incoming = before[crossing]
            impact_dt = _contact_dt(incoming, dt, parameters, contact_height)
            at_contact = _step(incoming, impact_dt, parameters)
            destination = index[crossing]
            valid[destination], contact_t[destination] = True, previous + impact_dt
            positions[destination], velocities[destination] = at_contact[:, :3], at_contact[:, 3:6]
            positions[destination, 2], reason[destination] = contact_height, "predicted_contact"
            active[destination] = False
        current[index] = after
        previous += dt
    return {"valid": valid, "time_to_contact": contact_t, "position": positions,
            "velocity": velocities, "reason": reason}


def _local_fit(times: np.ndarray, positions: np.ndarray, sigma: np.ndarray,
               acceleration: np.ndarray) -> dict[str, Any]:
    """Acceleration-compensated robust position/velocity fit at the last timestamp.

    Includes measured noise and observed residual dispersion in uncertainty;
    reports independent first/last-half velocity agreement to catch a position
    discontinuity that can look convincing in a very short fit.
    """
    centered = times - times[-1]
    design = np.column_stack((np.ones(len(times)), centered))
    corrected = positions - .5 * centered[:, None] ** 2 * acceleration
    weights = np.ones(len(times))
    coefficients = np.zeros((2, 3))
    for _ in range(8):
        coefficients = np.linalg.lstsq(design * np.sqrt(weights[:, None]),
                                        corrected * np.sqrt(weights[:, None]), rcond=None)[0]
        errors = np.linalg.norm((design @ coefficients - corrected) / sigma, axis=1)
        weights = np.minimum(1.0, 2.5 / np.maximum(errors, 1e-12))
    residuals = corrected - design @ coefficients
    normalized = np.linalg.norm(residuals / sigma, axis=1)
    inlier = normalized <= 4.5
    gram = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(gram)
    variance = np.maximum(sigma ** 2, np.sum(weights[:, None] * residuals ** 2, axis=0)
                          / max(1.0, float(weights.sum() - 2)))
    state_cov = np.zeros((6, 6))
    for axis in range(3):
        ids = [axis, axis + 3]
        state_cov[np.ix_(ids, ids)] = covariance * variance[axis]
    midpoint = (times[0] + times[-1]) / 2
    first, last = times <= midpoint, times > midpoint
    consistency = np.inf
    if np.count_nonzero(first) >= 4 and np.count_nonzero(last) >= 4:
        c1 = np.linalg.lstsq(design[first], corrected[first], rcond=None)[0]
        c2 = np.linalg.lstsq(design[last], corrected[last], rcond=None)[0]
        consistency = float(np.linalg.norm(c1[1] - c2[1]))
    return {"position": coefficients[0], "velocity": coefficients[1], "covariance": state_cov,
            "velocity_sigma": np.sqrt(np.diag(state_cov)[3:6]),
            "typical_error_m": float(np.median(np.linalg.norm(residuals, axis=1))),
            "inlier_fraction": float(inlier.mean()), "consistency_m_s": consistency,
            "span_s": float(times[-1] - times[0]), "samples": len(times)}


class IdentifiedFilter(OnlineFilter):
    """6D repaired baseline or 9D effective-Magnus position-only causal EKF."""

    def __init__(self, parameters: dict[str, Any], contact_height: float,
                 config: dict[str, Any] | None = None, augmented: bool = True,
                 *, impact_response: Callable | None = None):
        if impact_response is not None and not callable(impact_response):
            raise ValueError("impact_response must be callable or None")
        cfg = _configuration(config)
        parameters = dict(parameters)
        parameters.setdefault("impact_mode", "effective")
        parameters.setdefault("magnus_slope", .6)
        old_cfg = {k: cfg[k] for k in OLD_CONFIG}
        super().__init__(parameters["k_drag"], parameters["e_n"], parameters.get("alpha_t", .56),
                         contact_height, old_cfg, augmented)
        self.config, self.parameters = cfg, parameters
        self.impact_response = impact_response
        _validate(np.zeros((0, self.dimension)), np.zeros(0, bool), np.zeros(0), parameters, contact_height)
        self._observed_contact_candidate_t: float | None = None

    def _impact_state(self, state: np.ndarray) -> np.ndarray:
        """Optional first-impact map shared by propagation and recovery prior.

        The original physical mapping is unchanged when no callback is supplied.
        A callback receives a copied length-9 incident state and must preserve
        its position; projection onto the effective event plane is done here.
        """
        if self.impact_response is None:
            return _impact(state, self.parameters, self.contact_height)
        incoming = _padded(np.asarray(state, dtype=float)).copy()
        if incoming[5] < 0:
            outgoing = np.asarray(self.impact_response(incoming.copy()), dtype=float)
            if outgoing.shape != (9,) or not np.isfinite(outgoing).all():
                raise ValueError("impact_response must return a finite length-9 state")
            if not np.allclose(outgoing[:3], incoming[:3], rtol=0, atol=1e-12):
                raise ValueError("instantaneous impact_response must preserve position")
        else:
            outgoing = incoming
        outgoing = outgoing[:len(state)].copy()
        outgoing[2] = self.contact_height
        return outgoing

    def _impact_transition(self, state: np.ndarray) -> np.ndarray:
        if self.impact_response is None:
            return _impact_transition(state, self.parameters, self.contact_height)
        dimension = len(state)
        jacobian = np.empty((dimension, dimension))
        for axis in range(dimension):
            plus, minus = state.copy(), state.copy()
            epsilon = 1e-6 * max(1.0, abs(state[axis]))
            plus[axis] += epsilon
            minus[axis] -= epsilon
            jacobian[:, axis] = (self._impact_state(plus)-self._impact_state(minus))/(2*epsilon)
        return jacobian

    def snapshot(self) -> dict[str, Any]:
        result = super().snapshot()
        state = result["state"]
        b = np.zeros(3) if state is None or not self.augmented else state[6:9].copy()
        result["b_eff"] = b
        result["a_res"] = None if state is None else np.cross(b, state[3:6])
        result["state_definition"] = "p_v_effective_magnus" if self.augmented else "p_v"
        result["b_eff_units"] = "1/s"
        return result

    def _initialize(self) -> None:
        # The common 80-ms robust initialization estimates p,v with zero spin
        # prior. Its numerical RK4 flight is identical to rhs with b_eff=0.
        cfg = self.config
        self.config = {k: cfg[k] for k in OLD_CONFIG}
        try:
            super()._initialize()
        finally:
            self.config = cfg
        if self.augmented:
            self.state[6:9] = 0.0
            self.covariance[6:9, 6:9] = np.eye(3) * self.config["initial_magnus_sigma"] ** 2
        self._diagnostics["magnus_initialization"] = "zero_mean_effective_coefficient_prior"

    def _noise(self, dt: float) -> np.ndarray:
        q = np.zeros((self.dimension, self.dimension))
        for axis in range(3):
            ids = [axis, axis + 3]
            q[np.ix_(ids, ids)] = self.config["process_accel_sigma"] ** 2 * np.array(
                [[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        if self.augmented:
            q[6:9, 6:9] = np.eye(3) * self.config["magnus_drive_sigma"] ** 2 * dt
        return q

    def _advance(self, duration: float) -> None:
        if duration <= 0:
            return
        steps = max(1, int(np.ceil(duration / self.config["max_step_s"])))
        dt = duration / steps
        elapsed = 0.0
        for _ in range(steps):
            before = self.state.copy()
            after, transition = _step_with_transition(before, dt, self.parameters)
            immediate = not self.bounced and before[2] <= self.contact_height and before[5] < 0
            crossing = not self.bounced and before[2] > self.contact_height and after[2] <= self.contact_height and after[5] < 0
            if immediate or crossing:
                impact_dt = 0.0 if immediate else float(_contact_dt(before[None], dt, self.parameters, self.contact_height)[0])
                contact, first_transition = _step_with_transition(before, impact_dt, self.parameters)
                self.covariance = first_transition @ self.covariance @ first_transition.T + self._noise(impact_dt)
                reset_transition = self._impact_transition(contact)
                self.covariance = reset_transition @ self.covariance @ reset_transition.T
                self.covariance[:3, :3] += np.eye(3) * self.config["bounce_position_sigma"] ** 2
                self.covariance[3:6, 3:6] += np.eye(3) * self.config["bounce_velocity_sigma"] ** 2
                if self.augmented:
                    self.covariance[6:9, 6:9] += np.eye(3) * .25 * self.config["initial_magnus_sigma"] ** 2
                outgoing = self._impact_state(contact)
                self.state, last_transition = _step_with_transition(outgoing, dt - impact_dt, self.parameters)
                self.covariance = last_transition @ self.covariance @ last_transition.T + self._noise(dt - impact_dt)
                self.bounced = True
                self._diagnostics["predicted_contact_capture_t_s"] = float(self.last_t + elapsed + impact_dt)
            else:
                self.state = after
                self.covariance = transition @ self.covariance @ transition.T + self._noise(dt)
            elapsed += dt
        self.covariance = (self.covariance + self.covariance.T) / 2

    def _record_recent(self, timestamp: float, position: np.ndarray) -> None:
        if self._recent:
            last_t, last_position = self._recent[-1]
            if np.linalg.norm(position - last_position) > self.config["history_max_physical_speed_m_s"] * (timestamp - last_t) + .1:
                self._diagnostics["history_observation_excluded"] = "position_jump"
                return
        self._recent.append((timestamp, position.copy()))
        while self._recent and timestamp - self._recent[0][0] > self.config["history_window_s"]:
            self._recent.popleft()

    def _valid_fit(self, fit: dict[str, Any], minimum_span: float, minimum_samples: int) -> tuple[bool, str]:
        checks = [(fit["span_s"] >= minimum_span, "insufficient_time_span"),
                  (fit["samples"] >= minimum_samples, "insufficient_samples"),
                  (fit["typical_error_m"] <= self.config["local_fit_max_error_m"], "large_fit_residual"),
                  (fit["inlier_fraction"] >= self.config["local_fit_min_inlier_fraction"], "too_few_inliers"),
                  (np.max(fit["velocity_sigma"]) <= self.config["local_velocity_sigma_limit_m_s"], "velocity_uncertain"),
                  (fit["consistency_m_s"] <= max(self.config["local_fit_consistency_m_s"], .15 * np.linalg.norm(fit["velocity"])), "inconsistent_subwindows"),
                  (np.linalg.norm(fit["velocity"]) <= self.config["history_max_physical_speed_m_s"], "outside_physical_speed_prior")]
        for success, reason in checks:
            if not success:
                return False, reason
        return True, "consistent"

    def _fuse_local(self, fit: dict[str, Any], permit_position_shift: bool = False) -> bool:
        observation = np.r_[fit["position"], fit["velocity"]]
        covariance = fit["covariance"].copy()
        covariance[:3, :3] += np.eye(3) * self.config["recovery_position_sigma_m"] ** 2
        covariance[3:6, 3:6] += np.eye(3) * self.config["recovery_covariance_floor_m_s"] ** 2
        delta = observation - self.state[:6]
        innovation_cov = self.covariance[:6, :6] + covariance
        v_distance = float(delta[3:6] @ np.linalg.solve(innovation_cov[3:6, 3:6], delta[3:6]))
        self._diagnostics["recovery_velocity_prior_mahalanobis_squared"] = v_distance
        if v_distance > self.config["bounce_prior_velocity_gate"]:
            self._diagnostics["recovery_rejected"] = "velocity_inconsistent_with_physical_prior"
            return False
        # Gradual velocity correction: inflate fitted velocity noise until no
        # singular-value direction has gain larger than the configured bound.
        limit = self.config["recovery_max_velocity_gain"]
        for _ in range(8):
            innovation_cov = self.covariance[:6, :6] + covariance
            gain = np.linalg.solve(innovation_cov, self.covariance[:6, :]).T
            if np.linalg.norm(gain[3:6, 3:6], ord=2) <= limit + 1e-12:
                break
            covariance[3:6, 3:6] *= 2.0
        if permit_position_shift:
            # Causal reacquisition may follow a stable camera-position offset.
            # The position transition is explicit; velocity remains fused.
            self.state[:3] = observation[:3]
            delta[:3] = 0.0
        self.state += gain @ delta
        identity_minus_kh = np.eye(self.dimension)
        identity_minus_kh[:, :6] -= gain
        self.covariance = identity_minus_kh @ self.covariance @ identity_minus_kh.T + gain @ covariance @ gain.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        self._diagnostics["recovery_velocity_gain_norm"] = float(np.linalg.norm(gain[3:6, 3:6], ord=2))
        self._diagnostics["recovery_fit_span_s"] = fit["span_s"]
        self._diagnostics["recovery_fit_samples"] = fit["samples"]
        self._diagnostics["recovery_fit_velocity_sigma_m_s"] = fit["velocity_sigma"].tolist()
        return True

    def _confirm_bounce(self) -> bool:
        if self.bounce_confirmed or len(self._recent) < 16:
            return False
        times = np.array([x[0] for x in self._recent])
        positions = np.array([x[1] for x in self._recent])
        # Running median (not convolutional mean) rejects isolated height spikes.
        z = positions[:, 2]
        smooth_z = z.copy()
        smooth_z[1:-1] = np.median(np.stack((z[:-2], z[1:-1], z[2:])), axis=0)
        smooth_z[0], smooth_z[-1] = np.median(z[:2]), np.median(z[-2:])
        minimum = int(np.argmin(smooth_z))
        if minimum < 6 or smooth_z[minimum] > self.contact_height + self.config["bounce_observation_height_m"]:
            return False
        before = np.arange(max(0, minimum - 20), minimum)
        after = np.arange(minimum + 2, len(times))
        if len(after) < self.config["bounce_post_min_samples"]:
            self._diagnostics["bounce_recovery_wait"] = "insufficient_samples"
            return False
        if times[after[-1]] - times[after[0]] < self.config["bounce_post_min_span_s"]:
            self._diagnostics["bounce_recovery_wait"] = "insufficient_time_span"
            return False
        acc = dynamics(self.state, self.parameters)[3:6]
        pre_fit = _local_fit(times[before], positions[before], self.config["measurement_sigma_xyz"], acc)
        post_fit = _local_fit(times[after], positions[after], self.config["measurement_sigma_xyz"], acc)
        pre_ok, pre_reason = self._valid_fit(pre_fit, self.config["bounce_pre_min_span_s"], 8)
        post_ok, post_reason = self._valid_fit(post_fit, self.config["bounce_post_min_span_s"], self.config["bounce_post_min_samples"])
        if not pre_ok or not post_ok:
            self._diagnostics["bounce_recovery_rejected"] = f"pre:{pre_reason};post:{post_reason}"
            return False
        if pre_fit["velocity"][2] >= -.8 or post_fit["velocity"][2] <= .8:
            return False
        # If contact prediction was late, form an outgoing physical prior first.
        # Retain its horizontal direction and uncertainty; never install the
        # fitted outgoing velocity without checking and gradually fusing it.
        if not self.bounced:
            prior_state, prior_covariance = self.state.copy(), self.covariance.copy()
            self.state = self._impact_state(self.state)
            self.covariance[3:6, 3:6] += np.eye(3) * self.config["bounce_velocity_sigma"] ** 2
            self.covariance[:3, :3] += np.eye(3) * self.config["bounce_position_sigma"] ** 2
            self.state[:3] = post_fit["position"]
            if not self._fuse_local(post_fit):
                self.state, self.covariance = prior_state, prior_covariance
                return False
            self.bounced = True
        elif not self._fuse_local(post_fit):
            return False
        self.bounce_confirmed = True
        self._bounce_confirmation_t = float(times[-1])
        self._observed_contact_candidate_t = float(times[minimum])
        self._diagnostics["observed_bounce_minimum_s"] = float(times[minimum])
        self._diagnostics["observed_bounce_confirmation_s"] = float(times[-1])
        self._diagnostics["observed_bounce_reset"] = False
        self._diagnostics["observed_bounce_gradual_fusion"] = True
        self._consecutive_rejections = 0
        return True

    def _reacquire_from_history(self) -> bool:
        if self._consecutive_rejections < self.config["reacquire_after_rejections"] or not self._recent:
            return False
        current_t = self._recent[-1][0]
        start = current_t - max(self.config["reacquire_window_s"], self.config["bounce_post_min_span_s"])
        if self._observed_contact_candidate_t is not None:
            start = max(start, self._observed_contact_candidate_t + .005)
        history = [(t, p) for t, p in self._recent if t >= start]
        if len(history) < 10 or history[-1][0] - history[0][0] < self.config["reacquire_min_span_s"]:
            return False
        times, positions = np.array([x[0] for x in history]), np.array([x[1] for x in history])
        minimum = int(np.argmin(positions[:, 2]))
        near_contact = positions[minimum, 2] < self.contact_height + self.config["bounce_observation_height_m"]
        if near_contact and 2 <= minimum < len(times) - 2 and not self.bounce_confirmed:
            return False
        fit = _local_fit(times, positions, self.config["measurement_sigma_xyz"], dynamics(self.state, self.parameters)[3:6])
        okay, reason = self._valid_fit(fit, self.config["reacquire_min_span_s"], 10)
        if not okay:
            self._diagnostics["reacquisition_rejected"] = reason
            return False
        if not self._fuse_local(fit, permit_position_shift=True):
            return False
        self._diagnostics["reacquisition_span_s"] = fit["span_s"]
        self._consecutive_rejections = 0
        return True

    def update(self, t_s: float, position_xyz: np.ndarray | list[float]) -> dict[str, Any]:
        started = perf_counter()
        self._last_accepted = False
        try:
            position, t_s = np.asarray(position_xyz, dtype=float), float(t_s)
        except (TypeError, ValueError):
            self._last_reason = "invalid_observation"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        if not np.isfinite(t_s) or position.shape != (3,) or not np.all(np.isfinite(position)):
            self._last_reason = "invalid_observation"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        if self.last_t is not None and t_s <= self.last_t:
            self._last_reason = "duplicate_or_old_timestamp"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        duration = 0.0 if self.last_t is None else t_s - self.last_t
        self._diagnostics = {"capture_dt_s": duration, "timestamp_basis": "measurement_capture_time"}
        self._record_recent(t_s, position)
        if self.state is None:
            self._initial.append((t_s, position.copy()))
            self.last_t, self._last_accepted, self._last_reason = t_s, True, "initialization_buffer"
            if len(self._initial) >= 8 and t_s - self._initial[0][0] >= self.config["init_window_s"] - 1e-12:
                self._initialize()
                self._last_reason = "initialized"
            self._diagnostics["update_latency_ms"] = (perf_counter() - started) * 1000
            return self.snapshot()
        self._advance(duration)
        self.last_t = t_s
        confirmed = self._confirm_bounce()
        innovation = position - self.state[:3]
        measurement_covariance = np.diag(self.config["measurement_sigma_xyz"] ** 2)
        innovation_covariance = self.covariance[:3, :3] + measurement_covariance
        distance = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        self._diagnostics["innovation_mahalanobis_squared"] = distance
        self._diagnostics["observed_bounce_confirmed_this_frame"] = confirmed
        if distance > self.config["gate_threshold"]:
            self._last_reason = "mahalanobis_outlier"
            self._consecutive_rejections += 1
            if self._reacquire_from_history():
                self._last_reason, self._last_accepted = "reacquired_from_causal_history", True
        else:
            inflation = max(1.0, distance / self.config["soft_gate_threshold"])
            measurement_covariance *= inflation
            innovation_covariance = self.covariance[:3, :3] + measurement_covariance
            gain = np.linalg.solve(innovation_covariance, self.covariance[:3, :]).T
            self.state += gain @ innovation
            identity_minus_kh = np.eye(self.dimension)
            identity_minus_kh[:, :3] -= gain
            self.covariance = identity_minus_kh @ self.covariance @ identity_minus_kh.T + gain @ measurement_covariance @ gain.T
            self.covariance = (self.covariance + self.covariance.T) / 2
            self._last_accepted, self._last_reason, self._consecutive_rejections = True, "accepted", 0
            self._diagnostics["measurement_noise_inflation"] = inflation
        self._diagnostics["consecutive_rejections"] = self._consecutive_rejections
        self._diagnostics["update_latency_ms"] = (perf_counter() - started) * 1000
        return self.snapshot()

    def predict(self, t_s: float) -> dict[str, Any]:
        if not np.isfinite(t_s) or self.last_t is None or t_s < self.last_t:
            raise ValueError("Prediction timestamp must be finite and >= last capture time")
        result = self.snapshot()
        if self.state is None:
            return result
        duration = t_s - self.last_t
        predicted = forecast_states_batch(self.state[None], np.array([self.bounced]), [duration],
                                           self.parameters, self.contact_height, self.config)[0, 0]
        future_bounced = self.bounced
        if not future_bounced and duration > 0:
            future_bounced = bool(landing_batch(self.state[None], np.array([False]), self.parameters,
                                  self.contact_height, self.config, max_horizon_s=duration)["valid"][0])
        b = predicted[6:9] if self.augmented else np.zeros(3)
        result.update(state=predicted, p=predicted[:3].copy(), v=predicted[3:6].copy(),
                      b_eff=b.copy(), a_res=np.cross(b, predicted[3:6]), t_s=float(t_s),
                      bounced=future_bounced, covariance=None)
        result["diagnostics"].update(prediction_dt_s=float(duration), synthetic_measurement=False)
        return result
