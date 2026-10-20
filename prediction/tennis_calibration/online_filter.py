"""Causal position-only EKF and batched first-bounce trajectory prediction.

All times are capture timestamps in seconds; positions are metres in the
calibration frame. The 9-D variant estimates an equivalent residual acceleration,
not spin. No cleaning flags or offline impact labels enter this module.
"""
from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .flight import GRAVITY, initial_guess, simulate_batch_with_jacobian


DEFAULT_CONFIG = {
    "measurement_sigma_xyz": [0.006, 0.006, 0.008],
    "process_accel_sigma": 0.4,
    "residual_drive_sigma": 3.0,
    "tau_s": 0.6,
    "gate_threshold": 25.0,
    "soft_gate_threshold": 7.815,
    "init_window_s": 0.08,
    "max_step_s": 0.004,
    "initial_velocity_sigma": 0.8,
    "initial_residual_sigma": 5.0,
    "bounce_velocity_sigma": 1.5,
    "bounce_position_sigma": 0.015,
    "bounce_observation_height_m": 0.18,
    "residual_clip_m_s2": 30.0,
    "reacquire_after_rejections": 12,
    "reacquire_window_s": 0.04,
    "reacquire_min_span_s": 0.025,
}


def _configuration(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown online filter configuration: {sorted(unknown)}")
        result.update(config)
    sigma = np.broadcast_to(np.asarray(result["measurement_sigma_xyz"], dtype=float), (3,)).copy()
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("measurement_sigma_xyz must contain positive finite values")
    result["measurement_sigma_xyz"] = sigma
    for key, value in result.items():
        if key == "measurement_sigma_xyz":
            continue
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if result["soft_gate_threshold"] > result["gate_threshold"]:
        raise ValueError("soft_gate_threshold must not exceed gate_threshold")
    return result


def dynamics(state: np.ndarray, k_drag: float, tau_s: float) -> np.ndarray:
    """Continuous dynamics, supporting any leading batch dimensions."""
    state = np.asarray(state, dtype=float)
    if state.shape[-1] not in (6, 9):
        raise ValueError("State must have 6 or 9 components")
    derivative = np.zeros_like(state)
    velocity = state[..., 3:6]
    derivative[..., :3] = velocity
    derivative[..., 3:6] = GRAVITY - k_drag * np.linalg.norm(velocity, axis=-1, keepdims=True) * velocity
    if state.shape[-1] == 9:
        derivative[..., 3:6] += state[..., 6:9]
        derivative[..., 6:9] = -state[..., 6:9] / tau_s
    return derivative


def dynamics_jacobian(state: np.ndarray, k_drag: float, tau_s: float) -> np.ndarray:
    """Analytic Jacobian of the continuous dynamics (one state)."""
    state = np.asarray(state, dtype=float)
    dimension = len(state)
    if state.ndim != 1 or dimension not in (6, 9):
        raise ValueError("Expected one 6-D or 9-D state")
    jacobian = np.zeros((dimension, dimension))
    jacobian[:3, 3:6] = np.eye(3)
    velocity, speed = state[3:6], float(np.linalg.norm(state[3:6]))
    jacobian[3:6, 3:6] = -k_drag * (speed * np.eye(3) + np.outer(velocity, velocity) / max(speed, 1e-15))
    if dimension == 9:
        jacobian[3:6, 6:9] = np.eye(3)
        jacobian[6:9, 6:9] = -np.eye(3) / tau_s
    return jacobian


def _rk4(state: np.ndarray, step: float | np.ndarray, k_drag: float, tau_s: float) -> np.ndarray:
    h = np.asarray(step)
    if h.ndim:
        h = h[..., None]
    k1 = dynamics(state, k_drag, tau_s)
    k2 = dynamics(state + h * k1 / 2, k_drag, tau_s)
    k3 = dynamics(state + h * k2 / 2, k_drag, tau_s)
    k4 = dynamics(state + h * k3, k_drag, tau_s)
    return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _rk4_with_transition(state: np.ndarray, step: float, k_drag: float, tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the same RK4 stages; transition is not an Euler approximation."""
    identity = np.eye(len(state))
    k1 = dynamics(state, k_drag, tau_s)
    j1 = dynamics_jacobian(state, k_drag, tau_s)
    x2 = state + step * k1 / 2
    k2 = dynamics(x2, k_drag, tau_s)
    j2 = dynamics_jacobian(x2, k_drag, tau_s) @ (identity + step * j1 / 2)
    x3 = state + step * k2 / 2
    k3 = dynamics(x3, k_drag, tau_s)
    j3 = dynamics_jacobian(x3, k_drag, tau_s) @ (identity + step * j2 / 2)
    x4 = state + step * k3
    k4 = dynamics(x4, k_drag, tau_s)
    j4 = dynamics_jacobian(x4, k_drag, tau_s) @ (identity + step * j3)
    return (state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6,
            identity + step * (j1 + 2 * j2 + 2 * j3 + j4) / 6)


def _reset(state: np.ndarray, e_n: float, alpha_t: float, contact_height: float) -> np.ndarray:
    state = state.copy()
    state[..., 2] = contact_height
    state[..., 3:5] *= alpha_t
    state[..., 5] *= -e_n
    if state.shape[-1] == 9:
        state[..., 6:9] = 0
    return state


def _crossing_step(states: np.ndarray, bounced: np.ndarray, step: float,
                   k_drag: float, e_n: float, alpha_t: float, contact_height: float,
                   tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized step with one descending first-contact event per trajectory."""
    start = states.copy()
    flags = bounced.copy()
    immediate = ~flags & (start[:, 2] <= contact_height) & (start[:, 5] < 0)
    if np.any(immediate):
        start[immediate] = _reset(start[immediate], e_n, alpha_t, contact_height)
        flags[immediate] = True
    end = _rk4(start, step, k_drag, tau_s)
    crossing = ~flags & (start[:, 2] > contact_height) & (end[:, 2] <= contact_height) & (end[:, 5] < 0)
    if np.any(crossing):
        incoming = start[crossing]
        low, high = np.zeros(len(incoming)), np.full(len(incoming), step)
        # A fixed iteration count keeps all trajectories vectorized. At dt=4 ms
        # this resolves event time to < 1 microsecond.
        for _ in range(14):
            middle = (low + high) / 2
            above = _rk4(incoming, middle, k_drag, tau_s)[:, 2] > contact_height
            low = np.where(above, middle, low)
            high = np.where(above, high, middle)
        contact_time = (low + high) / 2
        impact = _rk4(incoming, contact_time, k_drag, tau_s)
        outgoing = _reset(impact, e_n, alpha_t, contact_height)
        end[crossing] = _rk4(outgoing, step - contact_time, k_drag, tau_s)
        flags[crossing] = True
    return end, flags


def forecast_states_batch(states: np.ndarray, bounced: np.ndarray, horizons: np.ndarray | list[float],
                          k_drag: float, e_n: float, alpha_t: float, contact_height: float,
                          config: dict[str, Any] | None = None) -> np.ndarray:
    """Return N x H x D future states without creating synthetic measurements.

    Horizons are nonnegative seconds after each input state, and must be sorted.
    ``bounced`` indicates whether first contact has already happened. This model
    deliberately does not add a second bounce.
    """
    cfg = _configuration(config)
    states, bounced = np.asarray(states, dtype=float), np.asarray(bounced, dtype=bool)
    horizons = np.asarray(horizons, dtype=float)
    if states.ndim != 2 or states.shape[1] not in (6, 9) or bounced.shape != (len(states),):
        raise ValueError("Expected states N x 6/9 and bounced N")
    if horizons.ndim != 1 or np.any(horizons < 0) or np.any(np.diff(horizons) < 0):
        raise ValueError("Horizons must be a sorted nonnegative 1-D array")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(horizons)):
        raise ValueError("Nonfinite forecast input")
    if not np.isfinite(k_drag) or k_drag < 0 or not np.isfinite(contact_height):
        raise ValueError("Invalid drag or contact height")
    if not (np.isfinite(e_n) and np.isfinite(alpha_t) and 0 <= e_n <= 1.5 and 0 <= alpha_t <= 1.5):
        raise ValueError("Invalid restitution parameters")
    result = np.empty((len(states), len(horizons), states.shape[1]))
    current, flags, previous = states.copy(), bounced.copy(), 0.0
    for j, horizon in enumerate(horizons):
        duration = float(horizon - previous)
        if duration:
            steps = max(1, int(np.ceil(duration / cfg["max_step_s"])))
            for _ in range(steps):
                current, flags = _crossing_step(current, flags, duration / steps, k_drag,
                                                e_n, alpha_t, contact_height, cfg["tau_s"])
        result[:, j] = current
        previous = float(horizon)
    return result


def forecast_batch(states: np.ndarray, bounced: np.ndarray, horizons: np.ndarray | list[float],
                   k_drag: float, e_n: float, alpha_t: float, contact_height: float,
                   config: dict[str, Any] | None = None) -> np.ndarray:
    """Return N x H x 3 forecast positions; see ``forecast_states_batch``."""
    return forecast_states_batch(states, bounced, horizons, k_drag, e_n, alpha_t,
                                 contact_height, config)[..., :3]


def landing_batch(states: np.ndarray, bounced: np.ndarray, k_drag: float, e_n: float,
                  alpha_t: float, contact_height: float,
                  config: dict[str, Any] | None = None, max_horizon_s: float = 3.0,
                  step_s: float = 0.004) -> dict[str, np.ndarray]:
    """Batched first descending contact in future, without applying its reset.

    Already-bounced states are invalid. A state below the effective plane but
    moving upward is allowed to rise above it before the descending contact.
    Input positions and plane share the measurement coordinate convention.
    """
    cfg = _configuration(config)
    states, bounced = np.asarray(states, dtype=float), np.asarray(bounced, dtype=bool)
    forecast_states_batch(states, bounced, [], k_drag, e_n, alpha_t, contact_height, cfg)
    if not np.isfinite(max_horizon_s) or max_horizon_s <= 0 or not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("Landing horizon and step must be positive and finite")
    count = len(states)
    valid = np.zeros(count, dtype=bool)
    time_to_contact = np.full(count, np.nan)
    positions, velocities = np.full((count, 3), np.nan), np.full((count, 3), np.nan)
    reasons = np.full(count, "no_contact_within_horizon", dtype=object)
    reasons[bounced] = "already_bounced"
    active = ~bounced.copy()
    current = states.copy()
    immediate = active & (current[:, 2] <= contact_height) & (current[:, 5] < 0)
    valid[immediate], time_to_contact[immediate], reasons[immediate] = True, 0.0, "contact_now"
    positions[immediate], velocities[immediate] = current[immediate, :3], current[immediate, 3:6]
    positions[immediate, 2] = contact_height
    active[immediate] = False
    previous = 0.0
    while previous < max_horizon_s - 1e-12 and np.any(active):
        dt = min(step_s, max_horizon_s - previous)
        indices = np.flatnonzero(active)
        before = current[indices]
        after = _rk4(before, dt, k_drag, cfg["tau_s"])
        crossing = (before[:, 2] > contact_height) & (after[:, 2] <= contact_height) & (after[:, 5] < 0)
        if np.any(crossing):
            incoming = before[crossing]
            low, high = np.zeros(len(incoming)), np.full(len(incoming), dt)
            for _ in range(14):
                middle = (low + high) / 2
                above = _rk4(incoming, middle, k_drag, cfg["tau_s"])[:, 2] > contact_height
                low, high = np.where(above, middle, low), np.where(above, high, middle)
            contact_time = (low + high) / 2
            impact = _rk4(incoming, contact_time, k_drag, cfg["tau_s"])
            destination = indices[crossing]
            valid[destination], time_to_contact[destination] = True, previous + contact_time
            positions[destination], velocities[destination] = impact[:, :3], impact[:, 3:6]
            positions[destination, 2], reasons[destination] = contact_height, "predicted_contact"
            active[destination] = False
        current[indices] = after
        previous += dt
    return {"valid": valid, "time_to_contact": time_to_contact, "position": positions,
            "velocity": velocities, "reason": reasons}


def _robust_line(times: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = times - times[-1]
    design = np.column_stack((np.ones(len(times)), centered))
    weights = np.ones(len(times))
    for _ in range(6):
        coefficients = np.linalg.lstsq(design * weights[:, None], positions * weights[:, None], rcond=None)[0]
        errors = np.linalg.norm(design @ coefficients - positions, axis=1)
        scale = max(0.002, 1.4826 * float(np.median(np.abs(errors - np.median(errors)))))
        weights = np.sqrt(np.minimum(1, 1.5 * scale / np.maximum(errors, 1e-12)))
    return coefficients[0], coefficients[1]


class OnlineFilter:
    """Position-only EKF using exclusively received capture-time observations.

    Hard-gated measurements still advance the physical prediction to capture
    time. Duplicate/old timestamps and malformed observations do not advance it.
    ``predict`` projects a copy to a requested time without changing last_t.
    Covariance is a model-based approximation, not a calibrated confidence bound.
    """

    def __init__(self, k_drag: float, e_n: float, alpha_t: float, contact_height: float,
                 config: dict[str, Any] | None = None, augmented: bool = True):
        self.config = _configuration(config)
        # Validate common physical parameters through the public batch function.
        forecast_states_batch(np.zeros((0, 9 if augmented else 6)), np.zeros(0, dtype=bool), [],
                              k_drag, e_n, alpha_t, contact_height, self.config)
        self.k_drag, self.e_n, self.alpha_t = float(k_drag), float(e_n), float(alpha_t)
        self.contact_height, self.augmented = float(contact_height), bool(augmented)
        self.dimension = 9 if augmented else 6
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.last_t: float | None = None
        self.bounced = False
        self.bounce_confirmed = False
        self._initial: list[tuple[float, np.ndarray]] = []
        self._recent: deque[tuple[float, np.ndarray]] = deque()
        self._diagnostics: dict[str, Any] = {}
        self._last_accepted = False
        self._last_reason = "not_initialized"
        self._consecutive_rejections = 0
        self._bounce_confirmation_t: float | None = None

    def snapshot(self) -> dict[str, Any]:
        state = None if self.state is None else self.state.copy()
        return {
            "initialized": state is not None, "accepted": self._last_accepted,
            "reason": self._last_reason, "t_s": self.last_t,
            "state": state, "p": None if state is None else state[:3].copy(),
            "v": None if state is None else state[3:6].copy(),
            "a_res": None if state is None else (state[6:9].copy() if self.augmented else np.zeros(3)),
            "covariance": None if self.covariance is None else self.covariance.copy(),
            "bounced": self.bounced, "bounce_confirmed": self.bounce_confirmed,
            "diagnostics": dict(self._diagnostics),
        }

    def _initialize(self) -> None:
        times = np.array([item[0] for item in self._initial])
        positions = np.array([item[1] for item in self._initial])
        relative = times - times[0]
        guess = initial_guess(times, positions)
        sigma = self.config["measurement_sigma_xyz"]
        cached: dict[str, np.ndarray] = {}

        def evaluate(candidate: np.ndarray) -> None:
            if "candidate" not in cached or not np.array_equal(candidate, cached["candidate"]):
                prediction, sensitivity = simulate_batch_with_jacobian(relative[None], candidate[None], self.k_drag,
                                                                         self.config["max_step_s"])
                cached.update(candidate=candidate.copy(), residual=((prediction[0] - positions) / sigma).ravel(),
                              jacobian=(sensitivity[0, :, :, :6] / sigma[None, :, None]).reshape(-1, 6))

        def residual(candidate: np.ndarray) -> np.ndarray:
            evaluate(candidate)
            return cached["residual"]

        def jacobian(candidate: np.ndarray) -> np.ndarray:
            evaluate(candidate)
            return cached["jacobian"]

        fit = least_squares(residual, guess, jac=jacobian, loss="soft_l1", f_scale=1.5,
                            max_nfev=40, xtol=1e-8, ftol=1e-8, gtol=1e-8)
        initial = np.r_[fit.x, np.zeros(3)] if self.augmented else fit.x.copy()
        final = forecast_states_batch(initial[None], np.array([False]), [float(relative[-1])],
                                      self.k_drag, self.e_n, self.alpha_t, self.contact_height, self.config)[0, 0]
        self.state = final
        self.covariance = np.diag(np.r_[sigma ** 2 * 4,
                                       np.full(3, self.config["initial_velocity_sigma"] ** 2),
                                       np.full(3, self.config["initial_residual_sigma"] ** 2) if self.augmented else []])
        # A window crossing contact cannot be safely fitted as one flight. Such
        # traces are flagged and the causal observed-bounce detector can recover.
        self.bounced = bool(np.any(positions[:, 2] <= self.contact_height) and final[5] > 0)
        self._diagnostics["initial_fit_converged"] = bool(fit.success)
        self._diagnostics["initial_observations"] = len(times)
        self._diagnostics["initial_span_s"] = float(relative[-1])
        self._initial.clear()

    def _noise(self, step: float) -> np.ndarray:
        q = np.zeros((self.dimension, self.dimension))
        acceleration = self.config["process_accel_sigma"] ** 2
        for axis in range(3):
            indices = [axis, axis + 3]
            q[np.ix_(indices, indices)] += acceleration * np.array([[step**3 / 3, step**2 / 2],
                                                                   [step**2 / 2, step]])
            if self.augmented:
                indices = [axis, axis + 3, axis + 6]
                # Integrated residual white noise; decay is already represented
                # by the RK4 transition. Small-step local covariance expansion.
                q[np.ix_(indices, indices)] += self.config["residual_drive_sigma"] ** 2 * np.array([
                    [step**5 / 20, step**4 / 8, step**3 / 6],
                    [step**4 / 8, step**3 / 3, step**2 / 2],
                    [step**3 / 6, step**2 / 2, step]])
        return q

    def _reset_covariance(self) -> None:
        jacobian = np.eye(self.dimension)
        jacobian[3, 3] = jacobian[4, 4] = self.alpha_t
        jacobian[5, 5] = -self.e_n
        if self.augmented:
            jacobian[6:9, 6:9] = 0
        self.covariance = jacobian @ self.covariance @ jacobian.T
        self.covariance[:3, :3] += np.eye(3) * self.config["bounce_position_sigma"] ** 2
        self.covariance[3:6, 3:6] += np.eye(3) * self.config["bounce_velocity_sigma"] ** 2
        if self.augmented:
            self.covariance[6:9, 6:9] = np.eye(3) * self.config["initial_residual_sigma"] ** 2
        # Contact-time uncertainty and saltation are approximated by inflation;
        # this covariance is not a certified calibrated landing uncertainty.

    def _advance(self, duration: float) -> None:
        if duration <= 0:
            return
        steps = max(1, int(np.ceil(duration / self.config["max_step_s"])))
        step = duration / steps
        for _ in range(steps):
            before = self.state.copy()
            after, transition = _rk4_with_transition(before, step, self.k_drag, self.config["tau_s"])
            immediate = not self.bounced and before[2] <= self.contact_height and before[5] < 0
            crossing = not self.bounced and before[2] > self.contact_height and after[2] <= self.contact_height and after[5] < 0
            if immediate or crossing:
                if immediate:
                    contact_time = 0.0
                else:
                    low, high = 0.0, step
                    for _ in range(14):
                        middle = (low + high) / 2
                        if _rk4(before, middle, self.k_drag, self.config["tau_s"])[2] > self.contact_height:
                            low = middle
                        else:
                            high = middle
                    contact_time = (low + high) / 2
                contact, first_transition = _rk4_with_transition(before, contact_time, self.k_drag, self.config["tau_s"])
                self.covariance = first_transition @ self.covariance @ first_transition.T + self._noise(contact_time)
                contact = _reset(contact, self.e_n, self.alpha_t, self.contact_height)
                self._reset_covariance()
                self.state, last_transition = _rk4_with_transition(contact, step - contact_time, self.k_drag, self.config["tau_s"])
                self.covariance = last_transition @ self.covariance @ last_transition.T + self._noise(step - contact_time)
                self.bounced = True
            else:
                self.state = after
                self.covariance = transition @ self.covariance @ transition.T + self._noise(step)
        self.covariance = (self.covariance + self.covariance.T) / 2

    def _record_recent(self, timestamp: float, position: np.ndarray) -> None:
        # Plausibility check precedes the bounce detector so one large position
        # outlier cannot create a false descending/ascending contact history.
        if self._recent:
            last_t, last_position = self._recent[-1]
            if np.linalg.norm(position - last_position) > 120 * (timestamp - last_t) + 0.1:
                return
        self._recent.append((timestamp, position.copy()))
        while self._recent and timestamp - self._recent[0][0] > 0.14:
            self._recent.popleft()

    def _confirm_bounce(self) -> bool:
        if self.bounce_confirmed or len(self._recent) < 9:
            return False
        times = np.array([item[0] for item in self._recent])
        positions = np.array([item[1] for item in self._recent])
        # Minimum of a 3-point median suppresses a lone z spike. Confirmation
        # requires actual rising observations; a forecast never confirms contact.
        smoothed_z = np.convolve(positions[:, 2], np.ones(3) / 3, mode="valid")
        minimum = int(np.argmin(smoothed_z)) + 1
        if minimum < 3 or len(times) - minimum < 4:
            return False
        if smoothed_z[minimum - 1] > self.contact_height + self.config["bounce_observation_height_m"]:
            return False
        before = np.arange(max(0, minimum - 8), minimum)
        after = np.arange(minimum + 1, len(times))
        if times[before[-1]] - times[before[0]] < 0.007 or times[after[-1]] - times[after[0]] < 0.007:
            return False
        _, velocity_before = _robust_line(times[before], positions[before])
        position, velocity = _robust_line(times[after], positions[after])
        if velocity_before[2] >= -0.8 or velocity[2] <= 0.8:
            return False
        # Both early and late predicted contacts can disagree with observations.
        # Re-estimate *only* the already observed postcontact side, never fit
        # across the velocity jump, even if prediction already reset its velocity.
        self.state[:3], self.state[3:6] = position, velocity
        if self.augmented:
            self.state[6:9] = 0
        self.covariance = np.diag(np.r_[self.config["measurement_sigma_xyz"] ** 2 * 4,
                                       np.full(3, self.config["bounce_velocity_sigma"] ** 2),
                                       np.full(3, self.config["initial_residual_sigma"] ** 2) if self.augmented else []])
        self.bounced = self.bounce_confirmed = True
        self._diagnostics["observed_bounce_minimum_s"] = float(times[minimum])
        self._diagnostics["observed_bounce_confirmation_s"] = float(times[-1])
        self._bounce_confirmation_t = float(times[-1])
        self._recent.clear()
        return True

    def _reacquire_from_history(self) -> bool:
        """Recover after sustained innovation failures using only recent history.

        A robust local line is adequate for reacquisition over 25--40 ms; its
        velocity uncertainty is intentionally inflated. The check rejects an
        isolated position jump and a window spanning an unconfirmed bounce.
        """
        if self._consecutive_rejections < self.config["reacquire_after_rejections"] or not self._recent:
            return False
        current_t = self._recent[-1][0]
        history = [(t, p) for t, p in self._recent
                   if current_t - t <= self.config["reacquire_window_s"]]
        if len(history) < 8 or history[-1][0] - history[0][0] < self.config["reacquire_min_span_s"]:
            return False
        times, positions = np.array([item[0] for item in history]), np.array([item[1] for item in history])
        if self.bounce_confirmed:
            confirmation = self._bounce_confirmation_t
            if confirmation is not None and times[0] < confirmation:
                return False
        minimum = int(np.argmin(positions[:, 2]))
        near_contact = positions[minimum, 2] < self.contact_height + self.config["bounce_observation_height_m"]
        if near_contact and 2 <= minimum < len(times) - 2:
            # Wait for the mode detector instead of blending opposite velocities.
            return False
        position, velocity = _robust_line(times, positions)
        reconstructed = position + (times - times[-1])[:, None] * velocity
        typical_error = float(np.median(np.linalg.norm(reconstructed - positions, axis=1)))
        maximum_error = max(0.02, 4 * float(np.max(self.config["measurement_sigma_xyz"])))
        if typical_error > maximum_error or np.linalg.norm(velocity) > 120:
            return False
        self.state[:3], self.state[3:6] = position, velocity
        if self.augmented:
            self.state[6:9] = 0
        self.covariance = np.diag(np.r_[self.config["measurement_sigma_xyz"] ** 2 * 4,
                                       np.full(3, self.config["bounce_velocity_sigma"] ** 2),
                                       np.full(3, self.config["initial_residual_sigma"] ** 2) if self.augmented else []])
        self._diagnostics["reacquisition_span_s"] = float(times[-1] - times[0])
        self._diagnostics["reacquisition_typical_error_m"] = typical_error
        self._consecutive_rejections = 0
        return True

    def update(self, t_s: float, position_xyz: np.ndarray | list[float]) -> dict[str, Any]:
        started = perf_counter()
        self._last_accepted = False
        try:
            position = np.asarray(position_xyz, dtype=float)
            t_s = float(t_s)
        except (TypeError, ValueError):
            self._last_reason = "invalid_observation"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        if not np.isfinite(t_s) or position.shape != (3,) or not np.all(np.isfinite(position)):
            self._last_reason = "invalid_observation"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        t_s = float(t_s)
        if self.last_t is not None and t_s <= self.last_t:
            self._last_reason = "duplicate_or_old_timestamp"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        duration = 0.0 if self.last_t is None else t_s - self.last_t
        self._diagnostics = {"capture_dt_s": duration, "timestamp_basis": "measurement_capture_time"}
        self._record_recent(t_s, position)
        if self.state is None:
            self._initial.append((t_s, position.copy()))
            self.last_t = t_s
            self._last_accepted = True
            self._last_reason = "initialization_buffer"
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
        self._diagnostics["observed_bounce_reset"] = confirmed
        if distance > self.config["gate_threshold"]:
            self._last_reason = "mahalanobis_outlier"
            self._consecutive_rejections += 1
            if self._reacquire_from_history():
                self._last_reason = "reacquired_from_causal_history"
                self._last_accepted = True
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
            if self.augmented:
                self.state[6:9] = np.clip(self.state[6:9], -self.config["residual_clip_m_s2"],
                                         self.config["residual_clip_m_s2"])
            self._last_accepted = True
            self._last_reason = "accepted"
            self._consecutive_rejections = 0
            self._diagnostics["measurement_noise_inflation"] = inflation
        self._diagnostics["consecutive_rejections"] = self._consecutive_rejections
        self._diagnostics["update_latency_ms"] = (perf_counter() - started) * 1000
        return self.snapshot()

    def predict(self, t_s: float) -> dict[str, Any]:
        """Project a copy to now/execution time; capture-time posterior stays intact."""
        if not np.isfinite(t_s) or self.last_t is None or t_s < self.last_t:
            raise ValueError("Prediction timestamp must be finite and >= last capture time")
        result = self.snapshot()
        if self.state is None:
            return result
        prediction = forecast_states_batch(self.state[None], np.array([self.bounced]), [t_s - self.last_t],
                                            self.k_drag, self.e_n, self.alpha_t, self.contact_height, self.config)[0, 0]
        future_bounced = self.bounced
        if not self.bounced and t_s > self.last_t:
            contact = landing_batch(self.state[None], np.array([False]), self.k_drag, self.e_n,
                                    self.alpha_t, self.contact_height, self.config,
                                    max_horizon_s=t_s - self.last_t,
                                    step_s=self.config["max_step_s"])
            future_bounced = bool(contact["valid"][0])
        result.update(state=prediction, p=prediction[:3].copy(), v=prediction[3:6].copy(),
                      a_res=prediction[6:9].copy() if self.augmented else np.zeros(3),
                      t_s=float(t_s), bounced=future_bounced)
        result["covariance"] = None  # posterior covariance is at capture time, not this future time
        result["diagnostics"]["prediction_dt_s"] = float(t_s - self.last_t)
        result["diagnostics"]["synthetic_measurement"] = False
        return result
