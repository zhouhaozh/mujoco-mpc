"""Isolated causal observer ablations; never used by the default player.

The baseline delegates to HeightPolicyPredictor exactly. Experimental updates
keep its raw-history, first-height event, warmup, and hold semantics, while
changing only measurement covariance, recovery gains, or the endpoint supplied
to a position update. Labels, future points, and per-evaluation fitted scales
are not accepted. An external training-only noise proxy is optional and is NOT
interpreted as calibrated measurement precision.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .height_policy import HeightPolicyPredictor, _PredictionOnlyFrame
from .identified_filter import dynamics, _local_fit


EXPERIMENTS = {
    "baseline": dict(adaptive=False, huber=False, protection=False, smoothing=False),
    "huber_only": dict(adaptive=False, huber=True, protection=False, smoothing=False),
    "adaptive_huber": dict(adaptive=True, huber=True, protection=False, smoothing=False),
    "protected_only": dict(adaptive=False, huber=False, protection=True, smoothing=False),
    "adaptive_protected": dict(adaptive=True, huber=True, protection=True, smoothing=False),
    "smooth_only": dict(adaptive=False, huber=False, protection=False, smoothing=True),
    "protected_smooth": dict(adaptive=False, huber=False, protection=True, smoothing=True),
    "adaptive_protected_smooth": dict(adaptive=True, huber=True, protection=True, smoothing=True),
}

FIXED_SETTINGS = {
    "huber_threshold_sigma": 2.5,
    "noise_proxy_geometric_blend_strength": .5,
    "noise_sigma_ratio_min": 1.,
    "noise_sigma_ratio_max": 2.,
    "quality_history_s": .040,
    "quality_sigma_ratio_max": 2.,
    "recovery_window_s": .080,
    "recovery_phase_variance_multiplier_start": 2.,
    "recovery_delta_v_budget_start_m_s": .20,
    "recovery_delta_v_budget_end_m_s": .80,
    "recovery_position_to_velocity_gain_start_s_inv": 3.,
    "recovery_position_to_velocity_gain_end_s_inv": 12.,
    "recovery_local_velocity_gain_start": .25,
    "smoothing_window_s": .020,
    "smoothing_min_span_s": .012,
    "smoothing_min_samples": 8,
    "smoothing_guard_s": .020,
    "smoothing_raw_height_guard_m": .080,
    "smoothing_max_endpoint_shift_m": .030,
}


class RobustHeightPredictor(HeightPolicyPredictor):
    """GP/height policy with a fixed, named experimental observer recipe.

    ``noise_sigma_xyz`` must be estimated outside this class from training
    groups only. It is bounded and geometrically blended with the original R.
    Every adaptive quality calculation inside this class uses already received
    raw measurements, without altering the scoring reference or history.
    """

    def __init__(self, bank, model_id="gp", policy="hold100", *, config=None,
                 hold_s=.1, experiment="baseline", noise_sigma_xyz=None):
        if experiment not in EXPERIMENTS:
            raise ValueError(f"unknown observer experiment: {experiment}")
        self.experiment = experiment
        self.experimental_flags = dict(EXPERIMENTS[experiment])
        self._velocity_budget_remaining = None
        self._recovery_fraction = None
        self._last_raw_position = None
        self.noise_proxy_sigma_xyz = None
        super().__init__(bank, model_id, policy, config=config, hold_s=hold_s)
        if noise_sigma_xyz is not None:
            sigma = np.asarray(noise_sigma_xyz, dtype=float)
            if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
                raise ValueError("training-only noise proxy must be finite positive XYZ")
            self.noise_proxy_sigma_xyz = sigma.copy()

    def snapshot(self):
        result = super().snapshot()
        result["observer_experiment"] = self.experiment
        result["diagnostics"]["observer_experiment"] = self.experiment
        return result

    def _near_contact(self, raw_position):
        if raw_position[2] <= self.contact_height + FIXED_SETTINGS["smoothing_raw_height_guard_m"]:
            return True
        # A modeled event can be early/late. Also bypass if any received point
        # in the prospective smoothing window is close to the surface, so a
        # now-high rising endpoint cannot hide a recent raw height minimum.
        recent = [(t, p) for t, p in self._recent
                  if t >= self.last_t - FIXED_SETTINGS["smoothing_window_s"]]
        if any(p[2] <= self.contact_height + FIXED_SETTINGS["smoothing_raw_height_guard_m"]
               for _, p in recent):
            return True
        if self.height_triggered:
            return self.last_t < self.height_contact_t_s + FIXED_SETTINGS["smoothing_guard_s"]
        if self.state[5] < 0:
            ttc = (self.state[2] - self.contact_height) / max(-self.state[5], .1)
            return ttc <= FIXED_SETTINGS["smoothing_window_s"] + FIXED_SETTINGS["smoothing_guard_s"]
        return False

    def _eligible_history(self, window, *, include_current):
        start = self.last_t - window
        if self.height_triggered:
            start = max(start, self.height_contact_t_s + .005)
        return [(t, p) for t, p in self._recent if t >= start
                and (include_current or t < self.last_t - 1e-12)]

    def _smoothed_endpoint(self, raw_position):
        """Robust acceleration-compensated fit at NOW, with no future samples."""
        self._diagnostics["smoothing_applied"] = False
        if self._near_contact(raw_position):
            self._diagnostics["smoothing_bypassed"] = "height_event_guard"
            return raw_position
        history = self._eligible_history(FIXED_SETTINGS["smoothing_window_s"], include_current=True)
        if (len(history) < FIXED_SETTINGS["smoothing_min_samples"]
                or history[-1][0] - history[0][0] < FIXED_SETTINGS["smoothing_min_span_s"]
                or abs(history[-1][0] - self.last_t) > 1e-12):
            self._diagnostics["smoothing_bypassed"] = "insufficient_same_phase_history"
            return raw_position
        times = np.array([t for t, _ in history])
        points = np.array([p for _, p in history])
        fit = _local_fit(times, points, self.config["measurement_sigma_xyz"],
                         dynamics(self.state, self.parameters)[3:6])
        if (fit["typical_error_m"] > self.config["local_fit_max_error_m"]
                or fit["inlier_fraction"] < self.config["local_fit_min_inlier_fraction"]):
            self._diagnostics["smoothing_bypassed"] = "poor_fit_quality"
            return raw_position
        change = fit["position"] - raw_position
        length = float(np.linalg.norm(change))
        if length > FIXED_SETTINGS["smoothing_max_endpoint_shift_m"]:
            change *= FIXED_SETTINGS["smoothing_max_endpoint_shift_m"] / length
        self._diagnostics.update(smoothing_applied=True,
                                 smoothing_history_start_s=float(times[0]),
                                 smoothing_history_end_s=float(times[-1]),
                                 smoothing_endpoint_shift_m=float(np.linalg.norm(change)))
        # R is never divided by sample count: overlapping causal windows are
        # correlated, and their fit precision is not calibrated sensor noise.
        return raw_position + change

    def _measurement_covariance(self, innovation):
        nominal = self.config["measurement_sigma_xyz"]
        sigma = nominal.copy()
        if self.experimental_flags["adaptive"]:
            if self.noise_proxy_sigma_xyz is not None:
                ratio = np.sqrt(self.noise_proxy_sigma_xyz / nominal)
                sigma *= np.clip(ratio, 1., 2.)
            # Separate XYZ quality ratios from the previous raw window. The
            # current innovation cannot inflate this history-only proxy.
            history = self._eligible_history(FIXED_SETTINGS["quality_history_s"], include_current=False)
            quality = np.ones(3)
            if len(history) >= 8 and history[-1][0] - history[0][0] >= .015:
                times = np.array([t for t, _ in history])
                points = np.array([p for _, p in history])
                centered = times - times[-1]
                design = np.column_stack((np.ones(len(times)), centered))
                acceleration = dynamics(self.state, self.parameters)[3:6]
                corrected = points - .5 * centered[:, None] ** 2 * acceleration
                coefficients = np.linalg.lstsq(design, corrected, rcond=None)[0]
                residual = corrected - design @ coefficients
                mad = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
                quality = np.clip(mad / sigma, 1., 2.)
                sigma *= quality
            phase = 1.
            if self.resume_t_s is not None:
                age = self.last_t - self.resume_t_s
                phase += max(0., 1. - age / FIXED_SETTINGS["recovery_window_s"])
            self._diagnostics["adaptive_quality_sigma_ratio_xyz"] = quality.tolist()
            self._diagnostics["adaptive_phase_variance_multiplier"] = phase
            covariance = np.diag(sigma ** 2 * phase)
        else:
            covariance = np.diag(sigma ** 2)
        if self.experimental_flags["huber"]:
            standardized = np.abs(innovation) / np.sqrt(np.diag(self.covariance[:3, :3] + covariance))
            weights = np.minimum(1., FIXED_SETTINGS["huber_threshold_sigma"] / np.maximum(standardized, 1e-12))
            covariance = np.diag(np.diag(covariance) / weights)
            self._diagnostics["huber_axis_weights"] = weights.tolist()
        self._diagnostics["experimental_measurement_variance_xyz"] = np.diag(covariance).tolist()
        return covariance

    def _start_recovery_budget(self):
        self._velocity_budget_remaining, self._recovery_fraction = None, None
        if not self.experimental_flags["protection"] or self.resume_t_s is None:
            return
        age = self.last_t - self.resume_t_s
        if not 0 <= age < FIXED_SETTINGS["recovery_window_s"]:
            return
        self._recovery_fraction = float(age / FIXED_SETTINGS["recovery_window_s"])
        self._velocity_budget_remaining = .20 + .60 * self._recovery_fraction
        self._diagnostics["recovery_delta_v_budget_m_s"] = self._velocity_budget_remaining
        self._diagnostics["recovery_delta_v_spent_m_s"] = 0.

    def _protected_gain(self, gain, delta, *, local=False):
        if self._velocity_budget_remaining is None:
            return gain
        gain = gain.copy()
        fraction = self._recovery_fraction
        position_cap = 3. + 9. * fraction
        position_norm = float(np.linalg.norm(gain[3:6, :3], ord=2))
        gain[3:6, :3] *= min(1., position_cap / max(position_norm, 1e-12))
        if local:
            local_cap = .25 + (self.config["recovery_max_velocity_gain"] - .25) * fraction
            velocity_norm = float(np.linalg.norm(gain[3:6, 3:6], ord=2))
            gain[3:6, 3:6] *= min(1., local_cap / max(velocity_norm, 1e-12))
        norm = float(np.linalg.norm(gain[3:6] @ delta))
        gain[3:6] *= min(1., self._velocity_budget_remaining / max(norm, 1e-12))
        spent = float(np.linalg.norm(gain[3:6] @ delta))
        self._velocity_budget_remaining = max(0., self._velocity_budget_remaining - spent)
        self._diagnostics["recovery_delta_v_spent_m_s"] += spent
        self._diagnostics["recovery_gain_protected"] = True
        return gain

    def _apply_gain(self, gain, delta, observation_covariance):
        """Joseph covariance uses exactly the gain that changed the mean."""
        self.state += gain @ delta
        ikh = np.eye(self.dimension)
        ikh[:, :len(delta)] -= gain
        self.covariance = ikh @ self.covariance @ ikh.T + gain @ observation_covariance @ gain.T
        self.covariance = (self.covariance + self.covariance.T) / 2

    def _fuse_local(self, fit, permit_position_shift=False):
        if self._velocity_budget_remaining is None:
            return super()._fuse_local(fit, permit_position_shift)
        observation = np.r_[fit["position"], fit["velocity"]]
        covariance = fit["covariance"].copy()
        covariance[:3, :3] += np.eye(3) * self.config["recovery_position_sigma_m"] ** 2
        covariance[3:6, 3:6] += np.eye(3) * self.config["recovery_covariance_floor_m_s"] ** 2
        delta = observation - self.state[:6]
        innovation_cov = self.covariance[:6, :6] + covariance
        distance = float(delta[3:6] @ np.linalg.solve(innovation_cov[3:6, 3:6], delta[3:6]))
        self._diagnostics["recovery_velocity_prior_mahalanobis_squared"] = distance
        if distance > self.config["bounce_prior_velocity_gate"]:
            self._diagnostics["recovery_rejected"] = "velocity_inconsistent_with_physical_prior"
            return False
        limit = self.config["recovery_max_velocity_gain"]
        for _ in range(8):
            innovation_cov = self.covariance[:6, :6] + covariance
            gain = np.linalg.solve(innovation_cov, self.covariance[:6, :]).T
            if np.linalg.norm(gain[3:6, 3:6], ord=2) <= limit + 1e-12:
                break
            covariance[3:6, 3:6] *= 2.
        if permit_position_shift:
            self.state[:3] = observation[:3]
            delta[:3] = 0.
        gain = self._protected_gain(gain, delta, local=True)
        self._apply_gain(gain, delta, covariance)
        self._diagnostics.update(recovery_velocity_gain_norm=float(np.linalg.norm(gain[3:6, 3:6], ord=2)),
                                 recovery_fit_span_s=fit["span_s"],
                                 recovery_fit_samples=fit["samples"],
                                 recovery_fit_velocity_sigma_m_s=fit["velocity_sigma"].tolist())
        return True

    def update(self, t_s, position_xyz):
        if self.experiment == "baseline":
            return super().update(t_s, position_xyz)
        if self._projection_only:
            raise RuntimeError("a restored posterior is projection-only")
        # Upstream has no R/gain hook. Keep this small update transaction aligned
        # with IdentifiedFilter.update, and preserve the HeightPolicy control
        # transfer before innovation, confirmation, and reacquisition.
        started = perf_counter()
        self._last_accepted = False
        try:
            position, t_s = np.asarray(position_xyz, dtype=float), float(t_s)
        except (TypeError, ValueError):
            self._last_reason = "invalid_observation"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        if not np.isfinite(t_s) or position.shape != (3,) or not np.isfinite(position).all():
            self._last_reason = "invalid_observation"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        if self.last_t is not None and t_s <= self.last_t:
            self._last_reason = "duplicate_or_old_timestamp"
            self._diagnostics = {"update_latency_ms": (perf_counter() - started) * 1000}
            return self.snapshot()
        duration = 0. if self.last_t is None else t_s - self.last_t
        self._diagnostics = {"capture_dt_s": duration, "timestamp_basis": "measurement_capture_time"}
        self._record_recent(t_s, position)
        self._last_raw_position = position.copy()
        if self.state is None:
            self._initial.append((t_s, position.copy()))
            self.last_t, self._last_accepted, self._last_reason = t_s, True, "initialization_buffer"
            if len(self._initial) >= 8 and t_s - self._initial[0][0] >= self.config["init_window_s"] - 1e-12:
                self._initialize()
                self.initialization_t_s = t_s
                self._last_reason = "initialized"
                if self.bounced:
                    self.height_event_status = "initialization_already_bounced_no_localized_event"
            self._diagnostics["update_latency_ms"] = (perf_counter() - started) * 1000
            return self.snapshot()
        self._inside_policy_update = True
        try:
            try:
                self._advance(duration)
            except _PredictionOnlyFrame:
                self.last_t = t_s
                self._last_reason = "height_policy_prediction_only"
                self._diagnostics.update(observation_assimilated=False, received_history_retained=True,
                                         update_latency_ms=(perf_counter() - started) * 1000)
                return self.snapshot()
        finally:
            self._inside_policy_update = False
        self.last_t = t_s
        self._start_recovery_budget()
        confirmed = self._confirm_bounce()
        measured = self._smoothed_endpoint(position) if self.experimental_flags["smoothing"] else position
        innovation = measured - self.state[:3]
        measurement_covariance = self._measurement_covariance(innovation)
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
            inflation = max(1., distance / self.config["soft_gate_threshold"])
            measurement_covariance *= inflation
            innovation_covariance = self.covariance[:3, :3] + measurement_covariance
            gain = np.linalg.solve(innovation_covariance, self.covariance[:3, :]).T
            gain = self._protected_gain(gain, innovation)
            self._apply_gain(gain, innovation, measurement_covariance)
            self._last_accepted, self._last_reason, self._consecutive_rejections = True, "accepted", 0
            self._diagnostics["measurement_noise_inflation"] = inflation
        if self.policy == "continuous" and self.bounced and not self.height_triggered:
            self.height_event_status = "observed_confirmation_preceded_model_crossing"
        self._diagnostics["consecutive_rejections"] = self._consecutive_rejections
        self._diagnostics["update_latency_ms"] = (perf_counter() - started) * 1000
        return self.snapshot()
