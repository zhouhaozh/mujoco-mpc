"""Causal position-only EKF with the frozen method-6 collision response.

The first impact in filtering, observed-contact recovery, and future projection
all use the same physics + GP velocity residual.  No model is fitted here.
This is an experimental online adapter, not a replacement of a frozen selected
model.  Its approximate covariance and conservative initialization floors are
not confidence intervals validated for coverage.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .identified_dynamics import impact_map
from .identified_filter import IdentifiedFilter, _contact_dt, _step


class GPOnlinePredictor(IdentifiedFilter):
    """Method-6 first-impact EKF and side-effect-free absolute-clock forecast.

    ``bank`` is an already loaded BounceModelBank.  ``seed_from_prefix`` accepts
    the fitted state at the final received precontact observation, not a state
    at a reference event.  The caller slices observations; future samples are
    rejected rather than silently retained in bounce/recovery history.
    """

    def __init__(self, bank, config: dict[str, Any] | None = None,
                 covariance_floor_sigmas=(.012, .25, .04149060165072289)):
        floor = np.asarray(covariance_floor_sigmas, dtype=float)
        if floor.shape != (3,) or not np.isfinite(floor).all() or np.any(floor <= 0):
            raise ValueError("positive finite p/v/b covariance floor sigmas required")
        self.bank = bank
        self.covariance_floor_sigmas = floor.copy()
        self._projection_only = False
        self._seed_metadata: dict[str, Any] = {}
        super().__init__(bank.parameters, bank.height, config=config, augmented=True,
                         impact_response=self._gp_impact)

    def _gp_impact(self, incoming: np.ndarray) -> np.ndarray:
        # The GP's local frame is undefined for an almost vertical incoming
        # ball.  Use the existing fixed physical law only in that degeneracy;
        # all ordinary descending incoming states use the frozen GP checkpoint.
        if np.linalg.norm(incoming[3:5]) < 1e-6:
            return impact_map(incoming, self.parameters)
        return self.bank.apply_state(incoming, "gp")

    @staticmethod
    def _validated_posterior(state, covariance, capture):
        state = np.asarray(state, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        try:
            capture = float(capture)
        except (TypeError, ValueError) as exc:
            raise ValueError("finite capture timestamp required") from exc
        if state.shape != (9,) or not np.isfinite(state).all():
            raise ValueError("one finite length-9 posterior state required")
        if covariance.shape != (9, 9) or not np.isfinite(covariance).all():
            raise ValueError("one finite 9 x 9 posterior covariance required")
        if not np.isfinite(capture):
            raise ValueError("finite capture timestamp required")
        if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-10):
            raise ValueError("posterior covariance must be symmetric")
        covariance = (covariance+covariance.T)/2
        values, vectors = np.linalg.eigh(covariance)
        tolerance = 1e-10*max(1., float(np.max(np.abs(values))))
        if np.min(values) < -tolerance:
            raise ValueError("posterior covariance must be positive semidefinite")
        # Roundoff-only negatives are removed.  No arbitrary covariance reset.
        covariance = (vectors*np.maximum(values, 0.))@vectors.T
        return state.copy(), covariance, capture

    def seed_from_prefix(self, fit, raw_pre_times, raw_pre_xyz):
        """Start online updates from a causal batch fit and received history.

        The initial covariance is fit covariance plus a diagonal conservative
        floor (position .012 m, velocity .25 m/s, effective b .04149 1/s by
        default).  Floors prevent the batch fit from treating approximate
        dynamics as exact when postcontact measurements arrive.  They are
        fixed defaults, not trained using future/test residuals.
        """
        if not fit.get("converged", False):
            raise ValueError("a converged precontact prefix fit is required")
        state, covariance, capture = self._validated_posterior(
            fit.get("state"), fit.get("covariance"), fit.get("capture_t_s"))
        times, positions = np.asarray(raw_pre_times, dtype=float), np.asarray(raw_pre_xyz, dtype=float)
        if times.ndim != 1 or len(times) == 0 or positions.shape != (len(times), 3):
            raise ValueError("nonempty precontact times N and XYZ N x 3 required")
        if not np.isfinite(times).all() or not np.isfinite(positions).all() or np.any(np.diff(times) <= 0):
            raise ValueError("received precontact history must be finite and strictly increasing")
        if np.any(times > capture+1e-10) or abs(times[-1]-capture) > 1e-10:
            raise ValueError("history must end at the fit capture timestamp; future observations prohibited")
        sigmas = np.repeat(self.covariance_floor_sigmas, 3)
        self.state, self.covariance, self.last_t = state, covariance+np.diag(sigmas**2), capture
        self.bounced = self.bounce_confirmed = False
        self._bounce_confirmation_t = self._observed_contact_candidate_t = None
        self._consecutive_rejections = 0
        self._initial.clear()
        self._recent.clear()
        for timestamp, position in zip(times, positions):
            self._record_recent(float(timestamp), position)
        self._last_accepted, self._last_reason = True, "seeded_from_received_precontact_prefix"
        self._projection_only = False
        self._seed_metadata = {
            "initialization": "same_full_precontact_batch_state_as_frozen_prediction",
            "seed_t_s": capture,
            "seed_raw_samples": int(len(times)),
            "covariance_floor_sigmas_p_v_b": self.covariance_floor_sigmas.tolist(),
            "covariance_floor_rule": "fit covariance plus diagonal variances",
            "uncertainty_interpretation": "local approximation with fixed conservative floor; coverage uncalibrated",
            "reference_contact_used_as_input": False,
            "collision_model": "frozen_method6_physics_plus_gp_velocity_residual",
            "vertical_frame_fallback": "fixed impact only if incident planar speed < 1e-6 m/s",
        }
        self._diagnostics = deepcopy(self._seed_metadata)
        return self.snapshot()

    def snapshot(self):
        result = super().snapshot()
        result["gp_online_adapter"] = True
        result["projection_only"] = self._projection_only
        result["seed_metadata"] = deepcopy(self._seed_metadata)
        return result

    def restore_snapshot(self, posterior):
        """Restore only a posterior for projection; this instance cannot update.

        State/covariance/capture clock and first-impact flags are copied.
        No observation, offline label, event clock, or recovery history enters
        the projector.  Call ``seed_from_prefix`` to start a live tracker.
        """
        if not posterior.get("initialized", False):
            raise ValueError("an initialized posterior snapshot is required")
        state, covariance, capture = self._validated_posterior(
            posterior.get("state"), posterior.get("covariance"), posterior.get("t_s"))
        for flag in ("bounced", "bounce_confirmed"):
            if not isinstance(posterior.get(flag), (bool, np.bool_)):
                raise ValueError("posterior first-impact flags must be boolean")
        if posterior["bounce_confirmed"] and not posterior["bounced"]:
            raise ValueError("confirmed first impact requires bounced=True")
        self.state, self.covariance, self.last_t = state, covariance, capture
        self.bounced = bool(posterior["bounced"])
        self.bounce_confirmed = bool(posterior["bounce_confirmed"])
        self._initial.clear()
        self._recent.clear()
        self._bounce_confirmation_t = self._observed_contact_candidate_t = None
        self._consecutive_rejections = 0
        self._projection_only = True
        self._last_accepted, self._last_reason = False, "restored_for_projection_only"
        self._seed_metadata = deepcopy(posterior.get("seed_metadata", {}))
        self._diagnostics = {"restored_for_projection_only": True,
                             "restored_capture_t_s": capture,
                             "future_observations_installed": False}
        return self.snapshot()

    def update(self, t_s, position_xyz):
        if self._projection_only:
            raise RuntimeError("a restored posterior is projection-only; live tracking requires seed_from_prefix")
        return super().update(t_s, position_xyz)

    def _project_step(self, state, bounced, dt):
        """Pure first-event step; callback does not receive observations."""
        before = state.copy()
        immediate = not bounced and before[2] <= self.contact_height and before[5] < 0
        if immediate:
            before, bounced = self._impact_state(before), True
        after = _step(before, dt, self.parameters)
        crossing = not bounced and before[2] > self.contact_height and after[2] <= self.contact_height and after[5] < 0
        if crossing:
            delta = float(_contact_dt(before[None], dt, self.parameters, self.contact_height)[0])
            contact = _step(before, delta, self.parameters)
            after = _step(self._impact_state(contact), dt-delta, self.parameters)
            bounced = True
        return after, bounced

    def _forecast_with_flags(self, target_times):
        targets = np.asarray(target_times, dtype=float)
        if targets.ndim != 1 or not np.isfinite(targets).all():
            raise ValueError("finite one-dimensional absolute target timestamps required")
        if self.state is None or self.last_t is None:
            raise ValueError("initialized GP posterior required for forecasting")
        if np.any(targets < self.last_t-1e-12):
            raise ValueError("forecast targets must not precede last received capture time")
        output, flags = np.empty((len(targets), 9)), np.empty(len(targets), dtype=bool)
        current, bounced, previous = self.state.copy(), bool(self.bounced), float(self.last_t)
        for index in np.argsort(targets, kind="stable"):
            duration = max(0., float(targets[index]-previous))
            if duration > 0:
                steps = max(1, int(np.ceil(duration/self.config["max_step_s"])))
                for _ in range(steps):
                    current, bounced = self._project_step(current, bounced, duration/steps)
            output[index], flags[index] = current, bounced
            previous = float(targets[index])
        return output, flags

    def predict_target_times(self, absolute_times):
        """Return H x 9 future states without advancing the received-data clock."""
        return self._forecast_with_flags(absolute_times)[0]

    def predict(self, t_s):
        predicted, flags = self._forecast_with_flags([float(t_s)])
        state = predicted[0]
        result = self.snapshot()
        result.update(state=state, p=state[:3].copy(), v=state[3:6].copy(),
                      b_eff=state[6:9].copy(), a_res=np.cross(state[6:9], state[3:6]),
                      t_s=float(t_s), bounced=bool(flags[0]), covariance=None)
        result["diagnostics"].update(prediction_dt_s=float(t_s-self.last_t), synthetic_measurement=False,
                                     collision_model="frozen_method6_physics_plus_gp_velocity_residual")
        return result
