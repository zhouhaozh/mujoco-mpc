"""Causal height-triggered observation policies for isolated experiments.

This module does not replace the selected filter or train collision models.
Every policy consumes only received t/XYZ observations.  A first impact is
detected by the inherited dynamics crossing the calibrated effective height,
before that frame's measurement can affect the posterior.  A hold suppresses
all observation assimilation, including recovery, while retaining received
history for a causal recovery after the hold.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .gp_online import GPOnlinePredictor
from .identified_dynamics import impact_map


class _PredictionOnlyFrame(Exception):
    """Internal control transfer after propagation and before assimilation."""


class HeightPolicyPredictor(GPOnlinePredictor):
    """An immutable collision checkpoint combined with a causal policy.

    ``continuous`` is numerically identical to the original adapter, including
    observed bounce recovery before a modeled height crossing. ``height_online``
    permits observations continuously but defers bounce confirmation/reset
    until the modeled height event; it is the matching control for ``hold100``.
    Ordinary causal precontact reacquisition is retained in all online policies.
    ``hold100`` suppresses observations until modeled contact + ``hold_s``;
    ``permanent_hold`` never resumes after that event. ``freeze_after_init``
    freezes at the natural initial posterior and then only propagates.
    ``init_hold100`` also freezes at initialization, but resumes at the first
    modeled height event + ``hold_s``. Unlike ``hold100``, it never incorporates
    further precontact observations after initialization.

    The first-observation warmup is inherited unchanged (normally >= 8 received
    points over >= 80 ms). Event time is the result of height-root localization,
    never an offline reference event or a preset timestamp. Uncertainty remains
    an uncalibrated local approximation, just as in the original filter.
    """

    policies = ("continuous", "height_online", "hold100", "permanent_hold",
                "freeze_after_init", "init_hold100")
    model_ids = ("fixed", "ridge1", "ridge2", "gp")

    def __init__(self, bank, model_id: str, policy: str, *,
                 config: dict[str, Any] | None = None, hold_s: float = .1):
        if model_id not in self.model_ids:
            raise ValueError(f"unknown collision model: {model_id}")
        if policy not in self.policies:
            raise ValueError(f"unknown observation policy: {policy}")
        if not np.isfinite(hold_s) or hold_s <= 0:
            raise ValueError("hold_s must be finite and positive")
        self.model_id, self.policy, self.hold_s = model_id, policy, float(hold_s)
        self.height_contact_t_s: float | None = None
        self.height_discovery_t_s: float | None = None
        self.resume_t_s: float | None = None
        self.initialization_t_s: float | None = None
        self.height_event_status = "not_yet_detected"
        self._inside_policy_update = False
        super().__init__(bank, config=config)

    def _gp_impact(self, incoming):
        # Match the GP adapter's physically defined degenerate vertical fallback.
        if np.linalg.norm(incoming[3:5]) < 1e-6:
            return impact_map(incoming, self.parameters)
        return self.bank.apply_state(incoming, self.model_id)

    @property
    def height_triggered(self):
        return self.height_contact_t_s is not None

    @property
    def hold_until_t_s(self):
        if self.policy in ("hold100", "init_hold100") and self.height_triggered:
            return self.height_contact_t_s + self.hold_s
        return None

    def _skip_observation_at(self, timestamp):
        if self.policy == "freeze_after_init":
            return self.initialization_t_s is not None
        if self.policy == "init_hold100":
            if self.initialization_t_s is None:
                return False
            return (not self.height_triggered
                    or timestamp < self.hold_until_t_s - 1e-12)
        if self.policy == "permanent_hold":
            return self.height_triggered
        return (self.policy == "hold100" and self.height_triggered
                and timestamp < self.hold_until_t_s - 1e-12)

    def _advance(self, duration):
        was_bounced = bool(self.bounced)
        super()._advance(duration)
        timestamp = float(self.last_t + duration)
        if not was_bounced and self.bounced and not self.height_triggered:
            contact = self._diagnostics.get("predicted_contact_capture_t_s")
            if contact is None:
                raise RuntimeError("modeled height crossing lacks event timestamp")
            self.height_contact_t_s = float(contact)
            self.height_discovery_t_s = timestamp
            self.height_event_status = "modeled_descending_height_crossing"
        if (self.policy in ("hold100", "init_hold100") and self.height_triggered
                and self.resume_t_s is None and not self._skip_observation_at(timestamp)):
            self.resume_t_s = timestamp
        if self._inside_policy_update and self._skip_observation_at(timestamp):
            # Parent update has validated/recorded this received observation and
            # propagated state/covariance. It has not updated last_t, recovered
            # from history, computed an innovation, or corrected the mean.
            raise _PredictionOnlyFrame

    def _height_recovery_permitted(self):
        return self.policy == "continuous" or self.height_triggered

    def _confirm_bounce(self):
        if not self._height_recovery_permitted():
            self._diagnostics["height_policy_recovery_deferred"] = "before_height_event"
            return False
        return super()._confirm_bounce()

    def _reacquire_from_history(self):
        # Reacquisition only fuses a consistent received local p/v window; it
        # never changes the first-impact flag. The inherited near-contact guard
        # already rejects a window straddling an unconfirmed observed minimum.
        # Blocking this during ordinary precontact flight prevents recovery from
        # camera-position offsets and confounds a hold-only policy comparison.
        # During a hold, _advance's control transfer bypasses this method too.
        return super()._reacquire_from_history()

    def update(self, t_s, position_xyz):
        started = perf_counter()
        was_initialized = self.state is not None
        self._inside_policy_update = True
        try:
            try:
                result = super().update(t_s, position_xyz)
            except _PredictionOnlyFrame:
                # t_s has already passed the inherited validation. This advances
                # the received-data clock once, without a synthetic measurement.
                self.last_t = float(t_s)
                self._last_accepted = False
                self._last_reason = "height_policy_prediction_only"
                self._diagnostics.update(
                    observation_assimilated=False,
                    received_history_retained=True,
                    update_latency_ms=(perf_counter() - started) * 1000)
                result = self.snapshot()
        finally:
            self._inside_policy_update = False
        if not was_initialized and result["initialized"]:
            self.initialization_t_s = float(result["t_s"])
            if result["bounced"]:
                self.height_event_status = "initialization_already_bounced_no_localized_event"
        if (self.policy == "continuous" and result["bounced"]
                and not self.height_triggered and was_initialized):
            self.height_event_status = "observed_confirmation_preceded_model_crossing"
        # Regenerate metadata after initialization/status changes; numerical
        # posterior arrays are untouched by this snapshot operation.
        return self.snapshot()

    def snapshot(self):
        result = super().snapshot()
        metadata = {
            "model_id": self.model_id,
            "policy": self.policy,
            "hold_s": self.hold_s,
            "height_triggered": self.height_triggered,
            "height_contact_t_s": self.height_contact_t_s,
            "height_discovery_t_s": self.height_discovery_t_s,
            "resume_t_s": self.resume_t_s,
            "hold_until_t_s": self.hold_until_t_s,
            "hold_active": bool(self.last_t is not None
                                and self._skip_observation_at(self.last_t)),
            "initialization_t_s": self.initialization_t_s,
            "height_event_status": self.height_event_status,
            "reference_contact_used_as_input": False,
            "height_trigger_basis": "propagated_state_z_crossing_effective_plane_descending",
            "freeze_history_policy": "received_points_retained_without_assimilation_until_resume",
        }
        result.update(metadata)
        result["diagnostics"].update(metadata)
        return result
