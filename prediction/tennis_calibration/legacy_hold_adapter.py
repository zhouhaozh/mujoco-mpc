"""Isolated observation-hold adapter for the immutable legacy six-state EKF.

The experiment owner supplies ``prediction_only`` from an already discovered
causal height event. This module neither reads contact labels nor decides when
to switch models. Received points remain available to causal recovery and the
original MLP history, while the held posterior only propagates.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .online_filter import OnlineFilter
from .online_predictor import OnlinePredictor


class _LegacyPredictionOnlyFrame(Exception):
    """Transfer control after propagation and before observation assimilation."""


class LegacyHoldFilter(OnlineFilter):
    """Legacy EKF6 with an externally controlled, per-received-frame hold.

    Set ``prediction_only`` before calling the existing ``update_filter_only``
    helper, or pass it as a one-frame keyword to ``update``. Normal updates keep
    the original numerical behavior, including initialization and recovery.
    A hold is supported only after the original filter has initialized; an
    early hold raises rather than silently fitting observations during a hold.

    Held measurements are recorded but never fused, replayed, or used to reset
    the current state. At resume, ordinary recovery may use *already received*
    history. The legacy MLP remains a distinct forecast head; its output should
    not be selected during a hold because its raw-history features still change.
    """

    def __init__(self, k_drag, e_n, alpha_t, contact_height, config=None,
                 augmented=False):
        if augmented:
            raise ValueError("legacy observation-hold adapter requires EKF6")
        self.prediction_only = False
        self._inside_hold_update = False
        super().__init__(k_drag, e_n, alpha_t, contact_height,
                         config=config, augmented=False)

    def _advance(self, duration):
        super()._advance(duration)
        if self._inside_hold_update and self.prediction_only:
            # OnlineFilter.update has already validated/recorded the frame,
            # but has not changed last_t or entered any observation recovery.
            raise _LegacyPredictionOnlyFrame

    def update(self, t_s, position_xyz, *, prediction_only=None):
        previous_mode = self.prediction_only
        if prediction_only is not None:
            if not isinstance(prediction_only, bool):
                raise TypeError("prediction_only must be a bool")
            self.prediction_only = prediction_only
        if not isinstance(self.prediction_only, bool):
            raise TypeError("prediction_only must be a bool")
        started = perf_counter()
        try:
            if self.prediction_only and self.state is None:
                raise ValueError("legacy observation hold requires prior initialization")
            self._inside_hold_update = True
            try:
                return super().update(t_s, position_xyz)
            except _LegacyPredictionOnlyFrame:
                self.last_t = float(t_s)
                self._last_accepted = False
                self._last_reason = "legacy_height_policy_prediction_only"
                self._diagnostics.update(
                    observation_assimilated=False,
                    received_history_retained=True,
                    update_latency_ms=(perf_counter() - started) * 1000)
                return self.snapshot()
        finally:
            self._inside_hold_update = False
            if prediction_only is not None:
                self.prediction_only = previous_mode


def create_held_legacy(run_dir: str | Path) -> OnlinePredictor:
    """Load the original predictor and replace only its observation adapter.

    Weights, physical parameters, feature schema and selected flags are kept.
    Create a fresh instance per trajectory: ``OnlinePredictor.reset`` would
    install the original OnlineFilter and remove this isolated adapter.
    """
    predictor = OnlinePredictor(run_dir)
    if predictor.config["augmented"] or predictor.config.get("selected_filter") != "physics6":
        raise ValueError("expected the frozen six-state legacy predictor bundle")
    predictor.filter = LegacyHoldFilter(**predictor.physical,
                                        config=predictor.config["filter"])
    return predictor
