"""Reusable live position-only ball predictor loaded from a frozen run bundle.

Each update accepts a capture timestamp and measured XYZ in the calibration
frame. ``now_t_s`` uses the same clock; no hardware latency is guessed. Frozen
validation flags choose physics or residual models, and inference never changes
network normalization, fitted parameters, or the validation selection.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .online_features import FEATURE_SCHEMA, base_features, landing_features, position_features
from .online_data import history_features
from .online_filter import OnlineFilter, forecast_batch, landing_batch
from .online_mlp import load_residual_mlp, predict_residual


class OnlinePredictor:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        if self.config.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Run feature schema does not match live inference schema")
        physical = self.config["physical"]
        self.physical = {key: float(physical[key]) for key in ("k_drag", "e_n", "alpha_t", "contact_height")}
        if not isinstance(self.config.get("augmented"), bool):
            raise ValueError("augmented must be a boolean")
        self.filter = OnlineFilter(**self.physical, config=self.config["filter"], augmented=self.config["augmented"])
        self.supported_horizons = np.asarray(self.config["supported_horizons_s"], dtype=float)
        self.trained_delays = np.asarray(self.config["trained_delays_s"], dtype=float)
        for name, values in (("supported_horizons_s", self.supported_horizons), ("trained_delays_s", self.trained_delays)):
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"Invalid {name}")
        self.position_model = self._load_optional("position_mlp_enabled", "mlp_position", 98,
                                                  FEATURE_SCHEMA["position_target_kind"])
        self.landing_model = self._load_optional("landing_mlp_enabled", "mlp_landing", 103,
                                                 FEATURE_SCHEMA["landing_target_kind"])
        self._history: deque[tuple[float, np.ndarray]] = deque()

    def _load_optional(self, flag: str, name: str, input_dim: int, target_kind: str):
        enabled = self.config.get(flag, False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{flag} must be a boolean")
        if not enabled:
            return None
        # Both layouts are supported explicitly; new bundles use models/name.
        directory = self.run_dir / "models" / name
        if not directory.exists():
            directory = self.run_dir / name
        model = load_residual_mlp(directory)
        if model["input_dim"] != input_dim or model["output_dim"] != 3 or model["target_kind"] != target_kind:
            raise ValueError(f"{name} dimensions/target semantics mismatch")
        return model

    def reset(self) -> None:
        """Begin a new tracked ball without changing fitted models/parameters."""
        self.filter = OnlineFilter(**self.physical, config=self.config["filter"], augmented=self.config["augmented"])
        self._history.clear()

    def update(self, capture_t_s: float, xyz: np.ndarray | list[float], *,
               now_t_s: float | None = None, horizons: list[float] | np.ndarray | None = None) -> dict:
        before = self.filter.last_t
        estimate = self.filter.update(capture_t_s, xyz)
        if self.filter.last_t is not None and self.filter.last_t != before:
            self._history.append((float(capture_t_s), np.asarray(xyz, dtype=float).copy()))
            # Keep a bracket before the 80 ms feature window and a little slack;
            # memory and feature work remain bounded during a long live stream.
            while len(self._history) > 2 and self._history[1][0] < self.filter.last_t - 0.12:
                self._history.popleft()
        result = self.predict(now_t_s=now_t_s, horizons=horizons)
        result["estimate"] = estimate
        return result

    def _mlp_delay_reason(self, delay_s: float) -> str | None:
        if abs(delay_s) > 1e-9:
            return "nonzero_delay_not_validated"
        if not np.any(np.isclose(self.trained_delays, delay_s, rtol=0, atol=1e-9)):
            return "delay_not_in_training_bundle"
        return None

    def predict(self, *, now_t_s: float | None = None,
                horizons: list[float] | np.ndarray | None = None) -> dict:
        started = perf_counter()
        snapshot = self.filter.snapshot()
        if horizons is None:
            horizons = self.supported_horizons.copy()
        horizons = np.asarray(horizons, dtype=float)
        if horizons.ndim != 1 or not np.all(np.isfinite(horizons)) or np.any(horizons < 0):
            raise ValueError("Forecast horizons must be finite nonnegative seconds")
        result = {"initialized": snapshot["initialized"], "estimate": snapshot,
                  "forecasts": [], "landing": {"valid": False, "reason": "not_initialized"}}
        if not snapshot["initialized"]:
            result["diagnostics"] = {"inference_latency_ms": (perf_counter() - started) * 1000}
            return result
        capture_time = float(snapshot["t_s"])
        current_time = capture_time if now_t_s is None else float(now_t_s)
        if not np.isfinite(current_time) or current_time < capture_time - 1e-12:
            raise ValueError("now_t_s must use the capture clock and be >= latest capture timestamp")
        delay = max(0.0, current_time - capture_time)
        physical_horizons, inverse = np.unique(horizons + delay, return_inverse=True)
        forecast = forecast_batch(snapshot["state"][None], np.array([snapshot["bounced"]]),
                                  physical_horizons, **self.physical, config=self.config["filter"])[0][inverse]
        raw_times = np.array([item[0] for item in self._history])
        raw_xyz = np.array([item[1] for item in self._history])
        history = history_features(raw_times, raw_xyz, len(raw_times) - 1,
                                   window_s=FEATURE_SCHEMA["history_window_s"],
                                   points=FEATURE_SCHEMA["history_points"])
        base = base_features(history, snapshot, delay)
        delay_reason = self._mlp_delay_reason(delay)
        for horizon, position in zip(horizons, forecast):
            corrected, correction, used = position.copy(), None, False
            reason = "disabled_by_validation_selection"
            if self.position_model is not None:
                if delay_reason:
                    reason = delay_reason
                elif not np.any(np.isclose(self.supported_horizons, horizon, rtol=0, atol=1e-9)):
                    reason = "horizon_not_validated"
                else:
                    try:
                        correction = predict_residual(self.position_model, position_features(base, horizon)[None])[0]
                        corrected += correction
                        used, reason = True, "validation_selected"
                    except FloatingPointError:
                        reason = "nonfinite_mlp_fallback_to_physics"
            result["forecasts"].append({"horizon_s": float(horizon), "target_t_s": current_time + float(horizon),
                                        "physics_position_xyz_m": position.copy(), "position_xyz_m": corrected,
                                        "correction_xyz_m": correction, "mlp_used": used, "mlp_reason": reason})
        physical_landing = landing_batch(snapshot["state"][None], np.array([snapshot["bounced"]]),
                                          **self.physical, config=self.config["filter"])
        if physical_landing["valid"][0]:
            original_time = capture_time + float(physical_landing["time_to_contact"][0])
            corrected_time, position = original_time, physical_landing["position"][0].copy()
            correction, used, reason = None, False, "disabled_by_validation_selection"
            if self.landing_model is not None:
                if delay_reason:
                    reason = delay_reason
                else:
                    single_landing = {key: value[0] for key, value in physical_landing.items()}
                    try:
                        correction = predict_residual(self.landing_model, landing_features(base, snapshot, single_landing)[None])[0]
                        position[:2] += correction[:2]
                        corrected_time += float(correction[2])
                        used, reason = True, "validation_selected"
                    except FloatingPointError:
                        reason = "nonfinite_mlp_fallback_to_physics"
            valid = corrected_time > current_time + 1e-12
            result["landing"] = {
                "valid": valid, "reason": "predicted_first_contact" if valid else "predicted_contact_not_in_future",
                "physics_contact_t_s": original_time, "contact_t_s": corrected_time,
                "time_to_contact_from_capture_s": corrected_time - capture_time,
                "remaining_from_now_s": corrected_time - current_time,
                "physics_position_xyz_m": physical_landing["position"][0].copy(), "position_xyz_m": position,
                "physics_contact_velocity_xyz_m_s": physical_landing["velocity"][0].copy(),
                "correction_xy_time": correction, "mlp_used": used, "mlp_reason": reason,
            }
        else:
            result["landing"] = {"valid": False, "reason": str(physical_landing["reason"][0])}
        result.update(capture_t_s=capture_time, now_t_s=current_time)
        result["diagnostics"] = {"delay_s": delay, "timestamp_basis": "measurement_capture_clock",
                                 "hardware_delay_inferred": False, "history_complete": history["history_complete"],
                                 "synthetic_measurement": False,
                                 "inference_latency_ms": (perf_counter() - started) * 1000}
        return result
