"""Reusable frozen precontact inference, independent of evaluation labels.

Input coordinates are metres, +z up, and timestamps are seconds. Call only
with the observations received before first ground contact. No reference
contact time, missing duration, cleaning flags or measured spin is required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .collision_prefix import estimate_prefix, predict_contact
from .collision_forecast import hybrid_forecast
from .collision_mapping import baseline_velocity, load_model, predict_velocity, predict_post_lift


class CollisionPredictor:
    def __init__(self, selected_path):
        path = Path(selected_path)
        self.selected = json.loads(path.read_text())
        if self.selected.get("schema") != "FROZEN_PRECONTACT/1":
            raise ValueError("Expected a frozen precontact model")
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.parameters = dict(self.selected["parameters"])
        self.height = float(self.selected["upstream"]["height"])
        spec = self.selected["selected_model"]
        self.velocity_model = (None if spec["velocity_model"] is None else
                               load_model(path.parent / "models" / spec["velocity_model"]))
        self.lift_model = (None if spec["lift_model"] is None else
                           load_model(path.parent / "models" / spec["lift_model"]))
        self.b_policy = spec["b_policy"]

    def predict(self, times_s, positions_m, cut_t_s, target_times_s=None):
        """Estimate incident state/event/outgoing state and frozen future path.

        Explicit target timestamps use the same clock as input observations.
        The default is 20/50/100/200/500 ms after *predicted* contact; formal
        evaluations instead supply fixed reference-clock targets themselves.
        This batch fit is not a 500/1000 Hz online update implementation.
        """
        started = perf_counter()
        fit = estimate_prefix(times_s, positions_m, cut_t_s, self.parameters,
                              self.height, self.selected["prefix_config"])
        contact = predict_contact(fit, self.parameters, self.height)
        result = {"valid": contact["valid"], "reason": contact["reason"],
                  "model_sha256": self.model_sha256, "fit": fit,
                  "contact": contact, "state_plus": None,
                  "target_times_s": None, "predicted_states": None,
                  "true_spin_output": False}
        if contact["valid"]:
            vm = np.asarray(contact["v_minus"])[None]
            bm = np.asarray(contact["b_minus"])[None]
            vp = (baseline_velocity(vm, self.parameters["e_n"], self.parameters["alpha_t"])
                  if self.velocity_model is None else predict_velocity(
                      self.velocity_model, vm, bm, self.parameters["e_n"], self.parameters["alpha_t"]))
            bp = bm if self.b_policy == "carry" else np.zeros_like(bm) if self.b_policy == "zero" else .5 * bm
            if self.lift_model is not None:
                bp = predict_post_lift(self.lift_model, vm, bm)
            plus = np.asarray(contact["state_minus"]).copy()
            plus[3:6], plus[6:9] = vp[0], bp[0]
            targets = (contact["contact_t_s"] + np.array([.02, .05, .1, .2, .5])
                       if target_times_s is None else np.asarray(target_times_s, dtype=float))
            if targets.ndim != 1 or not len(targets):
                raise ValueError("At least one absolute target timestamp required")
            predicted = hybrid_forecast(
                np.asarray(fit["state"])[None], np.array([fit["capture_t_s"]]),
                np.asarray(contact["state_minus"])[None], np.array([contact["contact_t_s"]]),
                plus[None], targets[None], self.parameters)[0]
            result.update(state_plus=plus, target_times_s=targets,
                          predicted_states=predicted,
                          normal_rebound_positive=bool(vp[0, 2] > 0))
        result["latency_ms"] = (perf_counter() - started) * 1000
        return result
