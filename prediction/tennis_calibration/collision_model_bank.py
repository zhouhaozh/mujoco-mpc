"""Frozen collision models shared by predictors and optional playback adapters.

This module has no viewer or MuJoCo dependency. Replay-only prefix helpers keep
their original causal cutoff semantics for compatibility with the player.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .collision_forecast import hybrid_forecast
from .collision_mapping import baseline_velocity, load_model, predict_velocity
from .collision_prefix import estimate_prefix, predict_contact


class BounceModelBank:
    """Load frozen maps, optionally using paths verified by a ModelBundle."""

    model_ids = ("fixed", "ridge1", "ridge2", "gp")
    model_names = {"fixed": "固定物理反弹", "ridge1": "线性残差反弹",
                   "ridge2": "二阶残差反弹", "gp": "GP 残差反弹（已选最优）"}
    checkpoint_names = {"ridge1": "velocity_0", "ridge2": "velocity_1", "gp": "velocity_3"}

    def __init__(self, selected_collision_path, *, model_paths=None):
        self.selected_collision_path = Path(selected_collision_path).resolve()
        self.checkpoint = json.loads(self.selected_collision_path.read_text())
        if self.checkpoint.get("schema") != "FROZEN_PRECONTACT/1":
            raise ValueError("a FROZEN_PRECONTACT/1 collision checkpoint is required")
        self.parameters = dict(self.checkpoint["parameters"])
        self.height = float(self.checkpoint["upstream"]["height"])
        self.prefix_config = dict(self.checkpoint["prefix_config"])
        if model_paths is None:
            model_paths = {name: self.selected_collision_path.parent / "models" / filename
                           for name, filename in self.checkpoint_names.items()}
        elif set(model_paths) != set(self.checkpoint_names):
            raise ValueError("explicit collision paths must contain ridge1/ridge2/gp")
        self.models = {name: load_model(Path(model_paths[name]))
                       for name in self.checkpoint_names}
        selected = self.checkpoint["selected_model"]
        if selected.get("b_policy") != "carry" or selected.get("lift_model") is not None:
            raise ValueError("the model bank supports the frozen effective-b carry selection")
        matches = [name for name, filename in self.checkpoint_names.items()
                   if filename == selected.get("velocity_model")]
        if len(matches) != 1:
            raise ValueError("the selected velocity checkpoint is not available in the model bank")
        self.selected_model_id = matches[0]
        self._prefix_cache = {}

    def _model(self, model_id):
        if model_id not in self.model_ids:
            raise ValueError(f"unknown collision model: {model_id}")
        return self.models.get(model_id)

    def apply_state(self, state9, model_id):
        """Apply an instantaneous velocity map, retaining position/effective b."""
        model = self._model(model_id)
        incoming = np.asarray(state9, dtype=float)
        if incoming.shape != (9,) or not np.isfinite(incoming).all() or incoming[5] >= 0:
            raise ValueError("a finite descending incident state of length 9 is required")
        outgoing = incoming.copy()
        if model is None:
            outgoing[3:6] = baseline_velocity(incoming[None, 3:6],
                self.parameters["e_n"], self.parameters["alpha_t"])[0]
        else:
            outgoing[3:6] = predict_velocity(model, incoming[None, 3:6], incoming[None, 6:9],
                self.parameters["e_n"], self.parameters["alpha_t"])[0]
        if not np.isfinite(outgoing).all():
            raise ValueError("collision checkpoint returned a nonfinite state")
        return outgoing

    def prepare_prefix(self, trace, gap_s=0.0):
        """Fit only raw prefix points; reference events define replay cuts only."""
        gap = float(gap_s)
        if not np.isfinite(gap) or gap < 0:
            raise ValueError("missing-tail duration must be finite and nonnegative")
        reference = trace.get("bounce_reference_s")
        if reference is None or not np.isfinite(reference):
            return {"valid": False, "reason": "no_firstcontact_reference_for_replay_cutoff",
                    "fit": None, "contact": None, "cut_t_s": None,
                    "raw_prefix_times": np.empty(0), "raw_prefix_xyz": np.empty((0, 3)),
                    "uses_post_contact_observations": False}
        cutoff = float(reference)-gap
        times, xyz = np.asarray(trace["raw_times"], dtype=float), np.asarray(trace["raw_xyz"], dtype=float)
        if times.ndim != 1 or xyz.shape != (len(times), 3):
            raise ValueError("raw times N and raw positions N x 3 are required")
        mask = times <= cutoff
        times, xyz = times[mask].copy(), xyz[mask].copy()
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(times).tobytes())
        digest.update(np.ascontiguousarray(xyz).tobytes())
        fingerprint = digest.hexdigest()
        # Model identity is absent: all heads share this same causal prefix fit.
        key = (cutoff, fingerprint)
        if key not in self._prefix_cache:
            fit = estimate_prefix(times, xyz, cutoff, self.parameters, self.height, self.prefix_config)
            contact = predict_contact(fit, self.parameters, self.height)
            self._prefix_cache[key] = {"valid": bool(contact["valid"]),
                "reason": contact["reason"], "fit": fit, "contact": contact,
                "cut_t_s": cutoff, "raw_prefix_times": times, "raw_prefix_xyz": xyz,
                "input_sha256": fingerprint, "uses_post_contact_observations": False}
        return deepcopy(self._prefix_cache[key])

    def predict(self, prepared, model_id, target_times):
        """Return H x 9 states at unchanged absolute target timestamps."""
        self._model(model_id)
        if not prepared.get("valid", False):
            raise ValueError(f"no valid precontact forecast: {prepared.get('reason')}")
        fit, contact = prepared["fit"], prepared["contact"]
        outgoing = self.apply_state(contact["state_minus"], model_id)
        targets = np.asarray(target_times, dtype=float)
        if targets.ndim != 1 or len(targets) == 0:
            raise ValueError("nonempty one-dimensional absolute target timestamps are required")
        return hybrid_forecast(np.asarray(fit["state"])[None], [fit["capture_t_s"]],
            np.asarray(contact["state_minus"])[None], [contact["contact_t_s"]],
            outgoing[None], targets[None], self.parameters)[0]
