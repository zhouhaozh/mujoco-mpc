"""Frozen precontact collision checkpoints for native MuJoCo playback.

The reference event is consulted only by the *replay harness* to select an
observation cutoff.  Neither that event nor the requested missing duration is
passed to the predictor.  Every model shares the same fitted incoming flight,
contact clock, and effective-b carry policy; only the collision velocity map
changes.  The corresponding real-engine adapter maps its own incident state.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .collision_model_bank import BounceModelBank  # compatible public re-export
from .identified_mujoco import StandaloneBall


class SelectableCollisionBall(StandaloneBall):
    """StandaloneBall with an optional first-impact response callback.

    ``None`` invokes the original adapter unchanged.  A callback receives a
    copy of the real MuJoCo incoming state, never the fitted offline state.
    This still uses an explicit impulse; native ground solver contact remains
    disabled.  Only the first descending crossing is mapped.
    """
    def __init__(self, *args, collision_response: Callable | None = None,
                 collision_model_id=None, **kwargs):
        if collision_response is not None and not callable(collision_response):
            raise ValueError("collision_response must be callable or None")
        self.collision_response = collision_response
        self.collision_model_id = collision_model_id
        super().__init__(*args, **kwargs)

    def _apply_first_impact(self, incoming, *, bracket_s):
        if self.collision_response is None:
            return super()._apply_first_impact(incoming, bracket_s=bracket_s)
        outgoing = np.asarray(self.collision_response(np.asarray(incoming).copy()), dtype=float)
        if outgoing.shape != (9,) or not np.isfinite(outgoing).all():
            raise ValueError("collision_response must return one finite length-9 state")
        if not np.allclose(outgoing[:3], incoming[:3], rtol=0, atol=1e-12):
            raise ValueError("an instantaneous collision_response must preserve position")
        plane_correction = float(self.contact_height-self.data.qpos[2])
        self.data.qpos[2] = self.contact_height
        self._set_velocity_and_b(outgoing[3:6], outgoing[6:9])
        self.bounced = True
        self.mujoco.mj_forward(self.model, self.data)
        actual = self.state()
        self.events.append({"event_type": "first_descending_effective_plane_crossing",
            "t_s": float(self.data.time), "impact_mode": "selectable_velocity_map",
            "collision_model_id": self.collision_model_id,
            "incoming_state": incoming.tolist(), "outgoing_state": actual.tolist(),
            "qpos_after_event": self.data.qpos.tolist(), "qvel_after_event": self.data.qvel.tolist(),
            "momentum_impulse_N_s": (self.mass*(actual[3:6]-incoming[3:6])).tolist(),
            "regime": "selected_frozen_collision_map", "native_contact_count": int(self.data.ncon),
            "position_plane_correction_m": plane_correction,
            "event_time_bracket_s": [float(t) for t in bracket_s],
            "definition": "real MuJoCo incident state + selected frozen explicit impulse; not native solver contact"})
