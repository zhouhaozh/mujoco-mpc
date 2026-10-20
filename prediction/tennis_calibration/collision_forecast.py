"""Absolute-clock, batched first-impact forecasting from frozen input states.

Each row has its own predicted contact time. Requested target timestamps must
be evaluated on the shared clock; they are never shifted to predicted contact.
This module uses free-flight dynamics before/after that one instantaneous event,
and intentionally does not introduce a second impact. No observations are read.
"""
from __future__ import annotations

import numpy as np

from .identified_dynamics import simulate_batch


def hybrid_forecast(capture_states, capture_times, contact_states_minus,
                    contact_times, states_plus, target_times, parameters,
                    *, max_step_s=0.004):
    """Forecast N independent cases at their requested N x H clock timestamps.

    ``contact_times`` and ``states_plus`` are the predicted event, never an
    evaluation contact label. ``contact_states_minus`` is retained in the API
    as the predicted preimpact state for validation/auditing; free-flight states
    before contact are propagated directly from capture. At the predicted event
    the output is postimpact, so targets earlier than a late predicted impact
    correctly remain preimpact. All states are finite N x 9 [p,v,b_eff].
    """
    capture_states = np.asarray(capture_states, dtype=float)
    contact_states_minus = np.asarray(contact_states_minus, dtype=float)
    states_plus = np.asarray(states_plus, dtype=float)
    capture_times = np.asarray(capture_times, dtype=float)
    contact_times = np.asarray(contact_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    if capture_states.ndim != 2 or capture_states.shape[1] != 9:
        raise ValueError("capture_states must have shape N x 9")
    count = len(capture_states)
    if count == 0 or contact_states_minus.shape != (count, 9) or states_plus.shape != (count, 9):
        raise ValueError("nonempty finite capture/contact/plus states N x 9 required")
    if capture_times.shape != (count,) or contact_times.shape != (count,):
        raise ValueError("capture_times and contact_times must have shape N")
    if target_times.ndim != 2 or target_times.shape[0] != count or target_times.shape[1] == 0:
        raise ValueError("target_times must have nonempty shape N x H")
    for array in (capture_states, capture_times, contact_states_minus,
                  contact_times, states_plus, target_times):
        if not np.isfinite(array).all():
            raise ValueError("all forecasting inputs must be finite")
    if np.any(contact_times < capture_times):
        raise ValueError("predicted contact cannot precede capture time")
    if np.any(target_times < capture_times[:, None]):
        raise ValueError("target timestamps cannot precede capture time")
    if np.any(np.diff(target_times, axis=1) < 0):
        raise ValueError("target timestamps must be nondecreasing within each case")
    if not np.isfinite(max_step_s) or max_step_s <= 0:
        raise ValueError("max_step_s must be positive and finite")
    k_drag = float(parameters["k_drag"])
    if not np.isfinite(k_drag) or k_drag < 0:
        raise ValueError("k_drag must be finite and nonnegative")
    until_contact = contact_times - capture_times
    pre_duration = np.minimum(target_times-capture_times[:, None], until_contact[:, None])
    post_duration = np.maximum(target_times-contact_times[:, None], 0)
    pre = simulate_batch(pre_duration, capture_states, k_drag, max_step_s)
    post = simulate_batch(post_duration, states_plus, k_drag, max_step_s)
    return np.where((target_times < contact_times[:, None])[..., None], pre, post)
