"""Read-only batched mean projection matching the scalar height-policy model.

No observations, event labels, filter history or covariance enter this function.
It preserves the scalar absolute-clock target ordering and RK4 discretization,
including the first descending height event and its 14-round root search.
"""
from __future__ import annotations

import numpy as np

from .identified_dynamics import impact_map
from .identified_filter import _contact_dt, _step


MODELS = ("fixed", "ridge1", "ridge2", "gp")


def _steps_compatible(states, dt, parameters):
    """Match the implicit 4-ms substeps of scalar identified_filter._step."""
    durations = np.broadcast_to(np.asarray(dt, dtype=float), (len(states),))
    counts = np.maximum(1, np.ceil(np.abs(durations) / .004).astype(np.int64))
    result = np.empty_like(states)
    for count in np.unique(counts):
        indices = np.flatnonzero(counts == count)
        result[indices] = _step(states[indices], durations[indices], parameters)
    return result


def _impact_batch(incoming, bank, model_id):
    outgoing = np.empty_like(incoming)
    for index, state in enumerate(incoming):
        # The scalar adapter defines a fixed physical fallback when the local
        # incident tangent frame is undefined; it does not query GP features.
        if np.linalg.norm(state[3:5]) < 1e-6:
            mapped = impact_map(state, bank.parameters)
        else:
            mapped = bank.apply_state(state.copy(), model_id)
        mapped = np.asarray(mapped, dtype=float)
        if mapped.shape != (9,) or not np.isfinite(mapped).all():
            raise ValueError("collision callback must return a finite length-9 state")
        if not np.allclose(mapped[:3], state[:3], rtol=0, atol=1e-12):
            raise ValueError("instantaneous collision callback must preserve position")
        outgoing[index] = mapped
    outgoing[:, 2] = bank.height
    return outgoing


def _project_step(states, flags, dt, bank, model_id):
    before, bounced = states.copy(), flags.copy()
    durations = np.broadcast_to(np.asarray(dt, dtype=float), (len(states),))
    immediate = ~bounced & (before[:, 2] <= bank.height) & (before[:, 5] < 0)
    if np.any(immediate):
        before[immediate] = _impact_batch(before[immediate], bank, model_id)
        bounced[immediate] = True
    after = _steps_compatible(before, durations, bank.parameters)
    crossing = (~bounced & (before[:, 2] > bank.height)
                & (after[:, 2] <= bank.height) & (after[:, 5] < 0))
    if np.any(crossing):
        incoming, event_steps = before[crossing], durations[crossing]
        if np.max(event_steps) <= .004:
            contact_dt = _contact_dt(incoming, event_steps, bank.parameters, bank.height)
        else:
            # Scalar _contact_dt's inner RK4 subdivision is per row. Larger
            # requested outer steps need row-wise root localization to preserve
            # that discretization; the default 4-ms path stays fully batched.
            contact_dt = np.asarray([
                _contact_dt(state[None], duration, bank.parameters, bank.height)[0]
                for state, duration in zip(incoming, event_steps)])
        contact = _steps_compatible(incoming, contact_dt, bank.parameters)
        after[crossing] = _steps_compatible(
            _impact_batch(contact, bank, model_id), event_steps - contact_dt, bank.parameters)
        bounced[crossing] = True
    return after, bounced


def forecast_posteriors(states, bounced, capture_times, horizons=(0., .1, .2, .5),
                        *, bank, model_id, max_step_s=.004):
    """Return N x H x 9 forecasts without changing any posterior input.

    ``capture_times`` are absolute received-data clocks; each target is its row's
    capture + horizon. Unsorted/duplicate nonnegative horizons are supported in
    the same order as the scalar ``predict_target_times`` interface. The zero
    horizon returns the posterior unchanged, even if an impact is immediately
    pending. Only a future projection can apply the first-impact map.
    """
    state = np.asarray(states, dtype=float)
    flags = np.asarray(bounced)
    capture = np.asarray(capture_times, dtype=float)
    offsets = np.asarray(horizons, dtype=float)
    if state.ndim != 2 or state.shape[1] != 9 or not np.isfinite(state).all():
        raise ValueError("finite N x 9 posterior states required")
    if flags.shape != (len(state),) or flags.dtype.kind != "b":
        raise ValueError("N boolean first-impact flags required")
    if capture.shape != (len(state),) or not np.isfinite(capture).all():
        raise ValueError("N finite absolute capture clocks required")
    if offsets.ndim != 1 or not np.isfinite(offsets).all() or np.any(offsets < 0):
        raise ValueError("finite nonnegative one-dimensional horizons required")
    if model_id not in MODELS:
        raise ValueError("unknown collision model")
    if not np.isfinite(max_step_s) or max_step_s <= 0:
        raise ValueError("positive finite maximum step required")
    if not np.isfinite(bank.height) or not np.isfinite(bank.parameters["k_drag"]) or bank.parameters["k_drag"] < 0:
        raise ValueError("finite effective height and nonnegative drag required")
    targets = capture[:, None] + offsets[None, :]
    if not np.isfinite(targets).all():
        raise ValueError("absolute target clocks overflowed")
    output = np.empty((len(state), len(offsets), 9), dtype=float)
    if len(state) == 0 or len(offsets) == 0:
        return output
    current, current_flags, previous = state.copy(), flags.copy(), capture.copy()
    for target_index in np.argsort(offsets, kind="stable"):
        duration = np.maximum(0., targets[:, target_index] - previous)
        active = np.flatnonzero(duration > 0)
        counts = np.maximum(1, np.ceil(duration[active] / max_step_s).astype(np.int64))
        for count in np.unique(counts):
            indices = active[counts == count]
            step = duration[indices] / count
            group, group_flags = current[indices], current_flags[indices]
            for _ in range(int(count)):
                group, group_flags = _project_step(group, group_flags, step, bank, model_id)
            current[indices], current_flags[indices] = group, group_flags
        output[:, target_index] = current
        previous = targets[:, target_index].copy()
    return output
