"""Causal replay inputs and offline-only labels for rolling tennis forecasts.

Offline ``keep`` flags and bounce annotations are deliberately never consulted
when producing replay observations, anchors, or history features. They may only
be used as held-out evaluation labels. Units and the measurement ground frame
remain the assumptions recorded by the original calibration.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

HORIZONS_S = (.1, .2, .5)


def _strict_rows(rows):
    """Accept finite rows in source order; discard only nonincreasing stamps."""
    indices = []
    previous = -np.inf
    for i, row in enumerate(rows):
        if np.all(np.isfinite(row)) and row[0] > previous:
            indices.append(i)
            previous = row[0]
    return np.asarray(indices, dtype=int)


def load_dataset(calibration_dir):
    """Return frozen-split traces with independent raw and label arrays.

    ``raw_*`` contains every finite, strictly chronological observation,
    including positional outliers and contact-neighborhood measurements.
    ``eval_*`` contains only the offline retained observations. Bounce reference
    fields are labels, never estimator inputs. The frozen split is validated
    rather than regenerated or selected using forecast errors.
    """
    root = Path(calibration_dir)
    annotations = json.loads((root / 'clean/annotations.json').read_text())
    split = json.loads((root / 'split.json').read_text())
    memberships = {}
    for name in ('train', 'validation', 'test'):
        for identity in split[name]:
            if identity in memberships:
                raise ValueError(f'duplicate split membership: {identity}')
            memberships[identity] = name
    groups = split.get('trajectory_to_group', {})
    # Stronger local two-sided impact fits are offline reference labels.
    impact_path = root / 'bounce/summary.json'
    impacts = {}
    if impact_path.exists():
        impacts = {r['trajectory_id']: r for r in
                   json.loads(impact_path.read_text()).get('impact_estimates', [])}
    metadata_path = root / 'clean/summary.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    traces = []
    seen = set()
    for annotation in annotations:
        identity = annotation['trajectory_id']
        if identity in seen or identity not in memberships:
            raise ValueError(f'annotation identity missing or duplicated: {identity}')
        seen.add(identity)
        clean_path = (root / 'clean' / annotation['cleaned_file']).resolve()
        if not clean_path.is_relative_to((root / 'clean').resolve()):
            raise ValueError('cleaned_file must remain within calibration clean directory')
        with clean_path.open(newline='') as stream:
            records = list(csv.DictReader(stream))
        rows = np.asarray([[float(r[k]) for k in ('t_s', 'x_m', 'y_m', 'z_m')]
                           for r in records], dtype=float).reshape(-1, 4)
        raw_indices = _strict_rows(rows)
        retained = np.asarray([r['keep'].lower() == 'true' for r in records])
        eval_indices = raw_indices[retained[raw_indices]]
        impact = impacts.get(identity, {})
        bounce_s = impact.get('t_b_s', annotation.get('first_bounce_t_s'))
        bounce_xy = impact.get('p_b_m')
        if bounce_xy is not None:
            bounce_xy = np.asarray(bounce_xy, dtype=float)[:2]
            if bounce_xy.shape != (2,) or not np.all(np.isfinite(bounce_xy)):
                bounce_xy = None
        events = [float(r['time_s']) for r in annotation.get('events', [])
                  if np.isfinite(r.get('time_s', np.nan))]
        if bounce_s is not None and np.isfinite(bounce_s):
            # Replace the annotated first event with the refined first event.
            first = annotation.get('first_bounce_t_s')
            events = [t for t in events if first is None or abs(t-first) > .05]
            events.append(float(bounce_s))
            bounce_s = float(bounce_s)
        else:
            bounce_s = None
        traces.append({
            'id': identity, 'trajectory_id': identity, 'split': memberships[identity],
            'groupid': groups.get(identity, annotation.get('session_id')),
            'raw_times': rows[raw_indices, 0], 'raw_xyz': rows[raw_indices, 1:].copy(),
            'eval_times': rows[eval_indices, 0], 'eval_xyz': rows[eval_indices, 1:].copy(),
            'bounce_reference_s': bounce_s, 'bounce_reference_xy': bounce_xy,
            'bounce_event_times_s': sorted(set(events)),
            'eval_scope_end_s': annotation.get('post_end_t_s'),
            'unit_assumption': metadata.get('unit_assumption'),
            'ground_assumption': metadata.get('ground_assumption'),
            'label_status': annotation.get('status'),
        })
    if seen != set(memberships):
        raise ValueError('frozen split and annotation identities differ')
    return traces


def causal_anchors(trace, initialization_s=.08, stride_s=.02):
    """Schedule first received observation at/after each elapsed-time slot.

    A gap may skip slots; it cannot create interpolated future observations.
    Extending the observation stream cannot change any earlier anchor.
    """
    if initialization_s < 0 or stride_s <= 0:
        raise ValueError('initialization must be nonnegative and stride positive')
    times = np.asarray(trace['raw_times'], dtype=float)
    if not len(times):
        return []
    due = float(times[0] + initialization_s)
    anchors = []
    for index, stamp in enumerate(times):
        if stamp + 1e-12 >= due:
            anchors.append({'current_index': index, 't_s': float(stamp),
                            'elapsed_s': float(stamp-times[0])})
            slots = max(1, int(np.floor((stamp-due+1e-12)/stride_s))+1)
            due += slots * stride_s
    return anchors


def history_features(times, xyz, current_index, window_s=.08, points=16):
    """Uniform past-time history, using only samples through current_index.

    Returns relative positions (translation invariant), valid-sample masks,
    normalized interpolation bracket gaps, elapsed-window fraction and a
    completeness flag. Missing early history or gaps > 12 ms are masked and
    filled with zeros. No extrapolation and no offline clean/bounce flag occurs.
    The flattened feature dimension is ``5*points+2``.
    """
    if window_s <= 0 or points < 2:
        raise ValueError('history window must be positive with at least 2 points')
    times = np.asarray(times, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (len(times), 3) or not 0 <= current_index < len(times):
        raise ValueError('history arrays or current_index are invalid')
    tt, pp = times[:current_index+1], xyz[:current_index+1]
    if not np.all(np.isfinite(tt)) or not np.all(np.diff(tt) > 0) or not np.all(np.isfinite(pp)):
        raise ValueError('history must be finite with strictly increasing timestamps')
    target = np.linspace(tt[-1]-window_s, tt[-1], points)
    positions = np.zeros((points, 3), dtype=float)
    mask = np.zeros(points, dtype=bool)
    gaps = np.zeros(points, dtype=float)
    for i, stamp in enumerate(target):
        if stamp < tt[0]-1e-12:
            continue
        right = int(np.searchsorted(tt, stamp, side='left'))
        if right < len(tt) and abs(tt[right]-stamp) <= 1e-12:
            positions[i] = pp[right]
            mask[i] = True
        elif 0 < right < len(tt):
            gap = tt[right]-tt[right-1]
            gaps[i] = gap
            if gap <= .012:
                fraction = (stamp-tt[right-1])/gap
                positions[i] = pp[right-1] + fraction*(pp[right]-pp[right-1])
                mask[i] = True
    relative = np.zeros_like(positions)
    relative[mask] = positions[mask]-pp[-1]
    elapsed = float(tt[-1]-tt[0])
    complete = bool(np.all(mask) and elapsed+1e-12 >= window_s)
    features = np.concatenate((relative.ravel(), mask.astype(float),
                               np.clip(gaps/window_s, 0, 1),
                               [min(1., elapsed/window_s), float(complete)]))
    return {'features': features, 'positions': positions, 'relative_xyz': relative,
            'sample_times': target, 'mask': mask, 'bracket_gaps_s': gaps,
            'history_complete': complete, 'elapsed_s': elapsed}


def _phase(anchor, target, bounce):
    if bounce is None:
        return 'unknown_bounce'
    if anchor < bounce and target < bounce:
        return 'pre_bounce'
    if anchor < bounce <= target:
        return 'cross_bounce'
    return 'post_bounce'


def future_labels(trace, anchor_t_s, horizons=HORIZONS_S, max_gap_s=.012,
                  bounce_padding_s=.003):
    """Offline labels at exact requested future times, with explicit missingness.

    A label cannot bridge a bounce or a bracket gap > 12 ms. Targets within
    3 ms of any offline bounce are omitted. These restrictions never remove
    observations from the replay input or alter estimator states.
    """
    times = np.asarray(trace['eval_times'], dtype=float)
    xyz = np.asarray(trace['eval_xyz'], dtype=float)
    bounce = trace.get('bounce_reference_s')
    events = np.asarray(trace.get('bounce_event_times_s', []), dtype=float)
    if not len(events) and bounce is not None:
        events = np.asarray([bounce])
    labels = []
    scope_end = trace.get('eval_scope_end_s')
    for horizon in horizons:
        if horizon < 0:
            raise ValueError('future horizons must be nonnegative')
        stamp = float(anchor_t_s+horizon)
        label = {'horizon_s': float(horizon), 'target_t_s': stamp,
                 'target_xyz': None, 'valid': False,
                 'phase': _phase(anchor_t_s, stamp, bounce), 'reason': ''}
        if scope_end is not None and stamp > scope_end+1e-12:
            label['reason'] = 'beyond_first_bounce_scope'
        elif not len(times) or stamp < times[0]-1e-12 or stamp > times[-1]+1e-12:
            label['reason'] = 'outside_observed_range'
        elif np.any(np.abs(events-stamp) <= bounce_padding_s):
            label['reason'] = 'bounce_neighborhood'
        else:
            right = int(np.searchsorted(times, stamp, side='left'))
            if right < len(times) and abs(times[right]-stamp) <= 1e-12:
                label.update(target_xyz=xyz[right].tolist(), valid=True)
            elif not 0 < right < len(times):
                label['reason'] = 'outside_observed_range'
            else:
                left = right-1
                gap = times[right]-times[left]
                if gap > max_gap_s:
                    label['reason'] = 'label_gap'
                elif np.any((events >= times[left]) & (events <= times[right])):
                    label['reason'] = 'bounce_bracket'
                else:
                    fraction = (stamp-times[left])/gap
                    target = xyz[left] + fraction*(xyz[right]-xyz[left])
                    label.update(target_xyz=target.tolist(), valid=True)
        labels.append(label)
    return labels


def _finite_vector(value, size):
    if value is None:
        return None
    vector = np.asarray(value, dtype=float)
    return vector if vector.shape == (size,) and np.all(np.isfinite(vector)) else None


def error_summary(errors):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        return {'count': 0, 'rmse_m': None, 'p50_m': None, 'p95_m': None, 'max_m': None}
    return {'count': int(len(errors)), 'rmse_m': float(np.sqrt(np.mean(errors**2))),
            'p50_m': float(np.quantile(errors, .5)), 'p95_m': float(np.quantile(errors, .95)),
            'max_m': float(np.max(errors))}


def _prediction_summary(rows):
    errors, per_trace = [], defaultdict(list)
    missing = Counter()
    labeled = failures = 0
    axes = []
    for row in rows:
        target = _finite_vector(row.get('target_xyz'), 3)
        if target is None or not row.get('valid', True):
            missing[row.get('reason') or 'missing_label'] += 1
            continue
        labeled += 1
        predicted = _finite_vector(row.get('predicted_xyz'), 3)
        if predicted is None:
            failures += 1
            missing[row.get('prediction_reason') or 'prediction_failure'] += 1
            continue
        delta = predicted-target
        error = float(np.linalg.norm(delta))
        errors.append(error)
        axes.append(delta)
        per_trace[row.get('trajectory_id', row.get('id', 'unknown'))].append(error)
    trace_rmse = [float(np.sqrt(np.mean(np.square(v)))) for v in per_trace.values()]
    return {
        'n_requested': len(rows), 'n_labeled': labeled, 'n_evaluated': len(errors),
        'n_prediction_failures': failures, 'n_missing_labels': len(rows)-labeled,
        'label_coverage': labeled/len(rows) if rows else None,
        'prediction_coverage_given_label': len(errors)/labeled if labeled else None,
        'missing_or_failure_reasons': dict(missing), 'position_error': error_summary(errors),
        'axis_rmse_m': np.sqrt(np.mean(np.square(axes), axis=0)).tolist() if axes else None,
        'trajectory_macro': {
            'n_trajectories': len(trace_rmse),
            'mean_per_trajectory_rmse_m': float(np.mean(trace_rmse)) if trace_rmse else None,
            'per_trajectory_rmse_distribution': error_summary(trace_rmse),
        },
    }


def aggregate_predictions(rows: Iterable[dict]):
    """Summarize rows for one model/split/scenario without hiding failures.

    Required row keys: trajectory_id, horizon_s, phase, target_xyz,
    predicted_xyz. ``valid``/``reason`` denote missing offline labels;
    ``prediction_reason`` denotes unavailable or rejected model output.
    Call separately per model/split/scenario so comparisons share exact labels.
    """
    rows = list(rows)
    horizons, phases, both = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        horizon = f"{float(row['horizon_s']):g}"
        phase = row.get('phase', 'unknown_bounce')
        horizons[horizon].append(row)
        phases[phase].append(row)
        both[f'{horizon}/{phase}'].append(row)
    report = _prediction_summary(rows)
    report['by_horizon_s'] = {k: _prediction_summary(v) for k, v in sorted(horizons.items())}
    report['by_phase'] = {k: _prediction_summary(v) for k, v in sorted(phases.items())}
    report['by_horizon_and_phase'] = {k: _prediction_summary(v) for k, v in sorted(both.items())}
    return report


def _landing_summary(rows):
    xy_error, time_error, paired = [], [], []
    missing = Counter()
    labeled = 0
    for row in rows:
        target = _finite_vector(row.get('target_xy'), 2)
        target_time = row.get('target_time_s')
        if target is None or target_time is None or not np.isfinite(target_time):
            missing['missing_bounce_reference'] += 1
            continue
        labeled += 1
        predicted = _finite_vector(row.get('predicted_xy'), 2)
        predicted_time = row.get('predicted_time_s')
        if predicted is None or predicted_time is None or not np.isfinite(predicted_time):
            missing[row.get('prediction_reason') or 'no_predicted_first_bounce'] += 1
            continue
        xy_error.append(float(np.linalg.norm(predicted-target)))
        time_error.append(float(predicted_time-target_time))
        paired.append(row)
    absolute = np.abs(time_error)
    return {
        'n_requested': len(rows), 'n_labeled': labeled, 'n_evaluated': len(paired),
        'prediction_coverage_given_label': len(paired)/labeled if labeled else None,
        'missing_or_failure_reasons': dict(missing), 'landing_xy_error': error_summary(xy_error),
        'time_error': {'count': len(time_error),
                       'rmse_s': float(np.sqrt(np.mean(np.square(time_error)))) if time_error else None,
                       'bias_s': float(np.mean(time_error)) if time_error else None,
                       'absolute_p50_s': float(np.quantile(absolute, .5)) if len(absolute) else None,
                       'absolute_p95_s': float(np.quantile(absolute, .95)) if len(absolute) else None,
                       'absolute_max_s': float(np.max(absolute)) if len(absolute) else None},
    }


def aggregate_landings(rows: Iterable[dict], remaining_bins=(0., .1, .2, .4, .8, np.inf)):
    """First-landing errors by offline actual remaining time (evaluation only).

    Rows use target_xy/target_time_s, predicted_xy/predicted_time_s and
    anchor_t_s. Anchors at/after the reference first bounce are not first-bounce
    predictions and are counted separately instead of scoring a second bounce.
    """
    rows = list(rows)
    active, post_count = [], 0
    for row in rows:
        actual = row.get('target_time_s')
        if actual is not None and np.isfinite(actual) and actual <= row['anchor_t_s']:
            post_count += 1
        else:
            active.append(row)
    report = _landing_summary(active)
    report['n_post_first_bounce_excluded'] = post_count
    groups = {}
    for left, right in zip(remaining_bins[:-1], remaining_bins[1:]):
        selected = [r for r in active if r.get('target_time_s') is not None
                    and left < r['target_time_s']-r['anchor_t_s'] <= right]
        groups[f'({left:g},{right:g}]'] = _landing_summary(selected)
    report['by_actual_remaining_s'] = groups
    return report
