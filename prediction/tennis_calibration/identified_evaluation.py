"""Strict paired regression comparison for identified tennis physics.

This module scores frozen rows; it contains no model-selection operation.
The historical test trajectories were previously exposed and are a reused
regression holdout, not a fresh blind assessment of this new model.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .online_data import aggregate_landings, aggregate_predictions


def read_prediction_rows(path):
    """Read either historical or new prediction/landing CSVs losslessly."""
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    vectors = ('target_xyz', 'predicted_xyz', 'target_xy', 'predicted_xy')
    numbers = ('horizon_s', 'anchor_t_s', 'capture_t_s', 'scenario_delay_s',
               'target_t_s', 'target_time_s', 'predicted_time_s', 'elapsed_s')
    for row in rows:
        for key in vectors:
            if key in row:
                row[key] = json.loads(row[key]) if row[key] else None
        for key in numbers:
            if key in row:
                row[key] = float(row[key]) if row[key] else None
        if 'valid' in row:
            row['valid'] = str(row['valid']).lower() in ('1', 'true')
    return rows


def _stamp(value):
    stamp = float(value)
    if not np.isfinite(stamp):
        raise ValueError('nonfinite comparison timestamp')
    return round(stamp, 12)


def _key(row, landing=False):
    result = (str(row.get('split', 'unknown')), str(row['trajectory_id']),
              _stamp(row.get('capture_t_s', row['anchor_t_s'])),
              _stamp(row.get('scenario_delay_s', 0.) or 0.))
    return result if landing else result + (_stamp(row['horizon_s']),)


def _vector(value, size):
    if value is None:
        return None
    vector = np.asarray(value, dtype=float)
    return vector if vector.shape == (size,) and np.all(np.isfinite(vector)) else None


def _same_value(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return aa.shape == bb.shape and np.allclose(aa, bb, rtol=0., atol=1e-10, equal_nan=True)
    if isinstance(a, (int, float, np.number)) and not isinstance(a, bool):
        return bool(np.isclose(a, b, rtol=0., atol=1e-10, equal_nan=True))
    return a == b


def _label_consistent(a, b, landing=False):
    fields = ('anchor_t_s', 'target_time_s', 'target_xy') if landing else (
        'anchor_t_s', 'target_t_s', 'target_xyz', 'phase', 'valid', 'reason')
    return all(_same_value(a.get(field), b.get(field)) for field in fields)


def _output_valid(row, landing=False):
    if landing:
        target_time, prediction_time = row.get('target_time_s'), row.get('predicted_time_s')
        return (target_time is not None and np.isfinite(target_time)
                and target_time > row['anchor_t_s']
                and _vector(row.get('target_xy'), 2) is not None
                and _vector(row.get('predicted_xy'), 2) is not None
                and prediction_time is not None and np.isfinite(prediction_time))
    return (row.get('valid', True) and _vector(row.get('target_xyz'), 3) is not None
            and _vector(row.get('predicted_xyz'), 3) is not None)


def _index(rows, landing=False):
    models = {}
    for row in rows:
        model = str(row['model'])
        bucket = models.setdefault(model, {})
        key = _key(row, landing)
        if key in bucket:
            raise ValueError(f'duplicate comparison row for {model}: {key}')
        bucket[key] = row
    return models


def compare_model_rows(rows, *, landing=False, strict_keys=True):
    """Return marginal coverage and scores on the shared successful subset.

    Strict mode requires every model to request the exact same anchor/horizon
    keys, including unavailable future labels. Label values must agree even
    when output fails. Shared-success scores supplement marginal scores and
    coverage; a model cannot improve its reported coverage by hiding a row.
    """
    rows = list(rows)
    models = _index(rows, landing)
    if not models:
        return {'models': {}, 'common_success_models': {}, 'n_common_keys': 0,
                'n_common_success': 0, 'n_label_mismatches': 0}
    keys = [set(bucket) for bucket in models.values()]
    common, union = set.intersection(*keys), set.union(*keys)
    missing = {model: len(union-set(bucket)) for model, bucket in models.items()}
    if strict_keys and any(missing.values()):
        raise ValueError(f'models requested different anchor/horizon keys: {missing}')
    names = sorted(models)
    mismatches = []
    for key in common:
        reference = models[names[0]][key]
        for model in names[1:]:
            if not _label_consistent(reference, models[model][key], landing):
                mismatches.append((key, model))
    if mismatches:
        raise ValueError(f'models used different evaluation labels: {mismatches[:3]}')
    shared_success = {key for key in common if all(
        _output_valid(models[model][key], landing) for model in names)}
    aggregator = aggregate_landings if landing else aggregate_predictions
    return {'models': {model: aggregator(list(bucket.values())) for model, bucket in models.items()},
            'common_success_models': {model: aggregator([bucket[key] for key in sorted(shared_success)])
                                      for model, bucket in models.items()},
            'n_union_keys': len(union), 'n_common_keys': len(common),
            'n_common_success': len(shared_success), 'n_label_mismatches': 0,
            'missing_keys_by_model': missing,
            'paired_subset_definition': 'Same labels and outputs available from every model; coverage reported separately.'}


def paired_comparison(prediction_rows, landing_rows, *, strict_keys=True):
    """Compare every split/delay separately without selecting a winner."""
    prediction_rows, landing_rows = list(prediction_rows), list(landing_rows)
    scenarios = sorted({(r.get('split', 'unknown'), float(r.get('scenario_delay_s', 0.) or 0.))
                        for r in prediction_rows + landing_rows})
    report = {'evaluation_role': 'Frozen-group regression comparison; historical test already exposed.',
              'scenarios': {}}
    for split, delay in scenarios:
        choose = lambda rows: [r for r in rows if r.get('split', 'unknown') == split
                               and np.isclose(float(r.get('scenario_delay_s', 0.) or 0.), delay, rtol=0., atol=1e-12)]
        report['scenarios'][f'{split}/delay={delay:g}'] = {
            'split': split, 'scenario_delay_s': delay,
            'prediction': compare_model_rows(choose(prediction_rows), strict_keys=strict_keys),
            'landing': compare_model_rows(choose(landing_rows), landing=True, strict_keys=strict_keys)}
    return report


def assert_fit_uses_training_only(trajectory_ids, frozen_split):
    """Validate membership of global parameter fit, independent of score."""
    ids = set(trajectory_ids)
    allowed = set(frozen_split['train'])
    forbidden = set(frozen_split.get('validation', [])) | set(frozen_split.get('test', []))
    if ids & forbidden or ids - allowed:
        raise ValueError('global physical parameter fit used nontraining trajectories')
    if not ids:
        raise ValueError('global physical parameter fit used no trajectories')
    return {'n_fit_trajectories': len(ids), 'n_training_available': len(allowed),
            'validation_or_test_in_fit': False}
