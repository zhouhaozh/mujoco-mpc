"""Ubuntu-only causal replay, validation selection, and held-out evaluation.

Offline labels never enter OnlineFilter or the feature constructors. All neural
normalizers fit training groups; validation freezes all decisions before test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .online_data import (load_dataset, causal_anchors, future_labels,
                          history_features, aggregate_predictions, aggregate_landings)
from .online_filter import (OnlineFilter, DEFAULT_CONFIG, forecast_batch,
                            forecast_states_batch, landing_batch)
from .online_mlp import fit_residual_mlp, predict_residual
from .online_features import base_features, position_features, landing_features, FEATURE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = np.array([.1, .2, .5])


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def replay(traces, physical, config, augmented, tag):
    """Replay every raw observation; cache only independently scheduled anchors."""
    cached, timing, diagnostics = {}, [], Counter()
    for n, trace in enumerate(traces):
        tracker = OnlineFilter(**physical, config=config, augmented=augmented)
        anchors = {a['current_index']: a for a in causal_anchors(trace)}
        records, initial = [], None
        for i, (stamp, xyz) in enumerate(zip(trace['raw_times'], trace['raw_xyz'])):
            snap = tracker.update(float(stamp), xyz)
            diagnostics[snap['reason']] += 1
            timing.append({'ms': snap['diagnostics']['update_latency_ms'],
                           'initialization': snap['reason'] == 'initialized'})
            if snap['initialized'] and initial is None:
                initial = {'state': snap['state'].copy(), 't_s': float(stamp)}
            if i in anchors:
                diagnostics['scheduled_anchors'] += 1
                diagnostics['initialized_anchors' if snap['initialized'] else 'uninitialized_anchors'] += 1
                history = history_features(trace['raw_times'], trace['raw_xyz'], i)
                records.append({'index': i, 't_s': float(stamp), 'snapshot': snap,
                                'history': history, 'initial': initial})
        cached[trace['id']] = records
        if (n+1) % 10 == 0 or n+1 == len(traces):
            print(f'REPLAY {tag} {n+1}/{len(traces)}', flush=True)
    return cached, timing, dict(diagnostics)


def predictions(traces, cached, physical, config, model_name, fixed=False, delay=0.):
    position_rows, landing_rows, learning = [], [], []
    for trace in traces:
        records = cached[trace['id']]
        usable = [r for r in records if r['snapshot']['initialized']]
        if not usable:
            raise RuntimeError('trajectory did not initialize: '+trace['id'])
        snaps = [r['snapshot'] for r in usable]
        stamps = np.array([r['t_s'] for r in usable])
        states = np.array([s['state'] for s in snaps])
        bounced = np.array([s['bounced'] for s in snaps])
        if fixed:
            first = usable[0]['initial']
            states = forecast_states_batch(first['state'][None], np.array([False]),
                        stamps-first['t_s'], **physical, config=config)[0]
            contact = landing_batch(first['state'][None], np.array([False]), **physical, config=config)
            bounced = ((stamps-first['t_s']) >= contact['time_to_contact'][0]
                       if contact['valid'][0] else np.zeros(len(stamps), dtype=bool))
        pp = forecast_batch(states, bounced, HORIZONS+delay, **physical, config=config)
        lands = landing_batch(states, bounced, **physical, config=config)
        for j, record in enumerate(usable):
            now = record['t_s']+delay
            identity = {'trajectory_id': trace['id'], 'split': trace['split'],
                        'groupid': trace['groupid'], 'model': model_name,
                        'capture_t_s': record['t_s'], 'anchor_t_s': now,
                        'scenario_delay_s': delay}
            labels = future_labels(trace, now, HORIZONS)
            # Every model requests the same labels, regardless of predicted bounce.
            base = base_features(record['history'], record['snapshot'], delay)
            lrow = dict(identity, target_time_s=trace['bounce_reference_s'],
                        target_xy=None if trace['bounce_reference_xy'] is None else trace['bounce_reference_xy'].tolist(),
                        predicted_time_s=None, predicted_xy=None,
                        prediction_reason=str(lands['reason'][j]))
            lx = None
            if lands['valid'][j] and lands['time_to_contact'][j] > delay:
                lrow.update(predicted_time_s=record['t_s']+float(lands['time_to_contact'][j]),
                            predicted_xy=lands['position'][j, :2].tolist(), prediction_reason='')
                lx = landing_features(base, record['snapshot'],
                        {k: v[j] for k, v in lands.items()})
            landing_rows.append(lrow)
            for k, label in enumerate(labels):
                row = dict(identity, **label, predicted_xyz=pp[j, k].tolist(), prediction_reason='')
                position_rows.append(row)
                learning.append({'position_row': row, 'landing_row': lrow,
                                 'position_X': position_features(base, HORIZONS[k]),
                                 'landing_X': lx, 'trajectory_id': trace['id']})
    return position_rows, landing_rows, learning


def macro(rows):
    return aggregate_predictions(rows)['trajectory_macro']['mean_per_trajectory_rmse_m']


def weights(identities):
    count = Counter(identities)
    return np.array([1./count[x] for x in identities])


def arrays(learning, split, landing=False):
    X, Y, ids = [], [], []
    seen = set()
    for item in learning:
        row = item['landing_row' if landing else 'position_row']
        if row['split'] != split:
            continue
        if landing:
            key = (item['trajectory_id'], row['capture_t_s'])
            if key in seen:
                continue
            seen.add(key)
            if (item['landing_X'] is None or row['target_xy'] is None
                    or row['target_time_s'] is None or row['target_time_s'] <= row['anchor_t_s']):
                continue
            feature = item['landing_X']
            target = np.r_[np.asarray(row['target_xy'])-row['predicted_xy'],
                           row['target_time_s']-row['predicted_time_s']]
        else:
            if not row['valid']:
                continue
            feature = item['position_X']
            target = np.asarray(row['target_xyz'])-row['predicted_xyz']
        X.append(feature); Y.append(target); ids.append(item['trajectory_id'])
    return np.asarray(X), np.asarray(Y), weights(ids)


def corrected(learning, model, landing_model):
    pr, lr = [], []
    corrections = predict_residual(model, np.array([x['position_X'] for x in learning]))
    seen = set()
    for i, item in enumerate(learning):
        row = dict(item['position_row'], model='physics_mlp')
        row['predicted_xyz'] = (np.asarray(row['predicted_xyz'])+corrections[i]).tolist()
        pr.append(row)
        row = dict(item['landing_row'], model='physics_mlp')
        key = (item['trajectory_id'], row['capture_t_s'])
        if key in seen:
            continue
        seen.add(key)
        if item['landing_X'] is not None:
            correction = predict_residual(landing_model, item['landing_X'][None])[0]
            row['predicted_xy'] = (np.asarray(row['predicted_xy'])+correction[:2]).tolist()
            row['predicted_time_s'] += float(correction[2])
            if row['predicted_time_s'] <= row['anchor_t_s']:
                row.update(predicted_xy=None, predicted_time_s=None,
                           prediction_reason='mlp_contact_before_now')
        lr.append(row)
    return pr, lr


def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r})
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list))
                             else v for k, v in row.items()})


def timing_summary(rows):
    result = {}
    for name, selected in [('all', rows), ('initialization', [r for r in rows if r['initialization']]),
                           ('steady_update', [r for r in rows if not r['initialization']])]:
        x = np.array([r['ms'] for r in selected])
        result[name] = {'n': len(x), 'p50_ms': float(np.quantile(x, .5)),
                        'p99_ms': float(np.quantile(x, .99)), 'max_ms': float(x.max()),
                        'over_1ms': int((x > 1.).sum()), 'over_2ms': int((x > 2.).sum())}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=Path, default=ROOT/'artifacts/tennis_calibration/20261008_calibration')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/tennis_online/20261008_online')
    args = parser.parse_args()
    if platform.system() != 'Linux' or Path(sys.prefix).resolve() != (ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Formal replay/training must run in Ubuntu project .venv-tennis')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    traces = load_dataset(args.calibration)
    flight = json.loads((args.calibration/'flight/shared_parameters.json').read_text())
    bounce = json.loads((args.calibration/'bounce/summary.json').read_text())['parameters']
    physical = dict(k_drag=flight['k_drag_per_m'], e_n=bounce['e_n'], alpha_t=bounce['alpha_t'],
                    contact_height=bounce['effective_contact_center_height_m'])
    val = [t for t in traces if t['split'] == 'validation']
    candidates = [('physics6', False, {}), ('physics9', True, {}),
                  ('physics9_fast', True, {'residual_drive_sigma': 6., 'tau_s': .3})]
    validation, caches, timer = [], {}, {}
    for name, augmented, overrides in candidates:
        cfg = dict(DEFAULT_CONFIG, **overrides)
        cache, times, diag = replay(val, physical, cfg, augmented, name)
        rows, land, _ = predictions(val, cache, physical, cfg, name)
        score = macro(rows)
        validation.append({'name': name, 'augmented': augmented, 'config': cfg,
                           'macro_rmse_m': score, 'prediction': aggregate_predictions(rows),
                           'landing': aggregate_landings(land), 'diagnostics': diag})
        caches[name] = cache; timer[name] = times
        dump(out/'validation_selection.json', validation)
        print(f'VALIDATION {name} macro_RMSE={score:.6f}', flush=True)
    chosen = min(validation, key=lambda c: c['macro_rmse_m'])
    config = dict(physical=physical, filter=chosen['config'], augmented=chosen['augmented'],
                  position_mlp_enabled=False, landing_mlp_enabled=False,
                  selected_filter=chosen['name'], supported_horizons_s=HORIZONS.tolist(),
                  trained_delays_s=[0.], feature_schema=FEATURE_SCHEMA)
    dump(out/'config.json', config)
    all_position, all_landing, selected_learning, times_report = [], [], [], {}
    other = [t for t in traces if t['split'] == 'train']
    development = [t for t in traces if t['split'] != 'test']
    best9 = min([c for c in validation if c['augmented']], key=lambda c: c['macro_rmse_m'])
    compare_names = ['physics6', best9['name']]
    all_caches, all_times = {}, {}
    for name in compare_names:
        if name in times_report:
            continue
        selection = next(c for c in validation if c['name'] == name)
        newcache, newtimes, diag = replay(other, physical, selection['config'], selection['augmented'], name)
        cache = dict(caches[name], **newcache)
        timing = timer[name]+newtimes
        all_caches[name], all_times[name] = cache, timing
        rows, land, learn = predictions(development, cache, physical, selection['config'], name)
        all_position.extend(rows); all_landing.extend(land)
        times_report[name] = {'update': timing_summary(timing), 'diagnostics_other_splits': diag}
        if name == chosen['name']:
            selected_learning = learn
            selected_cache = cache
        if name == 'physics6':
            r, l, _ = predictions(development, cache, physical, selection['config'], 'fixed_initial80ms', fixed=True)
            all_position.extend(r); all_landing.extend(l)
    print('MLP_POSITION_TRAIN_START', flush=True)
    tx, ty, tw = arrays(selected_learning, 'train')
    vx, vy, vw = arrays(selected_learning, 'validation')
    mlp, position_training = fit_residual_mlp(tx, ty, vx, vy, out/'mlp_position',
                                             sample_weights=tw, val_sample_weights=vw)
    print('MLP_POSITION_TRAIN_DONE', position_training['best_epoch'], flush=True)
    tx, ty, tw = arrays(selected_learning, 'train', True)
    vx, vy, vw = arrays(selected_learning, 'validation', True)
    print('MLP_LANDING_TRAIN_START', flush=True)
    lmlp, landing_training = fit_residual_mlp(tx, ty, vx, vy, out/'mlp_landing', sample_weights=tw,
                        val_sample_weights=vw, seed=20261009, target_kind='landing_correction_m_m_s')
    print('MLP_LANDING_TRAIN_DONE', landing_training['best_epoch'], flush=True)
    corrected_position, corrected_landing = corrected(selected_learning, mlp, lmlp)
    all_position.extend(corrected_position); all_landing.extend(corrected_landing)
    physical_val = [r for r in all_position if r['split'] == 'validation' and r['model'] == chosen['name']]
    mlp_val = [r for r in corrected_position if r['split'] == 'validation']
    config['position_mlp_enabled'] = macro(mlp_val) < macro(physical_val)
    # Landing selection uses validation only, with equal trajectory weight.
    def landing_macro(model, time=False):
        grouped = {}
        for row in all_landing:
            if (row['split'] == 'validation' and row['model'] == model and row['target_xy'] is not None
                    and row['target_time_s'] > row['anchor_t_s'] and row['predicted_xy'] is not None):
                error = (row['predicted_time_s']-row['target_time_s'] if time else
                         np.linalg.norm(np.array(row['predicted_xy'])-row['target_xy']))
                grouped.setdefault(row['trajectory_id'], []).append(error**2)
        return float(np.mean([np.sqrt(np.mean(v)) for v in grouped.values()]))
    plval = [r for r in all_landing if r['split'] == 'validation' and r['model'] == chosen['name']]
    mlval = [r for r in all_landing if r['split'] == 'validation' and r['model'] == 'physics_mlp']
    pcov = aggregate_landings(plval)['prediction_coverage_given_label']
    mcov = aggregate_landings(mlval)['prediction_coverage_given_label']
    config['validation_landing_selection'] = {
        'physics_xy_macro_rmse_m': landing_macro(chosen['name']),
        'mlp_xy_macro_rmse_m': landing_macro('physics_mlp'),
        'physics_time_macro_rmse_s': landing_macro(chosen['name'], True),
        'mlp_time_macro_rmse_s': landing_macro('physics_mlp', True),
        'physics_coverage': pcov, 'mlp_coverage': mcov}
    config['landing_mlp_enabled'] = (landing_macro('physics_mlp') < landing_macro(chosen['name'])
                    and landing_macro('physics_mlp', True) <= landing_macro(chosen['name'], True) and mcov >= pcov)
    config['selection_source'] = 'validation_groups_only_frozen_before_test_replay'
    dump(out/'config.json', config)
    config_sha = hashlib.sha256((out/'config.json').read_bytes()).hexdigest()
    dump(out/'training.json', {'position': position_training, 'landing': landing_training})
    print('CONFIG_FROZEN_BEFORE_TEST '+config_sha, flush=True)
    test = [t for t in traces if t['split'] == 'test']
    for name in compare_names:
        selection = next(c for c in validation if c['name'] == name)
        testcache, testtimes, diag = replay(test, physical, selection['config'], selection['augmented'], name+'/test')
        all_caches[name].update(testcache)
        rp, rl, learn = predictions(test, testcache, physical, selection['config'], name)
        all_position.extend(rp); all_landing.extend(rl)
        times_report[name] = {'update': timing_summary(all_times[name]+testtimes),
                             'diagnostics_test': diag}
        if name == chosen['name']:
            selected_cache = all_caches[name]
            rp, rl = corrected(learn, mlp, lmlp)
            all_position.extend(rp); all_landing.extend(rl)
        if name == 'physics6':
            rp, rl, _ = predictions(test, testcache, physical, selection['config'], 'fixed_initial80ms', fixed=True)
            all_position.extend(rp); all_landing.extend(rl)
    # Artificial arrival delays; held capture-time state, no future observations.
    delayed = {}
    for delay in (.01, .02, .04):
        rp, rl, _ = predictions(test, selected_cache, physical, chosen['config'], chosen['name'], delay=delay)
        all_position.extend(rp); all_landing.extend(rl)
        delayed[f'{delay:g}'] = {'prediction': aggregate_predictions(rp), 'landing': aggregate_landings(rl)}
    metrics = {}
    for split in ('train', 'validation', 'test'):
        metrics[split] = {}
        for name in sorted({r['model'] for r in all_position}):
            rp = [r for r in all_position if r['split'] == split and r['model'] == name and r['scenario_delay_s'] == 0]
            rl = [r for r in all_landing if r['split'] == split and r['model'] == name and r['scenario_delay_s'] == 0]
            metrics[split][name] = {'prediction': aggregate_predictions(rp), 'landing': aggregate_landings(rl)}
    write_csv(out/'predictions.csv', all_position)
    write_csv(out/'landings.csv', all_landing)
    report = {'schema': 'TENNIS_ONLINE/1', 'completed': True,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'hostname': platform.node(),
              'platform': platform.platform(), 'python': sys.version, 'python_prefix': sys.prefix,
              'calibration_dir': str(args.calibration), 'split_counts': dict(Counter(t['split'] for t in traces)),
              'causal_inputs': 'all raw finite monotonic tXYZ; no offline keep/bounce labels',
              'ground_truth': 'offline cleaned position interpolation and provisional local-fit first contact labels',
              'config': config, 'metrics': metrics, 'synthetic_delay_scenarios_test': delayed,
              'config_sha256_frozen_before_test': config_sha,
              'timing': times_report,
              'limits': ['units/ball reference/ground unconfirmed', 'first bounce only',
                         'residual acceleration is not true spin', '143 trajectories, 11 inferred groups',
                         'test groups provide offline generalization only; no live device test or MuJoCo deployment',
                         'covariance is model approximation, not calibrated prediction interval'],
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((ROOT/'tennis_calibration').glob('*.py'))}}
    dump(out/'summary.json', report)
    from .online_plots import make_online_plots
    make_online_plots(out, all_position, all_landing,
                      {'position': position_training['history'], 'landing': landing_training['history']}, report)
    print('TENNIS_ONLINE_REPORT='+str(out/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
