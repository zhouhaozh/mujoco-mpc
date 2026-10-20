"""Scientific plots of causal rolling forecasts, held-outs and coverage.

The row schema matches online_data.aggregate_predictions/aggregate_landings.
Pass rows in memory or save ``predictions.csv``/``landings.csv`` with vector
fields encoded as JSON. Test metrics never participate in model selection.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for key in ('target_xyz', 'predicted_xyz', 'target_xy', 'predicted_xy'):
            if key in row:
                row[key] = json.loads(row[key]) if row[key] else None
        for key in ('horizon_s', 'anchor_t_s', 'capture_t_s', 'scenario_delay_s',
                    'elapsed_s', 'target_time_s', 'predicted_time_s'):
            if key in row:
                row[key] = float(row[key]) if row[key] else None
        if 'valid' in row:
            row['valid'] = str(row['valid']).lower() in ('true', '1')
    return records


def _vector(value, size):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    return result if result.shape == (size,) and np.all(np.isfinite(result)) else None


def _position_error(row):
    target, predicted = _vector(row.get('target_xyz'), 3), _vector(row.get('predicted_xyz'), 3)
    if target is None or predicted is None or not row.get('valid', True):
        return None
    return float(np.linalg.norm(predicted-target))


def _method(row):
    return row.get('model', row.get('method', 'unknown'))


def _scenario(row):
    if 'scenario' in row:
        return row['scenario']
    return f"delay={float(row.get('scenario_delay_s', 0.) or 0.)*1000:g}ms"


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else np.nan


def _save(fig, output, filename):
    target = output/filename
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return str(target)


def make_online_plots(output_dir, predictions=None, landings=None, history=None,
                      summary=None, split='test', scenario=None):
    """Render model comparisons at identical requested horizons and labels.

    Rows include split/model/scenario in addition to online_data fields.
    ``history`` may be an MLP fit report, a report history list, or a dict of fit
    reports. ``elapsed_s`` is elapsed capture history, not device arrival time.
    Diagnostic history bins have different future targets and are labeled as
    such; they are not presented as a controlled initialization-window study.
    """
    del summary  # Reserved for compatibility with the complete replay report.
    root = Path(output_dir)
    output = root/'plots'
    output.mkdir(parents=True, exist_ok=True)
    predictions = list(predictions) if predictions is not None else _read_rows(root/'predictions.csv')
    landings = list(landings) if landings is not None else _read_rows(root/'landings.csv')
    prediction_rows = [r for r in predictions if r.get('split', split) == split]
    landing_rows = [r for r in landings if r.get('split', split) == split]
    scenarios = sorted(set(_scenario(r) for r in prediction_rows))
    if scenario is None and scenarios:
        scenario = 'delay=0ms' if 'delay=0ms' in scenarios else 'nominal' if 'nominal' in scenarios else scenarios[0]
    chosen = [r for r in prediction_rows if _scenario(r) == scenario]
    landing_chosen = [r for r in landing_rows if _scenario(r) == scenario]
    methods = sorted(set(_method(r) for r in chosen))
    colors = {method: plt.get_cmap('tab10')(i % 10) for i, method in enumerate(methods)}
    paths = []
    if chosen:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
        horizons = sorted(set(float(r['horizon_s']) for r in chosen))
        for ax, phase in zip(axes, ('all_phases', 'pre_bounce', 'cross_bounce', 'post_bounce')):
            for method in methods:
                values = []
                for horizon in horizons:
                    errors = [_position_error(r) for r in chosen if _method(r) == method
                              and (phase == 'all_phases' or r.get('phase') == phase)
                              and np.isclose(r['horizon_s'], horizon)]
                    values.append(_rmse([v for v in errors if v is not None]))
                ax.plot(np.asarray(horizons)*1000, values, 'o-', color=colors[method], label=method)
            ax.set(xlabel='Requested future horizon (ms)', ylabel='3D position RMSE (m)',
                   title=phase.replace('_', ' '))
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle(f'{split}: exact-time rolling forecasts; {scenario}')
        paths.append(_save(fig, output, 'rolling_horizon_errors.png'))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        for method in methods:
            medians, p95s, coverage = [], [], []
            for horizon in horizons:
                rows = [r for r in chosen if _method(r) == method and np.isclose(r['horizon_s'], horizon)]
                errors = [_position_error(r) for r in rows]
                values = [v for v in errors if v is not None]
                labeled = sum(_vector(r.get('target_xyz'), 3) is not None and r.get('valid', True) for r in rows)
                medians.append(float(np.quantile(values, .5)) if values else np.nan)
                p95s.append(float(np.quantile(values, .95)) if values else np.nan)
                coverage.append(len(values)/labeled if labeled else np.nan)
            x = np.asarray(horizons)*1000
            axes[0].plot(x, medians, 'o-', color=colors[method], label=f'{method}: P50')
            axes[0].plot(x, p95s, 'x--', color=colors[method], label=f'{method}: P95')
            axes[1].plot(x, coverage, 'o-', color=colors[method], label=method)
        axes[0].set(xlabel='Requested future horizon (ms)', ylabel='3D position error (m)', title='Error distribution')
        axes[1].set(xlabel='Requested future horizon (ms)', ylabel='Fraction of available labels',
                    title='Prediction coverage', ylim=(-.02, 1.03))
        for ax in axes:
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle(f'{split}, {scenario}; missing labels and prediction failures remain in report')
        paths.append(_save(fig, output, 'rolling_error_tail_and_coverage.png'))

    if landing_chosen:
        boundaries = np.asarray([0., .05, .1, .2, .4, .8, 1.5, 3.])
        centers = .5*(boundaries[:-1]+boundaries[1:])
        landing_methods = sorted(set(_method(r) for r in landing_chosen))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
        for i, method in enumerate(landing_methods):
            xy_p50, xy_p95, t_p50, t_p95, coverage = [], [], [], [], []
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                rows = [r for r in landing_chosen if _method(r) == method
                        and r.get('target_time_s') is not None
                        and left < r['target_time_s']-r['anchor_t_s'] <= right]
                xy, dt = [], []
                labeled = 0
                for row in rows:
                    target = _vector(row.get('target_xy'), 2)
                    if target is None:
                        continue
                    labeled += 1
                    predicted = _vector(row.get('predicted_xy'), 2)
                    predicted_time = row.get('predicted_time_s')
                    if predicted is None or predicted_time is None or not np.isfinite(predicted_time):
                        continue
                    xy.append(float(np.linalg.norm(predicted-target)))
                    dt.append(abs(predicted_time-row['target_time_s'])*1000)
                xy_p50.append(float(np.quantile(xy, .5)) if xy else np.nan)
                xy_p95.append(float(np.quantile(xy, .95)) if xy else np.nan)
                t_p50.append(float(np.quantile(dt, .5)) if dt else np.nan)
                t_p95.append(float(np.quantile(dt, .95)) if dt else np.nan)
                coverage.append(len(xy)/labeled if labeled else np.nan)
            color = colors.get(method, plt.get_cmap('tab10')(i % 10))
            axes[0].plot(centers, xy_p50, 'o-', color=color, label=f'{method}: P50')
            axes[0].plot(centers, xy_p95, 'x--', color=color, label=f'{method}: P95')
            axes[1].plot(centers, t_p50, 'o-', color=color, label=f'{method}: P50')
            axes[1].plot(centers, t_p95, 'x--', color=color, label=f'{method}: P95')
            axes[2].plot(centers, coverage, 'o-', color=color, label=method)
        axes[0].set(ylabel='First-landing XY error (m)', title='Landing location')
        axes[1].set(ylabel='Absolute first-landing time error (ms)', title='Landing time')
        axes[2].set(ylabel='Prediction coverage given reference', title='Coverage', ylim=(-.02, 1.03))
        for ax in axes:
            ax.set_xlabel('Actual remaining time to first bounce (s); evaluation only')
            ax.legend(fontsize=6)
            ax.grid(alpha=.2)
        fig.suptitle(f'{split}: first landing from evolving observations; {scenario}')
        paths.append(_save(fig, output, 'rolling_first_landing_errors.png'))

        # Choose two central trajectories by baseline error, never the best.
        # The rule is deterministic and is diagnostic selection only.
        reference_method = next((m for m in landing_methods if '9' in m and 'mlp' not in m.lower()),
                                next((m for m in landing_methods if 'mlp' not in m.lower()), landing_methods[0]))
        trace_errors = defaultdict(list)
        for row in landing_chosen:
            if _method(row) != reference_method:
                continue
            target, predicted = _vector(row.get('target_xy'), 2), _vector(row.get('predicted_xy'), 2)
            if target is not None and predicted is not None and row.get('target_time_s') is not None:
                if row['target_time_s'] > row['anchor_t_s']:
                    trace_errors[row.get('trajectory_id', row.get('id', 'unknown'))].append(
                        float(np.linalg.norm(predicted-target)))
        ranking = sorted(trace_errors, key=lambda identity: (float(np.median(trace_errors[identity])), identity))
        indices = sorted(set([max(0, len(ranking)//2-1), len(ranking)//2])) if ranking else []
        selected = [ranking[i] for i in indices]
        if selected:
            fig, axes = plt.subplots(1, len(selected), figsize=(7*len(selected), 4.5),
                                     squeeze=False, constrained_layout=True)
            for ax, identity in zip(axes.ravel(), selected):
                for i, method in enumerate(landing_methods):
                    rows = sorted([r for r in landing_chosen if _method(r) == method
                                   and r.get('trajectory_id', r.get('id')) == identity
                                   and r.get('target_time_s') is not None
                                   and r['target_time_s'] > r['anchor_t_s']], key=lambda r: r['anchor_t_s'])
                    remaining, errors = [], []
                    for row in rows:
                        target, predicted = _vector(row.get('target_xy'), 2), _vector(row.get('predicted_xy'), 2)
                        remaining.append(row['target_time_s']-row['anchor_t_s'])
                        errors.append(float(np.linalg.norm(predicted-target))
                                      if target is not None and predicted is not None else np.nan)
                    ax.plot(remaining, errors, color=colors.get(method, plt.get_cmap('tab10')(i % 10)), label=method)
                ax.invert_xaxis()
                ax.set(xlabel='Actual remaining time (s); evaluation only', ylabel='First-landing XY error (m)',
                       title=Path(identity).stem.replace('trajectory_', ''))
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
            fig.suptitle(f'{split}: two central trajectories ranked by {reference_method} median landing error')
            paths.append(_save(fig, output, 'rolling_landing_history_examples.png'))

    elapsed_rows = [r for r in chosen if r.get('elapsed_s') is not None
                    and np.isclose(r['horizon_s'], .2) and r.get('phase') == 'pre_bounce']
    if elapsed_rows:
        boundaries = [.08, .12, .2, .3, .5, .8, 1.2, 2., 4.]
        centers = .5*(np.asarray(boundaries[:-1])+boundaries[1:])
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for method in methods:
            values = []
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                errors = [_position_error(r) for r in elapsed_rows if _method(r) == method
                          and left <= r['elapsed_s'] < right]
                values.append(_rmse([v for v in errors if v is not None]))
            ax.plot(centers, values, 'o-', color=colors[method], label=method)
        ax.set(xlabel='Elapsed received capture history (s)', ylabel='200 ms future position RMSE (m)',
               title=f'{split}: rolling pre-bounce diagnostics; targets differ between bins')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        paths.append(_save(fig, output, 'rolling_history_diagnostics.png'))

    if len(scenarios) > 1:
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        x = np.arange(len(scenarios))
        for i, method in enumerate(sorted(set(_method(r) for r in prediction_rows))):
            values = []
            for name in scenarios:
                errors = [_position_error(r) for r in prediction_rows if _method(r) == method
                          and _scenario(r) == name and np.isclose(r['horizon_s'], .2)]
                values.append(_rmse([v for v in errors if v is not None]))
            ax.plot(x, values, 'o-', label=method, color=plt.get_cmap('tab10')(i % 10))
        ax.set_xticks(x, scenarios, rotation=15)
        ax.set(ylabel='200 ms future position RMSE (m)', title=f'{split}: arrival-delay and dropout scenarios')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
        paths.append(_save(fig, output, 'rolling_scenario_errors.png'))

    if history is not None:
        if isinstance(history, list):
            reports = {'MLP': {'history': history}}
        elif 'history' in history:
            reports = {'MLP': history}
        else:
            reports = history
        for name, report in reports.items():
            if isinstance(report, list):
                report = {'history': report}
            entries = report.get('history', [])
            if not entries:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            epochs = [r['epoch'] for r in entries]
            ax.plot(epochs, [r['train_output_rmse'] for r in entries], label='Training')
            ax.plot(epochs, [r['val_output_rmse'] for r in entries], label='Validation')
            if report.get('best_epoch') is not None:
                ax.axvline(report['best_epoch'], color='black', linestyle='--', label='Validation-selected epoch')
            ax.set(xlabel='Training epoch', ylabel='Output residual score (target units)',
                   title=f'{name}: train / validation only; test never selects epoch')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
            safe_name = ''.join(c if c.isalnum() else '_' for c in name.lower())
            paths.append(_save(fig, output, f'mlp_{safe_name}_learning_curve.png'))
    return paths
