"""Scientific figures for identified physics versus frozen historical methods.

Run on Ubuntu after model selection is frozen. These figures show reused test
regression data, not a fresh blind test. All accuracy curves use identical
labels and the shared successful subset; marginal coverage is shown explicitly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .identified_evaluation import paired_comparison, read_prediction_rows


def _number(value):
    return np.nan if value is None else float(value)


def _save(fig, output, filename):
    target = output/filename
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return str(target)


def _display(model):
    labels = {'fixed_initial80ms': 'Frozen first 80 ms', 'physics6': 'Previous EKF6',
              'physics9': 'Previous EKF9', 'physics_mlp': 'Previous EKF + MLP',
              'identified_physics': 'Identified physics',
              'identified_magnus': 'Identified Magnus EKF'}
    return labels.get(model, model.replace('_', ' '))


def _nominal_test(rows):
    return [row for row in rows if row.get('split') == 'test'
            and abs(float(row.get('scenario_delay_s', 0.) or 0.)) < 1e-12]


def make_identified_plots(out_dir, prediction_rows=None, landing_rows=None, summary=None):
    """Render fixed-horizon/phase, first-landing and coverage comparisons.

    Optional summary['observability'] is a list of small dictionaries with
    parameter/status/estimate/unit/condition fields, rendered verbatim without
    claiming unmeasured spin or contact material parameters were identified.
    """
    root = Path(out_dir)
    output = root/'plots'
    output.mkdir(parents=True, exist_ok=True)
    predictions = list(prediction_rows) if prediction_rows is not None else read_prediction_rows(root/'predictions.csv')
    landings = list(landing_rows) if landing_rows is not None else read_prediction_rows(root/'landings.csv')
    predictions, landings = _nominal_test(predictions), _nominal_test(landings)
    comparisons = paired_comparison(predictions, landings)['scenarios'].get('test/delay=0')
    if comparisons is None:
        return []
    preferred = ['fixed_initial80ms', 'physics6', 'physics9', 'physics_mlp']
    available = set(comparisons['prediction']['models']) | set(comparisons['landing']['models'])
    models = [name for name in preferred if name in available] + sorted(available-set(preferred))
    colors = {name: plt.get_cmap('tab10')(i % 10) for i, name in enumerate(models)}
    paths = []
    paired = comparisons['prediction']['common_success_models']
    marginal = comparisons['prediction']['models']
    horizons = sorted({float(row['horizon_s']) for row in predictions})
    if predictions:
        fig, axes = plt.subplots(1, 4, figsize=(17.4, 4.4), constrained_layout=True)
        for ax, phase in zip(axes, ('all', 'pre_bounce', 'cross_bounce', 'post_bounce')):
            for model in models:
                if model not in paired:
                    continue
                scores = []
                for horizon in horizons:
                    key = f'{horizon:g}' if phase == 'all' else f'{horizon:g}/{phase}'
                    source = paired[model]['by_horizon_s'] if phase == 'all' else paired[model]['by_horizon_and_phase']
                    metric = source.get(key, {})
                    scores.append(_number(metric.get('position_error', {}).get('rmse_m')))
                ax.plot(np.asarray(horizons)*1000, scores, 'o-', color=colors[model], label=_display(model))
            ax.set(xlabel='Future horizon (ms)', ylabel='3D position RMSE (m)',
                   title=phase.replace('_', ' '))
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        fig.suptitle('Reused test regression; identical targets and common successful forecasts; zero synthetic delay')
        paths.append(_save(fig, output, 'identified_horizon_phase_comparison.png'))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        for model in models:
            if model not in marginal:
                continue
            coverage, p95 = [], []
            for horizon in horizons:
                group = marginal[model]['by_horizon_s'].get(f'{horizon:g}', {})
                coverage.append(_number(group.get('prediction_coverage_given_label'))*100)
                score = paired[model]['by_horizon_s'].get(f'{horizon:g}', {})
                p95.append(_number(score.get('position_error', {}).get('p95_m')))
            axes[0].plot(np.asarray(horizons)*1000, coverage, 'o-', color=colors[model], label=_display(model))
            axes[1].plot(np.asarray(horizons)*1000, p95, 'o-', color=colors[model], label=_display(model))
        axes[0].set(ylabel='Coverage given valid target (%)', title='Marginal prediction coverage', ylim=(-2, 102))
        axes[1].set(ylabel='3D position P95 error (m)', title='Common successful subset: error tail')
        for ax in axes:
            ax.set_xlabel('Future horizon (ms)')
            ax.legend(fontsize=7)
            ax.grid(alpha=.25)
        fig.suptitle('Failures remain in coverage; previously exposed test groups')
        paths.append(_save(fig, output, 'identified_coverage_and_error_tail.png'))

    if landings:
        paired_land = comparisons['landing']['common_success_models']
        marginal_land = comparisons['landing']['models']
        bins = ['(0,0.1]', '(0.1,0.2]', '(0.2,0.4]', '(0.4,0.8]', '(0.8,inf]']
        x = np.arange(len(bins))
        fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
        for model in models:
            if model not in paired_land:
                continue
            xy, time, coverage = [], [], []
            for key in bins:
                paired_bin = paired_land[model]['by_actual_remaining_s'].get(key, {})
                marginal_bin = marginal_land[model]['by_actual_remaining_s'].get(key, {})
                xy.append(_number(paired_bin.get('landing_xy_error', {}).get('rmse_m')))
                time.append(_number(paired_bin.get('time_error', {}).get('rmse_s'))*1000)
                coverage.append(_number(marginal_bin.get('prediction_coverage_given_label'))*100)
            axes[0].plot(x, xy, 'o-', color=colors[model], label=_display(model))
            axes[1].plot(x, time, 'o-', color=colors[model], label=_display(model))
            axes[2].plot(x, coverage, 'o-', color=colors[model], label=_display(model))
        axes[0].set(ylabel='First-landing XY RMSE (m)', title='Common successful landing locations')
        axes[1].set(ylabel='First-landing time RMSE (ms)', title='Common successful landing times')
        axes[2].set(ylabel='Coverage given landing reference (%)', title='Marginal first-landing coverage', ylim=(-2, 102))
        for ax in axes:
            ax.set_xticks(x, bins, rotation=22)
            ax.set_xlabel('Actual remaining time (s): used only in evaluation')
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        fig.suptitle('Reused test regression; first contact only; post-contact anchors excluded')
        paths.append(_save(fig, output, 'identified_landing_remaining_time.png'))

    rows = (summary or {}).get('observability')
    if rows and isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
        fields = [field for field in ('parameter', 'status', 'estimate', 'unit', 'condition')
                  if any(field in row for row in rows)]
        if fields:
            cells = [[str(row.get(field, '')) for field in fields] for row in rows]
            fig, ax = plt.subplots(figsize=(15, max(3., .65*len(rows)+1.3)), constrained_layout=True)
            ax.axis('off')
            table = ax.table(cellText=cells, colLabels=fields, cellLoc='left', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.)
            ax.set_title('Parameter interpretation and observability conditions; no unmeasured true-spin claim')
            paths.append(_save(fig, output, 'identified_parameter_observability.png'))
    return paths
