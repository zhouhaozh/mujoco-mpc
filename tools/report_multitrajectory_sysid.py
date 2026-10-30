#!/usr/bin/env python3
"""Build the reproducible multi-trajectory system-identification report."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import fit_multitrajectory_dynamics as fit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'outputs/multitrajectory_sysid_20261030'
WINDOWS = ('full', 'evaluation', 'last_20ms')
MEASURES = ('base_xy', 'base_vxy', 'yaw_deg', 'wz', 'q_deg', 'dq', 'racket_vicon')


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False)+'\n')


def label(group):
    return f'S{group}' if group <= 5 else f'P{group-5}'


def replay_metrics(directory, evaluator):
    result = {}
    for group in range(1, 11):
        frames_path = directory/f'group_{group}/frames.jsonl'
        if not frames_path.exists():
            return None
        frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
        sim = fit.arrays(frames)
        real = fit.reference(evaluator.refs[group])
        result[str(group)] = {
            'full': fit.metric(sim, real, 0),
            'evaluation': fit.metric(sim, real, fit.PREFIX),
            'first_50ms': fit.metric(sim, real, 0, fit.PREFIX),
            'last_20ms': fit.metric(sim, real, sim['t'][-1]-.020),
            'stable_arrival': 'not_observed',
            'actual_contact': 'unconfirmed',
        }
    return result


def groups_for(kind):
    return {'all': list(range(1, 11)), 'smash': list(range(1, 6)),
            'peace': list(range(6, 11))}[kind]


def aggregate(metrics, groups, window='evaluation'):
    values = {}
    for measure in MEASURES:
        per_trial = np.asarray([metrics[str(g)][window][measure]['rmse']
                                for g in groups])
        vector = np.asarray([metrics[str(g)][window][measure]['vector_rmse']
                             for g in groups])
        values[measure] = {
            'rmse_equal_trial': np.sqrt(np.mean(per_trial*per_trial, axis=0)).tolist(),
            'vector_rmse_equal_trial': float(np.sqrt(np.mean(vector*vector))),
        }
    return values


def aggregate_stage(metrics):
    return {window: {kind: aggregate(metrics, groups_for(kind), window)
                     for kind in ('all', 'smash', 'peace')}
            for window in WINDOWS}


def collect_cv(output):
    combined, folds, errors = {}, [], []
    for index, expected in enumerate(fit.FOLDS, 1):
        result = load_json(output/f'fold_{index}/result.json')
        if result is None:
            folds.append({'fold': index, 'status': 'pending',
                          'expected_holdout_groups': list(expected)})
            continue
        holdout = tuple(result.get('holdout_groups', ()))
        if holdout != tuple(expected):
            errors.append(f'fold {index}: holdout {holdout} != expected {expected}')
            folds.append({'fold': index, 'status': 'invalid',
                          'holdout_groups': list(holdout)})
            continue
        fold_metrics = result.get('metrics', {})
        for group in holdout:
            key = str(group)
            if key in combined:
                errors.append(f'group {group} appears in multiple holdout folds')
            elif key not in fold_metrics:
                errors.append(f'fold {index} lacks metrics for holdout group {group}')
            else:
                combined[key] = fold_metrics[key]
        folds.append({'fold': index, 'status': 'complete',
                      'holdout_groups': list(holdout),
                      'evaluation_type': result.get('evaluation_type')})
    status = 'complete' if len(combined) == 10 and not errors else 'pending'
    return {'status': status, 'folds': folds, 'errors': errors,
            'metrics': combined,
            'description': 'retrospective prefix-conditioned suffix holdout'}


def noninferiority(candidate, baseline):
    if candidate is None:
        return {'status': 'pending', 'margin': 0.0, 'regressions': []}
    regressions, checks = [], 0
    for group in range(1, 11):
        key = str(group)
        if key not in candidate:
            continue
        for window in WINDOWS:
            for measure in MEASURES:
                for statistic in ('rmse', 'p95_abs', 'max_abs'):
                    before = np.asarray(baseline[key][window][measure][statistic])
                    after = np.asarray(candidate[key][window][measure][statistic])
                    for axis, (old, new) in enumerate(zip(before, after), 1):
                        checks += 1
                        if new > old + 1e-12:
                            regressions.append({'group': group, 'label': label(group),
                                                'window': window, 'measure': measure,
                                                'statistic': statistic,
                                                'axis': axis, 'baseline': float(old),
                                                'candidate': float(new),
                                                'increase': float(new-old)})
    regressions.sort(key=lambda row: row['increase'], reverse=True)
    return {'status': 'pass' if not regressions else 'fail', 'margin': 0.0,
            'checks': checks, 'regression_count': len(regressions),
            'regressions': regressions}


def pct(new, old):
    return 100*(new/old-1) if old else None


def trial_rows(baseline, initialized, candidate):
    rows = []
    for group in range(1, 11):
        key = str(group)
        row = {'group': group, 'label': label(group)}
        for stage, metrics in [('baseline', baseline), ('initialized', initialized),
                               ('fit_all', candidate)]:
            if metrics is None or key not in metrics:
                row[stage] = None
                continue
            row[stage] = {window: {
                'base_xy_m': metrics[key][window]['base_xy']['vector_rmse'],
                'base_vxy_m_s': metrics[key][window]['base_vxy']['vector_rmse'],
                'base_yaw_deg': metrics[key][window]['yaw_deg']['vector_rmse'],
                'base_wz_rad_s': metrics[key][window]['wz']['vector_rmse'],
                'q_deg': metrics[key][window]['q_deg']['rmse'],
                'dq_rad_s': metrics[key][window]['dq']['rmse'],
                'racket_vicon_m': metrics[key][window]['racket_vicon']['vector_rmse'],
            } for window in WINDOWS}
        if row['fit_all'] is not None:
            row['fit_all_change_percent_vs_baseline'] = {
                window: {name: pct(row['fit_all'][window][name],
                                  row['baseline'][window][name])
                         for name in ('base_xy_m', 'base_vxy_m_s', 'racket_vicon_m')}
                for window in WINDOWS}
        rows.append(row)
    return rows


def plot_overview(output, stages, destination):
    available = [(name, metrics) for name, metrics in stages.items()
                 if metrics is not None]
    x = np.arange(10)
    width = .8/max(1, len(available))
    colors = {'baseline': '#777777', 'initialized': '#4c78a8',
              'fit_all': '#f58518', 'cv_holdout': '#54a24b'}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for index, (name, metrics) in enumerate(available):
        offset = (index-(len(available)-1)/2)*width
        base = [metrics[str(g)]['evaluation']['base_xy']['vector_rmse']*100
                for g in range(1, 11) if str(g) in metrics]
        racket = [metrics[str(g)]['evaluation']['racket_vicon']['vector_rmse']*100
                  for g in range(1, 11) if str(g) in metrics]
        if len(base) == 10:
            axes[0, 0].bar(x+offset, base, width, label=name,
                           color=colors.get(name))
            axes[0, 1].bar(x+offset, racket, width, label=name,
                           color=colors.get(name))
    axes[0, 0].set_title('Base position RMSE after 50 ms')
    axes[0, 0].set_ylabel('cm')
    axes[0, 1].set_title('Racket vs Vicon RMSE after 50 ms')
    axes[0, 1].set_ylabel('cm')
    for axis in axes[0]:
        axis.set_xticks(x, [label(g) for g in range(1, 11)])
        axis.legend()
        axis.grid(axis='y', alpha=.25)

    joint_x = np.arange(5)
    for index, (name, metrics) in enumerate(available):
        if len(metrics) != 10:
            continue
        agg = aggregate(metrics, range(1, 11))
        offset = (index-(len(available)-1)/2)*width
        axes[1, 0].bar(joint_x+offset, agg['q_deg']['rmse_equal_trial'],
                       width, label=name, color=colors.get(name))
        axes[1, 1].bar(joint_x+offset, agg['dq']['rmse_equal_trial'],
                       width, label=name, color=colors.get(name))
    axes[1, 0].set_title('Joint position RMSE after 50 ms')
    axes[1, 0].set_ylabel('deg')
    axes[1, 1].set_title('Joint velocity RMSE after 50 ms')
    axes[1, 1].set_ylabel('rad/s')
    for axis in axes[1]:
        axis.set_xticks(joint_x, [f'J{j}' for j in range(1, 6)])
        axis.legend()
        axis.grid(axis='y', alpha=.25)
    fig.suptitle('Shared system identification: equal-trial metrics')
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def comparison(path):
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=',', names=True)
    return {name: data[name] for name in data.dtype.names}


def plot_trials(output, stages, plots):
    plots.mkdir(parents=True, exist_ok=True)
    colors = {'baseline': '#777777', 'initialized': '#4c78a8',
              'fit_all': '#f58518'}
    dirs = {'baseline': output/'baseline', 'initialized': output/'initialized',
            'fit_all': output/'fit_all/replay'}
    for group in range(1, 11):
        data = {name: comparison(path/f'group_{group}/comparison.csv')
                for name, path in dirs.items() if stages.get(name) is not None}
        data = {name: value for name, value in data.items() if value is not None}
        if not data:
            continue
        real = next(iter(data.values()))
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes[0, 0].plot(real['real_base_x_m'], real['real_base_y_m'], 'k--',
                        label='real')
        for name, values in data.items():
            axes[0, 0].plot(values['sim_base_x_m'], values['sim_base_y_m'],
                            color=colors[name], label=name)
        axes[0, 0].set_title('Base XY path')
        axes[0, 0].set_aspect('equal', adjustable='datalim')
        for component, style in [('vx_m_s', '-'), ('vy_m_s', ':')]:
            axes[0, 1].plot(real['t_s'], real[f'real_base_{component}'],
                            'k'+style, alpha=.6,
                            label=f'real {component[:2]}')
            for name, values in data.items():
                axes[0, 1].plot(values['t_s'], values[f'sim_base_{component}'],
                                style, color=colors[name], alpha=.8,
                                label=f'{name} {component[:2]}')
        axes[0, 1].set_title('Base velocity')
        for joint in range(1, 6):
            for name, values in data.items():
                error = np.rad2deg(values[f'sim_q_j{joint}_rad']-
                                   values[f'real_q_j{joint}_rad'])
                axes[1, 0].plot(values['t_s'], error, color=colors[name],
                                alpha=.65, label=f'{name} J{joint}')
                derror = (values[f'sim_dq_j{joint}_rad_s']-
                          values[f'real_dq_j{joint}_rad_s'])
                axes[1, 1].plot(values['t_s'], derror, color=colors[name],
                                alpha=.65, label=f'{name} J{joint}')
        axes[1, 0].set_title('Joint position error')
        axes[1, 0].set_ylabel('deg')
        axes[1, 1].set_title('Joint velocity error')
        axes[1, 1].set_ylabel('rad/s')
        for axis in axes.flat:
            axis.grid(alpha=.25)
            axis.set_xlabel('time (s)')
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].legend(fontsize=6, ncol=2)
        fig.suptitle(f'{label(group)} trajectory overlay')
        fig.tight_layout()
        fig.savefig(plots/f'{label(group)}_overlay.png', dpi=140)
        plt.close(fig)


def replay_manifest(output):
    replay = output/'fit_all/replay'
    paths = {'scene_path': 'scene.mjcf', 'robot_variant_path': 'variant.json',
             'motors_path': 'motors.json', 'chassis_path': 'chassis.json'}
    if not replay.is_dir() or not all((replay/path).is_file()
                                      for path in paths.values()):
        return False
    save(replay/'manifest.json', {
        'profile': 'shared_identification_candidate',
        'config_paths': paths,
        'scope': 'all ten previously inspected trajectories; in-sample candidate',
        'stable_arrival': 'not_observed',
        'actual_contact': 'unconfirmed',
        'arm_start_policy': load_json(output/'protocol.json').get('arm_start_policy', 'immediate'),
    })
    return True


def fmt(value, scale=1.0):
    return 'pending' if value is None else f'{value*scale:.3f}'


def report_markdown(summary):
    stages = summary['stages']
    lines = ['# Multi-trajectory MuJoCo system identification', '',
             f"Status: **{summary['status']}**. Results use ten previously inspected logs; validation is retrospective.", '',
             'Stable arrival is not observable in these recordings, and the logged planned event is not confirmed physical contact.', '',
             f"Arm start policy: `{summary['arm_start_policy']}`. Any initial holding policy is an unverified controller hypothesis, not a recovered hardware mode. Raw and applied commands are logged separately.", '',
             '`baseline` retains immediate command tracking. `initialized` uses the selected start policy and prefix-conditioned chassis state with the original parameters. `fit_all` adds shared parameter fitting. Compare both transitions separately; they answer different questions.', '',
             '## Equal-trial metrics after 50 ms', '',
             '| Stage | Class | Base position (cm) | Base velocity (m/s) | Racket vs Vicon (cm) |',
             '|---|---|---:|---:|---:|']
    for stage in ('baseline', 'initialized', 'fit_all', 'cv_holdout'):
        metrics = stages[stage]['metrics']
        if metrics is None or len(metrics) != 10:
            lines.append(f'| {stage} | all | pending | pending | pending |')
            continue
        aggregate_values = aggregate(metrics, range(1, 11))
        lines.append(f"| {stage} | all | {aggregate_values['base_xy']['vector_rmse_equal_trial']*100:.2f} | "
                     f"{aggregate_values['base_vxy']['vector_rmse_equal_trial']:.3f} | "
                     f"{aggregate_values['racket_vicon']['vector_rmse_equal_trial']*100:.2f} |")
        for kind in ('smash', 'peace'):
            a = aggregate(metrics, groups_for(kind))
            lines.append(f"| {stage} | {kind} | {a['base_xy']['vector_rmse_equal_trial']*100:.2f} | "
                         f"{a['base_vxy']['vector_rmse_equal_trial']:.3f} | "
                         f"{a['racket_vicon']['vector_rmse_equal_trial']*100:.2f} |")

    lines += ['', '## Equal-trial whole-record metrics', '',
              '| Stage | Base position (cm) | Base velocity (m/s) | Racket vs Vicon (cm) |',
              '|---|---:|---:|---:|']
    for stage in ('baseline', 'initialized', 'fit_all', 'cv_holdout'):
        metrics = stages[stage]['metrics']
        if metrics is None or len(metrics) != 10:
            lines.append(f'| {stage} | pending | pending | pending |')
            continue
        a = aggregate(metrics, range(1, 11), 'full')
        lines.append(f"| {stage} | {a['base_xy']['vector_rmse_equal_trial']*100:.2f} | "
                     f"{a['base_vxy']['vector_rmse_equal_trial']:.3f} | "
                     f"{a['racket_vicon']['vector_rmse_equal_trial']*100:.2f} |")

    lines += ['', '## Per-trajectory suffix metrics', '',
              '| Trial | Base baseline→all-fit (cm) | Velocity baseline→all-fit (m/s) | Vicon racket baseline→all-fit (cm) |',
              '|---|---:|---:|---:|']
    for row in summary['trials']:
        base = row['baseline']['evaluation']
        candidate = row['fit_all']['evaluation'] if row['fit_all'] else None
        lines.append(f"| {row['label']} | {base['base_xy_m']*100:.2f}→{fmt(candidate and candidate['base_xy_m'], 100)} | "
                     f"{base['base_vxy_m_s']:.3f}→{fmt(candidate and candidate['base_vxy_m_s'])} | "
                     f"{base['racket_vicon_m']*100:.2f}→{fmt(candidate and candidate['racket_vicon_m'], 100)} |")

    lines += ['', '## Per-axis suffix RMSE', '',
              '| Trial | q baseline→all-fit (deg, J1–J5) | dq baseline→all-fit (rad/s, J1–J5) |',
              '|---|---|---|']
    for row in summary['trials']:
        base = row['baseline']['evaluation']
        candidate = row['fit_all']['evaluation'] if row['fit_all'] else None
        q0 = ', '.join(f'{x:.2f}' for x in base['q_deg'])
        d0 = ', '.join(f'{x:.3f}' for x in base['dq_rad_s'])
        q1 = 'pending' if candidate is None else ', '.join(f'{x:.2f}' for x in candidate['q_deg'])
        d1 = 'pending' if candidate is None else ', '.join(f'{x:.3f}' for x in candidate['dq_rad_s'])
        lines.append(f"| {row['label']} | {q0} → {q1} | {d0} → {d1} |")

    lines += ['', '## Last 20 ms diagnostic', '',
              'This window is a trajectory-end diagnostic, not verified contact or stable arrival.', '',
              '| Trial | Base baseline→all-fit (cm) | Vicon racket baseline→all-fit (cm) |',
              '|---|---:|---:|']
    for row in summary['trials']:
        base = row['baseline']['last_20ms']
        candidate = row['fit_all']['last_20ms'] if row['fit_all'] else None
        lines.append(f"| {row['label']} | {base['base_xy_m']*100:.2f}→{fmt(candidate and candidate['base_xy_m'], 100)} | "
                     f"{base['racket_vicon_m']*100:.2f}→{fmt(candidate and candidate['racket_vicon_m'], 100)} |")

    if summary.get('arm_activation'):
        lines += ['', '## Applied command activation', '',
                  'Five entries are J1–J5. Zero means immediate tracking; null means no fresh command activated that axis during this record.', '',
                  '| Trial | Initially held axes | Activation times (s) |',
                  '|---|---|---|']
        for name, native in summary['arm_activation'].items():
            held = [f'J{i+1}' for i, value in enumerate(native.get('arm_axis_pending_initial', [])) if value]
            times = native.get('arm_axis_activation_times_s', [])
            lines.append(f"| {name} | {', '.join(held) or 'none'} | {json.dumps(times)} |")
    lines += ['', '## Strict non-inferiority', '']
    for stage in ('initialized', 'fit_all', 'cv_holdout'):
        check = summary['noninferiority'][stage]
        lines.append(f"- {stage}: {check['status']}; regressions="
                     f"{check.get('regression_count', 'pending')} with zero post-hoc margin.")
    lines += ['', 'This gate covers per-axis RMSE, P95 and maximum absolute error, including chassis yaw and yaw rate. Aggregate loss is not an acceptance criterion. Any listed per-trial/per-axis regression prevents a strict non-inferiority pass.', '',
              '## Interpretation', '',
              '- `fit_all` uses all ten records for parameter selection and is in-sample only.',
              '- `cv_holdout` contains only each fold’s held-out trajectories. Prefix data may condition the two-dimensional initial ramp state; suffix data is held out from selection.',
              '- These logs cannot prove stable arrival or actual ball contact. The report therefore does not claim complete sim-to-real alignment.', '',
              'See `summary.json` for full-window, suffix, last-20-ms, per-axis changes and every strict non-inferiority regression.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output/'baseline/manifest.json').exists():
        raise FileNotFoundError('baseline/manifest.json is required')

    evaluator = fit.evaluator(output)
    baseline = replay_metrics(output/'baseline', evaluator)
    initialized = replay_metrics(output/'initialized', evaluator)
    all_result = load_json(output/'fit_all/result.json')
    allfit = all_result.get('metrics') if all_result else None
    cv = collect_cv(output)
    cv_metrics = cv['metrics'] if cv['status'] == 'complete' else None
    stages = {
        'baseline': {'status': 'complete', 'metrics': baseline,
                     'aggregate': aggregate_stage(baseline)},
        'initialized': {'status': 'complete' if initialized else 'pending',
                        'metrics': initialized,
                        'aggregate': aggregate_stage(initialized) if initialized else None},
        'fit_all': {'status': 'complete' if allfit else 'pending',
                    'evaluation_type': all_result.get('evaluation_type') if all_result else None,
                    'metrics': allfit,
                    'aggregate': aggregate_stage(allfit) if allfit else None},
        'cv_holdout': {'status': cv['status'], 'description': cv['description'],
                       'folds': cv['folds'], 'errors': cv['errors'],
                       'metrics': cv_metrics,
                       'aggregate': aggregate_stage(cv_metrics) if cv_metrics else None},
    }
    summary = {
        'schema': 'MULTITRAJECTORY_SYSID_REPORT/1',
        'status': ('complete' if allfit and cv_metrics else 'pending'),
        'metric_aggregation': 'equal trajectory: sqrt(mean(per_trial_rmse^2))',
        'evaluation_start_s': fit.PREFIX,
        'arm_start_policy': load_json(output/'protocol.json').get('arm_start_policy', 'immediate'),
        'stable_arrival': 'not_observed',
        'actual_contact': 'unconfirmed',
        'fit_all_scope': 'in_sample_all_ten_previously_inspected_trajectories',
        'cross_validation_scope': 'retrospective_prefix_conditioned_suffix_holdout',
        'stages': stages,
        'trials': trial_rows(baseline, initialized, allfit),
        'noninferiority': {
            'initialized': noninferiority(initialized, baseline),
            'fit_all': noninferiority(allfit, baseline),
            'cv_holdout': noninferiority(cv_metrics, baseline),
        },
    }
    if (output/'fit_all/replay').exists():
        summary['arm_activation'] = {}
        for group in range(1, 11):
            native = load_json(output/f'fit_all/replay/group_{group}/summary.json')
            if native:
                summary['arm_activation'][label(group)] = {k: v for k, v in native.items()
                                                           if 'arm_' in k}
    save(output/'summary.json', summary)
    (output/'report.md').write_text(report_markdown(summary))
    plot_overview(output, {name: value['metrics'] for name, value in stages.items()},
                  output/'overview.png')
    plot_trials(output, {name: value['metrics'] for name, value in stages.items()},
                output/'plots')
    replay_manifest(output)
    print(json.dumps({'status': summary['status'],
                      'report': str(output/'report.md'),
                      'summary': str(output/'summary.json'),
                      'overview': str(output/'overview.png')}, indent=2))


if __name__ == '__main__':
    main()
