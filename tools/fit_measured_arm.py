#!/usr/bin/env python3
"""Fit X4 arm simulation parameters against measured command-replay logs."""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parameter_fit_support import Evaluator


def parameters_from_vector(x):
    # Log variables keep nonnegative parameters well-scaled over several decades.
    kp = np.exp(x[:5])
    kd = np.exp(x[5:10])
    armature = np.exp(x[10:15])
    damping = np.exp(x[15:20])
    return {'kp': kp.tolist(), 'kd': kd.tolist(), 'armature': armature.tolist(),
            'damping': damping.tolist(), 'frictionloss': [0.0]*5}


def vector_from_parameters(p):
    return np.log(np.r_[p['kp'], p['kd'], p['armature'], p['damping']])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/parameter_fit/arm'))
    parser.add_argument('--max-nfev', type=int, default=160)
    parser.add_argument('--worst-maxiter', type=int, default=10,
                        help='DE iterations for worst-axis J1/J2 and J5 refinement')
    parser.add_argument('--fixed-parameters', type=Path,
                        help='JSON parameters kept fixed during arm fitting (for example chassis)')
    parser.add_argument('--initial-parameters', type=Path,
                        help='JSON candidate used as the arm-search starting point')
    parser.add_argument('--local-maxfev', type=int, default=0,
                        help='optional Powell evaluations after each DE refinement')
    parser.add_argument('--gains-only', action='store_true',
                        help='refine only J1/J2 gains (keep passive joint properties fixed)')
    parser.add_argument('--rs04-cap', type=float,
                        help='fixed short-record RS04 torque cap in [40, 120] Nm')
    parser.add_argument('--de-popsize', type=int, default=4)
    parser.add_argument('--skip-j5', action='store_true')
    parser.add_argument('--skip-least-squares', action='store_true')
    parser.add_argument('--sequential-j12', action='store_true',
                        help='refine J1 then J2 separately instead of a shared worst objective')
    parser.add_argument('--friction-only', action='store_true',
                        help='jointly refine only J1/J2/J5 frictionloss')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.output/'work'
    train = Evaluator(work, (1,3,4))
    trials = (args.output/'trials.jsonl').open('a', buffering=1)
    state = {'best_cost': math.inf, 'best_parameters': None, 'best_metrics': None, 'evaluations': 0,
             'residual_size': 0}
    fixed = json.loads(args.fixed_parameters.read_text()) if args.fixed_parameters else {}
    fixed = fixed.get('parameters', fixed)
    if 'chassis_ramp' in fixed:
        # No training record commands yaw, so retain the identifiable baseline value.
        fixed['chassis_ramp']['ang_accel_limit_rad_s2'] = .5
    arm_fields = {'kp', 'kd', 'armature', 'damping', 'frictionloss'}

    def with_fixed(parameters):
        combined = {key: value for key, value in fixed.items() if key not in arm_fields}
        combined.update(parameters)
        return combined

    def evaluate_vector(x, full=False):
        p = with_fixed(parameters_from_vector(x))
        try:
            outputs = train.evaluate(p)
            residual = train.residual(outputs, 'arm', stride=5)
            cost = float(np.mean(residual**2))
            metrics = train.metrics(outputs) if full or cost < state['best_cost'] else None
            valid = True
        except Exception as error:
            residual = np.full(state['residual_size'] or 2570, 1e3)
            cost, metrics, valid = 1e6, None, False
            error_text = str(error)
        state['evaluations'] += 1
        if valid:
            state['residual_size'] = residual.size
        row = {'evaluation': state['evaluations'], 'cost': cost, 'valid': valid, 'parameters': p}
        if not valid:
            row['error'] = error_text
        trials.write(json.dumps(row)+'\n')
        if valid and cost < state['best_cost']:
            state.update(best_cost=cost, best_parameters=p, best_metrics=metrics, evaluations=state['evaluations'])
            write_json(args.output/'best.json', state)
            print(f"best {state['evaluations']}: cost={cost:.6g}", flush=True)
        return residual

    initial = {'kp': [0.02,0.02,20,20,0.02], 'kd': [30,30,5,3,.3],
               'armature': [.02,.02,.02,.01,.002], 'damping': [.01,.01,.01,.01,.001]}
    if args.resume and (args.output/'best.json').exists():
        initial = json.loads((args.output/'best.json').read_text())['best_parameters']
    if args.initial_parameters:
        loaded = json.loads(args.initial_parameters.read_text())
        initial = loaded.get('parameters', loaded)
    initial = {key: initial[key] for key in arm_fields if key in initial}
    x0 = vector_from_parameters(initial)
    # Gain bounds follow the motor protocol: J1-J4 use RS04/RS06 and J5 uses RS05.
    lower = np.log([1e-5]*5 + [1e-4]*5 + [1e-5]*5 + [1e-6]*5)
    upper = np.log([5000,5000,5000,5000,500] + [100,100,100,100,5] +
                   [5,5,1,1,.2] + [20,20,10,10,2])
    evaluate_vector(x0, full=True)
    if args.skip_least_squares:
        result = SimpleNamespace(success=True, message='least-squares stage skipped', nfev=0, x=x0)
    else:
        result = least_squares(evaluate_vector, x0, bounds=(lower, upper), max_nfev=args.max_nfev,
                               diff_step=.03, x_scale='jac', verbose=1)
        evaluate_vector(result.x, full=True)

    # Least squares minimizes aggregate sample error. Refine the axes that can be
    # hidden by that aggregate using the accepted per-group, per-axis thresholds.
    best = state['best_parameters']
    worst_trials = (args.output/'worst_trials.jsonl').open('a', buffering=1)

    def refine_axes(parameters, axes, fields, maxiter, seed, rs04_cap=False):
        base = json.loads(json.dumps(parameters))
        indices = [(field, axis) for field in fields for axis in axes]
        starts = np.log([max(base[field][axis], 1e-8) for field, axis in indices])
        bounds = []
        field_offset = {'kp': 0, 'kd': 5, 'armature': 10, 'damping': 15}
        for field, axis in indices:
            if field == 'frictionloss':
                bounds.append((np.log(1e-8), np.log(.5 if axis == 4 else 10.0)))
            else:
                i = field_offset[field] + axis
                bounds.append((lower[i], upper[i]))
        if args.rs04_cap is not None:
            if not 40 <= args.rs04_cap <= 120:
                parser.error('--rs04-cap must be between continuous 40 and peak 120 Nm')
            base.setdefault('motor_parameters', {}).setdefault('RS04', {})[
                'continuous_torque_nm'] = args.rs04_cap
        elif rs04_cap:
            base.setdefault('motor_parameters', {}).setdefault('RS04', {})[
                'continuous_torque_nm'] = 80.0
            starts = np.r_[starts, np.log(80.0)]
            bounds.append((np.log(40.0), np.log(120.0)))
        stage_best = {'score': math.inf, 'parameters': base, 'metrics': None}

        def objective(values):
            candidate = json.loads(json.dumps(base))
            for (field, axis), value in zip(indices, np.exp(values)):
                candidate[field][axis] = float(value)
            if rs04_cap and args.rs04_cap is None:
                candidate['motor_parameters']['RS04']['continuous_torque_nm'] = float(np.exp(values[-1]))
            try:
                outputs = train.evaluate(candidate)
                metrics = train.metrics(outputs)
                ratios = []
                for group in train.groups:
                    for axis in axes:
                        ratios += [metrics[group]['q_rmse_deg'][axis]/5.0,
                                   metrics[group]['dq_rmse'][axis]/.5]
                score = float(max(ratios) + .03*np.mean(np.square(ratios)))
                valid = True
            except Exception as error:
                score, metrics, valid = 1e6, None, False
                error_text = str(error)
            row = {'stage_axes': [axis+1 for axis in axes], 'score': score,
                   'valid': valid, 'parameters': candidate}
            if not valid:
                row['error'] = error_text
            worst_trials.write(json.dumps(row)+'\n')
            if valid and score < stage_best['score']:
                stage_best.update(score=score, parameters=candidate, metrics=metrics)
                write_json(args.output/f"best_worst_j{'_j'.join(str(a+1) for a in axes)}.json",
                           stage_best)
                print(f"worst J{','.join(str(a+1) for a in axes)}: {score:.6g}", flush=True)
            return score

        differential_evolution(objective, bounds, x0=starts, seed=seed, popsize=args.de_popsize,
                               maxiter=maxiter, tol=.01, polish=False, updating='immediate')
        if args.local_maxfev:
            local_start = [np.log(stage_best['parameters'][field][axis])
                           for field, axis in indices]
            if rs04_cap and args.rs04_cap is None:
                local_start.append(np.log(stage_best['parameters']['motor_parameters']['RS04'][
                    'continuous_torque_nm']))
            minimize(objective, local_start, method='Powell', bounds=bounds,
                     options={'maxfev': args.local_maxfev, 'xtol': .02, 'ftol': .002})
        return stage_best['parameters']

    if args.worst_maxiter:
        if args.friction_only:
            fields, axis_sets = ('frictionloss',), ((0,1,4),)
        else:
            fields = ('kp','kd') if args.gains_only else ('kp','kd','armature','damping')
            axis_sets = ((0,), (1,)) if args.sequential_j12 else ((0,1),)
        for stage, axes in enumerate(axis_sets):
            best = refine_axes(best, axes, fields, args.worst_maxiter,
                               20261028+stage, rs04_cap=not args.friction_only)
        if not args.skip_j5 and not args.friction_only:
            best = refine_axes(best, (4,), ('kp','kd','armature','damping'),
                               args.worst_maxiter, 20261029)
    train_outputs = train.evaluate(best, keep=args.output/'final_train')
    holdout = Evaluator(work, (2,5))
    holdout_outputs = holdout.evaluate(best, keep=args.output/'final_holdout')
    report = {'success': result.success, 'message': result.message,
              'optimizer_nfev': result.nfev, 'total_train_evaluations': train.count,
              'parameters': best, 'train_metrics': train.metrics(train_outputs),
              'holdout_metrics': holdout.metrics(holdout_outputs),
              'thresholds': {'q_rmse_deg': 5.0, 'dq_rmse_rad_s': .5}}
    all_metrics = {**report['train_metrics'], **report['holdout_metrics']}
    report['threshold_pass'] = all(all(v <= 5 for v in m['q_rmse_deg']) and
                                   all(v <= .5 for v in m['dq_rmse']) for m in all_metrics.values())
    write_json(args.output/'result.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
