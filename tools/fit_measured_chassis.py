#!/usr/bin/env python3
"""Fit chassis simulation parameters to measured base trajectories.

Production files and controller logic are never modified. Groups 1/3/4 are
used for fitting; groups 2/5 are evaluated only after search completes.
Run with /usr/bin/python3 (SciPy environment).
"""
import argparse
import json
import math

import numpy as np
from scipy.optimize import differential_evolution

from parameter_fit_support import Evaluator, ROOT

OUTPUT = ROOT / 'outputs/parameter_fit/chassis'


def decode(x):
    values = np.exp(x)
    return {
        'chassis_ramp': {
            'accel_limit_mps2': float(values[0]),
            'ang_accel_limit_rad_s2': float(values[1]),
            'dir_rate_limit_rad_s': float(values[2]),
        },
        'chassis_servo': {
            'steer_kp': float(values[3]), 'steer_kd': float(values[4]),
            'wheel_kp': float(values[5]), 'wheel_ki': float(values[6]),
            'wheel_int_max_rad': float(values[7]),
            'wheel_accel_mps2': float(values[8]),
        },
    }


def encode(parameters):
    ramp, servo = parameters['chassis_ramp'], parameters['chassis_servo']
    return np.log([
        ramp['accel_limit_mps2'], ramp['ang_accel_limit_rad_s2'],
        ramp['dir_rate_limit_rad_s'], servo['steer_kp'], servo['steer_kd'],
        servo['wheel_kp'], servo['wheel_ki'], servo['wheel_int_max_rad'],
        servo['wheel_accel_mps2'],
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maxiter', type=int, default=12)
    parser.add_argument('--popsize', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()
    if args.maxiter < 0 or args.popsize < 2:
        parser.error('maxiter must be nonnegative and popsize >= 2')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trials, best_path = OUTPUT/'trials.jsonl', OUTPUT/'best.json'
    evaluator = Evaluator(OUTPUT/'work', groups=(1, 3, 4))
    best = json.loads(best_path.read_text()) if best_path.exists() else {'objective': math.inf}

    def objective(x):
        nonlocal best
        parameters = decode(x)
        try:
            outputs = evaluator.evaluate(parameters)
            residual = evaluator.residual(outputs, kind='chassis', stride=5)
            value = float(np.sqrt(np.mean(residual**2)))
            metrics, status = evaluator.metrics(outputs), 'ok'
        except (ValueError, RuntimeError, TimeoutError) as error:
            value, metrics, status = 1e6, {}, f'failed: {error}'
        row = {'trial': evaluator.count, 'objective': value, 'parameters': parameters,
               'training_metrics': metrics, 'status': status}
        with trials.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        if value < best['objective']:
            best = row
            best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False)+'\n')
            print(f'trial {evaluator.count}: objective={value:.5f}', flush=True)
        return value

    # Fit constraints, not hardware ratings. Gain maxima remain below the RS06
    # protocol bounds (Kp=5000, Kd=100).
    physical_bounds = [
        (1., 30.), (.5, 30.), (1., 40.), (.1, 100.), (.02, 20.),
        (.05, 10.), (.001, 30.), (.01, 30.), (1., 40.),
    ]
    bounds = [(math.log(lo), math.log(hi)) for lo, hi in physical_bounds]
    x0 = encode(best['parameters']) if 'parameters' in best else None
    result = differential_evolution(objective, bounds, seed=args.seed, maxiter=args.maxiter,
                                    popsize=args.popsize, polish=False, x0=x0, workers=1,
                                    updating='immediate', tol=.002, atol=.002)
    parameters = decode(result.x)
    validation = Evaluator(OUTPUT/'validation_work', groups=(1, 2, 3, 4, 5))
    outputs = validation.evaluate(parameters, keep=OUTPUT/'final')
    names = ['accel_limit_mps2', 'ang_accel_limit_rad_s2', 'dir_rate_limit_rad_s',
             'steer_kp', 'steer_kd', 'wheel_kp', 'wheel_ki',
             'wheel_int_max_rad', 'wheel_accel_mps2']
    final = {
        'objective': float(result.fun), 'optimizer_success': bool(result.success),
        'optimizer_message': str(result.message), 'training_trials': evaluator.count,
        'training_groups': [1, 3, 4], 'holdout_groups': [2, 5],
        'parameters': parameters, 'all_group_metrics': validation.metrics(outputs),
        'bounds': {name: list(bound) for name, bound in zip(names, physical_bounds)},
        'limitations': [
            'No wheel/steer measurements exist, so individual chassis motors are not identifiable.',
            'Controller ramp/integrator history before the first frame is unavailable.',
            'Existing coordination and contact logic are unchanged.',
            'Bounds are fit constraints; fitted values are simulation parameters, not hardware ratings.',
        ],
    }
    (OUTPUT/'final.json').write_text(json.dumps(final, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
