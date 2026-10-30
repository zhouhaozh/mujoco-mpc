#!/usr/bin/env python3
"""Bounded, shared system identification with prefix-only initial-state estimates.

Uses existing native MuJoCo replay and isolated model snapshots. No production
configuration is changed. All validation on these previously inspected logs is
retrospective; unrecorded drive modes remain modeling assumptions.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import parameter_fit_support
from measured_dynamics_profiles import CURRENT_CONFIGS, PROJECT, snapshot_profile
from parameter_fit_support import Evaluator
from run_measured_dynamics import BINARY, compare, prepare

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path('/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹')
PREFIX = .050
FOLDS = [(1, 7), (3, 9), (4, 10), (2, 6), (5, 8)]
CHASSIS = [('chassis_ramp', 'accel_limit_mps2'), ('chassis_ramp', 'dir_rate_limit_rad_s'),
           ('chassis_servo', 'steer_kp'), ('chassis_servo', 'steer_kd'),
           ('chassis_servo', 'wheel_kp'), ('chassis_servo', 'wheel_accel_mps2')]
CHASSIS_BOUNDS = [(2., 30.), (2., 40.), (2., 60.), (.1, 8.), (.5, 15.), (2., 30.)]
ARM = [(field, j) for field in ('kp', 'kd') for j in range(5)]
ARM_BOUNDS = [(1e-5, 1000.)]*4 + [(1e-5, 100.)] + [(1e-4, 100.)]*4 + [(1e-4, 5.)]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_hash(payload):
    return hashlib.sha256(json.dumps(payload['commands'], sort_keys=True).encode()).hexdigest()


def sources(data):
    return sorted((data/'下位机轨迹').glob('*.csv')) + sorted((data/'和平球轨迹').glob('*.csv'))


def arrays(frames):
    return {key: np.asarray([f[field] for f in frames]) for key, field in
            [('t', 't'), ('q', 'q'), ('dq', 'dq'), ('base', 'base'),
             ('v', 'base_world_velocity'), ('racket', 'racket_xyz')]}


def reference(ref):
    return {'t': ref['t_s'], 'q': np.column_stack([ref[f'real_q_j{j}_rad'] for j in range(1, 6)]),
            'dq': np.column_stack([ref[f'real_dq_j{j}_rad_s'] for j in range(1, 6)]),
            'base': np.column_stack([ref[f'real_base_{k}'] for k in ('x_m', 'y_m', 'yaw_rad')]),
            'v': np.column_stack([ref[f'real_base_{k}'] for k in ('vx_m_s', 'vy_m_s', 'wz_rad_s')]),
            'racket': np.column_stack([ref[f'real_racket_{k}_m'] for k in 'xyz']),
            'vicon': np.column_stack([ref[f'vicon_racket_{k}_m'] for k in 'xyz'])}


def prefix_problem(payload, ref):
    """Restrict both inputs and reference to raw samples inside the prefix."""
    payload = copy.deepcopy(payload)
    payload['commands'] = [c for c in payload['commands'] if c['t_s'] <= PREFIX]
    end = payload['commands'][-1]['t_s']
    payload['duration_s'] = end
    real = reference(ref)
    mask = real['t'] <= end+1e-12
    return payload, {k: a[mask] for k, a in real.items()}


def error_blocks(sim, real, mask):
    return [(sim['q']-real['q'])[mask]/np.deg2rad(5),
            (sim['dq']-real['dq'])[mask]/.5,
            (sim['base'][:, :2]-real['base'][:, :2])[mask]/.05,
            (sim['v'][:, :2]-real['v'][:, :2])[mask]/.3,
            (sim['racket']-real['racket'])[mask]/.10]


def metric(sim, real, start=PREFIX, end=np.inf):
    m = (sim['t'] >= start-1e-12) & (sim['t'] <= end+1e-12)
    if not m.any():
        raise ValueError('empty evaluation window')
    def stats(e):
        e = e[m]
        return {'rmse': np.sqrt(np.mean(e*e, axis=0)).tolist(),
                'p95_abs': np.quantile(np.abs(e), .95, axis=0).tolist(),
                'max_abs': np.max(np.abs(e), axis=0).tolist(),
                'bias': np.mean(e, axis=0).tolist(),
                'vector_rmse': float(np.sqrt(np.mean(np.sum(e*e, axis=1))))}
    yaw = (sim['base'][:, 2:3]-real['base'][:, 2:3]+np.pi) % (2*np.pi)-np.pi
    return {'start_s': float(sim['t'][m][0]), 'end_s': float(sim['t'][m][-1]),
            'q_deg': stats(np.rad2deg(sim['q']-real['q'])), 'dq': stats(sim['dq']-real['dq']),
            'base_xy': stats(sim['base'][:, :2]-real['base'][:, :2]),
            'base_vxy': stats(sim['v'][:, :2]-real['v'][:, :2]), 'yaw_deg': stats(np.rad2deg(yaw)),
            'wz': stats(sim['v'][:, 2:3]-real['v'][:, 2:3]),
            'racket_fk': stats(sim['racket']-real['racket']),
            'racket_vicon': stats(sim['racket']-real['vicon'])}


def freeze_baseline(out, data):
    base = out/'baseline'
    paths = sources(data)
    if (base/'manifest.json').exists():
        manifest = json.loads((base/'manifest.json').read_text())
        expected = [manifest['sources'][str(g)] for g in range(1, 11)]
        if [str(p) for p in paths] != [m['path'] for m in expected]:
            raise ValueError('data selection changed; use a new output directory')
        if any(sha(p) != m['sha256'] for p, m in zip(paths, expected)):
            raise ValueError('source contents changed; use a new output directory')
        if sha(BINARY) != manifest['binary_sha256']:
            raise ValueError('native binary changed; use a new output directory')
        current_hashes = {k: sha(PROJECT/v) for k, v in CURRENT_CONFIGS.items()}
        if current_hashes != manifest['input_config_sha256']:
            raise ValueError('source configuration changed; use a new output directory')
        for key, path in manifest['config_paths'].items():
            if sha(Path(path)) != manifest['config_sha256'][key]:
                raise ValueError('frozen configuration was modified')
        for g in range(1, 11):
            payload = json.loads((base/f'group_{g}_input.json').read_text())
            if command_hash(payload) != manifest['command_sha256'][str(g)]:
                raise ValueError('frozen commands were modified')
        return
    if len(paths) != 10:
        raise ValueError(f'expected 10 recordings, found {len(paths)}')
    base.mkdir(parents=True, exist_ok=True)
    cfg = snapshot_profile('current', base/'config')
    manifest = {'config_paths': cfg,
                'config_contents': {k: json.loads(Path(v).read_text()) for k, v in cfg.items() if v.endswith('.json')},
                'config_sha256': {k: sha(Path(v)) for k, v in cfg.items()},
                'input_config_sha256': {k: sha(PROJECT/v) for k, v in CURRENT_CONFIGS.items()},
                'binary_sha256': sha(BINARY), 'sources': {}, 'command_sha256': {}}
    def run(item):
        g, path = item
        record, dq, velocity, payload = prepare(path)
        payload.update(project_root=str(base/'config'), **{k: Path(v).name for k, v in cfg.items()})
        input_path = base/f'group_{g}_input.json'
        save(input_path, payload)
        subprocess.run([str(BINARY), str(input_path), str(base/f'group_{g}')], check=True, capture_output=True)
        report = compare(base/f'group_{g}', record, dq, velocity, payload)
        return g, path, payload, report
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(run, enumerate(paths, 1)))
    for g, path, payload, report in results:
        manifest['sources'][g] = {'path': str(path), 'sha256': sha(path), 'label': f'{"S" if g <= 5 else "P"}{g if g <= 5 else g-5}'}
        manifest['command_sha256'][g] = command_hash(payload)
    save(base/'manifest.json', manifest)
    save(base/'metrics.json', {g: r for g, _, _, r in results})
    tree = ET.parse(cfg['scene_path'])
    p = {k: manifest['config_contents']['robot_variant_path']['arm_servo'][k] for k in ('kp', 'kd')}
    for field in ('armature', 'damping', 'frictionloss'):
        p[field] = [float(tree.find(f".//joint[@name='Joint{j}']").get(field, '0')) for j in range(1, 6)]
    c = manifest['config_contents']['chassis_path']
    p.update(chassis_ramp=c['ramp'], chassis_servo=c['servo'])
    save(out/'initial_parameters.json', p)
    print('Fresh baseline frozen for 10 trajectories', flush=True)


def evaluator(out, groups=range(1, 11), states=None, apply_policy=True):
    e = Evaluator(out/'work', groups, baseline_dir=out/'baseline', scene_path=out/'baseline/config/scene.mjcf')
    protocol = json.loads((out/'protocol.json').read_text())
    if apply_policy and protocol.get('arm_start_policy', 'immediate') != 'immediate':
        for payload in e.inputs.values():
            payload['arm_start_policy'] = protocol['arm_start_policy']
    if states is not None:
        for g in groups:
            state = states[str(g)]['state']
            if state is not None:
                e.inputs[g]['initial']['chassis_servo_state'] = state
    return e


def initial_estimates(out):
    p = json.loads((out/'initial_parameters.json').read_text())
    all_e = evaluator(out, apply_policy=False)
    reproduction = all_e.evaluate(p)
    for g, frames in reproduction.items():
        old = [json.loads(s) for s in (out/f'baseline/group_{g}/frames.jsonl').read_text().splitlines()]
        for k in ('qpos', 'qvel'):
            np.testing.assert_allclose([f[k] for f in frames], [f[k] for f in old], atol=1e-12, rtol=0)
    save(out/'baseline_reproduction.json', {'groups': 10, 'atol': 1e-12, 'binary_sha256': sha(BINARY)})
    estimates = {}
    for g in range(1, 11):
        e = evaluator(out, (g,))
        # Truncate simulator and observed reference before doing any hidden-state selection.
        # Commands and feedback share raw timestamps. End at the last raw sample
        # inside the prefix so reference interpolation cannot use a suffix sample.
        e.inputs[g], real = prefix_problem(e.inputs[g], e.refs[g])
        prefix_end = e.inputs[g]['duration_s']
        rows = []
        def score(state, prior=False):
            if state is None:
                e.inputs[g]['initial'].pop('chassis_servo_state', None)
            else:
                e.inputs[g]['initial']['chassis_servo_state'] = state
            sim = arrays(e.evaluate(p)[g])
            n = len(sim['t'])
            r = np.r_[((sim['base'][:, :2]-real['base'][:n, :2])/.01).ravel(),
                      ((sim['v'][:, :2]-real['v'][:n, :2])/.2).ravel()]/np.sqrt(n)
            if prior:
                r = np.r_[r, .1*np.asarray(state['ramped_body_velocity'])]
            rows.append({'state': copy.deepcopy(state), 'prefix_cost': float(r@r)})
            return r
        for state in (None, {}, {'ramped_body_velocity': [0., 0.]}):
            score(state)
        v = e.inputs[g]['initial']['base_world_velocity']
        if np.linalg.norm(v[:2]) > .1:
            # Only two latent coordinates; bounded by observed initial speed + 0.5 m/s.
            limit = min(3., np.linalg.norm(v[:2])+.5)
            fit = least_squares(lambda x: score({'ramped_body_velocity': x.tolist()}, True),
                                [0., 0.], bounds=(-limit, limit), diff_step=.05,
                                max_nfev=12, x_scale=1., ftol=.003, xtol=.003)
            score({'ramped_body_velocity': fit.x.tolist()})
        # Compare all options by the same data-only objective, not their penalized objective.
        candidates = [None, {}, {'ramped_body_velocity': [0., 0.]}]
        if np.linalg.norm(v[:2]) > .1:
            candidates.append({'ramped_body_velocity': fit.x.tolist()})
        selected = min(candidates, key=lambda s: float(np.sum(score(s)**2)))
        estimates[str(g)] = {'state': selected, 'prefix_end_s': prefix_end, 'trials': rows,
                              'scope': 'prefix-only effective controller initial state; hardware state unknown'}
        print(f'Initial state {g}: {selected}', flush=True)
    save(out/'initial_states.json', estimates)
    save(out/'initial_states_binding.json', {'baseline_sha256': sha(out/'baseline/manifest.json'),
                                           'protocol_sha256': sha(out/'protocol.json'),
                                           'states_sha256': sha(out/'initial_states.json')})
    e = evaluator(out, states=estimates)
    outputs = e.evaluate(p, keep=out/'initialized')
    write_metrics(out, out/'initialized', e, outputs)


def residual(e, outputs, baseline, kind, stride=5):
    pieces = []
    # Every trial and both classes have equal weight (folds retain equal class counts).
    weights = [1., 1., .2, .2, .15] if kind == 'arm' else [.15, .15, 1., 1., .15]
    for g in e.groups:
        s, r = arrays(outputs[g]), reference(e.refs[g])
        m = s['t'] >= PREFIX-1e-12
        selected = np.flatnonzero(m)[::stride]
        blocks = error_blocks(s, r, selected)
        for weight, block in zip(weights, blocks):
            pieces.append(np.sqrt(weight)*block.ravel()/np.sqrt(block.size*len(e.groups)))
        # Counter aggregate-loss compensation; these penalties are not an acceptance claim.
        old = error_blocks(baseline[g], r, selected)
        for a, b in zip(blocks, old):
            rmse = np.sqrt(np.mean(a*a, axis=0))
            old_rmse = np.sqrt(np.mean(b*b, axis=0))
            pieces.append(.25*np.maximum(0., (rmse-old_rmse)/np.maximum(old_rmse, .1))/np.sqrt(len(e.groups)))
        # Fixed final 20 ms is a diagnostic window, not a verified contact event.
        event = (s['t'] >= s['t'][-1]-.020-1e-12)
        for a in error_blocks(s, r, event):
            pieces.append(np.sqrt(.08/len(e.groups))*a.ravel()/np.sqrt(a.size))
    return np.concatenate(pieces)


def write_metrics(out, directory, e, outputs):
    result = {}
    manifest = json.loads((out/'baseline/manifest.json').read_text())
    for g, frames in outputs.items():
        payload = json.loads((directory/f'input_{g}.json').read_text())
        if command_hash(payload) != manifest['command_sha256'][str(g)]:
            raise AssertionError('original commands changed')
        record, dq, velocity, _ = prepare(Path(manifest['sources'][str(g)]['path']))
        # Existing compare validates causal command consumption, exact t0, finite physics.
        compare(directory/f'group_{g}', record, dq, velocity, payload)
        s, r = arrays(frames), reference(e.refs[g])
        result[g] = {'full': metric(s, r, 0), 'evaluation': metric(s, r),
                     'first_50ms': metric(s, r, 0, PREFIX),
                     'last_20ms': metric(s, r, s['t'][-1]-.020),
                     'stable_arrival': 'not_observed', 'actual_contact': 'unconfirmed'}
    save(directory/'metrics.json', result)
    return result


def fit(out, fold, chassis_nfev, arm_nfev):
    folder = out/('fit_all' if fold == 'all' else f'fold_{fold}')
    if (folder/'result.json').exists():
        print(f'Skipping completed {folder.name}', flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    initial = json.loads((out/'initial_parameters.json').read_text())
    states = json.loads((out/'initial_states.json').read_text())
    expected_binding = {'baseline_sha256': sha(out/'baseline/manifest.json'),
                        'protocol_sha256': sha(out/'protocol.json'),
                        'states_sha256': sha(out/'initial_states.json')}
    if json.loads((out/'initial_states_binding.json').read_text()) != expected_binding:
        raise ValueError('initial states do not match frozen baseline/protocol')
    holdout = () if fold == 'all' else FOLDS[int(fold)-1]
    train = tuple(g for g in range(1, 11) if g not in holdout)
    e = evaluator(out, train, states)
    baseline = {g: arrays(f) for g, f in e.evaluate(initial).items()}
    selected = copy.deepcopy(initial)
    stages = []
    with (folder/'trials.jsonl').open('w', buffering=1) as log:
        for stage, fields, bounds, nfev in [('chassis', CHASSIS, CHASSIS_BOUNDS, chassis_nfev),
                                             ('arm', ARM, ARM_BOUNDS, arm_nfev)]:
            fixed = copy.deepcopy(selected)
            x0 = np.log([fixed[k][j] for k, j in fields])
            lo, hi = np.log(bounds).T
            x0 = np.clip(x0, lo+1e-9, hi-1e-9)
            best = {'cost': float('inf')}
            calls = 0
            size = None
            def fun(x):
                nonlocal calls, size
                calls += 1
                p = copy.deepcopy(fixed)
                for (k, j), val in zip(fields, np.exp(x)):
                    p[k][j] = float(val)
                try:
                    outputs = e.evaluate(p)
                    r = np.r_[residual(e, outputs, baseline, stage), .01*(x-x0)]
                    if not np.isfinite(r).all():
                        raise ValueError('nonfinite residual')
                    size = len(r)
                    cost, error = float(r@r), None
                except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    if size is None:
                        raise
                    r, cost, error = np.full(size, 1000.), 1e9, str(exc)
                log.write(json.dumps({'stage': stage, 'call': calls, 'cost': cost, 'parameters': p, 'error': error})+'\n')
                if cost < best['cost']:
                    best.update(cost=cost, parameters=p, x=x.tolist())
                    save(folder/f'best_{stage}.json', best)
                    print(f'{folder.name}/{stage} {calls}: {cost:.5f}', flush=True)
                return r
            before = float(np.sum(fun(x0)**2))
            # Contact and actuator clipping can make the local Jacobian misleading.
            # A small deterministic coordinate search supplies a training-only start.
            for j in range(len(fields)):
                anchor = np.array(best['x'])
                for factor in (.5, .75, 1.25, 1.5):
                    trial = anchor.copy()
                    trial[j] = np.clip(anchor[j]+np.log(factor), lo[j]+1e-9, hi[j]-1e-9)
                    fun(trial)
            opt = least_squares(fun, np.array(best['x']), bounds=(lo, hi), diff_step=.005, x_scale=1.,
                                max_nfev=nfev, ftol=.002, xtol=.0001, gtol=1e-5)
            selected = best['parameters']
            stages.append({'stage': stage, 'before': before, 'after': best['cost'], 'calls': calls,
                           'message': opt.message, 'solver_success': bool(opt.success),
                           'parameters': fields, 'bounds': bounds})
    save(folder/'selection_frozen.json', {'parameters': selected, 'training_groups': train,
                                         'holdout_groups': holdout, 'stages': stages,
                                         'holdout_suffix_used_for_selection': False})
    all_e = evaluator(out, states=states)
    outputs = all_e.evaluate(selected, keep=folder/'replay')
    metrics = write_metrics(out, folder/'replay', all_e, outputs)
    save(folder/'result.json', {'parameters': selected, 'stages': stages, 'metrics': metrics,
                               'training_groups': train, 'holdout_groups': holdout,
                               'evaluation_type': 'in_sample' if not holdout else 'retrospective_suffix_holdout_with_prefix_conditioning',
                               'initial_estimation_prefix_s': PREFIX, 'evaluation_start_s': PREFIX})
    print(f'Completed {folder.name}', flush=True)


def main():
    global BINARY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'outputs/multitrajectory_sysid_20261030')
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--phase', choices=['prepare', 'fit'], required=True)
    ap.add_argument('--fold', choices=['all', '1', '2', '3', '4', '5'], default='all')
    ap.add_argument('--chassis-nfev', type=int, default=10)
    ap.add_argument('--arm-nfev', type=int, default=14)
    ap.add_argument('--arm-start-policy', choices=['immediate', 'hold_until_fresh_command'], default='immediate')
    ap.add_argument('--binary', type=Path, help='Explicit frozen native binary for replaying an older experiment')
    args = ap.parse_args()
    if args.binary:
        BINARY = args.binary.resolve()
        parameter_fit_support.BINARY = BINARY
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prepare':
        protocol = {'prefix_s': PREFIX, 'folds': FOLDS, 'shared_parameters': CHASSIS+ARM,
                                  'arm_start_policy': args.arm_start_policy,
                                  'chassis_bounds': CHASSIS_BOUNDS, 'arm_bounds': ARM_BOUNDS,
                                  'time_shift_fitted': False, 'torque_mass_inertia_fitted': False,
                                  'candidate_tolerances_are_not_formal_acceptance': True,
                                  'hidden_state_rule': 'raw-sample-bounded prefix selection; two bounded ramp velocities only when initial speed > 0.1 m/s',
                                  'observed_future_state_injected': False,
                                  'data_role': 'all logs previously inspected; retrospective validation only'}
        if (out/'protocol.json').exists():
            existing = json.loads((out/'protocol.json').read_text())
            # Older immediate-policy experiments predate the explicit field.
            existing.setdefault('arm_start_policy', 'immediate')
            if existing != protocol:
                raise ValueError('protocol changed; use a new output directory')
        else:
            save(out/'protocol.json', protocol)
        freeze_baseline(out, args.data)
        if not (out/'initial_states.json').exists():
            initial_estimates(out)
    else:
        freeze_baseline(out, args.data)
        fit(out, args.fold, args.chassis_nfev, args.arm_nfev)


if __name__ == '__main__':
    main()
