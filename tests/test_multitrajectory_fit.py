import copy
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import fit_multitrajectory_dynamics as fit


def interpolated_reference(suffix_value):
    raw_t = np.array([0.0, 0.0486, 0.0507, 0.1000])
    raw_y = np.array([0.0, 1.0, suffix_value, suffix_value + 1.0])
    t = np.arange(0.0, 0.1001, 0.001)
    y = np.interp(t, raw_t, raw_y)
    ref = {'t_s': t}
    for j in range(1, 6):
        ref[f'real_q_j{j}_rad'] = y + j
        ref[f'real_dq_j{j}_rad_s'] = y - j
    for name, offset in [('x_m', 1), ('y_m', 2), ('yaw_rad', 3),
                         ('vx_m_s', 4), ('vy_m_s', 5), ('wz_rad_s', 6)]:
        ref[f'real_base_{name}'] = y + offset
    for axis, offset in zip('xyz', (7, 8, 9)):
        ref[f'real_racket_{axis}_m'] = y + offset
        ref[f'vicon_racket_{axis}_m'] = y - offset
    return ref


def payload(suffix_command):
    commands = []
    for t, value in [(0.0, 0.0), (0.0486, 1.0),
                     (0.0507, suffix_command), (0.1000, suffix_command + 1.0)]:
        commands.append({'t_s': t, 'q_rad': [value]*5,
                         'dq_rad_s': [value]*5,
                         'base_twist': [value]*3})
    return {'duration_s': 0.1, 'commands': commands, 'initial': {}}


def reference_columns(t):
    zeros = np.zeros_like(t)
    ref = {'t_s': t}
    for j in range(1, 6):
        ref[f'real_q_j{j}_rad'] = zeros
        ref[f'real_dq_j{j}_rad_s'] = zeros
    for name in ('x_m', 'y_m', 'yaw_rad', 'vx_m_s', 'vy_m_s', 'wz_rad_s'):
        ref[f'real_base_{name}'] = zeros
    for axis in 'xyz':
        ref[f'real_racket_{axis}_m'] = zeros
        ref[f'vicon_racket_{axis}_m'] = zeros
    return ref


def constant_error_frames(t):
    return [{'t': float(x), 'q': [1.0]*5, 'dq': [1.0]*5,
             'base': [1.0, 1.0, 0.0],
             'base_world_velocity': [1.0, 1.0, 0.0],
             'racket_xyz': [1.0]*3} for x in t]


class PrefixIsolationTests(unittest.TestCase):
    def test_suffix_observations_and_commands_do_not_change_prefix_problem(self):
        before_payload, before_ref = fit.prefix_problem(
            payload(10.0), interpolated_reference(10.0))
        after_payload, after_ref = fit.prefix_problem(
            payload(-500.0), interpolated_reference(500.0))

        self.assertEqual(before_payload, after_payload)
        self.assertEqual(before_payload['duration_s'], 0.0486)
        self.assertEqual([c['t_s'] for c in before_payload['commands']],
                         [0.0, 0.0486])
        for key in before_ref:
            np.testing.assert_array_equal(before_ref[key], after_ref[key])
        self.assertLessEqual(before_ref['t'][-1], 0.0486)

    def test_folds_hold_out_every_trial_once_and_keep_four_per_class(self):
        held_out = [group for fold in fit.FOLDS for group in fold]
        self.assertCountEqual(held_out, range(1, 11))
        self.assertEqual(len(held_out), len(set(held_out)))
        for fold in fit.FOLDS:
            train = [g for g in range(1, 11) if g not in fold]
            self.assertEqual(sum(g <= 5 for g in train), 4)
            self.assertEqual(sum(g > 5 for g in train), 4)

    def test_constant_error_residual_is_sample_count_normalized(self):
        class Evaluation:
            groups = (1,)

        norms = []
        for count in (51, 101):
            t = np.linspace(0.05, 0.10, count)
            Evaluation.refs = {1: reference_columns(t)}
            frames = constant_error_frames(t)
            baseline = {1: fit.arrays(copy.deepcopy(frames))}
            value = fit.residual(Evaluation, {1: frames}, baseline, 'arm', stride=1)
            norms.append(np.linalg.norm(value))
        self.assertAlmostEqual(norms[0], norms[1], places=12)


if __name__ == '__main__':
    unittest.main()
