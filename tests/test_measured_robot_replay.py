import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from replay_measured_robot import (DEFAULT_MAPPING, DEFAULT_MODEL, Record, Robot,
                                   interpolate, lag_probe, read_record, validate_mapping)


class MeasuredReplayTests(unittest.TestCase):
    def fixture(self, directory, timestamps=(1792475766492., 1792475766494.), yaw=(179, -179)):
        header = ['local_timestamp', 'robotcurrent_x', 'robotcurrent_y', 'robotcurrent_angle', '#']
        header += [f'motor{i}_current_pos' for i in range(1, 6)]
        header += [f'{prefix}_{axis}' for prefix in ('vicon_racket', 'racket_pos') for axis in 'xyz']
        rows = [[t, 300, 20, angle, '#', 10, 20, 30, 40, 50, 3.2, .3, 1.4, 3.21, .31, 1.41, 0]
                for t, angle in zip(timestamps, yaw)]
        path = Path(directory) / 'log.csv'
        path.write_text('\t'.join(header) + '\n' + '\n'.join('\t'.join(map(str, r)) for r in rows))
        return path

    def test_mixed_units_epoch_origin_mapping_and_yaw_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            r = read_record(self.fixture(directory), DEFAULT_MAPPING)
        np.testing.assert_allclose(r.t, [0, .002])
        np.testing.assert_allclose(r.base[0], [3, .2, np.deg2rad(179)])
        self.assertAlmostEqual(r.base[1, 2], np.deg2rad(181))
        np.testing.assert_allclose(r.q[0], np.deg2rad([-10, -20, 30, 40, 50]))
        np.testing.assert_allclose(r.vicon[0], [3.2, .3, 1.4])
        self.assertEqual(r.extra_columns, 1)

    def test_order_and_offsets(self):
        mapping = copy.deepcopy(DEFAULT_MAPPING)
        mapping['joint_order'] = [2, 1, 3, 4, 5]
        mapping['joint_offset_deg'] = [5, 0, 0, 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            r = read_record(self.fixture(directory), mapping)
        np.testing.assert_allclose(r.q[0], np.deg2rad([-15, -10, 30, 40, 50]))

    def test_bad_clock_and_shifted_separator_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory, timestamps=(10, 10))
            with self.assertRaisesRegex(ValueError, 'increasing'):
                read_record(path, DEFAULT_MAPPING)
            path = self.fixture(directory)
            text = path.read_text().splitlines()
            text[1] = text[1].replace('#', '0')
            path.write_text('\n'.join(text))
            with self.assertRaisesRegex(ValueError, 'layout'):
                read_record(path, DEFAULT_MAPPING)

    def test_invalid_mapping_rejected(self):
        mapping = copy.deepcopy(DEFAULT_MAPPING)
        mapping['joint_sign'] = [1, 1, 0, 1, 1]
        with self.assertRaises(ValueError):
            validate_mapping(mapping)

    def test_known_lag_sign_and_same_support(self):
        t = np.linspace(0, .5, 251)
        points = np.column_stack([t, t*t, np.sin(8*t)])
        vicon = interpolate(t, points, t-.012)
        r = Record(Path('synthetic'), t*1000, t, np.zeros((len(t), 3)),
                   np.zeros((len(t), 5)), vicon, points, 0)
        probe = lag_probe(r, points)
        self.assertEqual(probe['best_shift_ms'], -12)
        self.assertLess(probe['best_rmse_m'], 1e-12)
        self.assertGreater(probe['zero_shift_rmse_m_same_support'], .01)

    def test_actual_model_fk_no_physics_and_j5_unobservable(self):
        robot = Robot(DEFAULT_MODEL)
        with patch.object(robot.mj, 'mj_step', side_effect=AssertionError('physics forbidden')):
            # First actual row of group 1, using the inferred directions and units.
            q = np.deg2rad([1.51595, -18.2552, -20.0835, -90.9693, .770559])
            p = robot.pose([2.81199, .135205, np.deg2rad(3)], q)
            np.testing.assert_allclose(p, [2.62295, .11567, 1.31885], atol=.001)
            orientation = robot.data.geom_xmat[robot.face].copy()
            q[4] += .7
            p2 = robot.pose([2.81199, .135205, np.deg2rad(3)], q)
            np.testing.assert_allclose(p, p2, atol=1e-12)
            self.assertGreater(np.linalg.norm(orientation-robot.data.geom_xmat[robot.face]), .1)


if __name__ == '__main__':
    unittest.main()
