import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from run_measured_dynamics import BINARY, prepare


DATA = Path('/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/下位机轨迹/r20261020_135606_877000.csv')


@unittest.skipUnless(BINARY.exists() and DATA.exists(),
                     'requires built x4_command_replay and measured trajectory')
class ArmStartPolicyTests(unittest.TestCase):
    def payload(self):
        payload = prepare(DATA)[3]
        payload['initial']['dq_rad_s'] = [0.0]*5
        first = np.asarray(payload['commands'][0]['q_rad'])
        for command in payload['commands']:
            command['q_rad'] = first.tolist()
            command['dq_rad_s'] = [0.0]*5
        return payload

    def run_payload(self, root, name, payload):
        source = root/f'{name}.json'
        source.write_text(json.dumps(payload))
        output = root/name
        result = subprocess.run([str(BINARY), str(source), str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        frames = [json.loads(line) for line in
                  (output/'frames.jsonl').read_text().splitlines()]
        summary = json.loads((output/'summary.json').read_text())
        return frames, summary

    def test_default_and_explicit_immediate_have_identical_physics(self):
        payload = self.payload()
        explicit = copy.deepcopy(payload)
        explicit['arm_start_policy'] = 'immediate'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy, legacy_summary = self.run_payload(root, 'legacy', payload)
            immediate, immediate_summary = self.run_payload(root, 'immediate', explicit)
        for old, new in zip(legacy, immediate):
            np.testing.assert_array_equal(old['qpos'], new['qpos'])
            np.testing.assert_array_equal(old['qvel'], new['qvel'])
            np.testing.assert_array_equal(new['applied_q_command'], new['q_command'])
            np.testing.assert_array_equal(new['applied_dq_command'], new['dq_command'])
            self.assertTrue(new['arm_command_active'])
        self.assertEqual(legacy_summary['arm_start_policy'], 'immediate')
        self.assertEqual(immediate_summary['arm_activation_time_s'], 0.0)
        self.assertEqual(immediate_summary['arm_axis_activation_times_s'], [0.0]*5)

    def test_hold_activates_only_when_fresh_command_is_consumed(self):
        payload = self.payload()
        payload['arm_start_policy'] = 'hold_until_fresh_command'
        initial_q = np.asarray(payload['initial']['q_rad'])
        raw_first = np.asarray(payload['commands'][0]['q_rad'])
        self.assertGreater(np.max(np.abs(initial_q-raw_first)), np.deg2rad(.5))
        payload['commands'][2]['q_rad'][0] += np.deg2rad(.6)
        expected_activation = np.ceil(payload['commands'][2]['t_s']/0.001-1e-12)*0.001
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, summary = self.run_payload(root, 'hold', payload)

        self.assertTrue(summary['arm_hold_entered'])
        self.assertEqual(summary['arm_axis_pending_initial'],
                         [True, True, False, False, False])
        self.assertAlmostEqual(summary['arm_axis_activation_times_s'][0],
                               expected_activation)
        self.assertIsNone(summary['arm_axis_activation_times_s'][1])
        self.assertIsNone(summary['arm_activation_time_s'])
        activation_index = int(round(expected_activation/0.001))
        for frame in frames[:activation_index]:
            self.assertFalse(frame['arm_command_active'])
            self.assertEqual(frame['arm_axis_command_active'],
                             [False, False, True, True, True])
            np.testing.assert_allclose(frame['applied_q_command'][:2], initial_q[:2], atol=0)
            np.testing.assert_array_equal(frame['applied_dq_command'][:2], [0.0]*2)
            np.testing.assert_array_equal(frame['q_command'], raw_first)
        self.assertFalse(frames[activation_index]['arm_command_active'])
        self.assertEqual(frames[activation_index]['arm_axis_command_active'],
                         [True, False, True, True, True])
        self.assertEqual(frames[activation_index]['applied_q_command'][0],
                         frames[activation_index]['q_command'][0])
        self.assertEqual(frames[activation_index]['applied_q_command'][1], initial_q[1])

        future = copy.deepcopy(payload)
        future['commands'][2]['q_rad'][0] += .2
        with tempfile.TemporaryDirectory() as directory:
            changed, _ = self.run_payload(Path(directory), 'future', future)
        for before, after in zip(frames[:activation_index+1],
                                 changed[:activation_index+1]):
            np.testing.assert_array_equal(before['qpos'], after['qpos'])
            np.testing.assert_array_equal(before['qvel'], after['qvel'])
        changed_index = activation_index + 1
        self.assertGreater(np.max(np.abs(np.asarray(frames[changed_index]['qvel'])-
                                        np.asarray(changed[changed_index]['qvel']))), 1e-10)

    def test_moving_initial_state_bypasses_hold(self):
        payload = self.payload()
        payload['arm_start_policy'] = 'hold_until_fresh_command'
        payload['initial']['dq_rad_s'][0] = .1
        with tempfile.TemporaryDirectory() as directory:
            frames, summary = self.run_payload(Path(directory), 'moving', payload)
        self.assertTrue(summary['arm_hold_eligible'])
        self.assertTrue(summary['arm_hold_entered'])
        self.assertEqual(summary['arm_axis_pending_initial'],
                         [False, True, False, False, False])
        self.assertEqual(frames[0]['arm_axis_command_active'],
                         [True, False, True, True, True])
        self.assertEqual(frames[0]['applied_q_command'][0],
                         frames[0]['q_command'][0])
        self.assertEqual(frames[0]['applied_q_command'][1],
                         payload['initial']['q_rad'][1])

    def test_fresh_velocity_command_activates_hold(self):
        payload = self.payload()
        payload['arm_start_policy'] = 'hold_until_fresh_command'
        payload['commands'][1]['dq_rad_s'][0] = .051
        expected = np.ceil(payload['commands'][1]['t_s']/0.001-1e-12)*0.001
        with tempfile.TemporaryDirectory() as directory:
            frames, summary = self.run_payload(Path(directory), 'velocity', payload)
        self.assertAlmostEqual(summary['arm_axis_activation_times_s'][0], expected)
        index = int(round(expected/.001))
        self.assertFalse(frames[index-1]['arm_axis_command_active'][0])
        self.assertTrue(frames[index]['arm_axis_command_active'][0])
        self.assertEqual(frames[index]['applied_dq_command'][0], .051)


if __name__ == '__main__':
    unittest.main()
