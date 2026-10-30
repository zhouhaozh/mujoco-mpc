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


SMASH = Path('/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/下位机轨迹/r20261020_135606_877000.csv')


@unittest.skipUnless(BINARY.exists() and SMASH.exists(),
                     'requires built x4_command_replay and measured smash trajectory')
class ChassisServoInitializationTests(unittest.TestCase):
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

    def test_default_is_deterministic_and_opt_in_initializes_moving_ramp(self):
        payload = prepare(SMASH)[3]
        initialized_payload = copy.deepcopy(payload)
        initialized_payload['initial']['chassis_servo_state'] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy, legacy_summary = self.run_payload(root, 'legacy', payload)
            legacy_copy, _ = self.run_payload(root, 'legacy_copy', copy.deepcopy(payload))
            initialized, initialized_summary = self.run_payload(
                root, 'initialized', initialized_payload)

        self.assertEqual(legacy, legacy_copy)
        self.assertFalse(legacy_summary['servo_state_initialized_consistently'])
        self.assertTrue(initialized_summary['servo_state_initialized_consistently'])
        np.testing.assert_array_equal(legacy[0]['qpos'], initialized[0]['qpos'])
        np.testing.assert_array_equal(legacy[0]['qvel'], initialized[0]['qvel'])
        initial_body = np.asarray(
            initialized_summary['servo_initialization']['initial_ramped_body_twist'])
        legacy_distance = np.linalg.norm(
            np.asarray(legacy[0]['base_twist_ramped']) - initial_body)
        initialized_distance = np.linalg.norm(
            np.asarray(initialized[0]['base_twist_ramped']) - initial_body)
        self.assertLess(initialized_distance, legacy_distance)
        self.assertTrue(initialized_summary['servo_history_unknown'])
        self.assertEqual(initialized_summary['servo_initialization']['wheel_integral_source'],
                         'zero')
        self.assertFalse(initialized_summary['future_measured_state_copied'])

    def test_body_velocity_override_is_validated_and_used(self):
        payload = prepare(SMASH)[3]
        payload['initial']['chassis_servo_state'] = {
            'ramped_body_velocity': [.25, -.1],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, summary = self.run_payload(root, 'override', payload)
            initial = summary['servo_initialization']['initial_ramped_body_twist']
            np.testing.assert_allclose(initial[:2], [.25, -.1], atol=0)
            self.assertLess(np.linalg.norm(
                np.asarray(frames[0]['base_twist_ramped']) - np.asarray(initial)), .2)

            invalid = copy.deepcopy(payload)
            invalid['initial']['chassis_servo_state']['ramped_body_velocity'] = [0]
            source = root/'invalid.json'
            source.write_text(json.dumps(invalid))
            result = subprocess.run([str(BINARY), str(source), str(root/'invalid')],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ramped_body_velocity must contain 2 values', result.stderr)


if __name__ == '__main__':
    unittest.main()
