import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import test_measured_dynamics
from measured_replay_initialization import DEFAULT_SETTINGS
from run_measured_dynamics import BINARY, prepare, run_profile


class CurrentReplayInitializationTests(unittest.TestCase):
    @unittest.skipUnless(BINARY.exists(), 'build x4_command_replay first')
    def test_current_default_applies_settings_and_legacy_restores_original_start(self):
        native_run = subprocess.run

        def replay_only(command, **kwargs):
            if len(command) > 1 and str(command[1]).endswith('plot_command_dynamics.py'):
                return subprocess.CompletedProcess(command, 0)
            return native_run(command, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = test_measured_dynamics.DynamicsReplayTests().write_log(root)
            lines = [row.split() for row in source.read_text().splitlines()]
            header = lines[0]
            for row in lines[1:]:
                row[header.index('motor1_current_speed')] = '.01'
                row[header.index('motor1_speed_cmd')] = '0'
                index = header.index('motor1_pos_cmd')
                row[index] = str(float(row[index])+100)
            source.write_text('\n'.join('\t'.join(row) for row in lines)+'\n')
            prepared = [prepare(source)]
            base_args = dict(base_command_frame='world', base_feedback_frame='world',
                             arm_mode='hybrid', no_gravity_feedforward=False)
            with patch('run_measured_dynamics.subprocess.run', side_effect=replay_only):
                for legacy in (False, True):
                    args = SimpleNamespace(**base_args, legacy_initialization=legacy)
                    run_profile(args, 'current', root/str(legacy), prepared)

            current = root/'False'
            legacy = root/'True'
            manifest = json.loads((current/'manifest.json').read_text())
            self.assertEqual(manifest['initialization_mode'], 'current_settings')
            self.assertEqual(manifest['initialization_by_group']['1']['status'], 'not_cached')
            snapshot = current/manifest['initialization_settings']['path']
            self.assertEqual(snapshot.read_bytes(), DEFAULT_SETTINGS.read_bytes())
            self.assertEqual(manifest['initialization_settings']['sha256'],
                             hashlib.sha256(snapshot.read_bytes()).hexdigest())
            frames = [json.loads(row) for row in (current/'group_1/frames.jsonl').read_text().splitlines()]
            old = [json.loads(row) for row in (legacy/'group_1/frames.jsonl').read_text().splitlines()]
            self.assertFalse(frames[0]['arm_axis_command_active'][0])
            self.assertTrue(frames[2]['arm_axis_command_active'][0])
            self.assertTrue(all(f['arm_axis_command_active'][0] for f in old))
            self.assertNotEqual(frames[0]['applied_q_command'][0], frames[0]['q_command'][0])
            self.assertEqual(old[0]['applied_q_command'], old[0]['q_command'])
            np.testing.assert_array_equal(frames[0]['qpos'], old[0]['qpos'])
            self.assertGreater(np.max(np.abs(np.asarray(frames[-1]['qvel'])-old[-1]['qvel'])), 1e-9)
            with (current/'group_1/comparison.csv').open() as stream:
                first = next(csv.DictReader(stream))
            self.assertEqual(float(first['arm_active_j1']), 0)
            self.assertNotEqual(first['applied_cmd_q_j1_rad'], first['cmd_q_j1_rad'])
            self.assertEqual(json.loads((legacy/'manifest.json').read_text())['initialization_mode'], 'legacy')


if __name__ == '__main__':
    unittest.main()
