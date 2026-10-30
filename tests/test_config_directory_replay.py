import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import test_measured_dynamics
import run_measured_dynamics as replay
from measured_dynamics_profiles import replay_scene

CONFIGS = ROOT/'sckd_sim/models/config/measured_dynamics'


class ConfigDirectoryReplayTests(unittest.TestCase):
    def invoke(self, *arguments, cwd=None):
        return subprocess.run([sys.executable, str(ROOT/'tools/run_measured_dynamics.py'),
                               *map(str, arguments)], cwd=cwd, capture_output=True,
                              text=True, timeout=60)

    def test_cli_requires_directory_and_explains_removed_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            for arguments, message in (
                ([], 'requires --config-dir'),
                (['--profile', 'current'], 'were removed'),
                (['--compare-profiles'], '--compare-config-dirs'),
                (['--legacy-initialization'], 'replay_initialization.json'),
                (['--config-dir', CONFIGS/'current', '--compare-config-dirs',
                  CONFIGS/'baseline', CONFIGS/'fitted'], 'not allowed with argument'),
            ):
                with self.subTest(arguments=arguments):
                    result = self.invoke(*arguments, '--output', output)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(output.exists())

    def test_incomplete_second_directory_fails_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root/'incomplete'
            incomplete.mkdir()
            result = self.invoke('--compare-config-dirs', CONFIGS/'current', incomplete,
                                 '--no-build', '--output', root/'run')
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('missing required files', result.stderr)
            self.assertFalse((root/'run').exists())

    def test_view_run_does_not_require_a_source_configuration(self):
        with patch.object(sys, 'argv', ['run_measured_dynamics.py', '--view-run', 'old_run']), \
                patch.dict(os.environ, {'DISPLAY': ':test'}), \
                patch.object(replay, 'view_result') as view:
            replay.main()
        view.assert_called_once_with(Path('old_run'), 1, .25)

    @unittest.skipUnless(replay.BINARY.exists(), 'build x4_command_replay first')
    def test_comparison_same_basename_independent_init_and_portable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Names have no semantics, and identical names must not collide in output.
            first, second = root/'one/custom', root/'two/custom'
            shutil.copytree(CONFIGS/'current', first)
            shutil.copytree(CONFIGS/'current', second)
            robot_path = first/'robot.json'
            robot = json.loads(robot_path.read_text())
            robot['motor_config'] = './motors.json'
            robot_path.write_text(json.dumps(robot))
            settings_path = second/'replay_initialization.json'
            settings = json.loads(settings_path.read_text())
            settings['enabled'] = False
            settings_path.write_text(json.dumps(settings))
            source = test_measured_dynamics.DynamicsReplayTests().write_log(root)
            output = root/'run'
            result = self.invoke('--compare-config-dirs', 'one/custom', 'two/custom',
                                 '--trajectory-file', source, '--no-build', '--output', output,
                                 cwd=root)
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            comparison = json.loads((output/'comparison.json').read_text())
            self.assertTrue(comparison['same_raw_experiment_inputs'])
            self.assertFalse(comparison['same_experiment_inputs'])
            self.assertEqual(comparison['config_dirs'],
                             {'config_1': str(first), 'config_2': str(second)})
            self.assertIn('初始化策略', (output/'comparison.md').read_text())
            for index, status in ((1, 'not_cached'), (2, 'disabled')):
                manifest = json.loads((output/f'config_{index}/manifest.json').read_text())
                self.assertEqual(manifest['initialization_by_group']['1']['status'], status)
                self.assertIn('replay_initialization.json', manifest['snapshot_sha256'])
                self.assertTrue(any(key.startswith('meshes/') for key in manifest['snapshot_sha256']))
                summary = json.loads((output/f'config_{index}/group_1/summary.json').read_text())
                self.assertTrue(summary['finite'])
                self.assertFalse(any(summary['warnings'].values()))
            # Output models must remain usable after both source directories disappear.
            shutil.rmtree(root/'one')
            shutil.rmtree(root/'two')
            moved = root/'moved_run'
            output.rename(moved)
            for index in (1, 2):
                model = mujoco.MjModel.from_xml_path(str(replay_scene(moved/f'config_{index}')))
                self.assertEqual(model.nq, 24)


if __name__ == '__main__':
    unittest.main()
