import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_x4_native_replay import prepare_run_config

PROJECT = ROOT / 'sckd_sim'
CONFIG = PROJECT / 'models/config/runtime_x4_950_replay.json'


CONFIG_DIRS = PROJECT / 'models/config/measured_dynamics'


class X4NativeConfigDirectoryTests(unittest.TestCase):
    def test_all_presets_snapshot_matching_mechanics_without_global_writes(self):
        sources = [p for directory in ('models/config', 'models/generated')
                   for p in (PROJECT / directory).rglob('*') if p.is_file()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            models = {}
            for name in ('baseline', 'fitted', 'current'):
                with self.subTest(config_dir=name):
                    output = Path(directory) / name
                    output.mkdir()
                    source = CONFIG_DIRS / name
                    runtime, cfg = prepare_run_config(CONFIG, output, source)
                    paths.append(cfg['scene_path'])
                    self.assertEqual(cfg['parameter_config_dir'], str(source.resolve()))
                    self.assertEqual(json.loads(runtime.read_text()), cfg)
                    self.assertEqual(Path(cfg['project_root']), PROJECT.resolve())
                    for key in ('scene_path', 'motors_path', 'robot_variant_path', 'chassis_path'):
                        self.assertTrue(Path(cfg[key]).is_relative_to(output))
                        self.assertTrue(Path(cfg[key]).is_file())
                    robot = json.loads(Path(cfg['robot_variant_path']).read_text())
                    self.assertEqual(robot['motor_config'], 'motors.json')
                    expected = json.loads((source / 'robot.json').read_text())
                    self.assertEqual(robot, expected)
                    model = mujoco.MjModel.from_xml_path(cfg['scene_path'])
                    models[name] = model
                    self.assertEqual((model.nq, model.nv, model.nu), (31, 29, 17))
                    self.assertEqual(model.geom_rgba[model.geom('racket_face').id, 3], 0)
                    manifest = json.loads((output / 'profile_manifest.json').read_text())
                    self.assertEqual(manifest['parameter_config_dir'], str(source.resolve()))
                    for filename, digest in manifest['snapshot_sha256'].items():
                        self.assertEqual(hashlib.sha256((output / 'config' / filename).read_bytes()).hexdigest(), digest)
            self.assertEqual(len(set(paths)), 3)
            baseline, fitted = models['baseline'], models['fitted']
            self.assertFalse(np.array_equal(baseline.dof_damping, fitted.dof_damping))
            self.assertFalse(np.array_equal(baseline.actuator_ctrlrange, fitted.actuator_ctrlrange))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    def test_custom_runtime_settings_survive_config_selection_and_relocation(self):
        original = json.loads(CONFIG.read_text())
        original.update(project_root=str(PROJECT), initial_base_xy_m=[2.1, .3],
                        native_mpc_library='third_party/native_mpc_x4_950/libnmpc.so.1',
                        prediction_period_ms=20, settle_s=.7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'custom.json'
            config.write_text(json.dumps(original))
            for name in ('baseline', 'fitted', 'current'):
                output = root / name
                output.mkdir()
                _, cfg = prepare_run_config(config, output, CONFIG_DIRS / name)
                for key in ('initial_base_xy_m', 'prediction_period_ms', 'settle_s', 'mpc_joint_sign'):
                    self.assertEqual(cfg[key], original[key])
                for key in ('native_mpc_library', 'trajectory_path', 'ball_mlp_model_path', 'ball_mlp_normalizer_path'):
                    self.assertEqual(cfg[key], str((PROJECT / original[key]).resolve()))
            self.assertEqual(json.loads(config.read_text()), original)

    def test_arbitrary_relocated_directory_is_selected_and_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relocated = root / 'arbitrary_robot_parameters'
            shutil.copytree(CONFIG_DIRS / 'baseline', relocated)
            first = root / 'first'
            first.mkdir()
            _, cfg = prepare_run_config(CONFIG, first, relocated)
            source_robot = (relocated / 'robot.json').read_bytes()
            source_motors = (relocated / 'motors.json').read_bytes()
            replay_meshdir = ET.parse(cfg['scene_path']).find('compiler').get('meshdir')
            self.assertFalse(Path(replay_meshdir).is_absolute())

            second = root / 'second'
            second.mkdir()
            _, copy = prepare_run_config(CONFIG, second, relocated)
            a = mujoco.MjModel.from_xml_path(cfg['scene_path'])
            b = mujoco.MjModel.from_xml_path(copy['scene_path'])
            self.assertEqual((a.nq, a.nv, a.nu), (b.nq, b.nv, b.nu))
            np.testing.assert_array_equal(a.dof_damping, b.dof_damping)
            np.testing.assert_array_equal(a.actuator_ctrlrange, b.actuator_ctrlrange)
            Path(copy['motors_path']).write_text('{}')
            self.assertEqual((relocated / 'robot.json').read_bytes(), source_robot)
            self.assertEqual((relocated / 'motors.json').read_bytes(), source_motors)
            self.assertEqual(Path(cfg['motors_path']).read_bytes(), source_motors)

            shutil.rmtree(relocated)
            portable = root / 'portable_replay'
            shutil.move(first, portable)
            moved_scene = portable / 'config/replay.mjcf'
            moved = mujoco.MjModel.from_xml_path(str(moved_scene))
            self.assertEqual((moved.nq, moved.nv, moved.nu), (31, 29, 17))
            manifest = json.loads((portable / 'profile_manifest.json').read_text())
            replay_digest = hashlib.sha256(moved_scene.read_bytes()).hexdigest()
            self.assertEqual(manifest['snapshot_sha256']['replay.mjcf'], replay_digest)

    def test_cli_requires_config_directory_and_explains_profile_migration(self):
        script = ROOT / 'tools/run_x4_native_replay.py'
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--config-dir CONFIG_DIR', result.stdout)
        self.assertNotIn('--profile', result.stdout)

        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--config-dir is required', result.stderr)

        result = subprocess.run([sys.executable, str(script), '--profile', 'current'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--profile has been replaced by --config-dir', result.stderr)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'config'
            shutil.copytree(CONFIG_DIRS / 'current', source)
            nested_output = source / 'nested_output'
            result = subprocess.run(
                [sys.executable, str(script), '--config-dir', str(source),
                 '--output', str(nested_output), '--no-build'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('--output cannot be inside --config-dir', result.stderr)
            self.assertFalse(nested_output.exists())

            missing_output = root / 'missing_output'
            result = subprocess.run(
                [sys.executable, str(script), '--config-dir', str(root / 'missing'),
                 '--output', str(missing_output), '--no-build'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Configuration directory does not exist', result.stderr)
            self.assertNotIn('Traceback', result.stderr)
            self.assertFalse(missing_output.exists())


if __name__ == '__main__':
    unittest.main()
