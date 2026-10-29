import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_x4_native_replay import prepare_run_config

PROJECT = ROOT / 'sckd_sim'
CONFIG = PROJECT / 'models/config/runtime_x4_950_replay.json'


class X4NativeProfileTests(unittest.TestCase):
    def test_all_profiles_snapshot_matching_mechanics_without_global_writes(self):
        sources = [p for directory in ('models/config', 'models/generated')
                   for p in (PROJECT / directory).rglob('*') if p.is_file()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            models = {}
            for name in ('baseline', 'fitted', 'current'):
                with self.subTest(profile=name):
                    output = Path(directory) / name
                    output.mkdir()
                    runtime, cfg = prepare_run_config(CONFIG, output, name)
                    paths.append(cfg['scene_path'])
                    self.assertEqual(cfg['parameter_profile'], name)
                    self.assertEqual(json.loads(runtime.read_text()), cfg)
                    self.assertEqual(Path(cfg['project_root']), PROJECT.resolve())
                    for key in ('scene_path', 'motors_path', 'robot_variant_path', 'chassis_path'):
                        self.assertTrue(Path(cfg[key]).is_relative_to(output))
                        self.assertTrue(Path(cfg[key]).is_file())
                    robot = json.loads(Path(cfg['robot_variant_path']).read_text())
                    self.assertEqual(robot['motor_config'], cfg['motors_path'])
                    source_robot = (PROJECT / 'models/config/x4_950.json' if name == 'current'
                                    else PROJECT / f'models/config/measured_dynamics/{name}/robot.json')
                    expected = json.loads(source_robot.read_text())
                    expected['motor_config'] = cfg['motors_path']
                    self.assertEqual(robot, expected)
                    model = mujoco.MjModel.from_xml_path(cfg['scene_path'])
                    models[name] = model
                    self.assertEqual((model.nq, model.nv, model.nu), (31, 29, 17))
                    self.assertEqual(model.geom_rgba[model.geom('racket_face').id, 3], 0)
                    manifest = json.loads((output / 'profile_manifest.json').read_text())
                    self.assertEqual(manifest['parameter_profile'], name)
                    for filename, digest in manifest['snapshot_sha256'].items():
                        self.assertEqual(hashlib.sha256((output / 'config' / filename).read_bytes()).hexdigest(), digest)
            self.assertEqual(len(set(paths)), 3)
            baseline, fitted = models['baseline'], models['fitted']
            self.assertFalse(np.array_equal(baseline.dof_damping, fitted.dof_damping))
            self.assertFalse(np.array_equal(baseline.actuator_ctrlrange, fitted.actuator_ctrlrange))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    def test_custom_runtime_settings_survive_profile_selection_and_relocation(self):
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
                _, cfg = prepare_run_config(config, output, name)
                for key in ('initial_base_xy_m', 'prediction_period_ms', 'settle_s', 'mpc_joint_sign'):
                    self.assertEqual(cfg[key], original[key])
                for key in ('native_mpc_library', 'trajectory_path', 'ball_mlp_model_path', 'ball_mlp_normalizer_path'):
                    self.assertEqual(cfg[key], str((PROJECT / original[key]).resolve()))
            self.assertEqual(json.loads(config.read_text()), original)

    def test_current_honors_custom_snapshot_paths_including_an_existing_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'first'
            first.mkdir()
            runtime, cfg = prepare_run_config(CONFIG, first, 'baseline')
            second = root / 'second'
            second.mkdir()
            _, copy = prepare_run_config(runtime, second, 'current')
            self.assertNotEqual(copy['scene_path'], cfg['scene_path'])
            a = mujoco.MjModel.from_xml_path(cfg['scene_path'])
            b = mujoco.MjModel.from_xml_path(copy['scene_path'])
            self.assertEqual((a.nq, a.nv, a.nu), (b.nq, b.nv, b.nu))
            np.testing.assert_array_equal(a.dof_damping, b.dof_damping)
            np.testing.assert_array_equal(a.actuator_ctrlrange, b.actuator_ctrlrange)
            source_bytes = Path(cfg['motors_path']).read_bytes()
            Path(copy['motors_path']).write_text('{}')
            self.assertEqual(Path(cfg['motors_path']).read_bytes(), source_bytes)

    def test_cli_exposes_profiles_and_rejects_unknown_values(self):
        script = ROOT / 'tools/run_x4_native_replay.py'
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--profile {baseline,fitted,current}', result.stdout)
        self.assertIn('default: current', result.stdout)
        result = subprocess.run([sys.executable, str(script), '--profile', 'unknown'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)


if __name__ == '__main__':
    unittest.main()
