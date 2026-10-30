import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))

import measured_dynamics_profiles as profiles
from run_measured_dynamics import BINARY, prepare


class MeasuredDynamicsProfileTests(unittest.TestCase):
    def write_log(self, directory):
        header = ['local_timestamp', 'robotcurrent_x', 'robotcurrent_y',
                  'robotcurrent_angle', '#']
        header += [f'motor{j}_current_pos' for j in range(1, 6)]
        header += [f'{prefix}_{axis}' for prefix in ('vicon_racket', 'racket_pos')
                   for axis in 'xyz']
        header += ['robotcurrent_vx', 'robotcurrent_vy', 'robotcurrent_vw']
        header += [f'motor{j}_{field}' for field in
                   ('current_speed', 'pos_cmd', 'speed_cmd') for j in range(1, 6)]
        header += ['robot_vx_cmd', 'robot_vy_cmd', 'robot_vw_cmd']
        lines = ['\t'.join(header)]
        for i, t in enumerate([0, 1.5, 3.2, 5.5]):
            values = [1792475766000+t, 300, 20, 3, '#', 1, 18, -20, -90, 0,
                      3.2, .2, 1.3, 3.2, .2, 1.3, 100, -50, .01,
                      .1, -.2, .3, -.4, .5, 1+i, 18, -20, -90, 0,
                      .2, .3, .4, .5, .6, 120+i, -60, .01, 0]
            lines.append('\t'.join(map(str, values)))
        path = Path(directory)/'test.csv'
        path.write_text('\n'.join(lines)+'\n')
        return path

    def test_named_sources_have_distinct_arm_and_torque_parameters(self):
        source = profiles.PROJECT/'models/config/measured_dynamics'
        baseline_robot = json.loads((source/'baseline/robot.json').read_text())
        fitted_robot = json.loads((source/'fitted/robot.json').read_text())
        baseline_motors = json.loads((source/'baseline/motors.json').read_text())
        fitted_motors = json.loads((source/'fitted/motors.json').read_text())

        self.assertNotEqual(baseline_robot['arm_servo']['kp'],
                            fitted_robot['arm_servo']['kp'])
        self.assertEqual(baseline_motors['motors']['RS04']['continuous_torque_nm'], 40)
        self.assertEqual(fitted_motors['motors']['RS04']['continuous_torque_nm'], 120)

    def test_configuration_directories_are_self_contained_and_relocatable(self):
        import mujoco

        source_root = profiles.PROJECT/'models/config/measured_dynamics'
        for name in ('current', 'baseline', 'fitted'):
            source = profiles.validate_config_dir(source_root/name)
            self.assertEqual(source, (source_root/name).resolve())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_root/'current'
            paths = profiles.snapshot_config_dir(source, root/'snapshot')
            self.assertEqual(set(paths), set(profiles.PROFILE_FILES))
            self.assertTrue((root/'snapshot/replay_initialization.json').is_file())
            self.assertTrue((root/'snapshot/provenance.json').is_file())
            self.assertTrue((root/'snapshot/meshes/Link1.STL').is_file())
            self.assertFalse(any(path.is_symlink() for path in (root/'snapshot').rglob('*')))
            self.assertFalse(os.path.samefile(source/'meshes/Link1.STL',
                                              root/'snapshot/meshes/Link1.STL'))
            self.assertEqual(json.loads(Path(paths['robot_variant_path']).read_text())['motor_config'],
                             'motors.json')
            self.assertEqual(ET.parse(paths['scene_path']).find('compiler').get('meshdir'), '.')

            relocated = root/'renamed_configuration'
            (root/'snapshot').rename(relocated)
            profiles.validate_config_dir(relocated)
            mujoco.MjModel.from_xml_path(str(relocated/'scene.mjcf'))

    def test_invalid_configuration_fails_before_snapshot_creation(self):
        source = profiles.PROJECT/'models/config/measured_dynamics/current'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root/'incomplete'
            incomplete.mkdir()
            for name in profiles.CONFIG_DIRECTORY_FILES:
                if name != 'chassis.json':
                    shutil.copy2(source/name, incomplete/name)
            (incomplete/'meshes').mkdir()
            destination = root/'snapshot'
            with self.assertRaisesRegex(FileNotFoundError, 'chassis.json'):
                profiles.snapshot_config_dir(incomplete, destination)
            self.assertFalse(destination.exists())

            escaping = root/'escaping'
            shutil.copytree(source, escaping)
            nested = escaping/'nested_snapshot'
            with self.assertRaisesRegex(ValueError, 'outside the source'):
                profiles.snapshot_config_dir(escaping, nested)
            self.assertFalse(nested.exists())

            original_scene = (escaping/'scene.mjcf').read_text()
            escaping_scene = original_scene.replace('meshdir="."', 'meshdir="../outside"')
            (escaping/'scene.mjcf').write_text(escaping_scene)
            with self.assertRaisesRegex(ValueError, r'without \.\.'):
                profiles.validate_config_dir(escaping)

            (escaping/'scene.mjcf').write_text('<mujoco>')
            with self.assertRaisesRegex(ValueError, 'not valid XML'):
                profiles.validate_config_dir(escaping)
            (escaping/'scene.mjcf').write_text(original_scene)

            initialization_path = escaping/'replay_initialization.json'
            original_initialization = initialization_path.read_text()
            initialization = json.loads(original_initialization)
            initialization['schema_version'] = 2
            initialization_path.write_text(json.dumps(initialization))
            with self.assertRaisesRegex(ValueError, 'schema_version'):
                profiles.validate_config_dir(escaping)
            initialization_path.write_text(original_initialization)

            robot_path = escaping/'robot.json'
            original_robot = robot_path.read_text()
            robot_path.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'must contain an object: robot.json'):
                profiles.validate_config_dir(escaping)
            robot_path.write_text(original_robot)

            missing_asset = root/'missing_asset'
            shutil.copytree(source, missing_asset)
            (missing_asset/'meshes/Link1.STL').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Link1.STL'):
                profiles.validate_config_dir(missing_asset)

    def test_snapshot_is_isolated_loadable_and_does_not_edit_current_config(self):
        current = [profiles.PROJECT/path for path in profiles.CURRENT_CONFIGS.values()]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in current}
        import mujoco

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'baseline'
            resolved = profiles.snapshot_profile('baseline', destination)
            self.assertEqual(set(resolved), set(profiles.CURRENT_CONFIGS))
            self.assertTrue(all(Path(path).is_absolute() for path in resolved.values()))
            with self.assertRaises(FileExistsError):
                profiles.snapshot_profile('baseline', destination)

            robot = json.loads(Path(resolved['robot_variant_path']).read_text())
            self.assertEqual(robot['motor_config'], resolved['motors_path'])
            compiler = ET.parse(resolved['scene_path']).find('compiler')
            self.assertTrue(Path(compiler.get('meshdir')).is_absolute())
            mujoco.MjModel.from_xml_path(resolved['scene_path'])

        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in current}
        self.assertEqual(before, after)

    def test_command_identity_ignores_only_configuration_locations(self):
        payload = {'schema': 'X4_COMMAND_REPLAY/1', 'project_root': '/one',
                   **{key: '/one/'+key for key in profiles.CURRENT_CONFIGS},
                   'initial': {'q_rad': [0]*5},
                   'commands': [{'t_s': 0, 'q_rad': [0]*5}]}
        relocated = copy.deepcopy(payload)
        relocated['project_root'] = '/two'
        for key in profiles.CURRENT_CONFIGS:
            relocated[key] = '/two/'+key
        self.assertEqual(profiles.command_identity(payload),
                         profiles.command_identity(relocated))

        changed_initial = copy.deepcopy(payload)
        changed_initial['initial']['q_rad'][0] = .1
        changed_command = copy.deepcopy(payload)
        changed_command['commands'][0]['q_rad'][0] = .1
        self.assertNotEqual(profiles.command_identity(payload),
                            profiles.command_identity(changed_initial))
        self.assertNotEqual(profiles.command_identity(payload),
                            profiles.command_identity(changed_command))

    def test_replay_scene_prefers_recorded_snapshot_and_rejects_missing_one(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            recorded = output/'profile/scene.mjcf'
            recorded.parent.mkdir()
            recorded.write_text('<mujoco/>')
            (output/'manifest.json').write_text(json.dumps({
                'config_paths': {'scene_path': 'profile/scene.mjcf'}}))

            original = profiles.CURRENT_CONFIGS['scene_path']
            profiles.CURRENT_CONFIGS['scene_path'] = 'models/generated/not-current.mjcf'
            try:
                self.assertEqual(profiles.replay_scene(output), recorded)
            finally:
                profiles.CURRENT_CONFIGS['scene_path'] = original

            recorded.unlink()
            with self.assertRaises(FileNotFoundError):
                profiles.replay_scene(output)

            (output/'manifest.json').write_text('{}')
            self.assertEqual(profiles.replay_scene(output),
                             profiles.PROJECT/profiles.CURRENT_CONFIGS['scene_path'])

    @unittest.skipUnless(BINARY.exists(), 'build x4_command_replay before native integration tests')
    def test_native_profiles_use_identical_experiment_input_but_diverge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = prepare(self.write_log(root))[3]
            snapshots = {name: profiles.snapshot_profile(name, root/name)
                         for name in ('baseline', 'fitted')}
            resolved_payloads = {}
            frames = {}
            for name, paths in snapshots.items():
                candidate = copy.deepcopy(payload)
                native_paths = {key: Path(path).name for key, path in paths.items()}
                candidate.update(project_root=str(root/name), **native_paths)
                resolved_payloads[name] = candidate
                source = root/f'{name}.json'
                source.write_text(json.dumps(candidate))
                output = root/f'{name}_run'
                result = subprocess.run([str(BINARY), str(source), str(output)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                frames[name] = [json.loads(line) for line in
                                (output/'frames.jsonl').read_text().splitlines()]
                summary = json.loads((output/'summary.json').read_text())
                self.assertTrue(summary['finite'])
                self.assertEqual(sum(summary['warnings'].values()), 0)

            self.assertEqual(profiles.command_identity(resolved_payloads['baseline']),
                             profiles.command_identity(resolved_payloads['fitted']))
            np.testing.assert_array_equal(frames['baseline'][0]['qpos'],
                                          frames['fitted'][0]['qpos'])
            np.testing.assert_array_equal(frames['baseline'][0]['qvel'],
                                          frames['fitted'][0]['qvel'])
            self.assertGreater(np.max(np.abs(
                np.asarray(frames['baseline'][-1]['qvel'])-
                np.asarray(frames['fitted'][-1]['qvel']))), 1e-9)


if __name__ == '__main__':
    unittest.main()
