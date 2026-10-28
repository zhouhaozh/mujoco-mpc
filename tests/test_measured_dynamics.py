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
from run_measured_dynamics import BINARY, CONFIGS, PROJECT, group_directories, prepare, world_velocity


class DynamicsReplayTests(unittest.TestCase):
    def write_log(self, directory, mutate_feedback=False):
        header = ['local_timestamp', 'robotcurrent_x', 'robotcurrent_y', 'robotcurrent_angle', '#']
        header += [f'motor{j}_current_pos' for j in range(1,6)]
        header += [f'{prefix}_{axis}' for prefix in ('vicon_racket','racket_pos') for axis in 'xyz']
        header += ['robotcurrent_vx','robotcurrent_vy','robotcurrent_vw']
        header += [f'motor{j}_{field}' for field in ('current_speed','pos_cmd','speed_cmd') for j in range(1,6)]
        header += ['robot_vx_cmd','robot_vy_cmd','robot_vw_cmd']
        lines = ['\t'.join(header)]
        for i, t in enumerate([0, 1.5, 3.2, 5.5]):
            values = [1792475766000+t, 300, 20, 3, '#', 1, 18, -20, -90, 0,
                      3.2, .2, 1.3, 3.2, .2, 1.3, 100, -50, .01,
                      .1, -.2, .3, -.4, .5, 1+i,18,-20,-90,0,
                      .2,.3,.4,.5,.6, 120+i, -60, .01, 0]
            if mutate_feedback and i:
                for name in ['robotcurrent_x','motor1_current_pos','motor3_current_speed','vicon_racket_z']:
                    values[header.index(name)] += 5
            lines.append('\t'.join(map(str,values)))
        path = Path(directory)/'test.csv'
        path.write_text('\n'.join(lines)+'\n')
        return path

    def test_unit_conversion_and_initial_velocity(self):
        with tempfile.TemporaryDirectory() as directory:
            record, dq, base_v, payload = prepare(self.write_log(directory))
        np.testing.assert_allclose(payload['initial']['base_world_velocity'], [.1,-.05,.01])
        np.testing.assert_allclose(payload['initial']['dq_rad_s'], [-.1,.2,.3,-.4,.5])
        np.testing.assert_allclose(payload['commands'][0]['q_rad'], np.deg2rad([-1,-18,-20,-90,0]))
        np.testing.assert_allclose(payload['commands'][0]['dq_rad_s'], [-.2,-.3,.4,.5,.6])
        np.testing.assert_allclose(payload['commands'][0]['base_twist'], [.12,-.06,.01])

    def test_later_feedback_not_in_native_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            before = prepare(self.write_log(directory))[3]
            after = prepare(self.write_log(directory, mutate_feedback=True))[3]
        self.assertEqual(before, after)
        self.assertNotIn('measurements', before)

    def test_body_velocity_transformed_once(self):
        result = world_velocity(np.array([[1.,0.,.2]]), np.array([np.pi/2]), 'body')
        np.testing.assert_allclose(result, [[0,1,.2]], atol=1e-15)

    def test_viewer_ignores_input_json_and_run_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name in ('group_1','group_2'):
                (p/name).mkdir()
            (p/'group_1_input.json').write_text('{}')
            (p/'group_1_run.log').write_text('')
            self.assertEqual([d.name for d in group_directories(p)], ['group_1','group_2'])

    @unittest.skipUnless(BINARY.exists(), 'build x4_command_replay before native integration tests')
    def test_native_initial_state_zoh_no_extrapolation_and_fresh_fk(self):
        import mujoco
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            payload = prepare(self.write_log(path))[3]
            source = path/'input.json'
            source.write_text(json.dumps(payload))
            run = path/'run'
            subprocess.run([str(BINARY),str(source),str(run)], check=True, capture_output=True)
            frames = [json.loads(s) for s in (run/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([f['command_index'] for f in frames], [0,0,1,1,2,2])
            np.testing.assert_allclose(frames[0]['q'], payload['initial']['q_rad'], atol=1e-14)
            np.testing.assert_allclose(frames[0]['dq'], payload['initial']['dq_rad_s'], atol=1e-14)
            np.testing.assert_allclose(frames[0]['base_world_velocity'], payload['initial']['base_world_velocity'], atol=1e-14)
            self.assertAlmostEqual(frames[-1]['t'], .005)
            self.assertIsNone(frames[0]['torque_interval_start_s'])
            self.assertEqual(frames[2]['torque_command_index'], 0)
            self.assertEqual(frames[2]['command_index'], 1)
            self.assertGreater(np.max(np.abs(np.array(frames[-1]['q'])-frames[0]['q'])), 1e-5)
            model = mujoco.MjModel.from_xml_path(str(PROJECT/CONFIGS['scene_path']))
            data = mujoco.MjData(model)
            data.qpos[:], data.qvel[:] = frames[-1]['qpos'], frames[-1]['qvel']
            mujoco.mj_forward(model,data)
            np.testing.assert_allclose(frames[-1]['racket_xyz'], data.geom_xpos[model.geom('racket_face').id], atol=1e-10)
            summary = json.loads((run/'summary.json').read_text())
            self.assertFalse(summary['future_measured_state_copied'])
            self.assertEqual(sum(summary['warnings'].values()), 0)
            # A command changed at 3.2ms cannot affect states before its consumption at 4ms.
            future = copy.deepcopy(payload)
            future['commands'][2]['q_rad'][0] += .1
            source.write_text(json.dumps(future))
            subprocess.run([str(BINARY),str(source),str(path/'future')], check=True, capture_output=True)
            changed = [json.loads(s) for s in (path/'future/frames.jsonl').read_text().splitlines()]
            for a,b in zip(frames[:5],changed[:5]):
                np.testing.assert_array_equal(a['qpos'],b['qpos'])
                np.testing.assert_array_equal(a['qvel'],b['qvel'])
            self.assertGreater(np.max(np.abs(np.array(frames[-1]['qvel'])-changed[-1]['qvel'])), 1e-9)
            # Existing output is rejected without replacing previous evidence.
            result = subprocess.run([str(BINARY),str(source),str(run)], capture_output=True)
            self.assertNotEqual(result.returncode,0)


if __name__ == '__main__':
    unittest.main()
