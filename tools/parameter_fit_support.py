"""Offline parameter-search support; invokes the unchanged native replay binary.

Candidates live in isolated temporary projects. No production parameter is written.
Requires the system Python NumPy/SciPy environment for the optimization drivers.
"""
import copy
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'outputs/measured_dynamics/baseline'
BINARY = ROOT / 'sckd_sim/build/x4_command_replay'


class Evaluator:
    def __init__(self, work, groups=(1, 3, 4), *, baseline_dir=None, scene_path=None):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.groups = tuple(groups)
        baseline = Path(baseline_dir) if baseline_dir is not None else BASELINE
        self.scene_path = Path(scene_path) if scene_path is not None else None
        self.manifest = json.loads((baseline/'manifest.json').read_text())
        self.refs = {}
        self.inputs = {}
        for g in groups:
            a = np.genfromtxt(baseline/f'group_{g}/comparison.csv', delimiter=',', names=True)
            self.refs[g] = {key: a[key] for key in a.dtype.names}
            self.inputs[g] = json.loads((baseline/f'group_{g}_input.json').read_text())
        self.count = 0

    def make_project(self, path, parameters):
        """Only data parameters: gains, limits, joint damping/friction/armature."""
        variant = copy.deepcopy(self.manifest['config_contents']['robot_variant_path'])
        variant['motor_config'] = str(path/'motors.json')
        motors = copy.deepcopy(self.manifest['config_contents']['motors_path'])
        if motors['torque_profile'] != 'continuous':
            raise ValueError('Offline fitter requires the continuous torque configuration')
        chassis = copy.deepcopy(self.manifest['config_contents']['chassis_path'])
        if 'kp' in parameters:
            variant['arm_servo']['kp'] = list(parameters['kp'])
        if 'kd' in parameters:
            variant['arm_servo']['kd'] = list(parameters['kd'])
        for key in ('ramp', 'servo'):
            chassis[key].update(parameters.get('chassis_'+key, {}))
        for name, values in parameters.get('motor_parameters', {}).items():
            motors['motors'][name].update(values)
        snapshot = ROOT/'outputs/parameter_fit/baseline_snapshot/x4_950_scene.mjcf'
        tree = ET.parse(self.scene_path if self.scene_path is not None else
                        snapshot if snapshot.exists() else ROOT/'sckd_sim/models/generated/x4_950_scene.mjcf')
        compiler = tree.getroot().find('compiler')
        compiler.set('meshdir', str(ROOT/'sckd_sim/models/source/x4_950'))
        for field in ('armature', 'damping', 'frictionloss'):
            if field in parameters:
                for j, value in enumerate(parameters[field], 1):
                    tree.find(f".//joint[@name='Joint{j}']").set(field, str(value))
        for actuator in tree.findall('.//actuator/motor'):
            name = actuator.get('joint')
            joint = next(j for j in variant['joints'] if j['joint'] == name)
            motor = motors['motors'][joint['motor']]
            cap = min(joint['mechanism_limit_nm'], motor['continuous_torque_nm'], motor['braking_torque_nm'])
            actuator.set('ctrlrange', f'{-cap} {cap}')
        tree.write(path/'scene.mjcf')
        for name, config in [('variant', variant), ('motors', motors), ('chassis', chassis)]:
            (path/f'{name}.json').write_text(json.dumps(config))

    def evaluate(self, parameters, keep=None):
        self.count += 1
        with tempfile.TemporaryDirectory(prefix='candidate-', dir=self.work) as temp:
            path = Path(temp)
            self.make_project(path, parameters)
            def run(g):
                payload = copy.deepcopy(self.inputs[g])
                payload.update(project_root=str(path), scene_path='scene.mjcf', robot_variant_path='variant.json',
                               motors_path='motors.json', chassis_path='chassis.json')
                source = path/f'input_{g}.json'
                source.write_text(json.dumps(payload))
                result = subprocess.run([str(BINARY), str(source), str(path/f'group_{g}')],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
                if result.returncode:
                    raise ValueError(f'group {g} unstable/invalid: {result.stderr.decode()[-400:]}')
                frames = [json.loads(line) for line in (path/f'group_{g}/frames.jsonl').read_text().splitlines()]
                return g, frames
            with ThreadPoolExecutor(max_workers=len(self.groups)) as pool:
                outputs = dict(pool.map(run, self.groups))
            if keep is not None:
                destination = Path(keep)
                destination.mkdir(parents=True, exist_ok=True)
                for child in path.iterdir():
                    if child.is_dir():
                        shutil.copytree(child, destination/child.name, dirs_exist_ok=True)
                    else:
                        shutil.copy2(child, destination/child.name)
                (destination/'parameters.json').write_text(json.dumps(parameters, indent=2)+'\n')
                # Make copied candidate inputs executable after the temp project is removed.
                variant_file = destination/'variant.json'
                variant = json.loads(variant_file.read_text())
                variant['motor_config'] = str((destination/'motors.json').resolve())
                variant_file.write_text(json.dumps(variant, indent=2)+'\n')
                for source in [*destination.glob('input_*.json'), *destination.glob('group_*/resolved_input.json')]:
                    payload = json.loads(source.read_text())
                    payload['project_root'] = str(destination.resolve())
                    source.write_text(json.dumps(payload, indent=2)+'\n')
            return outputs

    def residual(self, outputs, kind='arm', stride=5):
        pieces = []
        for g in self.groups:
            f = outputs[g]
            a = self.refs[g]
            if kind == 'arm':
                q = np.array([x['q'] for x in f])
                dq = np.array([x['dq'] for x in f])
                real_q = np.column_stack([a[f'real_q_j{j}_rad'] for j in range(1,6)])
                real_dq = np.column_stack([a[f'real_dq_j{j}_rad_s'] for j in range(1,6)])
                pieces += [((q-real_q)[::stride]/np.deg2rad(5)).ravel(), ((dq-real_dq)[::stride]/.5).ravel()]
            else:
                p = np.array([x['base'][:2] for x in f])
                v = np.array([x['base_world_velocity'][:2] for x in f])
                real_p = np.column_stack([a[f'real_base_{x}_m'] for x in 'xy'])
                real_v = np.column_stack([a[f'real_base_v{x}_m_s'] for x in 'xy'])
                pieces += [((p-real_p)[::stride]/.05).ravel(), ((v-real_v)[::stride]/.3).ravel()]
        return np.concatenate(pieces)

    def metrics(self, outputs):
        results = {}
        for g, frames in outputs.items():
            a = self.refs[g]
            q = np.array([x['q'] for x in frames])
            dq = np.array([x['dq'] for x in frames])
            qp = np.column_stack([a[f'real_q_j{j}_rad'] for j in range(1,6)])
            dqp = np.column_stack([a[f'real_dq_j{j}_rad_s'] for j in range(1,6)])
            p = np.array([x['base'][:2] for x in frames])
            v = np.array([x['base_world_velocity'][:2] for x in frames])
            pr = np.column_stack([a[f'real_base_{x}_m'] for x in 'xy'])
            vr = np.column_stack([a[f'real_base_v{x}_m_s'] for x in 'xy'])
            results[g] = {'q_rmse_deg': np.rad2deg(np.sqrt(np.mean((q-qp)**2,axis=0))).tolist(),
                          'dq_rmse': np.sqrt(np.mean((dq-dqp)**2,axis=0)).tolist(),
                          'base_rmse_m': float(np.sqrt(np.mean(np.sum((p-pr)**2,axis=1)))),
                          'base_velocity_rmse': float(np.sqrt(np.mean(np.sum((v-vr)**2,axis=1))))}
        return results
