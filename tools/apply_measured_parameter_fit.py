#!/usr/bin/env python3
"""Apply an offline fitted parameter profile; never edits controller code.

Reapply after regenerating MJCF to restore fitted armature/damping/friction.
Baseline originals are retained in outputs/parameter_fit/baseline_snapshot.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT/'sckd_sim/models'


def apply(parameters_path, dry_run=False):
    raw = json.loads(parameters_path.read_text())
    params = raw.get('parameters', raw.get('best_parameters', raw))
    variant = json.loads((MODELS/'config/x4_950.json').read_text())
    motors = json.loads((MODELS/'config/motors_x4_950.json').read_text())
    chassis = json.loads((MODELS/'config/chassis_x4_950.json').read_text())
    allowed = {'kp','kd','armature','damping','frictionloss','motor_parameters',
               'chassis_ramp','chassis_servo'}
    if set(params) - allowed:
        raise ValueError(f'Unsupported parameter fields: {set(params)-allowed}')
    if motors['torque_profile'] != 'continuous':
        raise ValueError('This fitted profile requires the continuous torque configuration')
    for key in ('kp','kd','armature','damping','frictionloss'):
        if key in params:
            if len(params[key]) != 5 or any(not math.isfinite(x) or x < 0 for x in params[key]):
                raise ValueError(f'{key} requires five finite nonnegative parameters')
    for key in ('kp','kd'):
        if key in params:
            for j, value in enumerate(params[key]):
                name = next(x['motor'] for x in variant['joints'] if x['joint']==f'Joint{j+1}')
                lo, hi = motors['motors'][name][key+'_range']
                if not lo <= value <= hi:
                    raise ValueError(f'{key}[{j}] exceeds the motor protocol range')
            variant['arm_servo'][key] = params[key]
    for name, fields in params.get('motor_parameters', {}).items():
        if set(fields) != {'continuous_torque_nm'}:
            raise ValueError('only effective torque saturation may change in motor parameters')
        value = fields['continuous_torque_nm']
        motor = motors['motors'][name]
        if not math.isfinite(value) or not 0 < value <= motor['peak_torque_nm']:
            raise ValueError('effective torque must be positive and no greater than nominal peak')
        motor.setdefault('nominal_continuous_torque_nm', motor['continuous_torque_nm'])
        motor.update(fields)
        motor['fit_note'] = 'Effective saturation for these <0.6 s replays, not a revised continuous hardware rating.'
    for key, fields in {
        'ramp': {'accel_limit_mps2','ang_accel_limit_rad_s2','dir_rate_limit_rad_s'},
        'servo': {'steer_kp','steer_kd','wheel_kp','wheel_ki','wheel_int_max_rad','wheel_accel_mps2'}
    }.items():
        updates = params.get('chassis_'+key, {})
        if set(updates)-fields:
            raise ValueError(f'Unsupported chassis {key} fields')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0
               for v in updates.values()):
            raise ValueError('Chassis parameters must be finite nonnegative numbers')
        chassis[key].update(updates)
    metadata = {'source': str(parameters_path.resolve()),
                'parameter_sha256': hashlib.sha256(parameters_path.read_bytes()).hexdigest(),
                'scope': 'Five short measured motion logs; simulation fit only; no hardware calibration claim.',
                'training_groups': [1,3,4], 'validation_groups': [2,5],
                'acceptance_thresholds': {'per_group_joint_angle_rmse_deg': 5, 'per_group_joint_velocity_rmse_rad_s': .5},
                'acceptance_status': 'Not certified by installer; inspect the independent replay report.',
                'runtime_logic_changed': False}
    variant['parameter_fit'] = metadata
    variant['arm_servo']['source'] = 'Offline parameter fit; original initialization is retained in baseline_snapshot/x4_950.json.'
    variant['arm_servo']['calibrated'] = False
    variant['fitted_joint_dynamics'] = {k: params[k] for k in ('armature','damping','frictionloss') if k in params}
    motors['parameter_fit'] = metadata
    chassis['parameter_fit'] = metadata
    chassis['ramp']['note'] = 'Fitted replay response limits; not measured hardware maximum acceleration.'
    chassis['servo']['note'] = 'Fitted against base motion; individual wheel/steer motor feedback is unavailable.'
    pending = {}
    for name, config in [('x4_950',variant),('motors_x4_950',motors),('chassis_x4_950',chassis)]:
        pending[MODELS/f'config/{name}.json'] = json.dumps(config,indent=2,ensure_ascii=False)+'\n'
    for name in ('x4_950_scene','x4_950_replay'):
        # Preserve current geometry/court; only update robot dynamics and ctrl limits.
        path = MODELS/f'generated/{name}.mjcf'
        tree = ET.parse(path)
        for field in ('armature','damping','frictionloss'):
            if field in params:
                for j,value in enumerate(params[field],1):
                    tree.find(f".//joint[@name='Joint{j}']").set(field,str(value))
        for actuator in tree.findall('.//actuator/motor'):
            joint = next(x for x in variant['joints'] if x['joint']==actuator.get('joint'))
            motor = motors['motors'][joint['motor']]
            cap = min(joint['mechanism_limit_nm'],motor['continuous_torque_nm'],motor['braking_torque_nm'])
            actuator.set('ctrlrange',f'{-cap} {cap}')
        ET.indent(tree,space='  ')
        text = "<?xml version='1.0' encoding='utf-8'?>\n"+ET.tostring(tree.getroot(),encoding='unicode')+'\n'
        pending[path] = text
        pending[path.with_suffix('.xml')] = text
        manifest_path = MODELS/f'generated/{name.replace("_scene", "")}_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['parameter_fit'] = {**metadata, 'applied_model_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                     'joint_dynamics': variant['fitted_joint_dynamics']}
        for item in manifest.get('joint_map', []):
            actuator = tree.find(f".//actuator/motor[@joint='{item['joint']}']")
            item['ctrlrange_nm'] = [float(x) for x in actuator.get('ctrlrange').split()]
        if 'generated_from' in manifest:
            manifest['generated_from']['motors_sha256'] = hashlib.sha256(pending[MODELS/'config/motors_x4_950.json'].encode()).hexdigest()
            manifest['generated_from']['config_sha256'] = hashlib.sha256(pending[MODELS/'config/x4_950.json'].encode()).hexdigest()
            manifest['edits'] = [x for x in manifest['edits'] if x.get('edit') != 'apply_measured_parameter_fit']
            manifest['edits'].append({'edit': 'apply_measured_parameter_fit', 'reason': metadata['scope']})
        pending[manifest_path] = json.dumps(manifest,indent=2,ensure_ascii=False)+'\n'
    audit = {str(path): {'before_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'after_sha256': hashlib.sha256(value.encode()).hexdigest()}
             for path,value in pending.items()}
    if not dry_run:
        for path,value in pending.items():
            path.write_text(value)
    return {**metadata, 'dry_run': dry_run, 'files': audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parameters',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    print(json.dumps(apply(args.parameters,args.dry_run),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
