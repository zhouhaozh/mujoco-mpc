#!/usr/bin/env python3
"""Compare measured motion with dynamics driven by the original logged setpoints.

Loads simulation parameters from an explicitly selected, self-contained directory. Only initial measured state and commands are passed to the native process.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import time

import numpy as np

from replay_measured_robot import DEFAULT_DATA, DEFAULT_MAPPING, ROOT, interpolate, read_record
from measured_dynamics_profiles import (CURRENT_CONFIGS, command_identity, replay_scene,
                                        snapshot_config_dir, validate_config_dir)
from measured_replay_initialization import apply_initialization

PROJECT = ROOT / 'sckd_sim'
BINARY = PROJECT / 'build/x4_command_replay'
CONFIGS = CURRENT_CONFIGS


def world_velocity(values, yaw, frame):
    result = values.copy()
    if frame == 'body':
        c, s = np.cos(yaw), np.sin(yaw)
        result[:, 0] = c*values[:, 0] - s*values[:, 1]
        result[:, 1] = s*values[:, 0] + c*values[:, 1]
    return result


def prepare(path, base_command_frame='world', base_feedback_frame='world',
            arm_mode='hybrid', gravity_feedforward=True):
    record = read_record(path, DEFAULT_MAPPING)
    lines = path.read_text().splitlines()
    header = lines[0].split()
    rows = [line.split() for line in lines[1:] if line.strip()]

    def columns(names):
        indices = [header.index(name) for name in names]
        out = np.array([[float(row[i]) for i in indices] for row in rows])
        if not np.isfinite(out).all():
            raise ValueError(f'{path}: nonfinite command/velocity')
        return out

    sign = np.array(DEFAULT_MAPPING['joint_sign'])
    measured_dq = columns([f'motor{j}_current_speed' for j in range(1, 6)]) * sign
    command_q = np.deg2rad(columns([f'motor{j}_pos_cmd' for j in range(1, 6)])) * sign
    command_dq = columns([f'motor{j}_speed_cmd' for j in range(1, 6)]) * sign
    base_v = columns(['robotcurrent_vx', 'robotcurrent_vy', 'robotcurrent_vw']) * [.001, .001, 1]
    base_v = world_velocity(base_v, record.base[:, 2], base_feedback_frame)
    base_command = columns(['robot_vx_cmd', 'robot_vy_cmd', 'robot_vw_cmd']) * [.001, .001, 1]
    payload = {'schema': 'X4_COMMAND_REPLAY/1', 'project_root': str(PROJECT), **CONFIGS,
               'initial': {'base_xy_m': record.base[0, :2].tolist(), 'base_yaw_rad': float(record.base[0, 2]),
                           'base_world_velocity': base_v[0].tolist(), 'q_rad': record.q[0].tolist(),
                           'dq_rad_s': measured_dq[0].tolist()},
               'duration_s': float(record.t[-1]), 'base_command_frame': base_command_frame,
               'arm_mode': arm_mode, 'gravity_feedforward': gravity_feedforward,
               'commands': [{'t_s': float(t), 'q_rad': q.tolist(), 'dq_rad_s': dq.tolist(), 'base_twist': b.tolist()}
                            for t, q, dq, b in zip(record.t, command_q, command_dq, base_command)]}
    return record, measured_dq, base_v, payload


def statistics(delta):
    delta = np.asarray(delta)
    return {'rmse_per_axis': np.sqrt(np.mean(delta**2, axis=0)).tolist(),
            'max_abs_per_axis': np.max(np.abs(delta), axis=0).tolist(),
            'final_per_axis': delta[-1].tolist(),
            'vector_rmse': float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))}


def compare(directory, record, measured_dq, measured_base_v, payload):
    frames = [json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
    t = np.array([frame['t'] for frame in frames])
    native_summary = json.loads((directory/'summary.json').read_text())
    if not native_summary['finite'] or not native_summary['time_advancement_sane'] or any(native_summary['warnings'].values()):
        raise RuntimeError(f'{directory}: invalid physics; inspect summary.json and run.log')
    if len(t) != native_summary['physics_steps'] + 1 or (np.diff(t) <= 0).any():
        raise ValueError('unexpected simulation timeline')
    real_q, real_dq = interpolate(record.t, record.q, t), interpolate(record.t, measured_dq, t)
    real_base, real_base_v = interpolate(record.t, record.base, t), interpolate(record.t, measured_base_v, t)
    real_fk, vicon = interpolate(record.t, record.logged_fk, t), interpolate(record.t, record.vicon, t)
    sim_q, sim_dq = np.array([f['q'] for f in frames]), np.array([f['dq'] for f in frames])
    sim_base = np.array([f['base'] for f in frames])
    sim_base[:, 2] = np.unwrap(sim_base[:, 2])
    sim_base_v = np.array([f['base_world_velocity'] for f in frames])
    sim_fk = np.array([f['racket_xyz'] for f in frames])
    indices = np.searchsorted(record.t, t+1e-12, side='right')-1
    if not np.array_equal(indices, [f['command_index'] for f in frames]):
        raise AssertionError('ZOH command index mismatch or future command consumed')
    expected_q = np.array([payload['commands'][i]['q_rad'] for i in indices])
    expected_dq = np.array([payload['commands'][i]['dq_rad_s'] for i in indices])
    np.testing.assert_array_equal(expected_q, [f['q_command'] for f in frames])
    np.testing.assert_array_equal(expected_dq, [f['dq_command'] for f in frames])
    np.testing.assert_array_equal([payload['commands'][i]['base_twist'] for i in indices],
                                  [f['base_twist_input'] for f in frames])
    for actual, measured in ((sim_q, real_q), (sim_dq, real_dq), (sim_base, real_base), (sim_base_v, real_base_v)):
        np.testing.assert_allclose(actual[0], measured[0], atol=1e-12, rtol=0)
    values = {'t_s': t}
    for kind, real, simulated, names in (
        ('base', real_base, sim_base, ['x_m', 'y_m', 'yaw_rad']),
        ('base', real_base_v, sim_base_v, ['vx_m_s', 'vy_m_s', 'wz_rad_s']),
        ('q', real_q, sim_q, [f'j{j}_rad' for j in range(1, 6)]),
        ('dq', real_dq, sim_dq, [f'j{j}_rad_s' for j in range(1, 6)]),
        ('racket', real_fk, sim_fk, ['x_m', 'y_m', 'z_m'])):
        for i, name in enumerate(names):
            values[f'real_{kind}_{name}'] = real[:, i]
            values[f'sim_{kind}_{name}'] = simulated[:, i]
    for j in range(5):
        values[f'cmd_q_j{j+1}_rad'] = expected_q[:, j]
        values[f'cmd_dq_j{j+1}_rad_s'] = expected_dq[:, j]
        values[f'applied_cmd_q_j{j+1}_rad'] = [f.get('applied_q_command', f['q_command'])[j] for f in frames]
        values[f'applied_cmd_dq_j{j+1}_rad_s'] = [f.get('applied_dq_command', f['dq_command'])[j] for f in frames]
        values[f'arm_active_j{j+1}'] = [int(f.get('arm_axis_command_active', [True]*5)[j]) for f in frames]
    for j, axis in enumerate('xyz'):
        values[f'vicon_racket_{axis}_m'] = vicon[:, j]
    with (directory/'comparison.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(values)
        writer.writerows(zip(*values.values()))
    yaw_error = (sim_base[:, 2:3]-real_base[:, 2:3]+np.pi) % (2*np.pi)-np.pi
    raw = np.array([f['tau_raw'] for f in frames[1:]])
    cap = np.array([f['tau_cap'] for f in frames[1:]])
    report = {'source': str(record.path), 'source_sha256': hashlib.sha256(record.path.read_bytes()).hexdigest(),
              'duration_compared_s': float(t[-1]), 'native_summary': native_summary,
              'initial_observed_state_exact': True, 'command_zoh_causal_and_exact': True,
              'joint_angle_deg': statistics(np.rad2deg(sim_q-real_q)),
              'joint_velocity_rad_s': statistics(sim_dq-real_dq),
              'base_xy_m': statistics(sim_base[:, :2]-real_base[:, :2]),
              'base_yaw_deg': statistics(np.rad2deg(yaw_error)),
              'base_velocity_xy_m_s': statistics(sim_base_v[:, :2]-real_base_v[:, :2]),
              'racket_vs_logged_fk_m': statistics(sim_fk-real_fk),
              'racket_vs_vicon_m': statistics(sim_fk-vicon),
              'arm_torque_clipped_fraction_per_axis': np.mean(np.abs(raw)>cap+1e-9, axis=0).tolist(),
              'max_abs_joint_speed_rad_s': np.max(np.abs(sim_dq), axis=0).tolist(),
              'base_height_range_m': [float(min(f['qpos'][2] for f in frames)), float(max(f['qpos'][2] for f in frames))]}
    (directory/'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


def write_report(output, manifest, reports):
    lines = ['# 实测命令驱动的动力学对照', '',
             f'参数配置：{manifest["config_dir"]}。本轮重新执行原生动力学计算。', '',
             '以实测底盘/机械臂位置和速度初始化，按原始日志命令及记录的起始控制规则运行；不再写入后续实测状态。', '',
             f'起始控制设置：{manifest.get("initialization_mode", "legacy")}。逐条设置来源、前缀长度及底盘初态保存在 manifest.json 的 initialization_by_group 中。', '',
             '本结果不是实际驱动器完整命令的复刻：日志没有 Kp/Kd、模式、前馈和各轮反馈。J5 位置命令恒为零，控制模式及姿态方向未确认。', '',
             '| 组 | 底盘 XY RMSE (m) | 拍心 vs 日志 FK RMSE (m) | 拍心 vs Vicon RMSE (m) | J1–J5 角度 RMSE (deg) |',
             '|---|---:|---:|---:|---|']
    for r in reports:
        angles = ', '.join(f'{v:.2f}' for v in r['joint_angle_deg']['rmse_per_axis'])
        lines.append(f'| {r["group"]} | {r["base_xy_m"]["vector_rmse"]:.3f} | {r["racket_vs_logged_fk_m"]["vector_rmse"]:.3f} | {r["racket_vs_vicon_m"]["vector_rmse"]:.3f} | {angles} |')
    lines += ['', '## 已检查', '', '- 初始可观测位置和速度一致；命令严格零阶保持，没有读取未来命令。',
              '- 原生进程输入仅包含初始状态与命令，不包含后续实测状态；1 ms 步长由实际 MuJoCo 动力学推进。',
              '- 每步状态有限、时钟连续，MuJoCo BADQPOS/BADQVEL/BADQACC 警告为零。',
              '- 本次回放不重新拟合参数或初态；已有初态可能来自该记录前缀估计，整段误差用于回放对照，不是新的独立验证。', '',
              '## 限制', '', *['- '+s for s in manifest['assumptions']], '',
              '每组 comparison.png 为五轴位置/速度对照，paths.png 为底盘、拍心轨迹误差；comparison.csv 和 frames.jsonl 分别保留原始命令、应用目标和逐轴激活状态。',
              'tau_* 字段对应前一个积分区间，详见 torque_interval_start_s/torque_command_index；不将日志 motor*_torque 当作下发前馈。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'metrics.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False)+'\n')


def group_directories(output):
    return sorted((p for p in output.glob('group_*') if p.is_dir() and p.name[6:].isdigit()),
                  key=lambda p: int(p.name[6:]))


def view_result(output, group=1, speed=.25):
    os.environ['MUJOCO_GL'] = 'glfw'
    import mujoco
    from replay_measured_robot import Robot
    from view_x4_native_replay import desktop_viewer
    directories = group_directories(output)
    if not directories:
        raise ValueError('no group directories to replay')
    all_frames, observed = [], []
    for d in directories:
        all_frames.append([json.loads(s) for s in (d/'frames.jsonl').read_text().splitlines()])
        with (d/'comparison.csv').open() as f:
            observed.append([{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)])
    scene_path = replay_scene(output)
    manifest = json.loads((output/'manifest.json').read_text())
    profile_label = manifest.get('config_dir', manifest.get('profile', output.name))
    sim = Robot(scene_path)
    real = Robot(scene_path)
    keys, cursor, paused, last = queue.SimpleQueue(), 0., False, time.monotonic()
    index = next((i for i, d in enumerate(directories) if d.name == f'group_{group}'), 0)
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    with desktop_viewer(sim.model, sim.data, key_callback=keys.put, show_left_ui=False, show_right_ui=False) as viewer:
        def camera():
            first = all_frames[index][0]['base']
            viewer.cam.lookat[:] = [first[0], first[1], .8]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 135, -20
        camera()
        while viewer.is_running():
            now = time.monotonic()
            if not paused:
                cursor += (now-last)*speed
            last = now
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    paused = not paused
                elif key in (ord('R'), ord('r')):
                    cursor = 0
                elif ord('1') <= key <= ord('9'):
                    desired = f'group_{key-ord("0")}'
                    for i, d in enumerate(directories):
                        if d.name == desired:
                            index, cursor = i, 0.
                            camera()
            frames = all_frames[index]
            if cursor > frames[-1]['t']+.2:
                cursor = 0
            i = min(len(frames)-1, max(0, int(cursor/.001)))
            frame, obs = frames[i], observed[index][i]
            with viewer.lock():
                sim.data.qpos[:], sim.data.qvel[:] = frame['qpos'], frame['qvel']
                sim.data.time = frame['t']
                mujoco.mj_forward(sim.model, sim.data)
                real.pose([obs[f'real_base_{x}'] for x in ('x_m','y_m','yaw_rad')],
                          [obs[f'real_q_j{j}_rad'] for j in range(1,6)])
                scene = viewer.user_scn
                scene.ngeom = 0
                mujoco.mjv_addGeoms(real.model, real.data, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
                for k in range(scene.ngeom):
                    scene.geoms[k].rgba[:] = [0, .9, 1, .25]
                for pos, color in ((sim.data.geom_xpos[sim.face], [1,.2,.1,1]),
                                   ([obs[f'vicon_racket_{x}_m'] for x in 'xyz'], [1,.1,.8,1])):
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.full(3,.025), np.asarray(pos), np.eye(3).ravel(), np.array(color,np.float32))
                    scene.ngeom += 1
            viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              'Dynamics vs measured\nRobot\nRacket markers\nTime\nControls\nLimit',
                              f'{directories[index].name}\nSolid = simulated; cyan ghost = measured\nRed = simulated; pink = Vicon\n{frame["t"]:.3f}s / {speed:g}x\nSpace pause | R restart | 1..5 select\nRecorded configuration: {profile_label}; hardware modes unknown'))
            viewer.sync()
            time.sleep(1/60)


def write_config_comparison(output):
    labels = ('config_1', 'config_2')
    manifests = {label: json.loads((output/label/'manifest.json').read_text()) for label in labels}
    first, second = (manifests[label] for label in labels)
    same_raw = first['raw_experiment_sha256'] == second['raw_experiment_sha256']
    if not same_raw:
        raise AssertionError('Configuration comparison measured initial states/commands/settings differ')
    same_effective = first['experiment_sha256'] == second['experiment_sha256']
    results = {label: json.loads((output/label/'metrics.json').read_text()) for label in labels}
    comparison = {'config_dirs': {label: manifests[label]['config_dir'] for label in labels},
                  'same_raw_experiment_inputs': same_raw,
                  'same_experiment_inputs': same_effective,
                  'raw_experiment_sha256': first['raw_experiment_sha256'],
                  'experiment_sha256': {label: manifests[label]['experiment_sha256'] for label in labels},
                  'metrics': results}
    (output/'comparison.json').write_text(json.dumps(comparison, indent=2, ensure_ascii=False)+'\n')
    lines = ['# 两套配置重新仿真对照', '',
             *[f'- {label}：{manifests[label]["config_dir"]}' for label in labels], '',
             '两套配置均重新执行原生动力学步进，共用同一次解析的实测初始状态和命令；原始输入 SHA-256 一致。',
             ('应用起始控制设置后的输入也一致。' if same_effective else
              '应用起始控制设置后的输入不同：本次差异包含初始化策略，不能单独归因于物理参数变化。'), '',
             '|组|配置|底盘 XY RMSE / m|J1–J5 角度 RMSE / °|J1–J5 速度 RMSE / rad/s|',
             '|---|---|---:|---|---|']
    for pair in zip(*(results[label] for label in labels)):
        for label, metrics in zip(labels, pair):
            q = ', '.join(f'{v:.3f}' for v in metrics['joint_angle_deg']['rmse_per_axis'])
            dq = ', '.join(f'{v:.3f}' for v in metrics['joint_velocity_rad_s']['rmse_per_axis'])
            lines.append(f'|{metrics["group"]}|{label}|{metrics["base_xy_m"]["vector_rmse"]:.4f}|{q}|{dq}|')
    lines += ['', '逐组轨迹图、实测叠加和独立模型快照分别保存在 config_1/、config_2/ 子目录。',
              '真机记录前的控制器内部状态仍未知。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(f'Fresh configuration comparison: {output/"comparison.md"}', flush=True)


def run_config(args, config_dir, output, prepared):
    config_dir = validate_config_dir(config_dir)
    output = output.resolve()
    config_paths = snapshot_config_dir(config_dir, output/'config')
    initialization_settings = output/'config/replay_initialization.json'
    print(f'Running fresh dynamics: config_dir={config_dir}, output={output}', flush=True)
    manifest = {'config_dir': str(config_dir), 'config_paths': {k: str(Path(v).relative_to(output)) for k,v in config_paths.items()},
                'raw_experiment_sha256': {}, 'experiment_sha256': {}, 'mapping': DEFAULT_MAPPING, 'base_command_frame': args.base_command_frame,
                'base_feedback_frame': args.base_feedback_frame, 'arm_mode': args.arm_mode,
                'gravity_feedforward': not args.no_gravity_feedforward,
                'initialization_mode': 'directory_settings',
                'initialization_by_group': {},
                'binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
                'snapshot_sha256': {str(p.relative_to(output/'config')): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((output/'config').rglob('*')) if p.is_file()},
                'config_sha256': {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in config_paths.items()},
                'config_contents': {k: json.loads(Path(v).read_text()) for k,v in config_paths.items() if v.endswith('.json')},
                'assumptions': [
                    '使用选定配置的仿真 Kp/Kd、连续档限矩、T-N 包络、底盘斜坡和指定重力前馈开关，不代表采集时的实际驱动器参数。',
                    'J5 位置命令恒为零，其实际模式与姿态方向尚未确认；仿真执行模式记录在 arm_mode 中。',
                    f'底盘反馈采用 {args.base_feedback_frame} 系，命令采用 {args.base_command_frame} 系；默认 world 由数据支持但无生产源码确认。',
                    '位置度转弧度，关节速度按 rad/s；底盘位置 cm 转 m、速度 mm/s 转 m/s、角速度按 rad/s。',
                    '原始日志约 2.1 ms 采样；仿真 1 ms 步长在下一物理拍消费已到达命令，最大额外取样滞后小于 1 ms。',
                    '仅可观测初始状态相同：缺失的舵角/轮速按平面无滑动重建；底盘高度/roll/pitch采用模型 home。',
                    '底盘控制初态遵循逐条 initialization 设置；未记录的 PI 积分仍为零，真实截取前历史未知。',
                    '起始保持属于已选择的仿真控制假设，不代表确认了采集时的真机驱动模式。',
                    'J5 位置对照依赖暂定方向；实测轮状态、真实模式、前馈、控制增益和延迟未包含于本批日志。',
                ]}
    manifest['initialization_settings'] = {
        'path': str(initialization_settings.relative_to(output)),
        'sha256': hashlib.sha256(initialization_settings.read_bytes()).hexdigest()}
    reports = []
    for number, (record, dq, base_v, original) in enumerate(prepared, 1):
        payload = {**original, 'project_root': str(output/'config'),
                   **{k: Path(v).name for k,v in config_paths.items()}}
        payload, metadata = apply_initialization(payload, record.path, settings_path=initialization_settings)
        manifest['raw_experiment_sha256'][str(number)] = command_identity(original)
        manifest['initialization_by_group'][str(number)] = metadata
        manifest['experiment_sha256'][str(number)] = command_identity(payload)
        input_path = output/f'group_{number}_input.json'
        input_path.write_text(json.dumps(payload, indent=2)+'\n')
        directory = output/f'group_{number}'
        with (output/f'group_{number}_run.log').open('w') as f:
            subprocess.run([str(BINARY), str(input_path.resolve()), str(directory.resolve())], stdout=f, stderr=subprocess.STDOUT, check=True)
        metrics = compare(directory, record, dq, base_v, payload)
        metrics['group'] = number
        metrics['initialization'] = metadata
        reports.append(metrics)
        print(f'Group {number}: base XY RMSE {metrics["base_xy_m"]["vector_rmse"]:.3f}m, '
              f'racket RMSE {metrics["racket_vs_vicon_m"]["vector_rmse"]:.3f}m', flush=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    write_report(output, manifest, reports)
    subprocess.run(['/usr/bin/python3', str(ROOT/'tools/plot_command_dynamics.py'), str(output.resolve())], check=True)
    print(f'Report: {output.resolve()/"report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--data-dir', type=Path, default=DEFAULT_DATA,
                        help='recording directory containing CSVs, or parent containing 下位机轨迹')
    inputs.add_argument('--trajectory-file', type=Path,
                        help='generate a replay for this single trajectory CSV')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/measured_dynamics'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    configs = parser.add_mutually_exclusive_group()
    configs.add_argument('--config-dir', type=Path,
                         help='self-contained configuration directory (required for a fresh replay)')
    configs.add_argument('--compare-config-dirs', type=Path, nargs=2, metavar=('DIR1', 'DIR2'),
                         help='rerun two configuration directories from the same measured state and commands')
    parser.add_argument('--profile', help=argparse.SUPPRESS)
    parser.add_argument('--compare-profiles', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--base-command-frame', choices=['world','body'], default='world')
    parser.add_argument('--base-feedback-frame', choices=['world','body'], default='world')
    parser.add_argument('--arm-mode', choices=['hybrid','j5_velocity'], default='hybrid')
    parser.add_argument('--no-gravity-feedforward', action='store_true')
    parser.add_argument('--legacy-initialization', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--view-run', type=Path, help='view an existing comparison without rerunning dynamics')
    parser.add_argument('--group', type=int, default=1,
                        help='initial viewer group; does not filter trajectory files')
    parser.add_argument('--speed', type=float, default=.25)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.profile is not None or args.compare_profiles:
        parser.error('--profile/--compare-profiles were removed; use --config-dir PATH or --compare-config-dirs DIR1 DIR2')
    if args.legacy_initialization:
        parser.error('--legacy-initialization was removed; set enabled=false in CONFIG_DIR/replay_initialization.json')
    if args.view_run and (args.config_dir or args.compare_config_dirs or args.view or args.trajectory_file):
        parser.error('--view-run cannot be combined with --trajectory-file, --config-dir, --compare-config-dirs or --view')
    if args.compare_config_dirs and args.view:
        parser.error('view each completed comparison with --view-run OUTPUT/config_1 or OUTPUT/config_2')
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    if args.view or args.view_run:
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            parser.error('viewer requires a desktop display')
    if args.view_run:
        view_result(args.view_run, args.group, args.speed)
        return
    if not (args.config_dir or args.compare_config_dirs):
        parser.error('fresh replay requires --config-dir PATH or --compare-config-dirs DIR1 DIR2')
    config_dirs = args.compare_config_dirs or [args.config_dir]
    try:
        config_dirs = [validate_config_dir(path) for path in config_dirs]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if any(args.output.is_relative_to(path) for path in config_dirs):
        parser.error('--output must be outside the source configuration directories')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be new or empty')
    if args.trajectory_file:
        path = args.trajectory_file.resolve()
        if not path.is_file() or path.suffix.lower() != '.csv':
            parser.error('--trajectory-file must point to an existing CSV file')
        paths = [path]
    else:
        paths = sorted(args.data_dir.glob('*.csv'))
        if not paths:
            paths = sorted((args.data_dir/'下位机轨迹').glob('*.csv'))
    if not paths:
        parser.error('no lower-level CSV logs found')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_build:
        with (args.output/'build.log').open('w') as f:
            subprocess.run(['cmake', '-S', str(PROJECT), '-B', str(PROJECT/'build')], stdout=f, stderr=subprocess.STDOUT, check=True)
            subprocess.run(['cmake', '--build', str(PROJECT/'build'), '--target', 'x4_command_replay', '-j4'], stdout=f, stderr=subprocess.STDOUT, check=True)
    prepared = [prepare(path, args.base_command_frame, args.base_feedback_frame,
                        args.arm_mode, not args.no_gravity_feedforward) for path in paths]
    if args.compare_config_dirs:
        for index, config_dir in enumerate(config_dirs, 1):
            run_config(args, config_dir, args.output/f'config_{index}', prepared)
        write_config_comparison(args.output)
    else:
        run_config(args, config_dirs[0], args.output, prepared)
    if args.view:
        view_result(args.output, args.group, args.speed)


if __name__ == '__main__':
    main()
