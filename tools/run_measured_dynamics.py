#!/usr/bin/env python3
"""Compare measured motion with dynamics driven by the original logged setpoints.

Uses current simulation gains, limits and gravity feedforward, not identified hardware
settings. Only initial measured state and commands are passed to the native process.
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

PROJECT = ROOT / 'sckd_sim'
BINARY = PROJECT / 'build/x4_command_replay'
CONFIGS = {'scene_path': 'models/generated/x4_950_scene.mjcf',
           'motors_path': 'models/config/motors_x4_950.json',
           'robot_variant_path': 'models/config/x4_950.json',
           'chassis_path': 'models/config/chassis_x4_950.json'}


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
    lines = ['# 实测命令驱动的动力学基线对照', '',
             '以实测底盘/机械臂位置和速度初始化；后续只向仿真输入日志中的位置、速度与底盘命令，使用当前仿真增益、限矩和重力前馈。', '',
             '本结果不是实际驱动器完整命令的复刻：日志没有 Kp/Kd、模式、前馈和各轮反馈。J5 位置命令恒为零，控制模式及姿态方向未确认。', '',
             '| 组 | 底盘 XY RMSE (m) | 拍心 vs 日志 FK RMSE (m) | 拍心 vs Vicon RMSE (m) | J1–J5 角度 RMSE (deg) |',
             '|---|---:|---:|---:|---|']
    for r in reports:
        angles = ', '.join(f'{v:.2f}' for v in r['joint_angle_deg']['rmse_per_axis'])
        lines.append(f'| {r["group"]} | {r["base_xy_m"]["vector_rmse"]:.3f} | {r["racket_vs_logged_fk_m"]["vector_rmse"]:.3f} | {r["racket_vs_vicon_m"]["vector_rmse"]:.3f} | {angles} |')
    lines += ['', '## 已检查', '', '- 初始可观测位置和速度一致；命令严格零阶保持，没有读取未来命令。',
              '- 原生进程输入仅包含初始状态与命令，不包含后续实测状态；1 ms 步长由实际 MuJoCo 动力学推进。',
              '- 每步状态有限、时钟连续，MuJoCo BADQPOS/BADQVEL/BADQACC 警告为零。',
              '- 对比阶段才插值实测状态，不平移时间、不拟合参数、不对齐终点。', '',
              '## 限制', '', *['- '+s for s in manifest['assumptions']], '',
              '每组 comparison.png 为五轴位置/速度对照，paths.png 为底盘、拍心轨迹误差；comparison.csv 和 frames.jsonl 保留数值。',
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
    sim = Robot(PROJECT/CONFIGS['scene_path'])
    real = Robot(PROJECT/CONFIGS['scene_path'])
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
                              f'{directories[index].name}\nSolid = simulated; cyan ghost = measured\nRed = simulated; pink = Vicon\n{frame["t"]:.3f}s / {speed:g}x\nSpace pause | R restart | 1..5 select\nCurrent simulation gains; hardware modes unknown'))
            viewer.sync()
            time.sleep(1/60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/measured_dynamics'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--base-command-frame', choices=['world','body'], default='world')
    parser.add_argument('--base-feedback-frame', choices=['world','body'], default='world')
    parser.add_argument('--arm-mode', choices=['hybrid','j5_velocity'], default='hybrid')
    parser.add_argument('--no-gravity-feedforward', action='store_true')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--view-run', type=Path, help='view an existing comparison without rerunning dynamics')
    parser.add_argument('--group', type=int, default=1, help='initial viewer group; analysis always covers all logs')
    parser.add_argument('--speed', type=float, default=.25)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    if args.view or args.view_run:
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            parser.error('viewer requires a desktop display')
    if args.view_run:
        view_result(args.view_run, args.group, args.speed)
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be new or empty')
    paths = sorted((args.data_dir/'下位机轨迹').glob('*.csv'))
    if not paths:
        parser.error('no lower-level CSV logs found')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_build:
        with (args.output/'build.log').open('w') as f:
            subprocess.run(['cmake', '-S', str(PROJECT), '-B', str(PROJECT/'build')], stdout=f, stderr=subprocess.STDOUT, check=True)
            subprocess.run(['cmake', '--build', str(PROJECT/'build'), '--target', 'x4_command_replay', '-j4'], stdout=f, stderr=subprocess.STDOUT, check=True)
    manifest = {'mapping': DEFAULT_MAPPING, 'base_command_frame': args.base_command_frame,
                'base_feedback_frame': args.base_feedback_frame, 'arm_mode': args.arm_mode,
                'gravity_feedforward': not args.no_gravity_feedforward,
                'binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
                'config_sha256': {k: hashlib.sha256((PROJECT/v).read_bytes()).hexdigest() for k,v in CONFIGS.items()},
                'config_contents': {k: json.loads((PROJECT/v).read_text()) for k,v in CONFIGS.items() if v.endswith('.json')},
                'assumptions': [
                    '使用当前未标定的仿真 Kp/Kd、连续档限矩、T-N 包络、底盘斜坡和重力前馈，不代表采集时的实际驱动器参数。',
                    '基线将五轴位置和速度同时送入混合控制；J5 位置命令恒为零，其实际模式与姿态方向尚未确认。',
                    f'底盘反馈采用 {args.base_feedback_frame} 系，命令采用 {args.base_command_frame} 系；默认 world 由数据支持但无生产源码确认。',
                    '位置度转弧度，关节速度按 rad/s；底盘位置 cm 转 m、速度 mm/s 转 m/s、角速度按 rad/s。',
                    '原始日志约 2.1 ms 采样；仿真 1 ms 步长在下一物理拍消费已到达命令，最大额外取样滞后小于 1 ms。',
                    '仅可观测初始状态相同：缺失的舵角/轮速按平面无滑动重建；底盘高度/roll/pitch采用模型 home。',
                    '截取前的底盘斜坡、积分器及驱动器内部状态未知，当前从默认控制器历史启动，不能视作完整内部状态相同。',
                    'J5 位置对照依赖暂定方向；实测轮状态、真实模式、前馈、控制增益和延迟未包含于本批日志。',
                ]}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    reports = []
    for number, path in enumerate(paths, 1):
        record, dq, base_v, payload = prepare(path, args.base_command_frame, args.base_feedback_frame,
                                             args.arm_mode, not args.no_gravity_feedforward)
        input_path = args.output/f'group_{number}_input.json'
        input_path.write_text(json.dumps(payload, indent=2)+'\n')
        directory = args.output/f'group_{number}'
        with (args.output/f'group_{number}_run.log').open('w') as f:
            subprocess.run([str(BINARY), str(input_path.resolve()), str(directory.resolve())], stdout=f, stderr=subprocess.STDOUT, check=True)
        metrics = compare(directory, record, dq, base_v, payload)
        metrics['group'] = number
        reports.append(metrics)
        print(f'Group {number}: base XY RMSE {metrics["base_xy_m"]["vector_rmse"]:.3f}m, '
              f'racket RMSE {metrics["racket_vs_vicon_m"]["vector_rmse"]:.3f}m', flush=True)
    write_report(args.output, manifest, reports)
    subprocess.run(['/usr/bin/python3', str(ROOT/'tools/plot_command_dynamics.py'), str(args.output.resolve())], check=True)
    print(f'Report: {args.output.resolve()/"report.md"}', flush=True)
    if args.view:
        view_result(args.output, args.group, args.speed)


if __name__ == '__main__':
    main()
