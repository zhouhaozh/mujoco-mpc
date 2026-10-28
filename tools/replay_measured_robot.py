#!/usr/bin/env python3
"""Replay measured X4 state and compare model, logged FK, and Vicon racket centres.

No physics steps, command replay, time correction, or geometry fitting are applied.
Run with the repository .venv/bin/python. Space pauses, R restarts, 1..5 selects a log.
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path('/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹')
DEFAULT_MODEL = ROOT / 'sckd_sim/models/generated/x4_950_scene.mjcf'
DEFAULT_MAPPING = {
    'joint_order': [1, 2, 3, 4, 5],
    'joint_sign': [-1, -1, 1, 1, 1],
    'joint_offset_deg': [0, 0, 0, 0, 0],
    'base_xy_scale': 0.01,
    'base_yaw_scale': math.pi / 180,
    'vicon_scale': 1.0,
}
COLORS = ([0, .9, 1, 1], [1, .15, .7, 1], [1, .8, .1, 1])


def validate_mapping(mapping):
    if set(mapping) != set(DEFAULT_MAPPING):
        raise ValueError('mapping keys must match mapping.json emitted by this script')
    if sorted(mapping['joint_order']) != [1, 2, 3, 4, 5]:
        raise ValueError('joint_order must be a permutation of 1..5')
    if len(mapping['joint_sign']) != 5 or not all(x in (-1, 1) for x in mapping['joint_sign']):
        raise ValueError('joint_sign must contain five +1/-1 values')
    offsets = np.asarray(mapping['joint_offset_deg'], dtype=float)
    if offsets.shape != (5,) or not np.isfinite(offsets).all():
        raise ValueError('joint_offset_deg must contain five finite values')
    for key in ('base_xy_scale', 'base_yaw_scale', 'vicon_scale'):
        if not math.isfinite(mapping[key]) or mapping[key] == 0:
            raise ValueError(f'{key} must be finite and nonzero')


@dataclass
class Record:
    path: Path
    timestamp_ms: np.ndarray
    t: np.ndarray
    base: np.ndarray
    q: np.ndarray
    vicon: np.ndarray
    logged_fk: np.ndarray
    extra_columns: int


def read_record(path, mapping):
    """Preserve # field positions; accept exactly one known unlabelled trailing value."""
    validate_mapping(mapping)
    lines = path.read_text().splitlines()
    header = lines[0].split()
    named = {name: i for i, name in enumerate(header) if name != '#'}
    required = ['local_timestamp', 'robotcurrent_x', 'robotcurrent_y', 'robotcurrent_angle']
    required += [f'motor{i}_current_pos' for i in range(1, 6)]
    required += [f'{prefix}_{axis}' for prefix in ('vicon_racket', 'racket_pos') for axis in 'xyz']
    if any(name not in named for name in required):
        raise ValueError(f'{path}: missing required columns')
    values, extras = [], set()
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.split()
        extra = len(fields) - len(header)
        if extra not in (0, 1) or any(fields[i] != '#' for i, h in enumerate(header) if h == '#'):
            raise ValueError(f'{path}:{number}: unexpected field layout')
        extras.add(extra)
        values.append([float(fields[named[name]]) for name in required])
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all() or (np.diff(a[:, 0]) <= 0).any():
        raise ValueError(f'{path}: need 2+ finite rows with strictly increasing timestamps')
    base = a[:, 1:4].copy()
    base[:, :2] *= mapping['base_xy_scale']
    base[:, 2] = np.unwrap(base[:, 2] * mapping['base_yaw_scale'])
    order = np.asarray(mapping['joint_order']) - 1
    q = np.deg2rad(a[:, 4:9][:, order] * mapping['joint_sign'] + mapping['joint_offset_deg'])
    return Record(path, a[:, 0], (a[:, 0] - a[0, 0]) / 1000, base, q,
                  a[:, 9:12] * mapping['vicon_scale'], a[:, 12:15], max(extras))


def interpolate(times, values, t):
    return np.array([np.interp(t, times, values[:, j]) for j in range(values.shape[1])]).T


class Robot:
    def __init__(self, path):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.home = self.data.qpos.copy()
        root = self.model.joint('root').id
        if self.model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('root must be a free joint')
        self.base_adr = int(self.model.jnt_qposadr[root])
        self.arm_adr = np.array([self.model.jnt_qposadr[self.model.joint(f'Joint{i}').id]
                                 for i in range(1, 6)])
        self.face = self.model.geom('racket_face').id

    def pose(self, base, q, t=0, full=True):
        self.data.qpos[:] = self.home
        b = self.base_adr
        self.data.qpos[b:b+2] = base[:2]
        self.data.qpos[b+3:b+7] = [math.cos(base[2]/2), 0, 0, math.sin(base[2]/2)]
        self.data.qpos[self.arm_adr] = q
        # Pose-only playback: no inferred wheel states or physical velocities.
        self.data.qvel[:] = 0
        self.data.time = t
        (self.mj.mj_forward if full else self.mj.mj_kinematics)(self.model, self.data)
        return self.data.geom_xpos[self.face].copy()

    def trajectory(self, record, q=None):
        q = record.q if q is None else q
        return np.array([self.pose(b, a, t, full=False) for b, a, t in zip(record.base, q, record.t)])


def errors(a, b):
    delta = a - b
    distance = np.linalg.norm(delta, axis=1)
    return {'rmse_m': float(np.sqrt(np.mean(distance**2))),
            'median_m': float(np.median(distance)), 'p95_m': float(np.percentile(distance, 95)),
            'max_m': float(distance.max()), 'bias_xyz_m': delta.mean(axis=0).tolist()}


def lag_probe(record, model_points):
    """Fixed support for every shift; positive lag compares model(t+lag) to Vicon(t)."""
    mask = (record.t >= .04) & (record.t <= record.t[-1] - .04)
    if mask.sum() < 5:
        return {'available': False}
    ts, target = record.t[mask], record.vicon[mask]
    scores = [errors(interpolate(record.t, model_points, ts + ms/1000), target)['rmse_m']
              for ms in range(-40, 41)]
    best = int(np.argmin(scores))
    return {'available': True, 'best_shift_ms': best - 40,
            'zero_shift_rmse_m_same_support': scores[40], 'best_rmse_m': scores[best],
            'at_search_boundary': best in (0, 80),
            'convention': 'compare model(t + shift) with Vicon(t); diagnostic only, not applied',
            'warning': 'Geometry, sensor filtering and unequal observation age can mimic delay.'}


def audit(robot, records, mapping, output, model_path):
    output.mkdir(parents=True, exist_ok=True)
    points = [robot.trajectory(r) for r in records]
    # Fix order and zero offsets, evaluate direction only against logged FK on group 1.
    # This is an internal-consistency probe, not an independent geometry calibration.
    candidates = []
    train = records[0]
    for sign in itertools.product((-1, 1), repeat=5):
        q = (train.q - np.deg2rad(mapping['joint_offset_deg'])) / mapping['joint_sign']
        q = q * sign + np.deg2rad(mapping['joint_offset_deg'])
        candidate = robot.trajectory(train, q)
        candidates.append({'sign': list(sign), 'train_logged_fk_rmse_m': errors(candidate, train.logged_fk)['rmse_m']})
    candidates.sort(key=lambda item: item['train_logged_fk_rmse_m'])
    report = {
        'model': str(model_path), 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'mapping': mapping, 'physics_stepped': False, 'time_shift_applied_ms': 0,
        'geometry_or_translation_fitted': False,
        'direction_probe_training_file': str(train.path), 'direction_probe_top4': candidates[:4],
        'limitations': [
            'J5 direction/zero cannot be identified from centre position: the centre lies on its axis.',
            'Joint order/zeros are checked for consistency, not uniquely identified by this direction probe.',
            'Logged FK is not independent ground truth; Vicon position is the external comparison.',
            'Vicon packet observation time is not available separately from local log time.',
            'Base height/roll/pitch and wheel angles are absent; use model home values and planar base.',
            'Vicon orientation convention is unverified; no racket-normal alignment claim.',
            'Units are supported by numerical consistency; CSV writer source was not located.',
        ], 'groups': [],
    }
    for number, (r, p) in enumerate(zip(records, points), 1):
        dt = np.diff(r.t)
        offsets = np.deg2rad(mapping['joint_offset_deg'])
        naive = robot.trajectory(r, (r.q - offsets) / mapping['joint_sign'] + offsets)
        yawless_base = r.base.copy()
        yawless_base[:, 2] = 0
        yawless = np.array([robot.pose(b, q, full=False) for b, q in zip(yawless_base, r.q)])
        group = {'group': number, 'source': str(r.path),
                 'source_sha256': hashlib.sha256(r.path.read_bytes()).hexdigest(),
                 'rows': len(r.t), 'first_timestamp_ms': float(r.timestamp_ms[0]),
                 'duration_s': float(r.t[-1]), 'sample_interval_ms_min_median_max':
                 (np.array([dt.min(), np.median(dt), dt.max()])*1000).tolist(),
                 'unlabelled_trailing_columns': r.extra_columns,
                 'model_vs_vicon': errors(p, r.vicon), 'model_vs_logged_fk': errors(p, r.logged_fk),
                 'logged_fk_vs_vicon': errors(r.logged_fk, r.vicon),
                 'all_positive_joint_sign_vs_vicon': errors(naive, r.vicon),
                 'zero_base_yaw_vs_logged_fk': errors(yawless, r.logged_fk),
                 'lag_probe': lag_probe(r, p)}
        report['groups'].append(group)
        with (output / f'group_{number}_comparison.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_ms', 't_s'] + [f'{prefix}_{axis}_m' for prefix in ('model', 'vicon', 'logged_fk') for axis in 'xyz'] + ['error_m'])
            for i in range(len(r.t)):
                writer.writerow([r.timestamp_ms[i], r.t[i], *p[i], *r.vicon[i], *r.logged_fk[i], float(np.linalg.norm(p[i]-r.vicon[i]))])
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    (output / 'mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    return points, report


def markers(scene, robot, record, points, t):
    mj = robot.mj
    scene.ngeom = min(scene.ngeom, scene.maxgeom)

    def line(a, b, color, width=.003):
        if np.linalg.norm(b-a) < 1e-9 or scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color, np.float32))
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1

    trajectories = [points, record.vicon, record.logged_fk]
    current = [robot.data.geom_xpos[robot.face].copy(), interpolate(record.t, record.vicon, t), interpolate(record.t, record.logged_fk, t)]
    for trajectory, pos, color in zip(trajectories, current, COLORS):
        indices = np.unique(np.linspace(0, len(trajectory)-1, min(100, len(trajectory))).astype(int))
        for a, b in zip(indices[:-1], indices[1:]):
            line(trajectory[a], trajectory[b], color)
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_SPHERE, np.full(3, .018), pos, np.eye(3).ravel(), np.array(color, np.float32))
            scene.ngeom += 1
    line(current[0], current[1], [1, 1, 1, 1], .004)
    return float(np.linalg.norm(current[0]-current[1]))


def set_camera(camera, record):
    centre = record.base[:, :2].mean(axis=0)
    camera.lookat[:] = [centre[0] + .2, centre[1], .9]
    camera.distance = 3.5
    camera.azimuth = 135
    camera.elevation = -20


def update_pose(robot, record, t):
    robot.pose(interpolate(record.t, record.base, t), interpolate(record.t, record.q, t), t)


def view(robot, records, points, report, group, speed):
    from view_x4_native_replay import desktop_viewer
    keys = queue.SimpleQueue()
    paused, cursor, last = False, 0., time.monotonic()
    with desktop_viewer(robot.model, robot.data, key_callback=keys.put,
                        show_left_ui=False, show_right_ui=False) as viewer:
        set_camera(viewer.cam, records[group])
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
                elif ord('1') <= key < ord('1')+len(records):
                    group, cursor = key-ord('1'), 0
                    set_camera(viewer.cam, records[group])
            r = records[group]
            if cursor > r.t[-1] + .2:
                cursor = 0
            t = min(cursor, r.t[-1])
            with viewer.lock():
                update_pose(robot, r, t)
                viewer.user_scn.ngeom = 0
                error = markers(viewer.user_scn, robot, r, points[group], t)
            rows = [
                ('Measured state replay', f'Group {group+1}: {r.path.name}'),
                ('Local time / speed', f'{t:.3f} / {r.t[-1]:.3f}s | {speed:g}x | {"paused" if paused else "looping"}'),
                ('Centre error / RMSE', f'{error*1000:.1f} / {report["groups"][group]["model_vs_vicon"]["rmse_m"]*1000:.1f} mm'),
                ('Markers and full paths', 'Cyan: model | Magenta: Vicon | Yellow: logged FK'),
                ('Mapping', 'See mapping.json; no fitted time shift or translation'),
                ('Limits', 'Pose only; J5 direction unverified; wheel states unavailable'),
                ('Controls', f'Space: pause | R: restart | 1..{len(records)}: select | mouse: orbit/zoom'),
            ]
            viewer.set_texts((robot.mj.mjtFont.mjFONT_NORMAL, robot.mj.mjtGridPos.mjGRID_TOPLEFT,
                              '\n'.join(a for a, _ in rows), '\n'.join(b for _, b in rows)))
            viewer.sync()
            time.sleep(1/60)


def render_video(robot, record, points, destination, speed):
    """Render the identical pose/overlay path to MP4 without a desktop window."""
    mj = robot.mj
    camera = mj.MjvCamera()
    set_camera(camera, record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.with_suffix('.png').exists():
        raise FileExistsError('video or same-stem PNG already exists; choose a new path')
    width, height, fps = 1280, 720, 30
    robot.model.vis.global_.offwidth = width
    robot.model.vis.global_.offheight = height
    legend = ("drawtext=text='Measured pose replay - no physics':x=20:y=20:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65,"
              "drawtext=text='Cyan = model   Magenta = Vicon   Yellow = logged FK':x=20:y=54:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65")
    command = ['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-vf', legend, '-pix_fmt', 'yuv420p', str(destination)]
    with mj.Renderer(robot.model, height=height, width=width) as renderer:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in range(math.ceil(record.t[-1]/speed*fps)+1):
                t = min(frame/fps*speed, record.t[-1])
                update_pose(robot, record, t)
                renderer.update_scene(robot.data, camera=camera)
                markers(renderer.scene, robot, record, points, t)
                pixels = renderer.render()
                if frame == 0:
                    # PNG via the already installed ffmpeg; no image-library dependency.
                    subprocess.run(['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                    '-s', f'{width}x{height}', '-i', '-', '-vf', legend, '-frames:v', '1', str(destination.with_suffix('.png'))],
                                   input=pixels.tobytes(), check=True)
                process.stdin.write(pixels.tobytes())
        finally:
            process.stdin.close()
            result = process.wait()
        if result:
            raise RuntimeError(f'ffmpeg exited {result}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--mapping', type=Path, help='override the full mapping.json emitted by a previous run')
    parser.add_argument('--group', type=int, default=1)
    parser.add_argument('--speed', type=float, default=.25)
    parser.add_argument('--headless', action='store_true', help='write analysis without opening the viewer')
    parser.add_argument('--video', type=Path, help='render the selected group to MP4 (also saves its first frame as PNG)')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/measured_robot_replay' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    if not args.headless and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('no desktop display; use --headless, optionally with --video')
    os.environ['MUJOCO_GL'] = 'egl' if args.headless else 'glfw'
    paths = sorted((args.data_dir / '下位机轨迹').glob('*.csv'))
    if not paths or len(paths) > 9 or not 1 <= args.group <= len(paths):
        parser.error('need 1..9 CSV files in 下位机轨迹 and a valid --group')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be a new or empty directory')
    if args.video and (args.video.suffix.lower() != '.mp4' or args.video.exists() or args.video.with_suffix('.png').exists()):
        parser.error('--video must be a new .mp4 path, with no existing same-stem PNG')
    mapping = json.loads(args.mapping.read_text()) if args.mapping else DEFAULT_MAPPING.copy()
    records = [read_record(path, mapping) for path in paths]
    robot = Robot(args.model.resolve())
    points, report = audit(robot, records, mapping, args.output, args.model.resolve())
    for group in report['groups']:
        print(f'Group {group["group"]}: model/Vicon RMSE={group["model_vs_vicon"]["rmse_m"]*1000:.2f} mm; '
              f'model/logged-FK RMSE={group["model_vs_logged_fk"]["rmse_m"]*1000:.3f} mm; '
              f'lag probe={group["lag_probe"].get("best_shift_ms", "n/a")} ms')
    print(f'Report: {args.output.resolve() / "report.json"}', flush=True)
    if args.video:
        render_video(robot, records[args.group-1], points[args.group-1], args.video, args.speed)
    if not args.headless:
        view(robot, records, points, report, args.group-1, args.speed)


if __name__ == '__main__':
    main()
