#!/usr/bin/env python3
"""Interactive playback of recorded X4 dynamics; never advances physics."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

# The desktop viewer needs GLFW even if offline rendering used EGL.
os.environ['MUJOCO_GL'] = 'glfw'
import mujoco
import mujoco.viewer
import numpy as np


@contextmanager
def desktop_viewer(model, data, **options):
    # launch_passive starts a daemon GUI thread. Handle.close only requests exit;
    # wait for its teardown before Python's atexit calls glfw.terminate.
    existing_threads = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, **options)
    gui_threads = set(threading.enumerate()) - existing_threads
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in gui_threads:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError('MuJoCo viewer did not finish closing')


def configure_camera(viewer, model, pose):
    # Set once so mouse orbit/zoom stays under user control.
    ball_qadr = model.jnt_qposadr[model.joint('ball_free').id]
    viewer.cam.lookat[:] = [(pose[0] + pose[ball_qadr]) / 2,
                           (pose[1] + pose[ball_qadr + 1]) / 2, 1.0]
    viewer.cam.distance = max(5.0, abs(pose[ball_qadr] - pose[0]) * 1.3)
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -15


def display_frame(viewer, model, data, frame, heading, status, controls):
    pose = np.asarray(frame['qpos'], dtype=float)
    velocity = np.asarray(frame['qvel'], dtype=float)
    if (pose.shape != (model.nq,) or velocity.shape != (model.nv,)
            or not np.isfinite(pose).all() or not np.isfinite(velocity).all()
            or not math.isfinite(frame['t'])):
        raise ValueError('recorded state does not match model')
    target = frame['target']
    remaining = target['t_hit_s'] - frame['replay_t'] if target['valid'] else None
    current = frame['phase'] == 'replay' and remaining is not None and remaining > 0
    if frame['phase'] == 'settle':
        prediction_status = 'Waiting for observations'
    elif frame['phase'] == 'stop':
        prediction_status = 'Recording ended'
    elif remaining is not None and remaining <= 0:
        prediction_status = 'Expired'
    else:
        prediction_status = 'Valid' if current else 'No valid future point'
    # t_hit_s and replay_t share the trajectory clock. Convert to simulation
    # time using the frame's actual offset, never the wall clock or view speed.
    hit_sim_time = frame['t'] - frame['replay_t'] + target['t_hit_s'] if current else None
    ball_qadr = model.jnt_qposadr[model.joint('ball_free').id]
    with viewer.lock():
        data.qpos[:] = pose
        data.qvel[:] = velocity
        data.time = float(frame['t'])
        mujoco.mj_forward(model, data)
        scene = viewer.user_scn
        scene.ngeom = 0
        markers = [(pose[ball_qadr:ball_qadr + 3], [1, 1, 0, 1], .03)]
        if current:
            markers.append((target['p_hit_m'], [1, .15, .8, .8], .05))
        for position, color, radius in markers:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.full(3, radius), np.asarray(position), np.eye(3).ravel(),
                               np.asarray(color, dtype=np.float32))
            scene.ngeom += 1
    rows = [
        (heading, 'Measured ball; no physical impact'),
        ('Simulation time / phase', f'{frame["t"]:.3f}s / {frame["phase"]}'),
        ('Status', status),
        ('MPC plan', 'executing' if frame['executing'] else 'inactive'),
        ('Trajectory time (s)', f'{frame["replay_t"]:.3f}' if frame['phase'] != 'settle' else 'Not started'),
        ('Prediction', prediction_status),
        ('Hit XYZ (world m)', '(' + ', '.join(f'{v:.3f}' for v in target['p_hit_m']) + ')' if current else '--'),
        ('Hit time (trajectory s)', f'{target["t_hit_s"]:.3f}' if current else '--'),
        ('Hit time (simulation s)', f'{hit_sim_time:.3f}' if current else '--'),
        ('Remaining (simulation s)', f'{remaining:.3f}' if current else '--'),
        ('Markers', 'Yellow: ball; pink: predicted hit'),
        ('Controls', controls),
    ]
    viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                      '\n'.join(label for label, _ in rows), '\n'.join(value for _, value in rows)))
    viewer.sync()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--speed', type=float, default=0.25)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('a graphical desktop is required (DISPLAY or WAYLAND_DISPLAY)')
    run = args.run.resolve()
    summary = json.loads((run / 'summary.json').read_text())
    frames = [json.loads(line) for line in (run / 'frames.jsonl').read_text().splitlines() if line.strip()]
    if not frames:
        raise ValueError('frames.jsonl is empty')
    model = mujoco.MjModel.from_xml_path(summary['scene'])
    data = mujoco.MjData(model)
    times = np.asarray([frame['t'] for frame in frames], dtype=float)
    poses = np.asarray([frame['qpos'] for frame in frames], dtype=float)
    velocities = np.asarray([frame['qvel'] for frame in frames], dtype=float)
    if (poses.shape != (len(frames), model.nq) or velocities.shape != (len(frames), model.nv)
            or not np.isfinite(times).all() or not np.isfinite(poses).all()
            or not np.isfinite(velocities).all() or (np.diff(times) <= 0).any()):
        raise ValueError('recorded states or timestamps do not match the model')
    keys = queue.SimpleQueue()
    paused = False
    cursor = float(times[0])
    last_wall = time.monotonic()
    with desktop_viewer(model, data, key_callback=keys.put,
                        show_left_ui=False, show_right_ui=False) as viewer:
        configure_camera(viewer, model, poses[0])
        print(f'Viewer opened: {run}\nRecorded simulation, looping at {args.speed:g}x. '
              'Space: pause/resume; R: restart; mouse: orbit/zoom; close window to exit.', flush=True)
        while viewer.is_running():
            now = time.monotonic()
            if not paused:
                cursor += (now - last_wall) * args.speed
            last_wall = now
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    paused = not paused
                elif key in (ord('R'), ord('r')):
                    cursor = float(times[0])
            if cursor > times[-1] + 0.15:
                cursor = float(times[0])
            index = max(0, min(len(frames) - 1, int(np.searchsorted(times, cursor, side='right') - 1)))
            frame = frames[index]
            display_frame(viewer, model, data, frame, 'X4 recorded simulation',
                          f'{args.speed:g}x / {"paused" if paused else "looping"}',
                          'Space: pause | R: restart')
            time.sleep(1 / 60)


if __name__ == '__main__':
    main()
