#!/usr/bin/env python3
"""Open a window first, then show X4 physics states as the native process runs."""
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import time

from run_x4_native_replay import native_environment, stop_process
from view_x4_native_replay import configure_camera, desktop_viewer, display_frame, mujoco


class FrameTail:
    """Read only newly flushed complete JSONL records, retaining a partial line."""
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.pending = b''

    def read(self):
        if not self.path.exists():
            return []
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            self.pending += stream.read()
            self.offset = stream.tell()
        lines = self.pending.split(b'\n')
        self.pending = lines.pop()
        return [json.loads(line) for line in lines if line.strip()]


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--speed', type=float, default=0.25, help='live simulation speed multiplier (default: 0.25)')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in (args.speed, args.timeout)):
        parser.error('--speed and --timeout must be finite and positive')
    config = args.config.resolve()
    run = args.run.resolve()
    cfg = json.loads(config.read_text())
    root = (config.parent / cfg['project_root']).resolve()
    cfg['native_mpc_library'] = str((root / cfg['native_mpc_library']).resolve())
    model = mujoco.MjModel.from_xml_path(str(root / cfg['scene_path']))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:2] = cfg['initial_base_xy_m']
    variant = json.loads((root / cfg['robot_variant_path']).read_text())
    for j, q in enumerate(variant['arm_servo']['standby_q_rad'], 1):
        data.qpos[model.jnt_qposadr[model.joint(f'Joint{j}').id]] = q
    reader = FrameTail(run / 'frames.jsonl')
    stats = {'mode': 'live', 'displayed_frames_while_running': 0,
             'displayed_mpc_frames_while_running': 0, 'received_frames': 0,
             'last_displayed_time': None, 'end_reason': 'starting'}
    process = None
    try:
        with desktop_viewer(model, data, show_left_ui=False, show_right_ui=False) as viewer, \
                (run / 'run.log').open('w') as log:
            viewer.cam.lookat[:] = [*cfg['initial_base_xy_m'], 1.0]
            viewer.cam.distance = 6
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -15
            viewer.set_texts((None, None, 'X4 LIVE SIMULATION', 'Starting controller and predictor...'))
            viewer.sync()
            print('Live window ready; starting native simulation now.', flush=True)
            process = subprocess.Popen([str(args.binary.resolve()), str(config), str(run),
                                        args.socket, str(args.speed)], cwd=run,
                                       env=native_environment(cfg), stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            frame = None
            finished = False
            while viewer.is_running():
                code = process.poll()
                # If the child has exited, this read also drains its final flush.
                incoming = reader.read()
                stats['received_frames'] += len(incoming)
                if incoming:
                    if frame is None:
                        configure_camera(viewer, model, incoming[0]['qpos'])
                    frame = incoming[-1]
                    live = code is None
                    display_frame(viewer, model, data, frame, 'X4 LIVE SIMULATION',
                                  f'Running at {args.speed:g}x' if live else 'Finished - final state',
                                  'Mouse: orbit/zoom | Close: stop/exit')
                    stats['last_displayed_time'] = frame['t']
                    if live:
                        stats['displayed_frames_while_running'] += 1
                        if stats['displayed_frames_while_running'] == 1:
                            print(f'Live state displayed while simulation_running=True, t={frame["t"]:.3f}s', flush=True)
                        if frame['executing']:
                            stats['displayed_mpc_frames_while_running'] += 1
                            if stats['displayed_mpc_frames_while_running'] == 1:
                                print(f'MPC motion displayed while simulation_running=True, t={frame["t"]:.3f}s', flush=True)
                if code is not None and not finished:
                    if code != 0:
                        raise RuntimeError(f'Native simulation exited {code}; see {run / "run.log"}')
                    if frame is None:
                        raise RuntimeError('Native simulation produced no display frames')
                    finished = True
                    stats['end_reason'] = 'completed'
                    display_frame(viewer, model, data, frame, 'X4 LIVE SIMULATION',
                                  'Finished - final state', 'Mouse: orbit/zoom | Close: exit')
                    (run / 'live_view.json').write_text(json.dumps(stats, indent=2) + '\n')
                    print(f'Simulation finished. Window holds final state; no replay. {run}', flush=True)
                if not finished and time.monotonic() - started > args.timeout:
                    raise TimeoutError(f'Simulation exceeded {args.timeout:g}s; see {run / "run.log"}')
                if not incoming:
                    viewer.sync()
                time.sleep(1 / 120)
            if not finished:
                stats['end_reason'] = 'window_closed_before_completion'
    except BaseException:
        stats['end_reason'] = 'error_or_interrupt'
        raise
    finally:
        stop_process(process)
        (run / 'live_view.json').write_text(json.dumps(stats, indent=2) + '\n')


if __name__ == '__main__':
    main()
