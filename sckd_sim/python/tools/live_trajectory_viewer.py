#!/usr/bin/env python3
"""Live MuJoCo view of trajectory replay, online prediction and robot swing."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import mujoco
import mujoco.viewer
import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = SIM_ROOT / "models/generated/x4_tennis_demo.mjcf"
DEFAULT_TRAJECTORY = SIM_ROOT / "examples/synthetic_trajectory.csv"
DEFAULT_BINARY = SIM_ROOT / "build/m2_intercept"

COLORS = {
    "history": np.array([.05, .90, .95, .92], np.float32),
    "forecast": np.array([1.00, .48, .06, 1.00], np.float32),
    "target": np.array([1.00, .18, .78, 1.00], np.float32),
    "ball": np.array([1.00, 1.00, .05, 1.00], np.float32),
}


class JsonlTail:
    """Incremental reader that never parses a writer's incomplete final line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""

    def read(self):
        if not self.path.exists():
            return []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        self.pending += chunk
        lines = self.pending.split(b"\n")
        self.pending = lines.pop()
        rows = []
        for line in lines:
            if line:
                rows.append(json.loads(line))
        return rows


def runtime_config(work: Path, trajectory: Path, trace: Path, strategy: str):
    return {
        "schema": "SCKD_RUNTIME_CONFIG/1",
        "path_base": "process_cwd",
        "values": {
            "scene_path": "models/generated/x4_tennis_demo.mjcf",
            "trials": 1,
            "swing_mode": "qp",
            "init_delay_s": .25,
            "seed": 2026,
            "base_mode": strategy,
            "serve_family": "measured",
            "fly_mode": "xb",
            "measured_replay_path": str(trajectory),
            "video_trace_path": str(trace),
            "viewer_enabled": True,
            "viewer_state_path": str(work / "viewer_state.bin"),
            "viewer_lock_path": str(work / "viewer_state.lock"),
        },
    }


def _path(scene, points, color, radius, *, limit=170):
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) < 2:
        return
    indices = np.unique(np.linspace(0, len(points)-1, min(len(points), limit+1)).astype(int))
    for left, right in zip(indices[:-1], indices[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        if np.linalg.norm(points[right] - points[left]) < 1e-9:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                             radius, points[left], points[right])
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1


def _sphere(scene, position, color, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, radius), np.asarray(position, float),
                        np.eye(3).ravel(), color)
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def _cross(scene, position, color, half=.14, radius=.018):
    point = np.asarray(position, float)
    for axis in (0, 2):
        offset = np.zeros(3)
        offset[axis] = half
        _path(scene, [point-offset, point+offset], color, radius, limit=1)


def populate_scene(scene, frame, history):
    """Presentation-only decoration; does not mutate model, data or controller."""
    scene.ngeom = 0
    _path(scene, history, COLORS["history"], .012, limit=130)
    forecast = frame.get("forecast") or {}
    future = [point[1:4] for point in forecast.get("grid", [])
              if len(point) >= 4 and point[0] >= frame["t"]-1e-12 and point[3] >= .025]
    _path(scene, future, COLORS["forecast"], .018, limit=150)
    if frame.get("target") is not None:
        _cross(scene, frame["target"], COLORS["target"])
    _sphere(scene, frame["ball"], COLORS["ball"], .065)


def apply_frame(viewer, model, data, frame, history):
    qpos, qvel = np.asarray(frame["qpos"], float), np.asarray(frame["qvel"], float)
    if qpos.shape != (model.nq,) or qvel.shape != (model.nv,):
        raise ValueError("trace qpos/qvel dimensions do not match the MuJoCo model")
    with viewer.lock():
        data.time = float(frame["t"])
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        populate_scene(viewer.user_scn, frame, history)
    forecast = frame.get("forecast") or {}
    status = "planning" if frame.get("target") is None else "swing plan active"
    viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL,
                      mujoco.mjtGridPos.mjGRID_TOPLEFT,
                      "LIVE TRAJECTORY INTERCEPTION\nTime\nStatus\nCyan\nOrange\nMagenta\nData",
                      f"\n{frame['t']:.3f} s\n{status}\nreceived history\nEKF future ({len(forecast.get('grid', []))} points)"
                      "\nplanned hit point\nCSV replay; no physical hit impulse"))
    viewer.sync()


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _log_tail(path: Path, limit=4000):
    try:
        text = path.read_text(errors="replace")
    except OSError as error:
        return f"<unable to read child log: {error}>"
    return text[-limit:].strip() or "<child log is empty>"


def _round_failure(message: str, log_path: Path):
    return RuntimeError(f"{message}\n--- child log tail ---\n{_log_tail(log_path)}")


def _clean_round_files(work: Path, *paths: Path):
    for path in (*paths, work / "viewer_state.bin", work / "viewer_state.lock"):
        path.unlink(missing_ok=True)


def run_round(viewer, model, data, args, work: Path, number: int):
    trace = work / f"round-{number}.jsonl"
    config_path = work / f"round-{number}.json"
    log_path = work / f"round-{number}.log"
    config_path.write_text(json.dumps(runtime_config(work, args.trajectory, trace, args.strategy), indent=2)+"\n")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("M2_", "BALL_", "N6_", "NMPC6_"))}
    command = [str(args.binary), "--config", str(config_path)]
    reader = JsonlTail(trace)
    history = deque(maxlen=500)
    frames = rendered_frames = prediction_frames = target_frames = 0
    live_announced = False
    process = None
    try:
        log = log_path.open("w")
        process = subprocess.Popen(command, cwd=SIM_ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        print(f"round {number}: child_pid={process.pid} trace={trace}", flush=True)
        while viewer.is_running():
            new_rows = reader.read()
            for frame in new_rows:
                frames += 1
                prediction_frames += int(frame.get("forecast") is not None)
                target_frames += int(frame.get("target") is not None)
                history.append(frame["ball"])
                if (not live_announced and frame.get("forecast") is not None
                        and frame.get("target") is not None):
                    print(f"live data received while process_running={process.poll() is None} "
                          f"t={frame['t']:.3f}", flush=True)
                    live_announced = True
            if new_rows:
                apply_frame(viewer, model, data, new_rows[-1], history)
                rendered_frames += 1
            if process.poll() is not None and not new_rows:
                break
            time.sleep(.004)
        code = process.poll()
        viewer_closed = not viewer.is_running() and code is None
        if code is None:
            stop_process(process)
            code = process.returncode
        if viewer_closed:
            print(f"round {number}: viewer closed; child_exit={code}", flush=True)
            return False
        if code not in (0, 1):
            raise _round_failure(f"m2_intercept exited with {code}", log_path)
        if frames == 0 or prediction_frames == 0 or target_frames == 0:
            raise _round_failure(
                f"incomplete live trace: frames={frames}, predictions={prediction_frames}, targets={target_frames}",
                log_path)
        print(f"round {number}: frames={frames} rendered_frames={rendered_frames} "
              f"prediction_frames={prediction_frames} "
              f"target_frames={target_frames} child_exit={code}", flush=True)
        log.close()
        _clean_round_files(work, trace, config_path, log_path)
        return True
    finally:
        stop_process(process)
        if 'log' in locals():
            log.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Live MuJoCo visualization of CSV observation, EKF forecast, hit point and robot swing.")
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--strategy", choices=("pfly", "nmpc6", "wbmpc"), default="pfly")
    parser.add_argument("--once", action="store_true",
                        help="run once and hold the final frame until the window closes")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.trajectory = args.trajectory.resolve(strict=True)
    args.binary = args.binary.resolve(strict=True)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("A graphical desktop is required (DISPLAY or WAYLAND_DISPLAY is unset).")
    if args.trajectory == DEFAULT_TRAJECTORY.resolve():
        print("data source: bundled synthetic trajectory CSV", flush=True)
    else:
        print(f"data source: CSV replay {args.trajectory}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    data = mujoco.MjData(model)
    with tempfile.TemporaryDirectory(prefix="sckd-live-") as temporary:
        with mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.lookat[:] = [1.0, -.5, 1.5]
            viewer.cam.distance = 8.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -18.0
            viewer.sync()
            round_number = 0
            while viewer.is_running():
                round_number += 1
                run_round(viewer, model, data, args, Path(temporary), round_number)
                if args.once:
                    while viewer.is_running():
                        time.sleep(.050)
                    break
                time.sleep(.35)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
