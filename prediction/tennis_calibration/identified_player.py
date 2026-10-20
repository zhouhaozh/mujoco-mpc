"""Native MuJoCo playback of the frozen identified predictor on Ubuntu.

Yellow is a free-running real MuJoCo ball initialized from the first causal
80-ms estimate. It is never overwritten by subsequent measurements. Blue is
the independent online EKF estimate; orange is its repeatedly updated future
forecast. Gray reference and green received observations are rendering-only.
The first impulse is the shared explicit event map, not native solver contact.
"""
from __future__ import annotations

import argparse
import faulthandler
from collections import deque
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import signal
import time

import numpy as np

from .identified_filter import IdentifiedFilter, forecast_states_batch
from .player_bounce import BounceModelBank, SelectableCollisionBall
from .online_data import load_dataset
from .player_scene import draw_environment
from .player_session import PlayerSession


class Replay:
    """Single raw stream; offline labels only determine display/task end."""
    def __init__(self, trace, selected, simulation):
        self.trace, self.selected, self.sim = trace, selected, simulation
        self.times = np.asarray(trace["raw_times"], float)
        self.positions = np.asarray(trace["raw_xyz"], float)
        self.origin = float(self.times[0])
        scope = trace.get("eval_scope_end_s")
        self.duration = float(min(self.times[-1], scope if scope is not None else self.times[-1])-self.origin)
        self.end_index = int(np.searchsorted(self.times, self.origin+self.duration, side="right"))
        self.tracker = IdentifiedFilter(selected["parameters"], selected["contact_height"],
                                        selected["filter"], selected.get("augmented", True))
        self.cursor = 0
        self.elapsed = 0.0
        self.initialized = False
        self.snapshot = None
        self.forecast = np.empty((0, 3))
        self.forecast_t = None
        self.last_forecast_elapsed = -np.inf
        self.filter_trail = []
        self.audit = []
        self.forecast_updates = 0
        self.sim.mujoco.mj_resetData(self.sim.model, self.sim.data)
        self.sim.data.qpos[:3] = self.positions[0]
        self.sim.data.qpos[3:7] = [1., 0., 0., 0.]
        self.sim.data.qvel[:] = 0
        self.sim.bounced = False
        self.sim.events.clear()
        self.sim.callback_calls = self.sim.engine_steps = 0
        self.sim.mujoco.mj_forward(self.sim.model, self.sim.data)

    def advance(self, elapsed):
        self.elapsed = min(max(float(elapsed), self.elapsed), self.duration)
        while self.cursor < self.end_index and self.times[self.cursor] <= self.origin+self.elapsed+1e-12:
            capture = float(self.times[self.cursor])
            snapshot = self.tracker.update(capture, self.positions[self.cursor])
            self.snapshot = snapshot
            self.cursor += 1
            if not snapshot["initialized"]:
                continue
            if not self.initialized:
                # One causal initial condition; no later qpos/qvel correction.
                initial = np.asarray(snapshot["state"], float)
                if len(initial) == 6:
                    initial = np.r_[initial, np.zeros(3)]
                self.sim.data.time = capture-self.origin
                self.sim.data.qpos[:3] = initial[:3]
                self.sim._set_velocity_and_b(initial[3:6], initial[6:9])
                self.sim.mujoco.mj_forward(self.sim.model, self.sim.data)
                self.initialized = True
            if capture-self.origin-self.last_forecast_elapsed >= .020-1e-12:
                future = forecast_states_batch(np.asarray(snapshot["state"])[None],
                    np.asarray([snapshot["bounced"]]), np.linspace(0, .5, 51),
                    self.selected["parameters"], self.selected["contact_height"], self.selected["filter"])
                self.forecast = future[0, :, :3]
                self.forecast_t = capture
                self.last_forecast_elapsed = capture-self.origin
                self.forecast_updates += 1
                self.filter_trail.append(np.asarray(snapshot["state"][:3]).tolist())
        if self.initialized:
            self.sim.advance_to(self.elapsed)
            state = self.sim.state()
            self.audit.append({"elapsed_s": self.elapsed, "capture_t_s": None if self.snapshot is None else self.snapshot["t_s"],
                "raw_rows_received": self.cursor, "qpos": self.sim.data.qpos.tolist(),
                "qvel": self.sim.data.qvel.tolist(), "simulation_state": state.tolist(),
                "filter_state": np.asarray(self.snapshot["state"]).tolist(),
                "simulation_bounced": self.sim.bounced, "filter_bounced": bool(self.snapshot["bounced"]),
                "forecast_capture_t_s": self.forecast_t, "forecast_updates": self.forecast_updates})


def _path_geoms(mujoco, scene, positions, rgba, *, limit=280, radius=.007):
    points = np.asarray(positions, float).reshape(-1, 3)
    if len(points) < 2:
        return
    indices = np.unique(np.linspace(0, len(points)-1, min(len(points), limit+1)).astype(int))
    for left, right in zip(indices[:-1], indices[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        if np.linalg.norm(points[right]-points[left]) < 1e-9:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, points[left], points[right])
        scene.ngeom += 1


def _sphere(mujoco, scene, position, rgba, radius=.055):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius]*3),
                      np.asarray(position), np.eye(3).ravel(), np.asarray(rgba))
    scene.ngeom += 1


def _draw(viewer, session):
    replay, controls = session.replay, session.controls
    mujoco = replay.sim.mujoco
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        session.style['court_visible'] = controls['visible']['court']
        draw_environment(mujoco, scene, session.style)
        replay.sim.model.geom_rgba[replay.sim.model.geom('ball_visual').id, 3] = float(controls['visible']['ball'])
        mask = replay.trace["eval_times"] <= replay.origin+replay.duration
        reference = replay.trace["eval_xyz"][mask]
        if controls['visible']['reference']:
            _path_geoms(mujoco, scene, reference, [.90, .92, .94, .80], limit=220, radius=.010)
        # No offline keep mask is applied to the received green stream.
        if controls['visible']['received']:
            _path_geoms(mujoco, scene, replay.positions[:replay.cursor], [.20, .94, .43, .85], limit=220, radius=.010)
        if controls['visible']['ball']:
            _path_geoms(mujoco, scene, [r['qpos'][:3] for r in replay.audit], [.98, .91, .15, .75], limit=90, radius=.009)
            _sphere(mujoco, scene, replay.sim.data.qpos[:3], [.96, .99, .15, 1.], radius=.070)
        if controls['visible']['estimate']:
            _path_geoms(mujoco, scene, replay.filter_trail, [.22, .70, 1., .98], limit=130, radius=.010)
            if replay.snapshot and replay.snapshot["initialized"]:
                _sphere(mujoco, scene, replay.snapshot["state"][:3], [.22, .70, 1., 1.], radius=.070)
        if controls['visible']['forecast']:
            _path_geoms(mujoco, scene, replay.forecast, [1., .48, .06, 1.], limit=51, radius=.016)
        if controls['view_mode'] == 'bounce' and controls['bounce_compare']:
            _path_geoms(mujoco, scene, getattr(replay, 'comparison', []), [.83, .44, 1., .95], limit=71, radius=.012)
    # set_texts takes the native viewer mutex itself. Calling it under
    # viewer.lock() deadlocks on the non-recursive mutex in MuJoCo 3.13.
    if time.perf_counter()-controls.get("last_text_wall", -np.inf) >= .100:
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "TENNIS TRAJECTORY REPLAY\nSample\nTime\nPlayback\nObservation\nYellow / Blue\nGray / Green / Orange",
            f"\n{session.pool_ids.index(session.trajectory_id)+1 if session.trajectory_id in session.pool_ids else '-'} / {len(session.pool_ids)} | {replay.trace['split'].upper()}"+
            f"\n{replay.elapsed:.3f} / {replay.duration:.3f} s"+
            f"\n{'PAUSED' if controls['paused'] else 'PLAYING'}  {controls['speed']:.2f}x"+
            (f"\nFrozen precontact | {controls['bounce_model'].upper()} | gap {controls['bounce_gap_s']*1000:.0f} ms"+
             "\nSelected simulation / selected prediction\nReference / frozen input / forecast; purple: fixed"
             if controls['view_mode'] == 'bounce' else
             "\nOnline rolling prediction\nFree simulation / current estimate\nReference / received / prediction")))
        controls["last_text_wall"] = time.perf_counter()
    viewer.sync()


def _camera(viewer, session):
    camera = session.style['camera']
    preset = session.controls['camera']
    with viewer.lock():
        viewer.cam.lookat[:] = camera['lookat']
        viewer.cam.distance = camera['distance']
        viewer.cam.azimuth = camera['azimuth']
        viewer.cam.elevation = camera['elevation']
        if preset == 'top':
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 90
            viewer.cam.distance *= 1.1
        elif preset == 'side':
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -8
        elif preset == 'follow':
            point = (session.replay.snapshot['state'][:3]
                     if session.replay.snapshot and session.replay.snapshot['initialized']
                     else session.replay.sim.data.qpos[:3])
            viewer.cam.lookat[:] = point
            viewer.cam.distance = 6.
            viewer.cam.elevation = -20
    session.camera_dirty = False


def _save(directory, replay, selected, started, headless):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"state_trace.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in replay.audit))
    movement = float(np.ptp(np.asarray([row["qpos"][:3] for row in replay.audit]), axis=0).max()) if replay.audit else 0.
    engine = getattr(replay, 'simulation_evidence', {
        'callback_calls': replay.sim.callback_calls,
        'engine_steps': replay.sim.engine_steps, 'events': replay.sim.events})
    summary = {"schema": "TENNIS_MUJOCO_NATIVE_PLAYER/1", "trajectory_id": replay.trace["id"],
        "selected_name": selected["selected_name"], "mode": "free_running_MuJoCo_plus_independent_online_EKF",
        "engine": replay.sim.mujoco.__version__, "platform": platform.platform(), "headless": headless,
        "wall_duration_s": time.perf_counter()-started, "replay_elapsed_s": replay.elapsed,
        "capture_time_origin_s": replay.origin, "future_horizon_s": .5,
        "forecast_capture_stride_s": .020, "display_frame_target_s": .033,
        "text_overlay_target_s": .100,
        "raw_rows_received": replay.cursor, "forecast_updates": replay.forecast_updates,
        "state_trace_rows": len(replay.audit), "qpos_motion_range_max_axis_m": movement,
        "callback_calls": engine['callback_calls'], "engine_steps": engine['engine_steps'],
        "events": engine['events'],
        "assertions": {"actual_qpos_moved": movement > .1, "future_prediction_repeatedly_updated": replay.forecast_updates > 1},
        "boundaries": ["raw observations only enter EKF", "gray reference is display only",
            "yellow initialized causally once then advances by MuJoCo", "first explicit impulse only; no native contact",
            "selected predictor remains independent of free-run simulator", "no robot model or gains loaded"]}
    if hasattr(replay, 'info'):
        summary.update(mode='frozen_precontact_bounce_MuJoCo', bounce=replay.info(),
                       collision_model_id=replay.model_id, forecast_updates=replay.forecast_updates,
                       selected_name='bounce_'+replay.model_id,
                       flight_predictor_name=selected['selected_name'],
                       forecast_capture_stride_s=None,
                       assertions={
                           'actual_qpos_moved': movement > .1,
                           'frozen_prediction_computed_once': replay.forecast_updates == 1,
                           'input_count_constant': bool(replay.audit) and all(row['raw_rows_received']==replay.cursor for row in replay.audit),
                           'real_engine_applied_selected_model': bool(engine['events']) and all(event.get('collision_model_id')==replay.model_id for event in engine['events'])},
                       boundaries=['only precontact observations fit the frozen initial state',
                                   'reference event defines UI cutoff and display end only',
                                   'each collision model uses identical prefix and absolute clocks',
                                   'real MuJoCo incident state drives the selected collision map',
                                   'postcontact observations are display-only',
                                   'first explicit impulse only; native contacts disabled'])
    (directory/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(directory), "summary": summary}, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", default="artifacts/tennis_calibration/20261008_calibration")
    parser.add_argument("--model", default="artifacts/tennis_identified/20261009_training_prior/selected_model.json")
    parser.add_argument("--collision-models", default="artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json")
    parser.add_argument("--view-mode", choices=['online', 'bounce'], default='online')
    parser.add_argument("--bounce-model", choices=['fixed', 'ridge1', 'ridge2', 'gp'], default='gp')
    parser.add_argument("--bounce-gap-ms", type=int, choices=[0,5,10,20,40,80], default=40)
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    parser.add_argument("--trajectory", help="Full identity or numeric list index")
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Engine/state smoke verification without a window")
    parser.add_argument("--no-panel", action="store_true", help="Keyboard-only native viewer")
    parser.add_argument("--max-seconds", type=float, default=0., help="Wall-clock run limit, zero = interactive")
    parser.add_argument("--output", default="artifacts/tennis_identified/20261009_native_player")
    args = parser.parse_args(argv)
    if platform.system() != "Linux":
        parser.error("Formal simulation and GUI playback run on Ubuntu only")
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if not np.isfinite(args.speed) or not .1 <= args.speed <= 4. or args.max_seconds < 0:
        parser.error("Speed must be 0.1 to 4.0; wall-clock limit must be nonnegative")
    selected = json.loads(Path(args.model).read_text())
    bank = BounceModelBank(args.collision_models)
    for key in ('k_drag','mass_kg','radius_m','inertia_factor','e_n','alpha_t'):
        if not np.isclose(selected['parameters'][key], bank.parameters[key], rtol=1e-10, atol=1e-12):
            parser.error('Collision checkpoints and selected flight model must share physical parameters')
    if not np.isclose(selected['contact_height'], bank.height, rtol=1e-10, atol=1e-12):
        parser.error('Collision checkpoints and flight model must share the event plane')
    traces = [trace for trace in load_dataset(args.calibration_dir)
              if trace.get("eval_scope_end_s") is not None
              and trace["eval_scope_end_s"] > trace["raw_times"][0]+.1]
    pool = [t for t in traces if args.split == 'all' or t['split'] == args.split]
    if not pool:
        parser.error("No trajectories with a first-bounce task interval")
    index = 0
    if args.trajectory:
        if args.trajectory.isdigit():
            index = int(args.trajectory) % len(pool)
        else:
            identities = [trace["id"] for trace in pool]
            if args.trajectory not in identities:
                parser.error("Trajectory identity not in selected split")
            index = identities.index(args.trajectory)
    commands = deque()
    def key_callback(key):
        commands.append(key)
    started = time.perf_counter()
    initial = np.r_[pool[index]["raw_xyz"][0], np.zeros(6)]
    sim = SelectableCollisionBall(initial, selected["parameters"], selected["contact_height"], timestep_s=.001)
    with sim:
        session = PlayerSession(sim, traces, selected, pool[index]['id'], [t['id'] for t in pool], speed=args.speed, loop=args.loop, bounce_bank=bank)
        session.controls['selection_split'] = args.split
        session.controls['bounce_model'] = args.bounce_model
        session.controls['bounce_gap_s'] = args.bounce_gap_ms/1000
        if args.view_mode == 'bounce':
            session.dispatch({'action':'view_mode','value':'bounce'})
        panel = None
        if args.headless:
            context = nullcontext(None)
        else:
            import mujoco.viewer
            context = mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback,
                                                   show_left_ui=False, show_right_ui=False)
        with context as viewer:
            if viewer:
                _camera(viewer, session)
                if not args.no_panel:
                    from .player_controls import ControlsWindow
                    panel = ControlsWindow(traces, commands.append, session.state())
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            last_state_wall = -np.inf
            previous_wall = time.perf_counter()
            try:
                while viewer is None or viewer.is_running():
                    if args.headless and not getattr(session.replay, 'valid', True):
                        break
                    now_wall = time.perf_counter()
                    if args.max_seconds and now_wall-started >= args.max_seconds:
                        break
                    delta_wall, previous_wall = now_wall-previous_wall, now_wall
                    handled = False
                    while commands:
                        command = commands.popleft()
                        with viewer.lock() if viewer else nullcontext():
                            if isinstance(command, dict):
                                session.dispatch(command)
                            else:
                                session.key(command)
                        handled = True
                        with (output/'interactions.jsonl').open('a') as stream:
                            stream.write(json.dumps({'command': command, 'state': session.state()}, ensure_ascii=False)+'\n')
                    if session.closed:
                        break
                    if panel and not panel.poll(session.state()):
                        break
                    if not session.controls['paused'] and not handled:
                        before = session.replay
                        if args.headless:
                            session.replay.advance(session.replay.elapsed+.020)
                        else:
                            with viewer.lock():
                                session.advance(min(.1, delta_wall))
                        if before is not session.replay:
                            _save(output, before, selected, started, args.headless)
                    if viewer:
                        if session.camera_dirty or session.controls['camera'] == 'follow':
                            _camera(viewer, session)
                        _draw(viewer, session)
                    if args.headless and session.replay.elapsed >= session.replay.duration-1e-9:
                        break
                    if now_wall-last_state_wall >= .100:
                        state = session.state()
                        state['process_id'] = os.getpid()
                        if panel:
                            def widget_tree(widget):
                                result=[]
                                for child in widget.winfo_children():
                                    if child.winfo_ismapped():
                                        item={'name':str(child), 'class':child.winfo_class(),
                                              'bounds':[child.winfo_rootx(),child.winfo_rooty(),child.winfo_width(),child.winfo_height()]}
                                        try: item['text']=str(child.cget('text'))
                                        except Exception: pass
                                        result.append(item); result.extend(widget_tree(child))
                                return result
                            state['control_widgets'] = widget_tree(panel.root)
                        temp=output/'player_state.tmp'
                        temp.write_text(json.dumps(state, ensure_ascii=False))
                        temp.replace(output/'player_state.json')
                        last_state_wall=now_wall
                    if viewer:
                        time.sleep(max(0., .033-(time.perf_counter()-now_wall)))
            finally:
                if panel:
                    panel.close()
                _save(output, session.replay, selected, started, args.headless)
    if args.headless and not getattr(session.replay, 'valid', True):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
