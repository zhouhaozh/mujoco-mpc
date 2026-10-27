import json
from contextlib import nullcontext
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sckd_sim/python/tools"))
import live_trajectory_viewer as live


class LiveTrajectoryViewerTests(unittest.TestCase):
    class FinishedProcess:
        pid = 12345

        def __init__(self, command, *, cwd, env, stdout, stderr, code=0, row=None,
                     rows=None, log=""):
            self.returncode = code
            stdout.write(log)
            stdout.flush()
            rows = rows if rows is not None else ([] if row is None else [row])
            if rows:
                config = json.loads(Path(command[-1]).read_text())
                Path(config["values"]["video_trace_path"]).write_text(
                    "".join(json.dumps(item)+"\n" for item in rows))

        def poll(self):
            return self.returncode

    def test_jsonl_tail_waits_for_complete_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            tail = live.JsonlTail(path)
            path.write_bytes(b'{"t":0}\n{"t":')
            self.assertEqual(tail.read(), [{"t": 0}])
            self.assertEqual(tail.read(), [])
            with path.open("ab") as stream:
                stream.write(b'1}\n')
            self.assertEqual(tail.read(), [{"t": 1}])

    def test_runtime_config_uses_unique_run_paths_and_live_pacing(self):
        work = Path("/tmp/live-test")
        config = live.runtime_config(work, Path("/data/trajectory.csv"),
                                     work / "trace.jsonl", "pfly")["values"]
        self.assertTrue(config["viewer_enabled"])
        self.assertEqual(config["measured_replay_path"], "/data/trajectory.csv")
        self.assertEqual(config["video_trace_path"], "/tmp/live-test/trace.jsonl")
        self.assertEqual(config["viewer_lock_path"], "/tmp/live-test/viewer_state.lock")

    def test_scene_contains_history_forecast_target_and_ball_without_model_mutation(self):
        model = mujoco.MjModel.from_xml_path(str(live.DEFAULT_MODEL))
        scene = mujoco.MjvScene(model, maxgeom=512)
        qpos0 = model.qpos0.copy()
        frame = {"t": .2, "ball": [1., 0., 1.], "target": [.4, -.2, 1.2],
                 "forecast": {"grid": [[.2, 1., 0., 1., 0., False],
                                          [.3, .7, -.1, .8, -1., False],
                                          [.4, .4, -.2, .5, -2., False]]}}
        live.populate_scene(scene, frame, [[1.2, .1, 1.1], [1., 0., 1.]])
        self.assertEqual(scene.ngeom, 6)  # history + 2 forecast + 2 target + ball
        self.assertTrue(all(scene.geoms[i].category == int(mujoco.mjtCatBit.mjCAT_DECOR)
                            for i in range(scene.ngeom)))
        np.testing.assert_array_equal(model.qpos0, qpos0)

    def test_cli_defaults_to_repeat_and_accepts_once(self):
        defaults = live.parse_args([])
        self.assertFalse(defaults.once)
        self.assertEqual(defaults.trajectory, live.SIM_ROOT / "examples/synthetic_trajectory.csv")
        self.assertTrue(defaults.trajectory.is_file())
        args = live.parse_args(["--once", "--strategy", "wbmpc"])
        self.assertTrue(args.once)
        self.assertEqual(args.strategy, "wbmpc")

    def test_stop_process_terminates_owned_child(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        live.stop_process(process)
        self.assertIsNotNone(process.returncode)

    def test_two_successful_rounds_leave_no_accumulated_files(self):
        row = {"t": .1, "forecast": {"grid": [[.1, 0, 0, 1, 0, False]]},
               "target": [0, 0, 1], "ball": [0, 0, 1], "qpos": [], "qvel": []}
        process = lambda *a, **k: self.FinishedProcess(*a, **k, row=row)
        args = type("Args", (), {"trajectory": Path("/data/trajectory.csv"),
                                  "strategy": "pfly", "binary": Path("/bin/true")})()
        viewer = type("Viewer", (), {"is_running": lambda self: True})()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(live.subprocess, "Popen", process), \
                mock.patch.object(live, "apply_frame"):
            work = Path(directory)
            self.assertTrue(live.run_round(viewer, None, None, args, work, 1))
            self.assertTrue(live.run_round(viewer, None, None, args, work, 2))
            self.assertEqual(list(work.iterdir()), [])

    def test_batch_coalesces_render_to_latest_frame_but_keeps_all_history(self):
        rows = [{"t": stamp, "forecast": {"grid": [[stamp, stamp, 0, 1, 0, False]]},
                 "target": [stamp, 0, 1], "ball": [stamp, 0, 1], "qpos": [], "qvel": []}
                for stamp in (.1, .2, .3)]
        process = lambda *a, **k: self.FinishedProcess(*a, **k, rows=rows)
        args = type("Args", (), {"trajectory": Path("/data/trajectory.csv"),
                                  "strategy": "pfly", "binary": Path("/bin/true")})()
        viewer = type("Viewer", (), {"is_running": lambda self: True})()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(live.subprocess, "Popen", process), \
                mock.patch.object(live, "apply_frame") as apply:
            live.run_round(viewer, None, None, args, Path(directory), 1)
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(apply.call_args.args[3]["t"], .3)
        self.assertEqual(list(apply.call_args.args[4]),
                         [[.1, 0, 1], [.2, 0, 1], [.3, 0, 1]])

    def test_apply_frame_sets_simulation_time_from_rendered_packet(self):
        model = mujoco.MjModel.from_xml_path(str(live.DEFAULT_MODEL))
        data = mujoco.MjData(model)
        frame = {"t": .73, "qpos": model.qpos0.tolist(),
                 "qvel": np.zeros(model.nv).tolist(), "ball": [0, 0, 1],
                 "forecast": {"grid": []}, "target": None}
        viewer = type("Viewer", (), {
            "user_scn": object(),
            "lock": lambda self: nullcontext(),
            "set_texts": mock.Mock(),
            "sync": lambda self: None,
        })()
        with mock.patch.object(live, "populate_scene"):
            live.apply_frame(viewer, model, data, frame, [])
        self.assertAlmostEqual(data.time, frame["t"])
        self.assertEqual(viewer.set_texts.call_args.args[0][0], mujoco.mjtFont.mjFONT_NORMAL)

    def test_child_failure_embeds_log_before_temporary_cleanup(self):
        process = lambda *a, **k: self.FinishedProcess(
            *a, **k, code=7, log="startup context\nfatal predictor detail\n")
        args = type("Args", (), {"trajectory": Path("/data/trajectory.csv"),
                                  "strategy": "pfly", "binary": Path("/bin/false")})()
        viewer = type("Viewer", (), {"is_running": lambda self: True})()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(live.subprocess, "Popen", process):
            with self.assertRaisesRegex(RuntimeError, "fatal predictor detail") as caught:
                live.run_round(viewer, None, None, args, Path(directory), 1)
            self.assertIn("child log tail", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
