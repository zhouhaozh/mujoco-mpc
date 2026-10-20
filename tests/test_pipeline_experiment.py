"""Ubuntu 执行的轻量 fixture 验收；不加载模型、不启动 MuJoCo 或真实预测服务。"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pipeline_experiment import (ExperimentSpec, RunAttempt, clean_environment, digest,
    file_resources, recorded_environment, resolved_runtime_config, run_logged, socket_service)
import evaluate_predictor_control
import run_control_ablation


class ExperimentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.csv"
        self.input.write_text("t,x,y,z\n0,1,2,3\n")
        self.destination = self.root / "run"
        self.spec = ExperimentSpec("fixture", (sys.executable, "fixture.py"), str(self.root),
            {"steer_rate_mode": "limited"}, {}, file_resources({"input": self.input}))

    def complete(self):
        with RunAttempt(self.destination, self.spec) as attempt:
            (self.destination / "control.log").write_text("fixture output\n")
            return attempt.complete({"hits": 1})

    def test_clean_environment_records_explicit_knobs_without_secrets(self):
        original = {"PATH": "/usr/bin", "M2_VIEW": "1", "BALL_DEDUP": "1", "N6_TRACE": "1",
                    "NMPC6_DEBUG": "1", "WB_CHASSIS": "1", "WM_DEBUG": "1", "WBO_DEBUG": "1",
                    "EXP_POS": "1.5", "SOME_TOKEN": "private"}
        env = clean_environment({"M2_STEER_RATE_MODE": "limited", "BALL_PREDICTOR_SOCKET": "/tmp/job.sock"}, original)
        self.assertNotIn("M2_VIEW", env)
        self.assertNotIn("N6_TRACE", env)
        self.assertNotIn("NMPC6_DEBUG", env)
        self.assertNotIn("BALL_DEDUP", env)
        for flag in ("WB_CHASSIS", "WM_DEBUG", "WBO_DEBUG"):
            self.assertNotIn(flag, env)
        recorded = recorded_environment(env, "/tmp/job.sock")
        self.assertEqual(recorded["BALL_PREDICTOR_SOCKET"], "{socket}")
        self.assertEqual(recorded["EXP_POS"], "1.5")
        self.assertNotIn("SOME_TOKEN", recorded)
        self.assertEqual(original["M2_VIEW"], "1")

    def test_resolved_runtime_config_accepts_legacy_and_rejects_conflicts(self):
        self.assertIsNone(resolved_runtime_config("legacy output"))
        self.assertEqual(resolved_runtime_config('[resolved config] {"seed":2026}\n'), {"seed": 2026})
        with self.assertRaises(ValueError):
            resolved_runtime_config('[resolved config] {"seed":1}\n[resolved config] {"seed":2}\n')

    def test_completed_result_reuses_only_matching_content_and_outputs(self):
        first = self.complete()
        with RunAttempt(self.destination, self.spec) as attempt:
            self.assertEqual(attempt.existing, first)
        (self.destination / "control.log").write_text("corrupted\n")
        with self.assertRaisesRegex(ValueError, "output missing/changed"):
            with RunAttempt(self.destination, self.spec):
                self.fail("corrupted output was reused")

    def test_changed_parameters_preserve_successful_result(self):
        self.complete()
        before = (self.destination / "run.json").read_bytes()
        changed = replace(self.spec, parameters={"steer_rate_mode": "both_unlimited"})
        with self.assertRaisesRegex(FileExistsError, "configuration"):
            with RunAttempt(self.destination, changed, retry_incomplete=True):
                self.fail("changed configuration was reused")
        self.assertEqual((self.destination / "run.json").read_bytes(), before)

    def test_result_metric_tampering_is_detected(self):
        result = self.complete()
        result["hits"] = 999
        (self.destination / "run.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "result fields changed"):
            with RunAttempt(self.destination, self.spec):
                self.fail("modified metrics were reused")

    def test_input_change_and_midrun_change_are_rejected(self):
        self.complete()
        self.input.write_text("different input")
        current = replace(self.spec, resources=file_resources({"input": self.input}))
        with self.assertRaises(FileExistsError):
            with RunAttempt(self.destination, current):
                self.fail("changed input was reused")
        other = self.root / "other"
        with self.assertRaisesRegex(RuntimeError, "resource changed"):
            with RunAttempt(other, current) as attempt:
                (other / "control.log").write_text("output")
                self.input.write_text("changed while running")
                attempt.complete({"hits": 1})
        self.assertFalse((other / "run.json").exists())
        self.assertEqual(json.loads((other / "run_manifest.json").read_text())["status"], "failed")

    def test_legacy_success_is_never_silently_skipped_or_overwritten(self):
        self.destination.mkdir()
        done = self.destination / "run.json"
        done.write_text('{"variant":"baseline"}\n')
        before = done.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "legacy result"):
            with RunAttempt(self.destination, self.spec, retry_incomplete=True):
                self.fail("legacy result was trusted")
        self.assertEqual(done.read_bytes(), before)

    def test_explicit_retry_archives_incomplete_attempt(self):
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with RunAttempt(self.destination, self.spec):
                (self.destination / "control.log").write_text("first attempt")
                raise RuntimeError("fixture failure")
        with self.assertRaisesRegex(FileExistsError, "incomplete run preserved"):
            with RunAttempt(self.destination, self.spec):
                self.fail("incomplete run was overwritten")
        with RunAttempt(self.destination, self.spec, retry_incomplete=True) as attempt:
            (self.destination / "control.log").write_text("second attempt")
            attempt.complete({"hits": 2})
        archives = list((self.destination / "attempts").iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "control.log").read_text(), "first attempt")
        self.assertEqual(json.loads((archives[0] / "run_manifest.json").read_text())["status"], "failed")

    def test_concurrent_attempt_cannot_archive_live_run(self):
        with RunAttempt(self.destination, self.spec) as first:
            with self.assertRaises(BlockingIOError):
                with RunAttempt(self.destination, self.spec, retry_incomplete=True):
                    self.fail("live run lock was ignored")
            (self.destination / "control.log").write_text("owned output")
            first.complete({"hits": 1})

    def test_timeout_preserves_log_and_marks_failure(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            with RunAttempt(self.destination, self.spec):
                run_logged([sys.executable, "-c", "import time; print('began',flush=True); time.sleep(5)"],
                    cwd=self.root, env=clean_environment(), log=self.destination / "control.log", timeout=.2)
        self.assertIn("began", (self.destination / "control.log").read_text())
        self.assertEqual(json.loads((self.destination / "run_manifest.json").read_text())["status"], "failed")
        self.assertFalse((self.destination / "run.json").exists())

    def test_service_failure_timeout_and_owned_socket_cleanup(self):
        path = self.root / "service.sock"
        with self.assertRaisesRegex(RuntimeError, "service exited"):
            with socket_service([sys.executable, "-c", "raise SystemExit(7)"], socket=path,
                    cwd=self.root, env=clean_environment(), log=self.root / "dead.log", startup_timeout=1):
                self.fail("dead service was ready")
        with self.assertRaises(TimeoutError):
            with socket_service([sys.executable, "-c", "import time; time.sleep(5)"], socket=path,
                    cwd=self.root, env=clean_environment(), log=self.root / "timeout.log", startup_timeout=.1):
                self.fail("service without socket was ready")
        code = "import socket,sys,time; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1]); s.listen(1); time.sleep(5)"
        with self.assertRaisesRegex(RuntimeError, "client failed"):
            with socket_service([sys.executable, "-c", code, str(path)], socket=path,
                    cwd=self.root, env=clean_environment(), log=self.root / "ready.log", startup_timeout=2) as process:
                self.assertTrue(path.exists())
                raise RuntimeError("client failed")
        self.assertIsNotNone(process.poll())
        self.assertFalse(path.exists())

    def test_service_never_removes_preexisting_socket(self):
        path = self.root / "owned.sock"
        with socket.socket(socket.AF_UNIX) as owner:
            owner.bind(str(path))
            with self.assertRaises(FileExistsError):
                with socket_service([sys.executable, "-c", "pass"], socket=path, cwd=self.root,
                        env=clean_environment(), log=self.root / "unused.log"):
                    self.fail("preexisting socket was replaced")
            self.assertTrue(path.exists())
        self.assertFalse((self.root / "unused.log").exists())


class ExistingRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runner-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.team = self.root / "team"
        for name in ("sckd_sim/build-predictor", "sckd_sim/models/config", "sckd_sim/models/generated", "sckd_sim/cpp/runtime", "tools"):
            (self.team / name).mkdir(parents=True)
        (self.team / "sckd_sim/cpp/runtime/fixture.h").write_text("// fixture runtime\n")
        (self.team / "sckd_sim/CMakeLists.txt").write_text("# fixture build\n")
        self.binary = self.team / "sckd_sim/build-predictor/m2_intercept"
        self.binary.write_text(f"#!{sys.executable}\n" + '''import json,os
from pathlib import Path
counter=Path(__file__).resolve().parents[2]/'invocations.txt'
counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')
if 'M2_CONTROL_DIAGNOSTICS' in os.environ:
    Path(os.environ['M2_CONTROL_DIAGNOSTICS']).write_text('{"kind":"state"}\\n')
if 'M2_VIDEO_TRACE' in os.environ:
    Path(os.environ['M2_VIDEO_TRACE']).write_text('{}\\n')
print('ENV='+json.dumps({k:os.environ.get(k) for k in ['M2_STEER_RATE_MODE','M2_CONTROL_VARIANT','M2_VIEW','N6_TRACE','BALL_DEDUP']}))
print('[resolved config] '+json.dumps({'steer_rate_mode':os.environ['M2_STEER_RATE_MODE'],'seed':2026}))
print('命中 1/1(100%) 平均击球点偏差 0.1m')
''')
        self.binary.chmod(0o755)
        for name in ("motors.json", "x4_tennis.json", "chassis_m1.json"):
            (self.team / "sckd_sim/models/config" / name).write_text("{}")
        (self.team / "sckd_sim/models/generated/x4_tennis_demo.mjcf").write_text("<mujoco/>")
        (self.team / "tools/control_ablation_cache.py").write_text('''import argparse,socket,time
p=argparse.ArgumentParser();p.add_argument('--serve');p.add_argument('--socket');a=p.parse_args()
s=socket.socket(socket.AF_UNIX);s.bind(a.socket);s.listen(1);time.sleep(30)
''')
        self.source = self.root / "source"
        self.source.mkdir()
        self.csv = self.source / "test.csv"
        self.csv.write_text("t_s,x_m,y_m,z_m\n0,1,2,3\n")
        self.item = {"sample_id": "test_01", "trajectory_id": "source_1", "csv": str(self.csv)}
        (self.source / "input_manifest.json").write_text(json.dumps({"items": [self.item]}))
        self.output = self.root / "output"
        (self.output / "cache").mkdir(parents=True)
        (self.output / "cache/test_01.jsonl").write_text("{}\n")
        (self.output / "identification").mkdir()
        (self.output / "identification/identified.json").write_text("{}")
        self.args = argparse.Namespace(team=self.team, source=self.source, root=self.output,
            steer_rate_mode="limited", video_trace=True, timeout=3, startup_timeout=2,
            retry_incomplete=False, samples=None, variants="baseline", modes="pfly")

    def test_ablation_runner_reuses_verified_result_and_records_effective_defaults(self):
        with patch.dict(os.environ, {"M2_VIEW": "1", "N6_TRACE": "1"}):
            first = run_control_ablation.run_one(self.args, "baseline", "pfly", self.item)
            second = run_control_ablation.run_one(self.args, "baseline", "pfly", self.item)
        self.assertEqual(first, second)
        self.assertEqual((self.team / "invocations.txt").read_text(), "1")
        self.assertEqual(first["steer_rate_mode"], "limited")
        resolved = first["experiment"]["spec"]
        self.assertEqual(resolved["parameters"]["init_delay_s"], .25)
        self.assertEqual(resolved["environment"]["M2_STEER_RATE_MODE"], "limited")
        self.assertNotIn("M2_VIEW", resolved["environment"])
        self.assertEqual(first["binary_sha256"], digest(self.binary))
        self.assertEqual(first["resolved_runtime_config"]["steer_rate_mode"], "limited")
        self.assertTrue(first["resolved_runtime_config_available"])
        self.assertFalse(list((self.team / ".tmp").glob("*.sock")))

    def test_ablation_runner_rejects_cache_binary_and_exp_knob_changes(self):
        run_control_ablation.run_one(self.args, "baseline", "pfly", self.item)
        done = self.output / "runs/baseline/pfly/test_01/run.json"
        before = done.read_bytes()
        cache = self.output / "cache/test_01.jsonl"
        cache.write_text('{"changed":true}\n')
        with self.assertRaises(FileExistsError):
            run_control_ablation.run_one(self.args, "baseline", "pfly", self.item)
        cache.write_text("{}\n")
        with patch.dict(os.environ, {"EXP_POS": "1.5"}):
            with self.assertRaises(FileExistsError):
                run_control_ablation.run_one(self.args, "baseline", "pfly", self.item)
        self.binary.write_text(self.binary.read_text() + "\n# changed binary\n")
        with self.assertRaises(FileExistsError):
            run_control_ablation.run_one(self.args, "baseline", "pfly", self.item)
        self.assertEqual(done.read_bytes(), before)
        self.assertEqual((self.team / "invocations.txt").read_text(), "1")

    def test_ablation_runner_rejects_source_changes_without_rebuilding(self):
        run_control_ablation.run_one(self.args, "baseline", "pfly", self.item)
        (self.team / "sckd_sim/cpp/runtime/fixture.h").write_text("// modified runtime\n")
        with self.assertRaisesRegex(FileExistsError, "configuration"):
            run_control_ablation.run_one(self.args, "baseline", "pfly", self.item)
        self.assertEqual((self.team / "invocations.txt").read_text(), "1")

    def test_predictor_control_runner_preserves_metrics_and_default_steering(self):
        args = argparse.Namespace(team=str(self.team), prediction=str(self.root / "unused_prediction"),
            out=str(self.root / "evaluation"), trials=1, profile=False, steer_rate_mode=None,
            timeout=3, startup_timeout=2, retry_incomplete=False)
        with patch.dict(os.environ, {"M2_CONTROL_VARIANT": "gains", "M2_VIEW": "1", "BALL_DEDUP": "1"}):
            first = evaluate_predictor_control.run((args, "legacy", "pfly", "fixed", 2026))
            second = evaluate_predictor_control.run((args, "legacy", "pfly", "fixed", 2026))
        self.assertEqual(first, second)
        self.assertEqual(first["hits"], 1)
        self.assertEqual(first["mean_plan_error_m"], .1)
        self.assertEqual(first["experiment"]["spec"]["parameters"]["steer_rate_mode"], "both_unlimited")
        environment = first["experiment"]["spec"]["environment"]
        self.assertEqual(environment["M2_CONTROL_VARIANT"], "baseline")
        self.assertNotIn("BALL_DEDUP", environment)
        self.assertNotIn("M2_VIEW", environment)
        self.assertEqual((self.team / "invocations.txt").read_text(), "1")

    def test_predictor_control_explicit_video_trace_survives_environment_cleanup(self):
        args = argparse.Namespace(team=str(self.team), prediction=str(self.root / "unused_prediction"),
            out=str(self.root / "evaluation"), trials=1, profile=False, video_trace=True,
            steer_rate_mode="limited", timeout=3, startup_timeout=2, retry_incomplete=False)
        result = evaluate_predictor_control.run((args, "dedup", "pfly", "fixed", 2026))
        destination = Path(args.out) / "dedup_pfly_fixed_2026"
        self.assertTrue((destination / "frames.jsonl").is_file())
        self.assertEqual(result["experiment"]["spec"]["environment"]["BALL_DEDUP"], "1")
        self.assertEqual(result["resolved_runtime_config"]["steer_rate_mode"], "limited")


if __name__ == "__main__":
    unittest.main()
