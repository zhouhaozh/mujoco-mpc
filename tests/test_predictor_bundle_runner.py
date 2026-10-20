"""Ubuntu轻量fixture：bundle解析委托预测环境，不加载模型或启动仿真。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_predictor_control as runner
from pipeline_experiment import ExperimentSpec, RunAttempt, digest


class PredictorBundleRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bundle-runner-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prediction = self.root / "prediction"
        (self.prediction / ".venv-tennis/bin").mkdir(parents=True)
        (self.prediction / "tennis_calibration").mkdir()
        (self.prediction / "tennis_calibration/fixture.py").write_text("# predictor source fixture\n")
        (self.prediction / "requirements.lock.txt").write_text("# dependency fixture\n")
        self.python = self.prediction / ".venv-tennis/bin/python"
        self.python.write_text(f"#!{sys.executable}\n" + '''import json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
with (root/'inspector_calls.jsonl').open('a') as stream:
    stream.write(json.dumps(sys.argv[1:])+'\\n')
(root/'inspector_runtime.json').write_text(json.dumps({'cwd':str(Path.cwd()),'pythonpath':os.environ.get('PYTHONPATH')}))
response=json.loads((root/'inspector_response.json').read_text())
if response.get('failure'):
    print(response['failure'],file=sys.stderr)
    raise SystemExit(7)
print(json.dumps(response['manifest']))
''')
        self.python.chmod(0o755)
        self.bundle = self.root / "relocated/bundles/revision/model_bundle.json"
        self.bundle.parent.mkdir(parents=True)
        self.bundle.write_text('{"fixture":"bundle manifest"}\n')
        self.resources = {}
        self.paths = {}
        names = ("selection", "collision", "identified", "noise", "head",
                 "collision_velocity_0", "collision_velocity_1", "collision_velocity_3")
        for name in names:
            path = self.root / "nonhistorical/resources" / (name + ".fixture")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + " fixture bytes\n")
            self.paths[name] = path
            self.resources[name] = {"path": os.path.relpath(path, self.bundle.parent),
                                    "resolved_path": str(path), "sha256": digest(path)}
        transform = {"R_world_from_model": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     "translation_world_from_model_m": [0, 0, .1]}
        canonical = json.dumps(transform, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode()
        self.resources["transform"] = {"inline": transform, "sha256": hashlib.sha256(canonical).hexdigest()}
        self.resolved = {"schema": "TENNIS_MODEL_BUNDLE/1", "fingerprint": "a" * 64,
                         "manifest_path": str(self.bundle), "resources": self.resources,
                         "settings": {"candidate": "fixture"}, "effective_config": {"transform": transform}}
        self.selection = self.paths["selection"]
        self.response = self.prediction / "inspector_response.json"
        self.write_response()

    def write_response(self, **extra):
        self.response.write_text(json.dumps({"manifest": self.resolved, **extra}))

    def inspect(self, **kwargs):
        return runner.inspect_predictor_bundle(self.prediction, self.selection, **kwargs)

    def test_bundle_inspection_uses_prediction_python_and_absolute_bundle_path(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            relative = self.bundle.relative_to(self.root)
            actual = self.inspect(model_bundle=relative)
        finally:
            os.chdir(previous)
        self.assertEqual(actual, self.resolved)
        call = json.loads((self.prediction / "inspector_calls.jsonl").read_text().splitlines()[-1])
        self.assertEqual(call[0], "-c")
        self.assertIn("from tennis_calibration.model_bundle import ModelBundle", call[1])
        self.assertEqual(call[2:], [str(self.prediction), "bundle", str(self.bundle)])

    def test_legacy_inspection_passes_selection_and_preserves_synthetic_factory(self):
        self.inspect()
        call = json.loads((self.prediction / "inspector_calls.jsonl").read_text().splitlines()[-1])
        self.assertEqual(call[2:], [str(self.prediction), "selection", str(self.selection)])
        self.assertIn("synthetic_world=True", call[1])

    def test_prediction_source_override_changes_cwd_pythonpath_and_hashed_source_only(self):
        source = self.root / "new_branch/prediction"
        package = source / "tennis_calibration"
        package.mkdir(parents=True)
        module = package / "fixture.py"
        module.write_text("# new branch predictor implementation\n")
        self.inspect(model_bundle=self.bundle, prediction_source=source)
        runtime = json.loads((self.prediction / "inspector_runtime.json").read_text())
        self.assertEqual(runtime["cwd"], str(source))
        self.assertEqual(runtime["pythonpath"].split(os.pathsep)[0], str(source))
        resources = runner.predictor_resources(self.prediction, self.selection,
            model_bundle=self.bundle, resolved_bundle=self.resolved, prediction_source=source)
        self.assertEqual(resources["predictor_python"]["path"], str(self.python))
        self.assertEqual(resources["predictor_code/fixture.py"], {"path": str(module), "sha256": digest(module)})
        self.assertNotEqual(resources["predictor_code/fixture.py"]["sha256"],
                            digest(self.prediction / "tennis_calibration/fixture.py"))
        legacy = runner.predictor_service_command(self.prediction, self.selection,
            self.root / "run", prediction_source=source)
        self.assertEqual(legacy[legacy.index("--project-root") + 1], str(self.prediction))
        bundled = runner.predictor_service_command(self.prediction, self.selection,
            self.root / "run", model_bundle=self.bundle, prediction_source=source)
        self.assertNotIn("--project-root", bundled)

    def test_explicit_binary_runs_and_is_hashed_without_requiring_default_build(self):
        team = self.root / "team"
        sim = team / "sckd_sim"
        for directory in ("cpp/runtime", "models/generated", "build-modularization"):
            (sim / directory).mkdir(parents=True)
        (sim / "cpp/runtime/fixture.h").write_text("// C++ source fixture\n")
        (sim / "models/generated/x4_tennis_demo.mjcf").write_text("<mujoco/>\n")
        (sim / "CMakeLists.txt").write_text("# build fixture\n")
        binary = sim / "build-modularization/m2_intercept"
        binary.write_text(f"#!{sys.executable}\n" +
            "print('[resolved config] {\"implementation\":\"modularization-fixture\"}')\n" +
            "print('命中 1/1(100%) 平均击球点偏差 0.1m')\n")
        binary.chmod(0o755)
        args = argparse.Namespace(team=team, prediction=self.prediction, out=self.root / "evaluation",
            binary=binary, trials=1, profile=False, timeout=3, startup_timeout=2,
            retry_incomplete=False)
        result = runner.run((args, "legacy", "pfly", "fixed", 2026))
        self.assertEqual(result["command"][0], str(binary))
        self.assertEqual(result["binary_sha256"], digest(binary))
        self.assertEqual(result["experiment"]["spec"]["parameters"]["binary"], str(binary))
        self.assertEqual(result["experiment"]["spec"]["resources"]["binary"]["path"], str(binary))
        self.assertFalse((sim / "build-predictor/m2_intercept").exists())
        output = Path(args.out) / "legacy_pfly_fixed_2026/summary.json"
        before = output.read_bytes()
        binary.write_text(binary.read_text() + "# changed compiled fixture\n")
        with self.assertRaisesRegex(FileExistsError, "configuration"):
            runner.run((args, "legacy", "pfly", "fixed", 2026))
        self.assertEqual(output.read_bytes(), before)

    def test_resources_follow_explicit_relocated_paths_and_keep_existing_labels(self):
        resolved = self.inspect(model_bundle=self.bundle)
        resources = runner.predictor_resources(self.prediction, self.selection,
            model_bundle=self.bundle, resolved_bundle=resolved)
        labels = {"selection": "selection", "collision": "collision_selection", "identified": "filter_selection",
                  "noise": "noise_proxy", "head": "residual_head", "collision_velocity_0": "collision_models/velocity_0",
                  "collision_velocity_1": "collision_models/velocity_1", "collision_velocity_3": "collision_models/velocity_3"}
        for name, label in labels.items():
            self.assertEqual(resources[label], {"path": str(self.paths[name]), "sha256": digest(self.paths[name])})
        self.assertEqual(resources["model_bundle"]["path"], str(self.bundle))
        self.assertIn("predictor_python", resources)
        self.assertIn("predictor_code/fixture.py", resources)
        self.assertIn("predictor_requirements", resources)
        self.assertNotIn("predictor_transform", resources)  # Inline content is in the resolved manifest.
        self.assertFalse((self.prediction / "artifacts").exists())

    def test_resource_tampering_after_inspection_is_rejected(self):
        resolved = self.inspect(model_bundle=self.bundle)
        self.paths["collision_velocity_1"].write_text("changed child resource\n")
        with self.assertRaisesRegex(RuntimeError, "changed after bundle inspection"):
            runner.predictor_resources(self.prediction, self.selection,
                model_bundle=self.bundle, resolved_bundle=resolved)

    def test_bundle_child_is_in_run_identity_and_reverified_at_completion(self):
        resources = runner.predictor_resources(self.prediction, self.selection, model_bundle=self.bundle)
        specification = ExperimentSpec("fixture", ("fixture",), str(self.root),
            {"predictor_bundle_fingerprint": self.resolved["fingerprint"],
             "predictor_resolved_manifest": self.resolved}, {}, resources)
        destination = self.root / "run"
        with self.assertRaisesRegex(RuntimeError, "resource changed during run"):
            with RunAttempt(destination, specification) as attempt:
                (destination / "control.log").write_text("fixture control output\n")
                self.paths["collision_velocity_0"].write_text("changed during run\n")
                attempt.complete({"hits": 1})
        self.assertFalse((destination / "run.json").exists())
        self.assertEqual(json.loads((destination / "run_manifest.json").read_text())["status"], "failed")

    def test_service_bundle_command_omits_legacy_frame_override(self):
        destination = self.root / "run"
        command = runner.predictor_service_command(self.prediction, self.selection,
                                                    destination, model_bundle=self.bundle)
        self.assertEqual(command[:3], [str(self.python), "-m", "tennis_calibration.controller_service"])
        self.assertEqual(command[command.index("--bundle") + 1], str(self.bundle))
        self.assertNotIn("--selection", command)
        self.assertNotIn("--synthetic-world", command)
        self.assertEqual(command[command.index("--socket") + 1], "{socket}")
        self.assertEqual(command[command.index("--resolved-manifest") + 1],
                         str(destination / "predictor_resolved_manifest.json"))
        legacy = runner.predictor_service_command(self.prediction, self.selection, destination)
        self.assertIn("--selection", legacy)
        self.assertIn("--synthetic-world", legacy)
        self.assertNotIn("--bundle", legacy)

    def test_service_resolved_manifest_must_equal_preflight_before_control(self):
        output = self.root / "service_manifest.json"
        output.write_text(json.dumps(self.resolved))
        self.assertEqual(runner.verify_predictor_manifest(output, self.resolved), self.resolved)
        changed = dict(self.resolved, fingerprint="b" * 64)
        output.write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError, "differs from preflight"):
            runner.verify_predictor_manifest(output, self.resolved)

    def test_failed_or_invalid_inspection_never_falls_back_to_legacy_paths(self):
        self.write_response(failure="fixture rejects changed model bundle")
        with self.assertRaisesRegex(RuntimeError, "fixture rejects changed model bundle"):
            self.inspect(model_bundle=self.bundle)
        self.resolved["fingerprint"] = "invalid"
        self.write_response()
        with self.assertRaisesRegex(ValueError, "invalid resolved model bundle"):
            self.inspect(model_bundle=self.bundle)

    def test_team_runner_import_has_no_prediction_dependency(self):
        program = ("import sys; import evaluate_predictor_control; "
                   "assert not any(k.startswith('tennis_calibration') for k in sys.modules); "
                   "assert 'numpy' not in sys.modules; assert 'scipy' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", program],
            cwd=Path(runner.__file__).parent, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
