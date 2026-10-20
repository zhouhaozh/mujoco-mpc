"""保护审计的独立临时fixture；不运行仿真/训练/绘图，不改真实仓库。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_time_window_preservation as preservation


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="preservation-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.team = self.base / "team"
        self.root = self.team / "sckd_sim/artifacts/frozen"
        self.root.mkdir(parents=True)
        self.old = {}
        for relative in ("sckd_sim/cpp/unchanged.cc", "sckd_sim/cpp/deployed.cc",
                         "sckd_sim/build-predictor/m2_intercept", "sckd_sim/build-modularization/m2_intercept"):
            path = self.team / relative
            resource = self.file(path, "original " + relative)
            self.old[str(path)] = resource["sha256"]
            aliases = {"sckd_sim/build-predictor/m2_intercept": "default_binary",
                       "sckd_sim/build-modularization/m2_intercept": "modular_binary"}
            self.file(self.root / "baseline" / aliases.get(relative, relative), path.read_text())
        self.json(self.root / "baseline_manifest.json", {"created": "fixture", "files": self.old})
        changed = self.team / "sckd_sim/cpp/deployed.cc"
        self.deployment = [{"path": str(changed), "before": self.old[str(changed)],
                            "sha256": self.file(changed, "authorized timing policy")["sha256"]}]
        added = self.team / "tools/new_timing.py"
        self.deployment.append({"path": str(added), "before": None,
                                "sha256": self.file(added, "new timing runner")["sha256"]})
        self.json(self.root / "deployment_manifest.json", self.deployment)
        model = self.file(self.team / "sckd_sim/models/robot.xml", "immutable model")
        solver = self.file(self.team / "sckd_sim/third_party/nmpc6/nlp.so", "immutable solver")
        self.json(self.root / "default_equivalence/input_manifest.json",
                  {"resources": {"model/robot.xml": model, "nmpc6/nlp.so": solver}})
        resources = {
            "input_manifest": self.file(self.team / "data/input_manifest.json", "original input metadata"),
            "cache_manifest/h1500": self.file(self.team / "cache/manifest.json", "original cache metadata")}
        ids = [f"test_{number:02}" for number in range(1, 36)]
        for sample in ids:
            resources["input/" + sample] = self.file(self.team / "data" / (sample + ".csv"), sample + " CSV")
            resources["frozen_cache/h1500/" + sample] = self.file(self.team / "cache" / (sample + ".jsonl"), sample + " cache")
        self.frozen = {"schema": "TIME_WINDOW_SEARCH_MANIFEST/1", "resources": resources,
            "runtime_resources": {"pfly": {"models/robot.xml": model, "nmpc6/nlp.so": solver}},
            "split": {"discovery": ids[:18], "confirmation": ids[18:]}}
        self.json(self.root / "stage1/search_manifest.json", self.frozen)

    def file(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {"path": str(path), "sha256": hashlib.sha256(content.encode()).hexdigest()}

    def json(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))

    def report(self):
        return preservation.audit(self.root, self.team, expected_branch="fixture", expected_head="abc",
            git_reader=lambda *args: {"passed": True, "branch": "fixture", "head": "abc", "staged_paths": []})

    def test_originals_backups_and_explicit_deployments_are_distinguished(self):
        report = self.report()
        self.assertTrue(report["passed"])
        self.assertTrue(report["coverage"]["passed"])
        self.assertEqual(report["counts"]["authorized_deployment"], 1)
        self.assertEqual(report["counts"]["authorized_new_file"], 1)
        self.assertEqual(report["coverage"]["original_input_samples"], 35)
        self.assertEqual(report["coverage"]["original_h1500_cache_samples"], 35)

    def test_protected_source_deployed_source_model_cache_and_backup_drift_are_exact(self):
        paths = [self.team / "sckd_sim/cpp/unchanged.cc", self.team / "sckd_sim/cpp/deployed.cc",
                 self.team / "sckd_sim/models/robot.xml", self.team / "cache/test_01.jsonl",
                 self.root / "baseline/default_binary"]
        for path in paths:
            with self.subTest(path=str(path)):
                original = path.read_bytes()
                try:
                    path.write_bytes(b"unexpected mutation")
                    report = self.report()
                    self.assertFalse(report["passed"])
                    self.assertEqual([(row["path"], row["status"]) for row in report["failures"]],
                                     [(str(path), "hash_mismatch")])
                finally:
                    path.write_bytes(original)

    def test_missing_resources_and_wrong_deployment_before_fail(self):
        path = self.team / "data/test_02.csv"
        path.unlink()
        self.deployment[0]["before"] = "0" * 64
        self.json(self.root / "deployment_manifest.json", self.deployment)
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"][0]["path"], str(path))
        self.assertEqual(report["failures"][0]["status"], "missing")
        self.assertEqual(report["issues"][0]["kind"], "deployment_before_mismatch")

    def test_conflicting_frozen_model_hash_and_incomplete_samples_fail(self):
        self.frozen["runtime_resources"]["pfly"]["models/robot.xml"]["sha256"] = "0" * 64
        del self.frozen["resources"]["frozen_cache/h1500/test_35"]
        self.json(self.root / "stage1/search_manifest.json", self.frozen)
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertEqual({row["kind"] for row in report["issues"]},
                         {"conflicting_evidence", "incomplete_protection_evidence"})

    def test_original_binaries_cannot_be_authorized_away(self):
        path = self.team / "sckd_sim/build-predictor/m2_intercept"
        self.deployment.append({"path": str(path), "before": self.old[str(path)],
                                "sha256": self.file(path, "replacement")["sha256"]})
        self.json(self.root / "deployment_manifest.json", self.deployment)
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertEqual(report["issues"][0]["kind"], "protected_binary_listed_for_deployment")
        self.assertEqual(report["failures"][0]["path"], str(path))

    def test_report_is_exclusive_and_cannot_write_through_frozen_directory_symlink(self):
        report = self.report()
        for path in (self.root / "new_audit.json", self.team / "sckd_sim/cpp/unchanged.cc"):
            with self.assertRaises(ValueError):
                preservation.write_report(path, report)
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            preservation.write_report(alias / "audit.json", report)
        out = self.base / "audit/report.json"
        preservation.write_report(out, report)
        with self.assertRaises(FileExistsError):
            preservation.write_report(out, report)

    def test_git_audit_requires_expected_branch_head_and_empty_staging(self):
        def result(value):
            return subprocess.CompletedProcess([], 0, value, "")
        for branch, head, staged, passed in (("fixture\n", "abc\n", "", True),
                ("other\n", "abc\n", "", False), ("fixture\n", "changed\n", "", False),
                ("fixture\n", "abc\n", "changed.cc\0", False)):
            with patch.object(preservation.subprocess, "run", side_effect=[result(branch), result(head), result(staged)]) as run:
                report = preservation.git_state(self.team, "fixture", "abc")
                self.assertEqual(report["passed"], passed)
                self.assertTrue(all(call.kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0" for call in run.call_args_list))
                self.assertTrue(all(call.args[0][1] == "--no-optional-locks" for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
