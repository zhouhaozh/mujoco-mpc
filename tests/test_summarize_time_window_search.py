"""终审的独立协议/完整性夹具；Ubuntu 执行，无 MuJoCo、绘图或控制求解。"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import summarize_time_window_search as final
from pipeline_experiment import ExperimentSpec, RunAttempt, digest, file_resources, write_json
import time_window_search as search


def recheck(contact=True, approaching=True, native=True, fallback=False):
    geometry = {"contact": contact, "approaching": approaching, "native_contact": contact and native,
        "native_approaching": approaching and native, "nonnative_contact": contact and not native,
        "nonnative_approaching": approaching and not native, "fallback_state_contact": contact and fallback,
        "fallback_state_approaching": approaching and fallback, "min_gap_m": -.01 if contact else .2,
        "first_contact_s": .5 if contact else None, "step_s": .001}
    return {scope: deepcopy(geometry) for scope in final.SCOPES} | {"fallback_any": fallback}


class StatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_known_tails_and_symmetry(self):
        self.assertEqual(final.exact_mcnemar(0, 0), 1)
        self.assertEqual(final.exact_mcnemar(4, 0), .125)
        self.assertEqual(final.exact_mcnemar(0, 4), .125)
        self.assertEqual(final.exact_mcnemar(3, 1), .625)
        self.assertEqual(final.exact_mcnemar(2, 2), 1)
        for bad in ((-1, 0), (True, 0), (1.5, 0)):
            with self.subTest(counts=bad), self.assertRaises(ValueError):
                final.exact_mcnemar(*bad)

    def test_paired_ids_preserve_losses_cache_difference_and_exact_p(self):
        left, right = [], []
        for sample, a, b in (("a", False, True), ("b", True, False), ("c", False, True)):
            left.append({"sample_id": sample, "input_sha256": sample, "cache_sha256": "old",
                         "reevaluation": recheck(a, a)})
            right.append({"sample_id": sample, "input_sha256": sample, "cache_sha256": "new",
                          "reevaluation": recheck(b, b)})
        paired = final.paired_summary(left, right, ["a", "b", "c"], "primary_1ms")
        self.assertEqual(paired["approaching"]["gained_sample_ids"], ["a", "c"])
        self.assertEqual(paired["approaching"]["lost_sample_ids"], ["b"])
        self.assertEqual(paired["approaching"]["exact_two_sided_mcnemar_p"], 1)
        self.assertFalse(paired["cache_hashes_identical"])
        right[0]["input_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "raw input hashes differ"):
            final.paired_summary(left, right, ["a", "b", "c"], "primary_1ms")

    def test_incomplete_analysis_keeps_denominator_and_null_rates(self):
        rows = [{"sample_id": "a", "reevaluation": recheck()}]
        result = final.phase_summary(rows, ["a", "b"], "primary_1ms")
        self.assertEqual(result["denominator"], 2)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["rates"]["approaching"])
        with self.assertRaises(ValueError):
            final.phase_summary(rows * 2, ["a", "b"], "primary_1ms")


class NativeApproachingTests(unittest.TestCase):
    def test_native_geometry_is_not_native_approaching(self):
        states = [
            {"t": 0., "planned": True, "active": "nmpc6", "fallback": False, "geometry": True, "approach": False},
            {"t": .5, "planned": True, "active": "wbmpc", "fallback": True, "geometry": True, "approach": True},
            {"t": 1., "planned": False, "active": "nmpc6", "fallback": False, "geometry": True, "approach": True},
        ]
        original = deepcopy(states)
        calls = []
        def adapter(rows, step, before, after):
            calls.append((step, before, after))
            return {"contact": any(row["planned"] and row["geometry"] for row in rows),
                "approaching": any(row["planned"] and row["geometry"] and row["approach"] for row in rows)}
        result = final.reevaluate_states(states, {"input_end_s": 2., "bounce_labels": [.2, .9]},
            {"hit_time_s": .5, "fallback": True}, "nmpc6", adapter)
        for scope in final.SCOPES:
            self.assertTrue(result[scope]["native_contact"])
            self.assertFalse(result[scope]["native_approaching"])
            self.assertTrue(result[scope]["nonnative_approaching"])
            self.assertTrue(result[scope]["fallback_state_approaching"])
        self.assertEqual(result["episode_bounds"]["evaluation_end_s"], .9)
        self.assertEqual(result["episode_bounds"]["required_episode_end_s"], .9)
        self.assertEqual(result["episode_bounds"]["unrecorded_tail_s"], 0)
        self.assertIn((.001, 10., 10.), calls)
        self.assertIn((.0005, .35, .35), calls)
        self.assertEqual(states, original)

    def test_secondary_endpoint_records_unobserved_tail_without_extrapolation(self):
        states = [{"t": t, "planned": True, "active": "nmpc6", "fallback": False} for t in (0., .996)]
        result = final.reevaluate_states(states, {"input_end_s": 2., "bounce_labels": [.2, 1.]},
            {"hit_time_s": .5, "fallback": False}, "nmpc6",
            lambda *args: {"contact": False, "approaching": False})
        bounds = result["episode_bounds"]
        self.assertEqual(bounds["evaluation_end_s"], .996)
        self.assertEqual(bounds["required_episode_end_s"], 1.)
        self.assertAlmostEqual(bounds["unrecorded_tail_s"], .004)

    def test_pfly_wait_eligibility_and_secondary_window_coverage(self):
        rows = [{"planned": True, "active": "pfly_or_wait", "arm_qp_active": flag} for flag in (False, True)]
        masked = final.masked_states(rows, lambda row: final.active_controller(row) == "pfly")
        self.assertEqual([row["planned"] for row in masked], [False, True])
        with self.assertRaisesRegex(ValueError, "does not cover"):
            final.reevaluate_states([{**rows[0], "t": 0., "fallback": False},
                                     {**rows[1], "t": 12., "fallback": False}],
                {"input_end_s": 12., "bounce_labels": []}, {"hit_time_s": 11., "fallback": False},
                "pfly", lambda *args: self.fail("uncovered window must fail before geometry"))


class IntegrityTests(unittest.TestCase):
    def test_completed_result_detects_artifact_and_resource_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resource = root / "source.py"
            resource.write_text("original")
            spec = ExperimentSpec("fixture", (), str(root), {}, {}, file_resources({"source": resource}))
            with RunAttempt(root / "run", spec) as attempt:
                (root / "run/diagnostics.jsonl").write_text("fixture\n")
                attempt.complete({"metric": 1})
            final.audit_completed_run(root / "run", final.HashAudit())
            original = (root / "run/diagnostics.jsonl").read_text()
            (root / "run/diagnostics.jsonl").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                final.audit_completed_run(root / "run", final.HashAudit())
            (root / "run/diagnostics.jsonl").write_text(original)
            resource.write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                final.audit_completed_run(root / "run", final.HashAudit())

    def test_hash_inventory_rechecks_after_initial_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("a")
            audit = final.HashAudit()
            audit.verify(path)
            path.write_text("b")
            with self.assertRaisesRegex(ValueError, "changed during analysis"):
                audit.recheck()


class FrozenSearchFixtureTests(unittest.TestCase):
    """Construct the wire files independently, without SearchRunner or solvers."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="final-timing-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.team, self.source, self.search = (self.root / name for name in ("team", "source", "search"))
        for path in (self.team / "sckd_sim/models/generated", self.source, self.search):
            path.mkdir(parents=True)
        self.scene = self.team / "sckd_sim/models/generated/x4_tennis_demo.mjcf"
        self.scene.write_text("<mujoco/>")
        self.binary, self.predictor, self.weight = (self.root / name for name in ("controller", "predictor.py", "weights.json"))
        for path in (self.binary, self.predictor, self.weight):
            path.write_text("fixture " + path.name)
        self.ids = [f"test_{i:02}" for i in range(1, 36)]
        self.split = search.fixed_split(self.ids, {"discovery": self.ids[:18], "confirmation": self.ids[18:]})
        self.items = []
        for sample in self.ids:
            csv = self.source / (sample + ".csv")
            csv.write_text("t,x,y,z\n0,0,0,0\n1,1,1,1\n")
            self.items.append({"sample_id": sample, "trajectory_id": sample, "csv": str(csv),
                               "input_end_s": 1., "bounce_labels": [], "source_sha256": digest(csv)})
        write_json(self.source / "input_manifest.json", {"items": self.items})
        producer, horizon = self.root / "forecast", self.root / "forecast/horizon_1500ms"
        self.cache = horizon / "cache"
        producer_spec = ExperimentSpec("prepare_time_window_caches", (), str(self.root),
            {"sample_ids": self.ids, "source": str(self.source)}, {},
            file_resources({"predictor_code/fixture.py": self.predictor, "model/weights": self.weight}))
        with RunAttempt(producer, producer_spec, completion="manifest.json") as attempt:
            self.cache.mkdir(parents=True)
            files = []
            for sample in self.ids:
                path = self.cache / (sample + ".jsonl")
                path.write_text('{"forecast":null}\n')
                files.append({"file": path.name, "sha256": digest(path)})
            cache_manifest = {"schema": "FORECAST_HORIZON_CACHES/1", "files": files,
                "strategy": "independent_forecast_per_horizon_and_sample", "bundle_fingerprint": "bundle"}
            write_json(self.cache / "manifest.json", cache_manifest)
            attempt.complete({"sample_count": 35, "base_bundle_fingerprint": "basebundle",
                "horizons": [{"cache_manifest": "horizon_1500ms/cache/manifest.json",
                    "cache_manifest_sha256": digest(self.cache / "manifest.json"), "bundle_fingerprint": "bundle"}]})
        self.jobs = []
        for mode in ("pfly", "nmpc6"):
            for role in ("base", "best"):
                self.jobs.append({"id": mode + "_" + role, "mode": mode})
        resources = {"input_manifest": self.source / "input_manifest.json", "cache_manifest/h1500": self.cache / "manifest.json"}
        resources.update({"input/" + item["sample_id"]: item["csv"] for item in self.items})
        resources.update({"frozen_cache/h1500/" + sample: self.cache / (sample + ".jsonl") for sample in self.ids})
        self.baselines = {mode: mode + "_base" for mode in ("pfly", "nmpc6")}
        self.selected = {mode: mode + "_best" for mode in ("pfly", "nmpc6")}
        self.manifest = {"schema": "TIME_WINDOW_SEARCH_MANIFEST/1", "spec": {"jobs": self.jobs},
            "resources": file_resources(resources), "runtime_resources": {mode: file_resources({"binary": self.binary,
                "models/scene": self.scene}) for mode in ("pfly", "nmpc6")}, "split": self.split,
            "baseline_jobs": self.baselines, "selection_rule": search.SELECTION_RULE, "controller_variant": "baseline"}
        write_json(self.search / "search_manifest.json", self.manifest)
        self.manifest_id = final.identity(self.manifest)
        confirmations = {}
        for request in self.jobs:
            job = {**request, "binary": str(self.binary), "policy": deepcopy(search.BASE_POLICY),
                "cache_id": "h1500", "cache": {"path": str(self.cache)}, "nlp_path": None,
                "controller_variant": "baseline", "steering": "both_unlimited"}
            policy = self.search / "policies" / (job["id"] + ".json")
            resolved = self.search / "jobs" / job["id"] / "resolved_job.json"
            for path, value in ((policy, job["policy"]), (resolved, job)):
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, value)
            summaries = {}
            for phase in ("discovery", "confirmation"):
                records = []
                for sample in self.split[phase]:
                    item = self.items[self.ids.index(sample)]
                    is_best = job["id"].endswith("best")
                    # Confirmation reverses discovery: frozen best must remain
                    # selected despite worse confirmation performance.
                    hit = is_best if phase == "discovery" else not is_best
                    runtime = {"values": {"time_windows_path": str(policy), "time_windows": job["policy"],
                        "control_variant": "baseline", "steer_rate_mode": "both_unlimited",
                        "base_mode": job["mode"], "init_delay_s": .25}}
                    trial_resources = file_resources({"input": item["csv"], "cache": self.cache / (sample + ".jsonl"),
                        "binary": self.binary, "timing_policy": policy, "resolved_job": resolved})
                    spec = ExperimentSpec("time_window_search", (str(self.binary),), str(self.team / "sckd_sim"),
                        {"search_manifest_identity": self.manifest_id, "resolved_job": job, "phase": phase,
                         "sample_id": sample, "trajectory_id": sample}, {}, trial_resources)
                    metric = {"job_id": job["id"], "phase": phase, "mode": job["mode"], "sample_id": sample,
                        "trajectory_id": sample, "hit_time_s": .5, "planned": True, "hit_time_covered": True,
                        "fallback": not is_best, "solver_calls": 1, "applied_solutions": 1,
                        "resolved_runtime_config": runtime, "native_at_geometry_contact": hit,
                        "geometric": {"contact": hit, "approaching": hit, "min_gap_m": -.01 if hit else .2,
                            "first_contact_s": .5 if hit else None, "step_s": .001},
                        **{field: trial_resources[label]["sha256"] for field, label in (("binary_sha256", "binary"),
                            ("input_sha256", "input"), ("cache_sha256", "cache"), ("policy_sha256", "timing_policy"))}}
                    destination = self.search / "runs" / phase / job["id"] / job["mode"] / sample
                    with RunAttempt(destination, spec) as attempt:
                        (destination / "diagnostics.jsonl").write_text("fixture\n")
                        (destination / "control.log").write_text("[resolved config] " + json.dumps(runtime) + "\n")
                        result = attempt.complete(metric)
                    metric.update(run_identity=result["experiment"]["identity"], run_result_sha256=result["experiment"]["result_sha256"])
                    records.append(metric)
                summary = search.summarize(job, phase, self.split[phase], records, [])
                summary["search_manifest_identity"] = self.manifest_id
                summaries[phase] = summary
                path = self.search / "summaries" / phase / (job["id"] + ".json")
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, summary)
            all35 = search.summarize(job, "all35_after_selection", self.ids,
                summaries["discovery"]["records"] + summaries["confirmation"]["records"], [])
            all35["search_manifest_identity"] = self.manifest_id
            path = self.search / "summaries/all35_after_selection" / (job["id"] + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, all35)
            confirmations[job["id"]] = {key: value for key, value in summaries["confirmation"].items() if key != "records"}
        selection = {"schema": "TIME_WINDOW_SELECTION/1", "selection_phase": "discovery",
            "search_manifest_identity": self.manifest_id, "selection_rule": search.SELECTION_RULE,
            "split": self.split, "baseline_jobs": self.baselines, "selected_by_mode": self.selected,
            "discovery_summary_sha256": {job["id"]: digest(self.search / "summaries/discovery" / (job["id"] + ".json")) for job in self.jobs}}
        write_json(self.search / "selection.json", selection)
        selection_sha = digest(self.search / "selection.json")
        write_json(self.search / "confirmation_plan.json", {"schema": "TIME_WINDOW_CONFIRMATION_PLAN/1",
            "selection_sha256": selection_sha, "selection_data": "discovery_only", "sample_ids": self.split["confirmation"],
            "job_ids": [job["id"] for job in self.jobs]})
        write_json(self.search / "confirmation_summary.json", {"schema": "TIME_WINDOW_CONFIRMATION/1",
            "complete": True, "selection_sha256": selection_sha, "baseline_jobs": self.baselines,
            "selected_by_mode": self.selected, "confirmation": confirmations})

    @staticmethod
    def evaluate(path, item, mode, recorded):
        hit = recorded["geometric"]["approaching"]
        result = recheck(hit, hit, native=int(item["sample_id"][-2:]) % 2 == 0, fallback=recorded["fallback"])
        if item["sample_id"] == "test_01":
            result["primary_0_5ms"]["contact"] = not hit
        return result

    def test_full_four_groups_keep_frozen_selection_and_all35_denominator(self):
        with redirect_stdout(io.StringIO()):
            report = final.summarize_search(self.search, self.team, self.source, self.root / "analysis", evaluator=self.evaluate)
        self.assertTrue(report["complete"])
        self.assertEqual(report["selected_by_mode"], self.selected)
        self.assertEqual(len(report["records"]), 4)
        for mode in ("pfly", "nmpc6"):
            best = report["groups"][mode]["selected"]
            self.assertEqual(best["phases"]["all35_after_selection"]["primary_1ms"]["denominator"], 35)
            self.assertEqual(best["phases"]["discovery"]["primary_1ms"]["counts"]["approaching"], 18)
            self.assertEqual(best["phases"]["confirmation"]["primary_1ms"]["counts"]["approaching"], 0)
            self.assertEqual(best["step_sensitivity"]["contact"], ["test_01"])
            self.assertEqual(report["paired"][mode]["confirmation"]["primary_1ms"]["approaching"]["lost"], 17)
        markdown = (self.root / "analysis/analysis.md").read_text()
        self.assertIn("不是新盲测", markdown)
        self.assertIn("原生接近几何", markdown)
        inventory = final.load(self.root / "analysis/verified_files.json")
        self.assertEqual(final.identity(inventory), report["integrity"]["inventory_sha256"])
        final.audit_completed_run(self.root / "analysis", final.HashAudit())
        with self.assertRaisesRegex(ValueError, "new directory"):
            final.summarize_search(self.search, self.team, self.source, self.root / "analysis", evaluator=self.evaluate)

    def test_changed_policy_rejects_completion_before_creating_analysis(self):
        path = self.search / "policies/pfly_base.json"
        policy = final.load(path)
        policy["pfly_init_delay_s"] = .123
        write_json(path, policy)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            final.summarize_search(self.search, self.team, self.source, self.root / "bad_analysis", evaluator=self.evaluate)
        self.assertFalse((self.root / "bad_analysis").exists())

    def test_incomplete_confirmation_never_marks_analysis_complete(self):
        path = self.search / "confirmation_summary.json"
        value = final.load(path)
        value["complete"] = False
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "incomplete/error"):
            final.summarize_search(self.search, self.team, self.source, self.root / "incomplete", evaluator=self.evaluate)
        self.assertFalse((self.root / "incomplete").exists())


if __name__ == "__main__":
    unittest.main()
