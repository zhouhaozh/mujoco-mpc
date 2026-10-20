"""时间窗口搜索协议/身份/发现与确认隔离；只运行临时假程序，不运行MuJoCo。"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import time_window_search as search


def metric(sample, *, mode="pfly", hit=True, source="input", cache="cache"):
    return {"mode": mode, "sample_id": sample, "trajectory_id": sample, "planned": True,
        "hit_time_covered": True, "fallback": False, "solver_calls": 1, "applied_solutions": 1,
        "qp_attempts": 1, "qp_accepts": 1, "solve_times_ms": [2.], "input_sha256": source,
        "cache_sha256": cache, "native_at_geometry_contact": hit,
        "geometric": {"contact": hit, "approaching": hit, "min_gap_m": -.02 if hit else .10,
                      "first_contact_s": .5 if hit else None, "step_s": .001}}


class SearchProtocolTests(unittest.TestCase):
    def test_metrics_and_search_evaluation_do_not_import_matplotlib(self):
        # A fresh process prevents an already-imported plotting module from hiding
        # accidental coupling. MuJoCo geometry is stubbed; the metrics reader is real.
        code = r'''
import importlib.abc,json,sys,tempfile,types
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class NoPlotting(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname == 'matplotlib' or fullname.startswith('matplotlib.'):
            raise ModuleNotFoundError('matplotlib deliberately unavailable')
sys.meta_path.insert(0,NoPlotting())
from control_diagnostic_metrics import norm,stats,STATUS
assert norm([3,4]) == 5
assert stats([None,1,3])['median'] == 2
assert STATUS[0] == '严格'
import time_window_search
geometry=types.ModuleType('analyze_control_ablation')
geometry.motor_audit=lambda states: None
geometry.contact=lambda *args: {'contact':False,'approaching':False,'min_gap_m':None}
sys.modules['analyze_control_ablation']=geometry
mj=types.ModuleType('mujoco')
mj.MjModel=types.SimpleNamespace(from_xml_path=lambda path: object())
mj.MjData=lambda model: object()
sys.modules['mujoco']=mj
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);dest=root/'pfly'/'test_01';dest.mkdir(parents=True)
    row={'kind':'state','t':0,'p_reference':[0,0,0],'p_actual':[0,0,0],
         'p_command_actualbase':[0,0,0],'p_nominal_actual':[0,0,0],
         'tau_raw':[0]*5,'tau_cap':[1]*5,'tau':[0]*5,'planned':False,
         'goal_time':-1,'fallback':False}
    (dest/'diagnostics.jsonl').write_text(json.dumps(row)+'\n')
    runner=time_window_search.SearchRunner.__new__(time_window_search.SearchRunner)
    runner.raw={};runner.team=root
    result=runner.evaluate_trial(dest,{'trajectory_id':'test_01','input_end_s':1,'bounce_labels':[]})
    assert not result['planned'] and not result['geometric']['contact']
    assert result['motor_audit_passed']
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
'''
        result = subprocess.run([sys.executable, "-c", code,
            str(Path(__file__).resolve().parents[1] / "tools")], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_policy_merge_retains_defaults_and_rejects_non_timing_knobs(self):
        updated = search.merged_policy(search.BASE_POLICY, {"pfly_init_delay_s": .1, "nmpc6": {"stop_s": .02}})
        self.assertEqual(updated["nmpc6"]["strict_window_s"], .1)
        self.assertEqual(updated["nmpc6"]["stop_s"], .02)
        self.assertEqual(search.BASE_POLICY["pfly_init_delay_s"], .25)
        with self.assertRaises(ValueError):
            search.merged_policy(search.BASE_POLICY, {"torque_cap": 999})
        with self.assertRaises(ValueError):
            search.merged_policy(search.BASE_POLICY, {"pfly_duration_min_s": 2})
        with self.assertRaises(ValueError):
            search.merged_policy(search.BASE_POLICY, {"planner_min_lead_s": 0})
        with self.assertRaises(ValueError):
            search.merged_policy(search.BASE_POLICY, {"planner_max_horizon_s": .9})
        with self.assertRaises(ValueError):
            search.merged_policy(search.BASE_POLICY, {"nmpc6": {"dt_ratios": [.2, .3, .5]}})
        valid = search.merged_policy(search.BASE_POLICY, {"nmpc6": {
            "dt_ratios": [.2, .3, .5], "allow_nonuniform_dt": True}})
        self.assertEqual(valid["nmpc6"]["dt_ratios"], [.2, .3, .5])
        with self.assertRaises(ValueError):
            search.merged_policy(search.BASE_POLICY, {"nmpc6": {
                "dt_ratios": [1 / 3 + 1e-10, 1 / 3 - 1e-10, 1 / 3]}})

    def test_discovery_confirmation_split_is_fixed_disjoint_and_complete(self):
        ids = [f"test_{number:02}" for number in range(1, 36)]
        split = search.fixed_split(ids, {"seed": 31})
        self.assertEqual(split, search.fixed_split(list(reversed(ids)), {"seed": 31}))
        self.assertEqual(len(split["discovery"]), 18)
        self.assertEqual(len(split["confirmation"]), 17)
        self.assertFalse(set(split["discovery"]) & set(split["confirmation"]))
        bad = {"discovery": ids[:18], "confirmation": ids[17:34]}
        with self.assertRaises(ValueError):
            search.fixed_split(ids, bad)

    def test_incomplete_candidate_retains_denominator_and_is_ineligible(self):
        expected = ["a", "b"]
        job = {"id": "failed", "mode": "pfly"}
        failed = search.summarize(job, "discovery", expected, [metric("a")], [{"sample_id": "b"}])
        self.assertEqual(failed["denominator"], 2)
        self.assertIsNone(failed["approaching_contact_rate"])
        self.assertFalse(failed["complete"])
        valid = search.summarize({"id": "valid", "mode": "pfly"}, "discovery", expected,
                                 [metric("a", hit=False), metric("b", hit=False)], [])
        self.assertEqual(search.ranked_best([failed, valid])["job_id"], "valid")

    def test_paired_change_preserves_gained_lost_ids_and_marks_cache_change(self):
        baseline = search.summarize({"id": "base", "mode": "pfly"}, "discovery", ["a", "b"],
            [metric("a", hit=False), metric("b", hit=True)], [])
        candidate = search.summarize({"id": "candidate", "mode": "pfly"}, "discovery", ["a", "b"],
            [metric("a", hit=True, cache="new"), metric("b", hit=False, cache="new")], [])
        delta = search.paired_change(baseline, candidate)
        self.assertEqual(delta["contact"]["gained_sample_ids"], ["a"])
        self.assertEqual(delta["contact"]["lost_sample_ids"], ["b"])
        self.assertEqual(delta["contact"]["delta_percentage_points"], 0)
        self.assertFalse(delta["cache_hashes_identical"])
        candidate["records"][0]["input_sha256"] = "changed"
        with self.assertRaises(ValueError):
            search.paired_change(baseline, candidate)

    def test_runtime_must_acknowledge_policy_and_matching_init_delay(self):
        job = {"mode": "pfly", "controller_variant": "baseline", "steering": "both_unlimited",
               "policy": deepcopy(search.BASE_POLICY), "nlp_path": None}
        values = {"time_windows_path": "/tmp/policy.json", "time_windows": deepcopy(job["policy"]),
                  "base_mode": "pfly", "control_variant": "baseline", "steer_rate_mode": "both_unlimited",
                  "init_delay_s": .25}
        search.assert_runtime_consumed({"values": values}, job, Path("/tmp/policy.json"))
        values["init_delay_s"] = .1
        with self.assertRaises(RuntimeError):
            search.assert_runtime_consumed({"values": values}, job, Path("/tmp/policy.json"))
        with self.assertRaises(RuntimeError):
            search.assert_runtime_consumed(None, job, Path("/tmp/policy.json"))


class SearchRunnerFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="timing-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.team = self.root / "team"
        for directory in ("sckd_sim/build-predictor", "sckd_sim/models/generated", "sckd_sim/cpp/runtime",
                          "sckd_sim/third_party/nmpc6", "tools"):
            (self.team / directory).mkdir(parents=True)
        (self.team / "sckd_sim/CMakeLists.txt").write_text("# fixture\n")
        (self.team / "sckd_sim/models/generated/x4_tennis_demo.mjcf").write_text("<mujoco/>")
        (self.team / "sckd_sim/cpp/runtime/fixture.h").write_text("// unchanged source\n")
        (self.team / "sckd_sim/third_party/nmpc6/nlp.so").write_text("fixture solver")
        self.binary = self.team / "sckd_sim/build-predictor/m2_intercept"
        self.binary.write_text(f"#!{sys.executable}\n" + '''import json,os,sys
from pathlib import Path
p=Path(os.environ['M2_TIME_WINDOWS_PATH']);policy=json.loads(p.read_text())
Path(os.environ['M2_CONTROL_DIAGNOSTICS']).write_text('{}\\n')
values={'time_windows_path':str(p),'time_windows':policy,'control_variant':os.environ['M2_CONTROL_VARIANT'],
        'steer_rate_mode':os.environ['M2_STEER_RATE_MODE'],'base_mode':sys.argv[6],'init_delay_s':float(sys.argv[4]),
        'nmpc6_nlp_path':os.environ.get('M2_NMPC6_NLP','third_party/nmpc6/nlp.so')}
print('[resolved config] '+json.dumps({'values':values}))
calls=Path(__file__).resolve().parents[2]/'calls.jsonl'
with calls.open('a') as f:f.write(json.dumps({'policy':p.stem,'csv':os.environ['M2_MEASURED_REPLAY'],
    'EXP_POS':os.environ.get('EXP_POS'),'delay':sys.argv[4]})+'\\n')
''')
        self.binary.chmod(0o755)
        (self.team / "tools/control_ablation_cache.py").write_text('''import argparse,socket,time
p=argparse.ArgumentParser();p.add_argument('--serve');p.add_argument('--socket');a=p.parse_args()
s=socket.socket(socket.AF_UNIX);s.bind(a.socket);s.listen(1);time.sleep(30)
''')
        self.source = self.root / "source"
        self.source.mkdir()
        self.cache = self.root / "cache"
        self.cache.mkdir()
        items = []
        for number in range(1, 36):
            name = f"test_{number:02}"
            csv = self.source / (name + ".csv")
            csv.write_text("t_s,x_m,y_m,z_m\n0,0,0,1\n")
            (self.cache / (name + ".jsonl")).write_text("{}\n")
            items.append({"sample_id": name, "trajectory_id": name, "csv": str(csv)})
        (self.source / "input_manifest.json").write_text(json.dumps({"items": items}))
        self.output = self.root / "results"
        self.spec = {"schema": search.SEARCH_SCHEMA, "team": str(self.team), "source": str(self.source),
            "binary": str(self.binary), "out": str(self.output), "cache_map": {"h1500": str(self.cache)},
            "timeout_s": 2, "startup_timeout_s": 2,
            "jobs": [{"id": "base", "mode": "pfly"},
                     {"id": "optimized", "mode": "pfly", "patch": {"pfly_init_delay_s": .2}}]}
        self.spec_path = self.root / "search.json"
        self.spec_path.write_text(json.dumps(self.spec))

    def evaluator(self, path, item):
        phase = path.parents[2].name
        job = path.parents[1].name
        # Confirmation deliberately reverses discovery ranking. It must not reselect.
        hit = (job == "optimized") if phase == "discovery" else (job == "base")
        return metric(item["sample_id"], mode=path.parent.name, hit=hit)

    def runner(self, **kwargs):
        return search.SearchRunner(self.spec_path, trial_evaluator=self.evaluator, **kwargs)

    def test_trial_reuses_hashes_and_removes_acceptance_experiment_environment(self):
        runner = self.runner()
        job = runner.resolve_job(self.spec["jobs"][1])
        sample = runner.by_sample[runner.split["discovery"][0]]
        with patch.dict(os.environ, {"EXP_POS": "10"}):
            first = runner.run_trial(job, "discovery", sample)
            second = runner.run_trial(job, "discovery", sample)
        self.assertEqual(first, second)
        calls = [json.loads(line) for line in (self.team / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["EXP_POS"])
        self.assertEqual(float(calls[0]["delay"]), .2)
        (self.cache / (sample["sample_id"] + ".jsonl")).write_text('{"changed":true}\n')
        with self.assertRaisesRegex(RuntimeError, "resource changed during run"):
            runner.run_trial(job, "discovery", sample)

    def test_manifest_rejects_resource_drift_before_first_trial(self):
        runner = self.runner()
        job = runner.resolve_job(self.spec["jobs"][0])
        sample = runner.by_sample[runner.split["discovery"][0]]
        for path in (Path(sample["csv"]), self.cache / (sample["sample_id"] + ".jsonl"), self.binary):
            with self.subTest(resource=path.name):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    with self.assertRaisesRegex(RuntimeError, "resource changed during run"):
                        runner.run_trial(job, "discovery", sample)
                finally:
                    path.write_bytes(original)

    def test_full_policy_seed_preserves_original_baseline(self):
        seeded = search.merged_policy(search.BASE_POLICY, {"pfly_init_delay_s": .1,
            "nmpc6": {"freeze_window_s": .7}})
        self.spec["jobs"][1] = {"id": "seed", "mode": "pfly", "policy": seeded,
            "patch": {"target_lock_s": .12}}
        self.spec_path.write_text(json.dumps(self.spec))
        runner = self.runner()
        baseline = runner.resolve_job(self.spec["jobs"][0])
        seed = runner.resolve_job(self.spec["jobs"][1])
        self.assertEqual(baseline["policy"], search.BASE_POLICY)
        self.assertEqual(seed["policy"]["pfly_init_delay_s"], .1)
        self.assertEqual(seed["policy"]["nmpc6"]["freeze_window_s"], .7)
        self.assertEqual(seed["policy"]["target_lock_s"], .12)

    def test_confirmation_requires_discovery_selection_and_spec_never_overwrites(self):
        runner = self.runner()
        with self.assertRaisesRegex(RuntimeError, "frozen selection"):
            runner.confirm()
        self.spec["jobs"][1]["patch"]["pfly_init_delay_s"] = .1
        self.spec_path.write_text(json.dumps(self.spec))
        with self.assertRaises(FileExistsError):
            self.runner()

    def test_coordinate_base_uses_only_completed_discovery_candidates(self):
        self.spec["jobs"].append({"id": "coordinate", "mode": "pfly", "base_best_of": ["base", "optimized"],
                                  "patch": {"target_lock_s": .1}})
        self.spec_path.write_text(json.dumps(self.spec))
        runner = self.runner()
        for request in self.spec["jobs"][:2]:
            job = runner.resolve_job(request)
            expected = runner.split["discovery"]
            hits = request["id"] == "optimized"
            summary = search.summarize(job, "discovery", expected, [metric(sid, hit=hits) for sid in expected], [])
            summary["search_manifest_identity"] = runner.manifest_identity
            runner.store_summary(summary)
        coordinate = runner.resolve_job(self.spec["jobs"][-1])
        self.assertEqual(coordinate["base_job"], "optimized")
        self.assertEqual(coordinate["policy"]["pfly_init_delay_s"], .2)
        self.assertEqual(coordinate["policy"]["target_lock_s"], .1)

    def test_nonuniform_job_requires_explicit_independent_solver_library(self):
        self.spec["jobs"] = [{"id": "n6", "mode": "nmpc6", "patch": {
            "nmpc6": {"dt_ratios": [.2, .3, .5], "allow_nonuniform_dt": True}}}]
        self.spec_path.write_text(json.dumps(self.spec))
        runner = self.runner()
        with self.assertRaisesRegex(ValueError, "independent nlp_path"):
            runner.resolve_job(self.spec["jobs"][0])

    def test_search_never_evaluates_confirmation_and_confirmation_does_not_reselect(self):
        runner = self.runner()
        with redirect_stdout(io.StringIO()):
            selection = runner.search()
        self.assertEqual(selection["selected_by_mode"], {"pfly": "optimized"})
        self.assertFalse((self.output / "runs/confirmation").exists())
        discovery_calls = [json.loads(line) for line in (self.team / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(discovery_calls), 36)
        self.assertEqual({Path(row["csv"]).stem for row in discovery_calls}, set(runner.split["discovery"]))
        with patch.object(search, "ranked_best", side_effect=AssertionError("confirmation must not select")):
            with redirect_stdout(io.StringIO()):
                confirmed = runner.confirm()
        self.assertTrue(confirmed["complete"])
        self.assertEqual(confirmed["selected_by_mode"], {"pfly": "optimized"})
        self.assertEqual(confirmed["paired"]["pfly"]["confirmation"]["contact"]["delta_count"], -17)
        self.assertEqual(confirmed["paired"]["pfly"]["all35_after_selection"]["trials"], 35)
        with self.assertRaisesRegex(RuntimeError, "discovery is frozen"):
            runner.search()

    def test_technical_trial_failure_is_retained_and_excludes_job_from_selection(self):
        runner = self.runner()
        job = runner.resolve_job(self.spec["jobs"][0])
        bad = runner.split["discovery"][0]
        def run_trial(job, phase, item):
            if item["sample_id"] == bad:
                raise RuntimeError("fixture technical failure")
            return {**metric(item["sample_id"]), "experiment": {"identity": "fixture", "result_sha256": "fixture"}}
        with patch.object(runner, "run_trial", side_effect=run_trial):
            with redirect_stdout(io.StringIO()):
                summary = runner.run_job(job, "discovery")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["denominator"], 18)
        self.assertEqual(summary["completed_trials"], 17)
        self.assertEqual(summary["technical_failures"][0]["sample_id"], bad)
        self.assertIsNone(summary["geometry_contact_rate"])


if __name__ == "__main__":
    unittest.main()
