"""Ubuntu 时间窗口搜索：冻结18条发现集，17条只用于事先锁定候选的确认。

读取 TIME_WINDOW_SEARCH/1 JSON；不内置搜索矩阵，不改源码/力矩/速度/拍速门。
每个任务给出id/mode，以及policy、patch、base_job或base_best_of。路径相对spec。
缓存须事先生成，cache_map将horizon标签映射到冻结cache目录；本程序不重训或
偷偷重建预测缓存。全部35条为反复使用的历史test，确认结果也不是新盲测。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

from pipeline_experiment import (ExperimentSpec, RunAttempt, STEER_RATE_MODES,
    clean_environment, digest, file_resources, recorded_environment, resolved_runtime_config,
    run_logged, socket_path, socket_service, team_resources, write_json)

SEARCH_SCHEMA = "TIME_WINDOW_SEARCH/1"
POLICY_SCHEMA = "SCKD_TIME_WINDOWS/1"
BASE_POLICY = {"schema": POLICY_SCHEMA, "planner_min_lead_s": .9, "planner_max_horizon_s": 2.5,
    "target_lock_s": .15, "refine_before_arm_active": False, "pfly_init_delay_s": .25,
    "pfly_duration_min_s": .30, "pfly_duration_max_s": 1.8, "pfly_legacy_lead_s": .020,
    "pfly_ready_transition_s": .25, "nmpc6": {"dt_ratios": [1 / 3, 1 / 3, 1 / 3],
    "freeze_window_s": 1., "stop_s": .012, "strict_window_s": .100, "allow_nonuniform_dt": False}}
SELECTION_RULE = ["approaching_contacts_desc", "geometry_contacts_desc", "fallback_trials_asc",
                  "minimum_gap_p95_asc", "job_id_asc"]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_once(path, payload):
    """同内容只读复用；不覆盖历史spec、策略或已冻结的选型。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise FileExistsError(f"different existing artifact preserved: {path}; use a new output directory")
    else:
        write_json(path, payload, exclusive=True)


def merged_policy(base, patch):
    if not isinstance(patch, dict):
        raise ValueError("policy/patch must be an object")
    result = deepcopy(base)
    for key, value in patch.items():
        if key not in BASE_POLICY:
            raise ValueError(f"unknown timing field (non-timing knobs are forbidden): {key}")
        if key == "nmpc6":
            if not isinstance(value, dict) or set(value) - set(BASE_POLICY["nmpc6"]):
                raise ValueError("unknown nmpc6 timing field")
            result[key].update(deepcopy(value))
        else:
            result[key] = deepcopy(value)
    if result["schema"] != POLICY_SCHEMA:
        raise ValueError("unsupported timing policy schema")
    for key, value in result.items():
        if key in ("schema", "nmpc6"):
            continue
        if key == "refine_before_arm_active":
            if type(value) is not bool:
                raise ValueError("refine_before_arm_active must be boolean")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"timing field must be finite and nonnegative: {key}")
    if result["planner_min_lead_s"] <= 0 or result["pfly_duration_min_s"] <= 0:
        raise ValueError("planner lead and duration minimum must be positive")
    if result["pfly_duration_min_s"] > result["pfly_duration_max_s"]:
        raise ValueError("duration minimum exceeds maximum")
    if result["planner_min_lead_s"] >= result["planner_max_horizon_s"]:
        raise ValueError("planner horizon must exceed minimum lead")
    nmpc = result["nmpc6"]
    if type(nmpc["allow_nonuniform_dt"]) is not bool:
        raise ValueError("allow_nonuniform_dt must be boolean")
    ratios = nmpc["dt_ratios"]
    if not isinstance(ratios, list) or len(ratios) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("dt_ratios requires three finite positive values")
    if not math.isclose(sum(ratios), 1., rel_tol=0, abs_tol=1e-9):
        raise ValueError("dt_ratios must sum to one")
    if not nmpc["allow_nonuniform_dt"] and not ratios[0] == ratios[1] == ratios[2]:
        raise ValueError("nonuniform dt requires explicit allow_nonuniform_dt")
    for key in ("freeze_window_s", "stop_s", "strict_window_s"):
        value = nmpc[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid nmpc6 timing value: {key}")
    return result


def fixed_split(sample_ids, specification):
    ids = sorted(sample_ids)
    if len(ids) != 35 or len(set(ids)) != 35:
        raise ValueError("the fixed historical test protocol requires exactly 35 unique sample IDs")
    if "discovery" in specification or "confirmation" in specification:
        discovery = list(specification.get("discovery", []))
        confirmation = list(specification.get("confirmation", []))
    else:
        shuffled = list(ids)
        random.Random(specification.get("seed", 20261019)).shuffle(shuffled)
        discovery, confirmation = sorted(shuffled[:18]), sorted(shuffled[18:])
    if len(discovery) != 18 or len(confirmation) != 17 or len(set(discovery + confirmation)) != 35 or set(discovery + confirmation) != set(ids):
        raise ValueError("split must cover all35 exactly once as discovery18/confirmation17")
    return {"discovery": discovery, "confirmation": confirmation,
            "historical_data_not_blind": True, "selection_data": "discovery_only"}


def ranked_best(summaries):
    valid = [summary for summary in summaries if summary.get("complete")]
    if not valid:
        raise ValueError("no complete discovery candidate is available for selection")
    def score(summary):
        gap = summary["minimum_gap_m"]["p95"]
        return (-summary["approaching_contacts"], -summary["geometry_contacts"],
                summary["fallback_trials"], math.inf if gap is None else gap, summary["job_id"])
    return min(valid, key=score)


def distribution(values):
    import numpy as np
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "median": None, "p95": None, "mean": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite metric")
    return {"n": len(values), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "mean": float(np.mean(values)), "max": max(values)}


def summarize(job, phase, expected, records, failures):
    samples = [record["sample_id"] for record in records]
    if len(samples) != len(set(samples)) or set(samples) - set(expected):
        raise ValueError("duplicate/unexpected trial records")
    complete = not failures and set(samples) == set(expected)
    count = len(expected)
    geometric = sum(record["geometric"]["contact"] for record in records)
    approaching = sum(record["geometric"]["approaching"] for record in records)
    return {"schema": "TIME_WINDOW_JOB_METRICS/1", "job_id": job["id"], "mode": job["mode"],
        "phase": phase, "complete": complete, "expected_trials": count, "completed_trials": len(records),
        "sample_ids": list(expected), "denominator": count, "technical_failures": failures,
        "geometry_contacts": geometric, "approaching_contacts": approaching,
        "geometry_contact_rate": geometric / count if complete else None,
        "approaching_contact_rate": approaching / count if complete else None,
        "native_geometry_contacts": sum(record.get("native_at_geometry_contact", False) for record in records),
        "fallback_trials": sum(record["fallback"] for record in records),
        "planned_trials": sum(record["planned"] for record in records),
        "covered_trials": sum(record["hit_time_covered"] for record in records),
        "minimum_gap_m": distribution([record["geometric"]["min_gap_m"] for record in records]),
        "actual_to_goal_m": distribution([record.get("actual_to_goal_m") for record in records]),
        "solve_ms": distribution([value for record in records for value in record.get("solve_times_ms", [])]),
        "solver_calls": sum(record["solver_calls"] for record in records),
        "applied_solutions": sum(record["applied_solutions"] for record in records),
        "qp_attempts": sum(record.get("qp_attempts", 0) for record in records),
        "qp_accepts": sum(record.get("qp_accepts", 0) for record in records),
        "resolved_job": job, "records": records,
        "interpretation": "historical measured-reference geometric interception; includes fallback; not physical outgoing-ball hit rate"}


def paired_change(baseline, candidate):
    if not baseline["complete"] or not candidate["complete"]:
        raise ValueError("paired comparison requires complete trials on both sides")
    left = {record["sample_id"]: record for record in baseline["records"]}
    right = {record["sample_id"]: record for record in candidate["records"]}
    if set(left) != set(right):
        raise ValueError("paired comparison sample sets differ")
    if any(left[sid]["input_sha256"] != right[sid]["input_sha256"] for sid in left):
        raise ValueError("paired comparison input CSV content changed")
    out = {"baseline_job": baseline["job_id"], "candidate_job": candidate["job_id"],
           "trials": len(left), "input_hashes_identical": True,
           "cache_hashes_identical": all(left[sid]["cache_sha256"] == right[sid]["cache_sha256"] for sid in left)}
    for field in ("contact", "approaching"):
        gained = sorted(sid for sid in left if not left[sid]["geometric"][field] and right[sid]["geometric"][field])
        lost = sorted(sid for sid in left if left[sid]["geometric"][field] and not right[sid]["geometric"][field])
        out[field] = {"gained_sample_ids": gained, "lost_sample_ids": lost,
                      "delta_count": len(gained) - len(lost), "delta_percentage_points": 100 * (len(gained) - len(lost)) / len(left)}
    out["fallback_delta"] = candidate["fallback_trials"] - baseline["fallback_trials"]
    return out


def assert_runtime_consumed(runtime, job, policy_path):
    if not isinstance(runtime, dict) or not isinstance(runtime.get("values"), dict):
        raise RuntimeError("candidate binary did not report a resolved runtime configuration")
    values = runtime["values"]
    expected = {"time_windows_path": str(policy_path), "control_variant": job["controller_variant"],
                "steer_rate_mode": job["steering"], "base_mode": job["mode"],
                "init_delay_s": job["policy"]["pfly_init_delay_s"]}
    for key, value in expected.items():
        if values.get(key) != value:
            raise RuntimeError(f"binary did not consume requested {key}: {values.get(key)!r} != {value!r}")
    if values.get("time_windows") != job["policy"]:
        raise RuntimeError("binary resolved time window policy differs from the requested policy")
    if job.get("nlp_path") and Path(values.get("nmpc6_nlp_path", "")).resolve() != Path(job["nlp_path"]).resolve():
        raise RuntimeError("binary did not consume explicit NMPC solver library")


class SearchRunner:
    def __init__(self, spec_path, *, out=None, retry_incomplete=False, trial_evaluator=None):
        self.spec_path = Path(spec_path).resolve()
        self.raw = json.loads(self.spec_path.read_text())
        if self.raw.get("schema") != SEARCH_SCHEMA:
            raise ValueError(f"search spec schema must be {SEARCH_SCHEMA}")
        self.base = self.spec_path.parent
        self.team = self.path(self.raw["team"])
        self.source = self.path(self.raw["source"])
        self.out = Path(out).resolve() if out is not None else self.path(self.raw["out"])
        self.binary = self.path(self.raw["binary"])
        self.retry = retry_incomplete
        self.evaluator = trial_evaluator or self.evaluate_trial
        self.controller_variant = self.raw.get("controller_variant", "baseline")
        if self.controller_variant != "baseline":
            raise ValueError("time-window search fixes controller_variant=baseline; controller/acceptance changes are out of scope")
        self.steering = self.raw.get("steering", "both_unlimited")
        if self.steering not in STEER_RATE_MODES:
            raise ValueError("unknown explicit steering mode")
        policy = self.raw.get("policy", {})
        if isinstance(policy, str):
            policy = json.loads(self.path(policy).read_text())
        self.policy = merged_policy(BASE_POLICY, policy)
        self.items = json.loads((self.source / "input_manifest.json").read_text())["items"]
        self.by_sample = {item["sample_id"]: item for item in self.items}
        self.split = fixed_split([item["sample_id"] for item in self.items], self.raw.get("split", {}))
        for item in self.items:
            csv = Path(item["csv"])
            item["csv"] = str((csv if csv.is_absolute() else self.source / csv).resolve())
            for field in ("source_sha256", "input_sha256"):
                if item.get(field) and item[field] != digest(item["csv"]):
                    raise ValueError(f"input manifest content mismatch: {item['sample_id']}")
        self.cache_map = {}
        for key, value in self.raw["cache_map"].items():
            metadata = {"path": value} if isinstance(value, str) else dict(value)
            metadata["path"] = str(self.path(metadata["path"]))
            self.cache_map[key] = metadata
        if "h1500" not in self.cache_map:
            raise ValueError("cache_map must explicitly provide default h1500")
        self.jobs = self.raw["jobs"]
        maximum = self.raw.get("max_configs", 80)
        if not isinstance(maximum, int) or not 1 <= maximum <= 80 or not 1 <= len(self.jobs) <= maximum:
            raise ValueError("search requires 1..max_configs jobs with max_configs<=80")
        self.by_job = {}
        for job in self.jobs:
            name = job["id"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name) or name in self.by_job:
                raise ValueError("job IDs must be unique safe path components")
            if job["mode"] not in ("nmpc6", "pfly"):
                raise ValueError("time-window search supports nmpc6 and pfly")
            parents = ([job["base_job"]] if job.get("base_job") else []) + job.get("base_best_of", [])
            if job.get("base_job") and job.get("base_best_of"):
                raise ValueError("use base_job or base_best_of, not both")
            for parent in parents:
                if parent not in self.by_job or self.by_job[parent]["mode"] != job["mode"]:
                    raise ValueError("coordinate-search bases must be earlier jobs of the same method")
            self.by_job[name] = job
        self.baselines = dict(self.raw.get("baseline_jobs", {}))
        for job in self.jobs:
            self.baselines.setdefault(job["mode"], job["id"])
        for mode, name in self.baselines.items():
            if name not in self.by_job or self.by_job[name]["mode"] != mode:
                raise ValueError("invalid baseline_jobs mapping")
        self.out.mkdir(parents=True, exist_ok=True)
        resources = {"input_manifest": self.source / "input_manifest.json",
                     "search_spec": self.spec_path, "runner": Path(__file__),
                     "runner_common": Path(__file__).with_name("pipeline_experiment.py"),
                     "cache_service": self.team / "tools/control_ablation_cache.py",
                     "geometry_evaluator": Path(__file__).with_name("interception_metrics.py"),
                     "geometry_adapter": Path(__file__).with_name("analyze_control_ablation.py"),
                     "diagnostic_metrics": Path(__file__).with_name("control_diagnostic_metrics.py")}
        if isinstance(self.raw.get("policy"), str):
            resources["base_policy"] = self.path(self.raw["policy"])
        for job in self.jobs:
            if job.get("binary"):
                resources["candidate_binary/" + job["id"]] = self.path(job["binary"])
            if job.get("nlp_path"):
                resources["candidate_nlp/" + job["id"]] = self.path(job["nlp_path"])
        if self.raw.get("nlp_path"):
            resources["default_nlp"] = self.path(self.raw["nlp_path"])
        for item in self.items:
            resources["input/" + item["sample_id"]] = item["csv"]
        for key, cache in self.cache_map.items():
            manifest = Path(cache["path"]) / "manifest.json"
            if manifest.exists():
                resources["cache_manifest/" + key] = manifest
            for item in self.items:
                resources[f"frozen_cache/{key}/{item['sample_id']}"] = Path(cache["path"]) / (item["sample_id"] + ".jsonl")
        self.resources = file_resources(resources)
        self.runtime_resources = {mode: team_resources(self.team, mode, binary=self.binary) for mode in self.baselines}
        manifest = {"schema": "TIME_WINDOW_SEARCH_MANIFEST/1", "spec": self.raw,
            "resources": self.resources, "runtime_resources": self.runtime_resources,
            "split": self.split, "baseline_jobs": self.baselines, "selection_rule": SELECTION_RULE,
            "controller_variant": self.controller_variant, "steering": self.steering,
            "policy_defaults": self.policy, "historical_data_not_blind": True}
        write_once(self.out / "search_manifest.json", manifest)
        self.manifest_identity = identity(manifest)

    def path(self, value):
        path = Path(value)
        return (path if path.is_absolute() else self.base / path).resolve()

    def summary_path(self, phase, name):
        return self.out / "summaries" / phase / (name + ".json")

    def read_summary(self, phase, name):
        path = self.summary_path(phase, name)
        if not path.exists():
            raise FileNotFoundError(f"required earlier discovery result is unavailable: {path}")
        result = json.loads(path.read_text())
        if result.get("search_manifest_identity") != self.manifest_identity or result["phase"] != phase:
            raise ValueError("summary does not belong to this frozen search/split")
        return result

    def resolve_job(self, request):
        parent = None
        considered = []
        if request.get("base_best_of"):
            considered = [self.read_summary("discovery", name) for name in request["base_best_of"]]
            parent = ranked_best(considered)["resolved_job"]
        elif request.get("base_job"):
            summary = self.read_summary("discovery", request["base_job"])
            if not summary["complete"]:
                raise ValueError("base_job is incomplete")
            parent = summary["resolved_job"]
        policy = deepcopy(parent["policy"] if parent else self.policy)
        if "policy" in request:
            policy = merged_policy(policy, request["policy"])
        policy = merged_policy(policy, request.get("patch", {}))
        cache_id = request.get("cache_id", parent["cache_id"] if parent else "h1500")
        if cache_id not in self.cache_map:
            raise ValueError(f"unknown frozen cache_id: {cache_id}")
        binary = str(self.path(request["binary"])) if request.get("binary") else (parent["binary"] if parent else str(self.binary))
        nlp = str(self.path(request["nlp_path"])) if request.get("nlp_path") else (
            parent.get("nlp_path") if parent else str(self.path(self.raw["nlp_path"])) if self.raw.get("nlp_path") else None)
        if request["mode"] == "nmpc6" and policy["nmpc6"]["allow_nonuniform_dt"] and not nlp:
            raise ValueError("nonuniform NMPC dt requires an explicit independent nlp_path")
        resolved = {"id": request["id"], "mode": request["mode"], "policy": policy,
            "cache_id": cache_id, "cache": self.cache_map[cache_id], "binary": binary,
            "nlp_path": nlp, "controller_variant": self.controller_variant, "steering": self.steering,
            "base_job": parent["id"] if parent else None,
            "base_best_of": request.get("base_best_of", []),
            "ineligible_base_candidates": [x["job_id"] for x in considered if not x["complete"]]}
        write_once(self.out / "policies" / (request["id"] + ".json"), policy)
        write_once(self.out / "jobs" / request["id"] / "resolved_job.json", resolved)
        return resolved

    def frozen_resource(self, path):
        """Return the construction-time identity, including inherited binary/NLP choices."""
        wanted = str(Path(path).resolve())
        sources = [self.resources, *self.runtime_resources.values()]
        for resources in sources:
            for resource in resources.values():
                if resource["path"] == wanted:
                    return dict(resource)
        raise ValueError(f"trial requested a resource absent from its frozen search manifest: {wanted}")

    def run_trial(self, job, phase, item):
        name = item["sample_id"]
        dest = self.out / "runs" / phase / job["id"] / job["mode"] / name
        policy_path = self.out / "policies" / (job["id"] + ".json")
        cache = Path(job["cache"]["path"]) / (name + ".jsonl")
        sock = socket_path(self.team, "timewin")
        overrides = {"M2_CONTROL_VARIANT": job["controller_variant"], "M2_STEER_RATE_MODE": job["steering"],
            "M2_CONTROL_DIAGNOSTICS": str(dest / "diagnostics.jsonl"), "M2_MEASURED_REPLAY": item["csv"],
            "BALL_PREDICTOR_SOCKET": str(sock), "M2_TIME_WINDOWS_PATH": str(policy_path)}
        if job.get("nlp_path"):
            overrides["M2_NMPC6_NLP"] = job["nlp_path"]
        env = clean_environment(overrides)
        # Existing EXP_* acceptance experiments must not leak into timing-only baseline search.
        for key in list(env):
            if key.startswith("EXP_"):
                del env[key]
        delay = job["policy"]["pfly_init_delay_s"]
        scene = self.raw.get("scene", "models/generated/x4_tennis_demo.mjcf")
        command = [job["binary"], scene, "1", "qp", format(delay, ".17g"),
                   str(self.raw.get("seed", 2026)), job["mode"], "measured", "xb"]
        service = [sys.executable, str(self.team / "tools/control_ablation_cache.py"),
                   "--serve", str(cache), "--socket", "{socket}"]
        extras = {"timing_policy": policy_path,
                  "resolved_job": self.out / "jobs" / job["id"] / "resolved_job.json"}
        base_resources = self.runtime_resources[job["mode"]]
        # The search manifest freezes all35/cache variants once. A trial rechecks only
        # the files it consumes, avoiding rehashing every other trajectory per solve.
        common = {key: value for key, value in self.resources.items() if key in (
            "input_manifest", "search_spec", "runner", "runner_common", "cache_service", "geometry_evaluator",
            "geometry_adapter", "diagnostic_metrics", "base_policy")}
        cache_manifest = "cache_manifest/" + job["cache_id"]
        if cache_manifest in self.resources:
            common[cache_manifest] = self.resources[cache_manifest]
        resources = {**common, **base_resources, **file_resources(extras),
                     "input": self.frozen_resource(item["csv"]),
                     "cache": self.frozen_resource(cache),
                     "binary": self.frozen_resource(job["binary"])}
        if job.get("nlp_path"):
            resources["selected_nlp"] = self.frozen_resource(job["nlp_path"])
        timeout = self.raw.get("timeout_s", 600)
        startup = self.raw.get("startup_timeout_s", 5)
        spec = ExperimentSpec("time_window_search", tuple(command), str(self.team / "sckd_sim"),
            {"search_manifest_identity": self.manifest_identity, "phase": phase, "resolved_job": job,
             "sample_id": name, "trajectory_id": item["trajectory_id"], "timeout_s": timeout,
             "startup_timeout_s": startup, "geometry_step_s": self.raw.get("geometry_step_s", .001),
             "all_expected_sample_ids": self.split[phase]}, recorded_environment(env, sock), resources, tuple(service))
        with RunAttempt(dest, spec, retry_incomplete=self.retry) as attempt:
            if attempt.existing is not None:
                return attempt.existing
            started = time.monotonic()
            with socket_service([str(sock) if value == "{socket}" else value for value in service],
                    socket=sock, cwd=self.team, env=env, log=dest / "cache_service.log", startup_timeout=startup):
                process = run_logged(command, cwd=self.team / "sckd_sim", env=env,
                                     log=dest / "control.log", timeout=timeout)
            if process.returncode not in (0, 1) or not (dest / "diagnostics.jsonl").is_file():
                raise RuntimeError(f"trial exited {process.returncode} or did not record diagnostics")
            runtime = resolved_runtime_config((dest / "control.log").read_text())
            assert_runtime_consumed(runtime, job, policy_path)
            metric = self.evaluator(dest, item)
            record = {**metric, "job_id": job["id"], "phase": phase, "variant": "baseline",
                "steer_rate_mode": job["steering"], "cache_id": job["cache_id"],
                "sample_id": name, "trajectory_id": item["trajectory_id"],
                "binary_sha256": resources["binary"]["sha256"], "input_sha256": resources["input"]["sha256"],
                "cache_sha256": resources["cache"]["sha256"], "policy_sha256": resources["timing_policy"]["sha256"],
                "returncode": process.returncode, "wall_s": time.monotonic() - started,
                "resolved_runtime_config": runtime}
            write_json(dest / "metrics.json", record, exclusive=True)
            return attempt.complete(record)

    def evaluate_trial(self, dest, item):
        import mujoco
        from analyze_control_ablation import contact, motor_audit
        from control_diagnostic_metrics import read_case
        states, events, applies, metric = read_case(dest, item)
        motor_audit(states)
        scene = self.raw.get("scene", "models/generated/x4_tennis_demo.mjcf")
        if not hasattr(self, "_evaluation_model"):
            self._evaluation_model = mujoco.MjModel.from_xml_path(str(self.team / "sckd_sim" / scene))
        model = self._evaluation_model
        geometric = contact(model, mujoco.MjData(model), states, item, metric["hit_time_s"], self.raw.get("geometry_step_s", .001))
        active = None
        if geometric.get("first_contact_s") is not None:
            at = max((row for row in states if row["t"] <= geometric["first_contact_s"]), key=lambda row: row["t"], default=states[0])
            active = at["active"]
            if active == "pfly_or_wait":
                active = "pfly" if at["arm_qp_active"] else "wait"
        return {**metric, "geometric": geometric, "motor_audit_passed": True,
            "active_at_geometry_contact": active, "native_at_geometry_contact": geometric["contact"] and active == metric["mode"],
            "qp_attempts": states[-1].get("qp_attempts", 0), "qp_accepts": states[-1].get("qp_accepts", 0),
            "solve_times_ms": [event["ms"] for event in events]}

    def store_summary(self, summary):
        path = self.summary_path(summary["phase"], summary["job_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = json.loads(path.read_text())
            if old != summary:
                write_once(self.out / "summary_history" / summary["phase"] / summary["job_id"] / (identity(old) + ".json"), old)
        write_json(path, summary)

    def run_job(self, job, phase):
        expected = self.split[phase]
        records, failures = [], []
        for sample in expected:
            try:
                result = self.run_trial(job, phase, self.by_sample[sample])
                compact = {key: value for key, value in result.items() if key != "experiment"}
                compact["run_identity"] = result["experiment"]["identity"]
                compact["run_result_sha256"] = result["experiment"]["result_sha256"]
                records.append(compact)
            except Exception as error:
                manifest = self.out / "runs" / phase / job["id"] / job["mode"] / sample / "run_manifest.json"
                previous = json.loads(manifest.read_text()) if manifest.exists() else {}
                detail = previous.get("error", {"type": type(error).__name__, "message": str(error)})
                failures.append({"sample_id": sample, "error": detail,
                                 "run_identity": previous.get("identity"), "status": "technical_failure"})
            write_json(self.out / "progress.json", {"phase": phase, "job_id": job["id"],
                "attempted_trials": len(records) + len(failures), "completed_trials": len(records),
                "technical_failures": len(failures), "expected_trials": len(expected), "current_sample": sample})
            print(f"{phase} {job['id']} {len(records)+len(failures)}/{len(expected)} complete={len(records)} failures={len(failures)}", flush=True)
        summary = summarize(job, phase, expected, records, failures)
        summary["search_manifest_identity"] = self.manifest_identity
        self.store_summary(summary)
        return summary

    def search(self, only_jobs=None):
        if (self.out / "confirmation_plan.json").exists():
            raise RuntimeError("discovery is frozen after confirmation starts; use --phase confirm to resume")
        requested = set(only_jobs) if only_jobs else set(self.by_job)
        if requested - set(self.by_job):
            raise ValueError("unknown requested job ID")
        for request in self.jobs:
            if request["id"] in requested:
                self.run_job(self.resolve_job(request), "discovery")
        missing = [name for name in self.by_job if not self.summary_path("discovery", name).exists()]
        if missing:
            return {"complete": False, "pending_jobs": missing, "confirmation_started": False}
        summaries = {name: self.read_summary("discovery", name) for name in self.by_job}
        selected = {}
        for mode, baseline_id in self.baselines.items():
            if not summaries[baseline_id]["complete"]:
                raise RuntimeError(f"baseline {baseline_id} has technical failures; selection cannot proceed")
            candidates = [summary for summary in summaries.values() if summary["mode"] == mode]
            selected[mode] = ranked_best(candidates)["job_id"]
            for candidate in candidates:
                if candidate["complete"]:
                    write_once(self.out / "paired" / "discovery" / (candidate["job_id"] + ".json"),
                               paired_change(summaries[baseline_id], candidate))
        selection = {"schema": "TIME_WINDOW_SELECTION/1", "search_manifest_identity": self.manifest_identity,
            "selection_phase": "discovery", "selection_rule": SELECTION_RULE, "selected_by_mode": selected,
            "baseline_jobs": self.baselines, "split": self.split,
            "ineligible_jobs": [name for name, summary in summaries.items() if not summary["complete"]],
            "discovery_summary_sha256": {name: digest(self.summary_path("discovery", name)) for name in self.by_job},
            "historical_data_not_blind": True}
        write_once(self.out / "selection.json", selection)
        return selection

    def confirm(self):
        selection_path = self.out / "selection.json"
        if not selection_path.exists():
            raise RuntimeError("confirmation requires a frozen selection from --phase search")
        selection = json.loads(selection_path.read_text())
        if selection.get("search_manifest_identity") != self.manifest_identity or selection.get("selection_phase") != "discovery":
            raise ValueError("selection does not belong to this discovery-only search")
        for mode, name in selection["selected_by_mode"].items():
            if name not in self.by_job or self.by_job[name]["mode"] != mode:
                raise ValueError("selection contains an invalid method/job mapping")
        # Verify identities, never reselect using confirmation outcomes.
        for name, checksum in selection["discovery_summary_sha256"].items():
            if digest(self.summary_path("discovery", name)) != checksum:
                raise ValueError("discovery results changed after selection was frozen")
        wanted = []
        for mode in sorted(self.baselines):
            for name in (self.baselines[mode], selection["selected_by_mode"][mode]):
                if name not in wanted:
                    wanted.append(name)
        write_once(self.out / "confirmation_plan.json", {"schema": "TIME_WINDOW_CONFIRMATION_PLAN/1",
            "selection_sha256": digest(selection_path), "job_ids": wanted,
            "sample_ids": self.split["confirmation"], "selection_data": "discovery_only"})
        confirmations = {}
        for name in wanted:
            job = self.read_summary("discovery", name)["resolved_job"]
            confirmations[name] = self.run_job(job, "confirmation")
        all35 = {}
        for name in wanted:
            discovered = self.read_summary("discovery", name)
            confirmed = confirmations[name]
            combined = summarize(discovered["resolved_job"], "all35_after_selection",
                sorted(self.by_sample), discovered["records"] + confirmed["records"],
                discovered["technical_failures"] + confirmed["technical_failures"])
            combined["search_manifest_identity"] = self.manifest_identity
            combined["selection_bias_note"] = "18 discovery cases participated in selecting this candidate; all35 is not an independent validation"
            self.store_summary(combined)
            all35[name] = combined
        paired = {}
        for mode, baseline_id in self.baselines.items():
            best_id = selection["selected_by_mode"][mode]
            paired[mode] = {}
            for phase, table in (("confirmation", confirmations), ("all35_after_selection", all35)):
                if table[baseline_id]["complete"] and table[best_id]["complete"]:
                    delta = paired_change(table[baseline_id], table[best_id])
                    paired[mode][phase] = delta
                    write_once(self.out / "paired" / phase / (mode + ".json"), delta)
        result = {"schema": "TIME_WINDOW_CONFIRMATION/1", "selection_sha256": digest(selection_path),
            "complete": all(summary["complete"] for summary in confirmations.values()),
            "selected_by_mode": selection["selected_by_mode"], "baseline_jobs": self.baselines,
            "paired": paired, "historical_data_not_blind": True,
            "confirmation": {name: {key: value for key, value in summary.items() if key != "records"}
                             for name, summary in confirmations.items()}}
        write_json(self.out / "confirmation_summary.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("search", "confirm"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", help="search only: run a comma-separated subset; selection waits for all jobs")
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args()
    if args.phase == "confirm" and args.jobs:
        parser.error("confirmation job IDs are determined by the frozen discovery selection")
    runner = SearchRunner(args.spec, out=args.out, retry_incomplete=args.retry_incomplete)
    result = runner.search(args.jobs.split(",") if args.jobs else None) if args.phase == "search" else runner.confirm()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.phase == "confirm" and not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
