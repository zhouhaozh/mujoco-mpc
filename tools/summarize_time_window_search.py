"""Ubuntu 冻结时间窗口搜索终审；仅新增 JSON/中文 Markdown，不重选或绘图。

主口径保持计划击球时刻前后各 0.35s、有限盒球几何与 -0.30m/s 接近门。
另以相同适配器复核 0.5ms 分类，及前后各 10s 的已规划全程次口径。
原生控制贡献在同一网格上屏蔽非原生状态后单独计算 approaching。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

from pipeline_experiment import (ExperimentSpec, RunAttempt, digest, file_resources,
                                 resolved_runtime_config, write_json)
from time_window_search import (SELECTION_RULE, assert_runtime_consumed, fixed_split,
                                ranked_best, summarize)

SCHEMA = "TIME_WINDOW_FINAL_ANALYSIS/1"
PHASES = ("discovery", "confirmation", "all35_after_selection")
SCOPES = ("primary_1ms", "primary_0_5ms", "episode_1ms")


def identity(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


class HashAudit:
    """一个路径只能有一个冻结内容；开始和完成前各核验一次。"""
    def __init__(self):
        self.files = {}

    def verify(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        key = str(path)
        if key in self.files:
            require(expected is None or self.files[key] == expected,
                    f"conflicting frozen SHA256: {path}")
            return self.files[key]
        actual = digest(path)
        require(expected is None or actual == expected, f"SHA256 mismatch: {path}")
        self.files[key] = actual
        return actual

    def resources(self, resources):
        require(isinstance(resources, dict) and resources, "empty resource inventory")
        for resource in resources.values():
            self.verify(resource["path"], resource["sha256"])

    def recheck(self):
        for path, expected in sorted(self.files.items()):
            require(digest(path) == expected, f"resource changed during analysis: {path}")

    def inventory(self):
        return [{"path": path, "sha256": sha} for path, sha in sorted(self.files.items())]


def audit_completed_run(path, audit, *, completion="run.json"):
    path = Path(path)
    result = load(path / completion)
    metadata = result.get("experiment", {})
    require(metadata.get("schema") == "PIPELINE_EXPERIMENT/1"
            and metadata.get("status") == "completed", f"uncompleted experiment: {path}")
    require(metadata.get("identity") == identity(metadata["spec"]), f"spec identity mismatch: {path}")
    metric = {key: value for key, value in result.items() if key != "experiment"}
    require(metadata.get("result_sha256") == identity(metric), f"result identity mismatch: {path}")
    require(load(path / "run_manifest.json") == metadata, f"run manifest differs: {path}")
    audit.resources(metadata["spec"]["resources"])
    artifacts = metadata.get("artifacts")
    require(isinstance(artifacts, dict) and artifacts, f"missing output hashes: {path}")
    for relative, sha in artifacts.items():
        artifact = (path / relative).resolve()
        require(artifact.is_relative_to(path.resolve()), f"artifact escaped run directory: {artifact}")
        audit.verify(artifact, sha)
    audit.verify(path / completion)
    audit.verify(path / "run_manifest.json")
    return result


def exact_mcnemar(gained, lost):
    """双侧精确 McNemar：2*Binom(n, .5) 下较小不一致格的左尾。"""
    require(type(gained) is int and type(lost) is int and min(gained, lost) >= 0,
            "McNemar discordant counts must be nonnegative integers")
    n = gained + lost
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / (2 ** n)) if n else 1.


def active_controller(row):
    active = row["active"]
    return ("pfly" if row["arm_qp_active"] else "wait") if active == "pfly_or_wait" else active


def masked_states(states, predicate):
    # The adapter interpolates poses/velocities identically. Only eligibility at
    # the same left state changes, exactly as its original planned gate does.
    return [{**row, "planned": bool(row["planned"] and predicate(row))} for row in states]


def reevaluate_states(states, item, metric, mode, contact_call):
    """contact_call(states, step, before, after) uses the frozen contact adapter."""
    require(len(states) >= 2, "at least two recorded states are required for interpolation")
    hit = metric["hit_time_s"]
    required_end = min(item["input_end_s"], item["bounce_labels"][1]
                       if len(item["bounce_labels"]) > 1 else item["input_end_s"])
    scope_end = min(required_end, states[-1]["t"])
    planned_times = [row["t"] for row in states if row["planned"] and row["t"] <= scope_end]
    if hit is not None and planned_times:
        require(min(planned_times) >= hit - 10 and scope_end <= hit + 10,
                "the fixed +/-10s secondary window does not cover the planned episode")
    native = masked_states(states, lambda row: active_controller(row) == mode)
    nonnative = masked_states(states, lambda row: active_controller(row) != mode)
    fallback = masked_states(states, lambda row: bool(row["fallback"]))
    output = {}
    for name, step, width in (("primary_1ms", .001, .35), ("primary_0_5ms", .0005, .35),
                              ("episode_1ms", .001, 10.)):
        geometry = contact_call(states, step, width, width)
        require(not geometry["approaching"] or geometry["contact"], "approaching without geometry contact")
        geometry = {**geometry}
        for label, eligible in (("native", native), ("nonnative", nonnative), ("fallback_state", fallback)):
            # An empty eligibility set, or no full-scope contact, is provably a
            # negative subset without further geometry calls.
            subset = contact_call(eligible, step, width, width) if geometry["contact"] and any(
                row["planned"] for row in eligible) else {"contact": False, "approaching": False}
            require(not subset["approaching"] or geometry["approaching"], "subset approaching exceeds full scope")
            require(not subset["contact"] or geometry["contact"], "subset contact exceeds full scope")
            geometry[label + "_contact"] = bool(subset["contact"])
            geometry[label + "_approaching"] = bool(subset["approaching"])
        output[name] = geometry
    output["episode_bounds"] = {"recorded_start_s": states[0]["t"], "recorded_end_s": states[-1]["t"],
        "required_episode_end_s": required_end, "evaluation_end_s": scope_end,
        "unrecorded_tail_s": max(0., required_end - states[-1]["t"]),
        "first_planned_state_s": min(planned_times) if planned_times else None,
        "endpoint": "min(second_bounce_if_labelled,input_end,recording_end)",
        "eligibility": "original adapter planned flag at interpolation left state"}
    output["fallback_any"] = bool(metric["fallback"])
    return output


class DiagnosticEvaluator:
    def __init__(self, team, scene, audit, resources):
        import mujoco
        import analyze_control_ablation as adapter
        import control_diagnostic_metrics as diagnostics
        import interception_metrics as geometry
        for label, module in (("geometry_adapter", adapter), ("diagnostic_metrics", diagnostics),
                              ("geometry_evaluator", geometry)):
            require(Path(module.__file__).resolve() == Path(resources[label]["path"]).resolve(),
                    f"evaluation imported a different frozen source: {label}")
            audit.verify(module.__file__, resources[label]["sha256"])
        scene = (Path(team) / "sckd_sim" / scene).resolve(strict=True)
        require(str(scene) in audit.files, "evaluation scene is absent from frozen runtime resources")
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        face = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "racket_face")
        ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        require(face >= 0 and ball >= 0, "scene lacks racket_face/ball_geom")
        self.halfsize, self.radius = tuple(self.model.geom_size[face]), float(self.model.geom_size[ball, 0])
        self.mj, self.adapter, self.diagnostics, self.geometry = mujoco, adapter, diagnostics, geometry

    def __call__(self, path, item, mode, recorded):
        states, _, _, metric = self.diagnostics.read_case(path, item)
        self.adapter.motor_audit(states)
        for key in ("mode", "sample_id", "trajectory_id", "planned", "hit_time_s", "hit_time_covered", "fallback"):
            require(metric[key] == recorded[key], f"diagnostic metric differs: {path}: {key}")
        def call(eligible, step, before, after):
            protocol = self.geometry.InterceptionProtocol(self.halfsize, self.radius, step,
                before, after, -.30, .1)
            return self.adapter.contact(self.model, self.mj.MjData(self.model), eligible, item,
                                       metric["hit_time_s"], step, protocol=protocol)
        return reevaluate_states(states, item, metric, mode, call)

    def protocol(self):
        base = self.geometry.InterceptionProtocol(self.halfsize, self.radius).as_dict()
        return {"primary": base,
            "refinement": {**base, "interpolation_step_s": .0005},
            "secondary_episode": {**base, "window_before_s": 10., "window_after_s": 10.}}


def audit_cache_provenance(manifest, audit, sample_ids, source):
    """Follow each independent-H cache back to the completed predictor producer."""
    producers = {}
    for label, resource in manifest["resources"].items():
        if not label.startswith("cache_manifest/"):
            continue
        path = Path(resource["path"])
        cache = load(path)
        require(cache.get("schema") == "FORECAST_HORIZON_CACHES/1",
                f"cache lacks independently auditable horizon provenance: {path}")
        require(cache.get("strategy") == "independent_forecast_per_horizon_and_sample",
                f"cache was not produced independently per H: {path}")
        files = cache["files"]
        require({Path(row["file"]).stem for row in files} == set(sample_ids)
                and len(files) == 35, f"cache does not cover exactly all35: {path}")
        for row in files:
            audit.verify(path.parent / row["file"], row["sha256"])
        producer = path.parent.parent.parent
        if producer not in producers:
            result = audit_completed_run(producer, audit, completion="manifest.json")
            spec = result["experiment"]["spec"]
            require(spec["runner"] == "prepare_time_window_caches", "wrong cache producer runner")
            require(result.get("sample_count") == 35 and set(spec["parameters"]["sample_ids"]) == set(sample_ids),
                    "cache producer is incomplete or covers a different sample set")
            require(Path(spec["parameters"]["source"]).resolve() == source, "cache producer source differs")
            require(any(key.startswith("predictor_code/") for key in spec["resources"])
                    and any(key.startswith("model/") for key in spec["resources"]),
                    "cache producer lacks predictor-source/model hashes")
            producers[producer] = result
        result = producers[producer]
        entry = [row for row in result["horizons"]
                 if (producer / row["cache_manifest"]).resolve() == path.resolve()]
        require(len(entry) == 1 and entry[0]["cache_manifest_sha256"] == resource["sha256"]
                and entry[0]["bundle_fingerprint"] == cache["bundle_fingerprint"],
                f"cache manifest is not the completed producer output: {path}")
    require(producers, "no verifiable forecast cache producer found")
    return [{"path": str(path / "manifest.json"), "identity": result["experiment"]["identity"],
             "base_bundle_fingerprint": result["base_bundle_fingerprint"]}
            for path, result in sorted(producers.items())]


def phase_summary(records, expected, scope):
    require(len({row["sample_id"] for row in records}) == len(records), "duplicate analysis sample")
    require({row["sample_id"] for row in records} <= set(expected), "unexpected analysis sample")
    complete = {row["sample_id"] for row in records} == set(expected)
    denominator = len(expected)
    names = ("contact", "approaching", "native_contact", "native_approaching", "nonnative_approaching",
             "fallback_state_approaching")
    counts = {name: sum(bool(row["reevaluation"][scope][name]) for row in records) for name in names}
    counts["fallback_trials"] = sum(bool(row["reevaluation"]["fallback_any"]) for row in records)
    return {"complete": complete, "denominator": denominator, "completed_trials": len(records),
        "counts": counts, "rates": {name: count / denominator if complete else None for name, count in counts.items()},
        "sample_ids": list(expected)}


def paired_summary(left, right, expected, scope):
    lmap, rmap = ({row["sample_id"]: row for row in rows} for rows in (left, right))
    require(len(lmap) == len(left) and len(rmap) == len(right) and len(set(expected)) == len(expected),
            "paired comparison contains duplicate samples")
    require(set(lmap) == set(rmap) == set(expected), "paired comparison requires every expected sample")
    require(all(lmap[sid]["input_sha256"] == rmap[sid]["input_sha256"] for sid in expected),
            "paired raw input hashes differ")
    output = {"trials": len(expected), "input_hashes_identical": True,
        "cache_hashes_identical": all(lmap[sid]["cache_sha256"] == rmap[sid]["cache_sha256"] for sid in expected)}
    for field in ("contact", "approaching", "native_approaching", "nonnative_approaching", "fallback_state_approaching"):
        gained, lost, both, neither = [], [], [], []
        for sid in expected:
            a, b = (bool(table[sid]["reevaluation"][scope][field]) for table in (lmap, rmap))
            (both if a and b else lost if a else gained if b else neither).append(sid)
        output[field] = {"gained_sample_ids": sorted(gained), "lost_sample_ids": sorted(lost),
            "both_sample_ids": sorted(both), "neither_sample_ids": sorted(neither),
            "gained": len(gained), "lost": len(lost), "delta_count": len(gained) - len(lost),
            "delta_percentage_points": 100 * (len(gained) - len(lost)) / len(expected),
            "exact_two_sided_mcnemar_p": exact_mcnemar(len(gained), len(lost))}
    return output


def read_frozen_search(root, team, source, audit):
    manifest = load(root / "search_manifest.json")
    require(manifest.get("schema") == "TIME_WINDOW_SEARCH_MANIFEST/1", "unsupported search manifest")
    manifest_id = identity(manifest)
    audit.verify(root / "search_manifest.json")
    audit.resources(manifest["resources"])
    require(set(manifest["runtime_resources"]) == {"pfly", "nmpc6"}, "both native controllers are required")
    for resources in manifest["runtime_resources"].values():
        audit.resources(resources)
    require(manifest["controller_variant"] == "baseline", "controller variant changed during timing search")
    require(manifest["selection_rule"] == SELECTION_RULE, "selection rule changed")
    require(float(manifest["spec"].get("geometry_step_s", .001)) == .001, "search primary geometry must use 1ms")
    input_path = source / "input_manifest.json"
    require(input_path.resolve() == Path(manifest["resources"]["input_manifest"]["path"]).resolve(),
            "--source differs from frozen search source")
    items = load(input_path)["items"]
    require(len(items) == 35 and len({item["sample_id"] for item in items}) == 35, "source must contain exactly all35")
    by_sample = {}
    for raw in items:
        item = deepcopy(raw)
        csv = Path(item["csv"])
        item["csv"] = str((csv if csv.is_absolute() else source / csv).resolve(strict=True))
        frozen = manifest["resources"]["input/" + item["sample_id"]]
        require(item["csv"] == str(Path(frozen["path"]).resolve()), "source CSV path differs from frozen input")
        for field in ("source_sha256", "input_sha256"):
            if item.get(field):
                require(item[field] == frozen["sha256"], f"input manifest SHA differs: {item['sample_id']}")
        by_sample[item["sample_id"]] = item
    split = fixed_split(by_sample, manifest["split"])
    require(split == manifest["split"], "split differs from frozen discovery18/confirmation17")
    split["all35_after_selection"] = sorted(by_sample)
    selection, confirmation, plan = (load(root / filename) for filename in
                                     ("selection.json", "confirmation_summary.json", "confirmation_plan.json"))
    for filename in ("selection.json", "confirmation_summary.json", "confirmation_plan.json"):
        audit.verify(root / filename)
    require(selection.get("schema") == "TIME_WINDOW_SELECTION/1"
            and selection.get("search_manifest_identity") == manifest_id
            and selection.get("selection_phase") == "discovery"
            and selection.get("selection_rule") == SELECTION_RULE, "invalid frozen discovery selection")
    require(selection["split"] == manifest["split"], "selection split changed")
    require(confirmation.get("schema") == "TIME_WINDOW_CONFIRMATION/1" and confirmation.get("complete"),
            "confirmation has incomplete/error trials")
    require(plan.get("schema") == "TIME_WINDOW_CONFIRMATION_PLAN/1"
            and plan["selection_data"] == "discovery_only"
            and plan["sample_ids"] == split["confirmation"], "invalid confirmation plan")
    require(plan["selection_sha256"] == confirmation["selection_sha256"] == digest(root / "selection.json"),
            "selection changed after confirmation freeze")
    require(selection["baseline_jobs"] == confirmation["baseline_jobs"] == manifest["baseline_jobs"],
            "baseline job mapping differs")
    require(selection["selected_by_mode"] == confirmation["selected_by_mode"]
            and set(selection["selected_by_mode"]) == {"pfly", "nmpc6"}, "selected job mapping differs")
    discovery = {}
    expected_jobs = {job["id"] for job in manifest["spec"]["jobs"]}
    require(set(selection["discovery_summary_sha256"]) == expected_jobs, "selection omitted discovery candidates")
    for job, sha in selection["discovery_summary_sha256"].items():
        path = root / "summaries/discovery" / (job + ".json")
        audit.verify(path, sha)
        summary = load(path)
        require(summary["search_manifest_identity"] == manifest_id and summary["phase"] == "discovery"
                and summary["job_id"] == job, "discovery summary identity differs")
        discovery[job] = summary
    for mode, selected in selection["selected_by_mode"].items():
        require(ranked_best([row for row in discovery.values() if row["mode"] == mode])["job_id"] == selected,
                "frozen selection fails its discovery-only ranking audit")
    wanted = set(selection["baseline_jobs"].values()) | set(selection["selected_by_mode"].values())
    require(set(plan["job_ids"]) == wanted and len(plan["job_ids"]) == len(wanted), "confirmation plan jobs differ")
    require(set(confirmation["confirmation"]) == wanted, "confirmation summary jobs differ")
    # --team is an explicit runtime root, never an implicit alternate model.
    scene = (team / "sckd_sim" / manifest["spec"].get("scene", "models/generated/x4_tennis_demo.mjcf")).resolve()
    require(str(scene) in audit.files, "--team scene differs from frozen runtime resources")
    return manifest, manifest_id, selection, confirmation, split, by_sample


def audit_trial(root, phase, job, sample, item, record, manifest_id, team, audit):
    path = root / "runs" / phase / job["id"] / job["mode"] / sample
    result = audit_completed_run(path, audit)
    metadata = result["experiment"]
    compact = {key: value for key, value in result.items() if key != "experiment"}
    compact.update(run_identity=metadata["identity"], run_result_sha256=metadata["result_sha256"])
    require(compact == record, f"summary/run record differs: {path}")
    spec, parameters = metadata["spec"], metadata["spec"]["parameters"]
    require(spec["runner"] == "time_window_search" and parameters["search_manifest_identity"] == manifest_id
            and parameters["resolved_job"] == job and parameters["phase"] == phase
            and parameters["sample_id"] == sample and parameters["trajectory_id"] == item["trajectory_id"],
            f"trial request differs from frozen search: {path}")
    require(Path(spec["cwd"]).resolve() == (team / "sckd_sim").resolve()
            and Path(spec["command"][0]).resolve() == Path(job["binary"]).resolve(), "trial binary/cwd differs")
    policy = root / "policies" / (job["id"] + ".json")
    resources = spec["resources"]
    expected_paths = {"input": item["csv"], "binary": job["binary"],
        "cache": str(Path(job["cache"]["path"]) / (sample + ".jsonl")), "timing_policy": str(policy),
        "resolved_job": str(root / "jobs" / job["id"] / "resolved_job.json")}
    for label, expected in expected_paths.items():
        require(Path(resources[label]["path"]).resolve() == Path(expected).resolve(), f"trial resource path differs: {label}")
    for field, label in (("binary_sha256", "binary"), ("input_sha256", "input"),
                         ("cache_sha256", "cache"), ("policy_sha256", "timing_policy")):
        require(result[field] == resources[label]["sha256"], f"trial recorded hash differs: {field}")
    require(load(policy) == job["policy"] and load(expected_paths["resolved_job"]) == job, "frozen policy/job differs")
    runtime = resolved_runtime_config((path / "control.log").read_text())
    require(runtime == result["resolved_runtime_config"], "recorded/actual runtime configuration differs")
    assert_runtime_consumed(runtime, job, policy)
    return path


def markdown_report(report):
    phase_label = {"discovery": "发现18", "confirmation": "确认17", "all35_after_selection": "合并35"}
    lines = ["# 时间窗口搜索终审", "", "全部源码、模型、二进制、输入、缓存、策略及已完成运行的 SHA256 核验通过。",
        "候选仅按冻结的发现18结果选定；确认17未参与重选。全部35条均为反复使用的历史数据，合并35包含选型偏差，确认17也不是新盲测。", ""]
    for scope, title in (("primary_1ms", "主口径：计划击球时刻 ±0.35s，1ms"),
                         ("episode_1ms", "次口径：已规划全回合记录内，±10s，1ms")):
        lines += ["## " + title, "", "| 控制器 | 分组 | 子集 | 几何 | 接近几何 | 原生接近几何 | 回退样本 |",
            "|---|---|---|---:|---:|---:|---:|"]
        details = []
        for mode in ("pfly", "nmpc6"):
            for role, label in (("baseline", "原基线"), ("selected", "冻结最优")):
                group = report["groups"][mode][role]
                for phase in PHASES:
                    result = group["phases"][phase][scope]
                    n, counts = result["denominator"], result["counts"]
                    cell = lambda key: f"{counts[key]}/{n} ({100*counts[key]/n:.1f}%)"
                    lines.append(f"| {mode} | {label} | {phase_label[phase]} | {cell('contact')} | {cell('approaching')} | {cell('native_approaching')} | {counts['fallback_trials']}/{n} |")
        lines += ["", "| 控制器 | 子集 | 接近几何新增 / 丢失 | 差值 | 精确双侧 McNemar p |",
                  "|---|---|---:|---:|---:|"]
        for mode in ("pfly", "nmpc6"):
            for phase in PHASES:
                delta = report["paired"][mode][phase][scope]["approaching"]
                lines.append(f"| {mode} | {phase_label[phase]} | {delta['gained']} / {delta['lost']} | {delta['delta_percentage_points']:+.1f} 个百分点 | {delta['exact_two_sided_mcnemar_p']:.6g} |")
                details.append(f"{mode} {phase_label[phase]} 新增：{', '.join(delta['gained_sample_ids']) or '无'}；丢失：{', '.join(delta['lost_sample_ids']) or '无'}。")
        for detail in details:
            lines += ["", detail]
        lines.append("")
    lines += ["## 主口径插值步长复核", "", "| 控制器 | 分组 | 1ms 与 0.5ms 几何分类变化 | 接近分类变化 | 原生接近分类变化 |", "|---|---|---:|---:|---:|"]
    for mode in ("pfly", "nmpc6"):
        for role in ("baseline", "selected"):
            diff = report["groups"][mode][role]["step_sensitivity"]
            lines.append(f"| {mode} | {role} | {len(diff['contact'])} | {len(diff['approaching'])} | {len(diff['native_approaching'])} |")
    lines += ["", "## 回合记录边界", "", "| 控制器 | 分组 | 末帧早于应有回合终点的样本 | 最大未记录尾段 |", "|---|---|---:|---:|"]
    for mode in ("pfly", "nmpc6"):
        for role in ("baseline", "selected"):
            coverage = report["groups"][mode][role]["episode_recording_coverage"]
            lines.append(f"| {mode} | {role} | {coverage['trials_with_unrecorded_tail']}/35 | {1000*coverage['max_unrecorded_tail_s']:.3f} ms |")
    lines += ["", "原生接近几何由当前 active 控制器与完整接近门共同判定；不以原生几何接触代替。回退样本表示该次运行出现过 fallback；回退状态接近贡献另见 JSON。一个样本可同时有原生和非原生接近接触，贡献计数不可直接相加。",
        "", "全回合次口径仅在原适配器 planned 标志为真时计数，终点为第二次触地标注、输入终点及记录终点的最小值。未记录尾段不外推，也不视为已观测的未接触时间；该口径的未命中仅表示记录范围内未接触。尺寸、法向、速度门及端点规则保持原样。",
        "", "p 值为配对精确双侧 McNemar，未做多重比较校正；发现集与合并集仅供探索描述。几何拦截含回退，不能解释为真实碰撞、出球质量或实机击球成功率。",
        "", f"核验文件 {report['integrity']['file_count']} 个；平台 {report['platform']['system']} / {report['platform']['machine']}；UTC {report['created_at_utc']}。", ""]
    return "\n".join(lines)


def summarize_search(root, team, source, out, *, evaluator=None):
    root, team, source, out = (Path(path).resolve() for path in (root, team, source, out))
    require(not out.exists(), "analysis --out must be a new directory; existing output is preserved")
    require(not out.is_relative_to(root) and not out.is_relative_to(source), "analysis output must be separate from frozen search/source")
    audit = HashAudit()
    manifest, manifest_id, selection, confirmation, split, items = read_frozen_search(root, team, source, audit)
    producers = audit_cache_provenance(manifest, audit, items, source)
    if evaluator is None:
        evaluator = DiagnosticEvaluator(team, manifest["spec"].get("scene", "models/generated/x4_tennis_demo.mjcf"),
                                        audit, manifest["resources"])
    groups, paired, all_records, by_job = {}, {}, {}, {}
    for mode in ("pfly", "nmpc6"):
        groups[mode] = {}
        for role, job_id in (("baseline", selection["baseline_jobs"][mode]),
                             ("selected", selection["selected_by_mode"][mode])):
            if job_id not in by_job:
                summaries = {phase: load(root / "summaries" / phase / (job_id + ".json")) for phase in PHASES}
                job = summaries["discovery"]["resolved_job"]
                require(job["id"] == job_id and job["mode"] == mode, "resolved job method differs")
                for phase, summary in summaries.items():
                    audit.verify(root / "summaries" / phase / (job_id + ".json"))
                    require(summary["search_manifest_identity"] == manifest_id and summary["phase"] == phase
                            and summary["resolved_job"] == job, "summary resolved job/search identity differs")
                    expected = summarize(job, phase, split[phase], summary["records"], summary["technical_failures"])
                    require(all(summary[key] == value for key, value in expected.items()), "summary aggregate fields differ")
                    require(summary["complete"], f"incomplete {phase} trials: {job_id}; denominator retained in source")
                compact_confirmation = {key: value for key, value in summaries["confirmation"].items() if key != "records"}
                require(compact_confirmation == confirmation["confirmation"][job_id], "confirmation final/phase summaries differ")
                combined = summaries["discovery"]["records"] + summaries["confirmation"]["records"]
                require(combined == summaries["all35_after_selection"]["records"], "all35 is not the exact discovery+confirmation union")
                records = []
                for recorded in combined:
                    sample, phase = recorded["sample_id"], recorded["phase"]
                    require(phase in ("discovery", "confirmation") and sample in split[phase], "record phase/split differs")
                    path = audit_trial(root, phase, job, sample, items[sample], recorded, manifest_id, team, audit)
                    recheck = evaluator(path, items[sample], mode, recorded)
                    for field in ("contact", "approaching"):
                        require(recheck["primary_1ms"][field] == recorded["geometric"][field],
                                f"primary classification does not reproduce: {job_id}/{sample}/{field}")
                    records.append({"sample_id": sample, "phase": phase, "input_sha256": recorded["input_sha256"],
                        "cache_sha256": recorded["cache_sha256"], "binary_sha256": recorded["binary_sha256"],
                        "policy_sha256": recorded["policy_sha256"], "run_identity": recorded["run_identity"],
                        "reevaluation": recheck})
                    print(f"analysis {job_id} {len(records)}/35 {sample}", flush=True)
                by_job[job_id] = records
                all_records[job_id] = records
            records = by_job[job_id]
            groups[mode][role] = {"job_id": job_id,
                "phases": {phase: {scope: phase_summary([row for row in records if row["sample_id"] in split[phase]],
                    split[phase], scope) for scope in SCOPES} for phase in PHASES},
                "step_sensitivity": {field: sorted(row["sample_id"] for row in records
                    if row["reevaluation"]["primary_1ms"][field] != row["reevaluation"]["primary_0_5ms"][field])
                    for field in ("contact", "approaching", "native_approaching")},
                "episode_recording_coverage": {
                    "trials_with_unrecorded_tail": sum(row["reevaluation"].get("episode_bounds", {}).get("unrecorded_tail_s", 0.) > 1e-12 for row in records),
                    "max_unrecorded_tail_s": max(row["reevaluation"].get("episode_bounds", {}).get("unrecorded_tail_s", 0.) for row in records),
                    "tail_sample_ids": sorted(row["sample_id"] for row in records if row["reevaluation"].get("episode_bounds", {}).get("unrecorded_tail_s", 0.) > 1e-12)}}
        left, right = (by_job[groups[mode][role]["job_id"]] for role in ("baseline", "selected"))
        paired[mode] = {phase: {scope: paired_summary([row for row in left if row["sample_id"] in split[phase]],
            [row for row in right if row["sample_id"] in split[phase]], split[phase], scope) for scope in SCOPES} for phase in PHASES}
    audit.recheck()
    inventory = audit.inventory()
    report = {"schema": SCHEMA, "complete": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "machine": platform.machine(), "python": sys.version},
        "search_root": str(root), "search_manifest_identity": manifest_id,
        "selection_sha256": digest(root / "selection.json"), "selected_by_mode": selection["selected_by_mode"],
        "baseline_jobs": selection["baseline_jobs"], "selection_verified_using": "frozen_discovery_only",
        "historical_data_not_blind": True, "all35_contains_discovery_selection_bias": True,
        "split": split, "protocols": evaluator.protocol() if hasattr(evaluator, "protocol") else {"fixture": True},
        "native_attribution": "original planned AND current active controller==method; identical geometry/approach grid",
        "fallback_attribution": "fallback_any is any state flag; fallback_state_approaching is same-grid planned AND fallback flag",
        "contribution_counts_may_overlap": True, "groups": groups, "paired": paired, "records": all_records,
        "cache_producers": producers, "integrity": {"passed": True, "file_count": len(inventory),
            "inventory_sha256": identity(inventory), "checked_at_start_and_before_completion": True},
        "interpretation": "历史测量参考有限拍面几何拦截，含回退；非真实碰撞/出球质量/实机命中率",
        "statistics": "paired exact two-sided McNemar, unadjusted; discovery/all35 exploratory"}
    # Only small identity anchors are duplicated into the analysis RunAttempt;
    # verified_files.json preserves the full transitive content inventory.
    anchors = {"analysis_runner": Path(__file__), "search_manifest": root / "search_manifest.json",
        "selection": root / "selection.json", "confirmation": root / "confirmation_summary.json",
        "confirmation_plan": root / "confirmation_plan.json", "source_manifest": source / "input_manifest.json",
        "analysis_common": Path(__file__).with_name("pipeline_experiment.py"),
        "search_runner": Path(__file__).with_name("time_window_search.py")}
    spec = ExperimentSpec("summarize_time_window_search", tuple(sys.argv), str(Path.cwd()),
        {"schema": SCHEMA, "search_manifest_identity": manifest_id, "inventory_sha256": identity(inventory),
         "root": str(root), "team": str(team), "source": str(source), "split": split}, {}, file_resources(anchors))
    with RunAttempt(out, spec) as attempt:
        write_json(out / "verified_files.json", inventory, exclusive=True)
        write_json(out / "analysis.json", report, exclusive=True)
        (out / "analysis.md").write_text(markdown_report(report))
        audit.recheck()
        return attempt.complete(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "team", "source", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("simulation diagnostics and result generation must run on Ubuntu")
    try:
        result = summarize_search(args.root, args.team, args.source, args.out)
        print(json.dumps({"complete": result["complete"], "out": str(args.out.resolve()),
                          "verified_files": result["integrity"]["file_count"]}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError) as error:
        parser.exit(2, f"time-window analysis incomplete: {error}\n")


if __name__ == "__main__":
    main()
