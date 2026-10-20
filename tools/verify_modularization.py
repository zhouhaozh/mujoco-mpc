"""Ubuntu上串行核对重构前后控制行为；本文件仅依赖Python标准库。

默认：pfly/nmpc6/wbmpc × fixed/rand/gs，同seed、每组1球，保留旧EKF行为。
--prediction ROOT 为合成球启用相同的外部预测服务。
--cache-root ROOT 可另加实测缓存配对；支持ROOT/cache或直接缓存目录。
缓存源由cache/manifest.json的source定位，或--input-manifest显式给定；
默认取首个有缓存样本，--samples可指定逗号分隔列表，不生成/重写输入缓存。

输出必须为新目录。耗时字段不参与数值等价比较，但墙钟门导致的采用/拒收
变化单独报告，依然判为不等价；不豁免其后发生的状态/执行命令差异。
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time


SCHEMA = "SCKD_MODULARIZATION_CHECK/1"
MODES = ("pfly", "nmpc6", "wbmpc")
FAMILIES = ("fixed", "rand", "gs")
# 只剔除确定为墙钟耗时的键；t、contact_time、plan_age_s、plan_ms等仿真量保留。
WALL_FIELDS = frozenset({"ms", "solve_ms", "wall_s", "elapsed_ms", "forecast_compute_ms",
    "process_wall_s", "mean_ms", "max_ms", "p95_ms", "p99_ms", "total_ms", "first_ms"})
TRACE_EXTENSIONS = frozenset({"schema", "runtime_identity"})
CONTROL_PREFIXES = ("M2_", "NMPC6_", "N6_", "BALL_", "EXP_")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def resource(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a regular resource: {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def add_resource(resources, label, path):
    resources[label] = resource(path)


def add_tree(resources, label, path):
    path = Path(path).resolve(strict=True)
    for item in sorted(path.rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts:
            add_resource(resources, f"{label}/{item.relative_to(path)}", item)


def clean_environment():
    # 只清除会改变控制实验的开关；不记录整个环境或凭据。
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(CONTROL_PREFIXES) and key not in ("WM_DEBUG", "WBO_DEBUG", "WB_CHASSIS")}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    return env


def environment_record(env):
    return {key: value for key, value in sorted(env.items())
            if key.startswith(CONTROL_PREFIXES) or key in (
                "WM_DEBUG", "WBO_DEBUG", "WB_CHASSIS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "PYTHONHASHSEED", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH")}


def stop_owned(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


@contextmanager
def service_process(command, cwd, env, path, log, startup_timeout):
    if path.exists():
        raise FileExistsError(path)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + startup_timeout
            while not path.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"prediction service exited {process.returncode}; see {log}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"prediction startup timeout; see {log}")
                time.sleep(.05)
            yield
        finally:
            stop_owned(process)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached_inputs(args):
    if not args.cache_root:
        if args.samples or args.input_manifest:
            raise ValueError("--samples/--input-manifest require --cache-root")
        return [], {}
    root = args.cache_root.resolve(strict=True)
    cache = root / "cache" if (root / "cache").is_dir() else root
    cache_manifest = cache / "manifest.json"
    resources = {}
    if args.input_manifest:
        manifest = args.input_manifest.resolve(strict=True)
    elif (root / "input_manifest.json").is_file():
        manifest = root / "input_manifest.json"
    elif cache_manifest.is_file() and load_json(cache_manifest).get("source"):
        source = Path(load_json(cache_manifest)["source"])
        if not source.is_absolute():
            source = (cache_manifest.parent / source).resolve()
        manifest = source / "input_manifest.json"
    else:
        raise FileNotFoundError("cannot locate cached input_manifest.json; use --input-manifest")
    manifest = manifest.resolve(strict=True)
    add_resource(resources, "measured/input_manifest", manifest)
    if cache_manifest.is_file():
        add_resource(resources, "measured/cache_manifest", cache_manifest)
    if (manifest.parent / "transform.json").is_file():
        add_resource(resources, "measured/transform", manifest.parent / "transform.json")
    items = load_json(manifest)["items"]
    requested = args.samples.split(",") if args.samples else None
    available = {str(item["sample_id"]): item for item in items
                 if (cache / (str(item["sample_id"]) + ".jsonl")).is_file()}
    selected = requested if requested is not None else sorted(available)[:1]
    if not selected:
        raise ValueError("no measured sample has an existing prediction cache")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate sample id in --samples")
    result = []
    for sample in selected:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample) or sample in (".", ".."):
            raise ValueError(f"unsafe sample id: {sample!r}")
        if sample not in available:
            raise ValueError(f"unknown sample or missing cache: {sample}")
        item = available[sample]
        csv_path = Path(item["csv"])
        if not csv_path.is_absolute():
            csv_path = manifest.parent / csv_path
        csv_path = csv_path.resolve(strict=True)
        cached = (cache / (sample + ".jsonl")).resolve(strict=True)
        add_resource(resources, f"measured/{sample}/input", csv_path)
        add_resource(resources, f"measured/{sample}/cache", cached)
        result.append({"sample_id": sample, "csv": str(csv_path), "cache": str(cached),
                       "trajectory_id": item.get("trajectory_id")})
    return result, resources


def collect_resources(args, measured_resources):
    sim = args.team / "sckd_sim"
    resources = dict(measured_resources)
    add_resource(resources, "binary/baseline", args.baseline)
    add_resource(resources, "binary/candidate", args.candidate)
    add_resource(resources, "verification_script", Path(__file__))
    add_resource(resources, "cmake", sim / "CMakeLists.txt")
    add_resource(resources, "scene", sim / args.scene)
    add_tree(resources, "model", sim / "models")
    add_tree(resources, "cpp_sources", sim / "cpp")
    for item in sorted((sim / "third_party/nmpc6").glob("*.so")):
        add_resource(resources, "nmpc6/" + item.name, item)
    if args.cache_root:
        add_resource(resources, "cache_service", args.team / "tools/control_ablation_cache.py")
    if args.prediction:
        root = args.prediction
        selection = args.selection or root / "artifacts/tennis_controller/20261014/selected.json"
        add_resource(resources, "prediction/selection", selection)
        add_resource(resources, "prediction/python", args.predictor_python or root / ".venv-tennis/bin/python")
        add_tree(resources, "prediction/source", root / "tennis_calibration")
        add_resource(resources, "prediction/collision_selection", root / "artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json")
        add_tree(resources, "prediction/collision_models", root / "artifacts/tennis_collision/20261009_frozen_pre/models")
        add_resource(resources, "prediction/filter_selection", root / "artifacts/tennis_identified/20261009_training_prior/selected_model.json")
        add_resource(resources, "prediction/noise_proxy", root / "artifacts/tennis_robust/20261013/validation_v2/noise_proxy.json")
        head = load_json(selection).get("head")
        if head:
            add_resource(resources, "prediction/residual_head", root / head)
    return resources


def resolved_config(log):
    prefix = "[resolved config] "
    records = [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    if not records:
        return None
    if any(value != records[0] for value in records):
        raise ValueError("conflicting resolved configuration records")
    return records[0]


def verify_config(config, log, expected):
    mismatches = []
    if config is not None:
        values = config.get("values", {})
        for key, wanted in expected.items():
            if key in values and values[key] != wanted:
                mismatches.append({"field": key, "expected": wanted, "actual": values[key]})
            elif key not in values:
                mismatches.append({"field": key, "expected": wanted, "actual": "missing"})
        evidence = "binary_resolved_values"
    else:
        # 历史二进制没有resolved config；显式argv/env提供身份，已有两行stdout核对开关。
        for label, key in (("control variant", "control_variant"), ("steer rate mode", "steer_rate_mode")):
            found = re.search(r"\[" + re.escape(label) + r"\] ([^;\n]+)", log)
            if not found or found.group(1).strip() != expected[key]:
                mismatches.append({"field": key, "expected": expected[key],
                                   "actual": found.group(1).strip() if found else "missing stdout"})
        evidence = "explicit_argv_env_plus_legacy_stdout"
    return {"passed": not mismatches, "evidence": evidence, "mismatches": mismatches}


def run_binary(args, binary, side, case, dest):
    dest.mkdir(parents=True, exist_ok=False)
    diagnostics = dest / "diagnostics.jsonl"
    log_path = dest / "control.log"
    config = {"scene_path": args.scene, "trials": 1, "swing_mode": "qp", "init_delay_s": .25,
              "seed": args.seed, "base_mode": case["mode"], "serve_family": case["family"],
              "fly_mode": case["rule"], "control_variant": args.variant,
              "steer_rate_mode": args.steer_rate_mode or (
                  "both_unlimited" if args.variant == "baseline" and case["mode"] in ("pfly", "nmpc6") else "limited")}
    command = [str(binary), args.scene, "1", "qp", ".25", str(args.seed),
               case["mode"], case["family"], case["rule"]]
    env = clean_environment()
    env["M2_CONTROL_VARIANT"] = args.variant
    env["M2_CONTROL_DIAGNOSTICS"] = str(diagnostics)
    if args.steer_rate_mode:
        env["M2_STEER_RATE_MODE"] = args.steer_rate_mode
    if "csv" in case:
        env["M2_MEASURED_REPLAY"] = case["csv"]
    result = {"side": side, "case": case, "binary": resource(binary), "expected_config": config,
              "command": command, "cwd": str(args.team / "sckd_sim"), "completed": False}
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="sckd-modular-") as temporary:
            sock = Path(temporary) / "predictor.sock"
            service_command = None
            service_cwd = args.team
            if "cache" in case:
                service_command = [sys.executable, str(args.team / "tools/control_ablation_cache.py"),
                                   "--serve", case["cache"], "--socket", str(sock)]
            elif args.prediction:
                selection = args.selection or args.prediction / "artifacts/tennis_controller/20261014/selected.json"
                python = args.predictor_python or args.prediction / ".venv-tennis/bin/python"
                service_command = [str(python), "-m", "tennis_calibration.controller_service",
                                   "--selection", str(selection), "--socket", str(sock), "--synthetic-world"]
                service_cwd = args.prediction
            if service_command:
                env["BALL_PREDICTOR_SOCKET"] = str(sock)
            result["service_command"] = service_command
            result["environment"] = environment_record(env)
            dump_new(dest / "invocation.json", result)

            def execute():
                with log_path.open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=args.team / "sckd_sim", env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        return process.wait(timeout=args.timeout)
                    finally:
                        stop_owned(process)

            if service_command:
                with service_process(service_command, service_cwd, env, sock,
                                     dest / "service.log", args.startup_timeout):
                    result["returncode"] = execute()
            else:
                result["returncode"] = execute()
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if result["returncode"] not in (0, 1):
            raise RuntimeError(f"unexpected binary returncode {result['returncode']}")
        if "=== M2 拦截验收 ===" not in log or not diagnostics.is_file():
            raise RuntimeError("missing trial completion summary or diagnostics")
        if "csv" in case and "[MEASURED REPLAY]" not in log:
            raise RuntimeError("missing measured replay marker")
        actual = resolved_config(log)
        result["config_validation"] = verify_config(actual, log, config)
        dump_new(dest / "configuration.json", {"expected": config, "actual": actual,
                                              "validation": result["config_validation"]})
        # 必须至少有一条state，不能把两个空文件当作等价。
        rows = read_trace(diagnostics)
        result["trace_counts"] = dict(Counter(row["kind"] for row in rows))
        result["metadata_validation"] = validate_metadata(rows)
        if not result["trace_counts"].get("state"):
            raise RuntimeError("diagnostics contain no state records")
        result["completed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["wall_s"] = time.monotonic() - started
    result["artifacts"] = {path.name: resource(path) for path in sorted(dest.iterdir()) if path.is_file()}
    dump_new(dest / "run.json", result)
    return result


def read_trace(path):
    rows = []
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=invalid)
            if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
                raise ValueError(f"invalid diagnostic record {path}:{line_number}")
            stamp = row.get("t")
            if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                raise ValueError(f"missing/invalid simulation time {path}:{line_number}")
            rows.append(row)
    return rows


def keyed(rows):
    result = {}
    seen = Counter()
    for row in rows:
        identity = (row["kind"], round(row["t"] * 1_000_000_000))
        occurrence = seen[identity]
        seen[identity] += 1
        result[identity + (occurrence,)] = row
    return result


def validate_metadata(rows):
    """只认可本次定义的两个新增顶层字段；旧capture_time继续原样比较。"""
    required = {"clock_id", "track_id", "observation_seq", "state_seq", "last_capture_s", "last_receive_s"}
    previous = {}
    tracks = set()
    errors = []
    present = 0
    for index, row in enumerate(rows):
        if not (TRACE_EXTENSIONS & row.keys()):
            continue
        present += 1
        if row.get("schema") != "SCKD_CONTROL_TRACE/1":
            errors.append({"row": index, "reason": "unexpected_trace_schema"})
        identity = row.get("runtime_identity")
        if not isinstance(identity, dict) or set(identity) != required:
            errors.append({"row": index, "reason": "invalid_runtime_identity_fields"})
            continue
        track = identity["track_id"]
        if identity["clock_id"] != "simulation" or not isinstance(track, str) or not re.fullmatch(r"trial-[0-9]+", track):
            errors.append({"row": index, "reason": "invalid_clock_or_track"})
            continue
        tracks.add(track)
        sequences = (identity["observation_seq"], identity["state_seq"])
        if any(type(value) is not int or value < 0 for value in sequences):
            errors.append({"row": index, "reason": "invalid_sequence"})
            continue
        if track in previous:
            old_sequences, old_capture = previous[track]
            if any(current < old for current, old in zip(sequences, old_sequences)):
                errors.append({"row": index, "reason": "sequence_regressed"})
            if identity["last_capture_s"] == old_capture and sequences[0] != old_sequences[0]:
                errors.append({"row": index, "reason": "duplicate_capture_incremented_observation_seq"})
        previous[track] = (sequences, identity["last_capture_s"])
        capture, receive = identity["last_capture_s"], identity["last_receive_s"]
        if (capture is None) != (receive is None):
            errors.append({"row": index, "reason": "partial_observation_time"})
        elif capture is not None:
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (capture, receive)):
                errors.append({"row": index, "reason": "invalid_observation_time"})
            elif capture > receive + 1e-9 or receive > row["t"] + .001001:
                # state旧t字段写的是本tick开始，receive最多领先该字段一个1ms tick。
                errors.append({"row": index, "reason": "noncausal_observation_time"})
    if present and present != len(rows):
        errors.append({"reason": "partial_trace_identity_coverage", "present": present, "rows": len(rows)})
    return {"passed": not errors, "extension_present": bool(present), "validated_rows": present,
            "tracks": sorted(tracks), "errors": errors[:100], "errors_truncated": len(errors) > 100}


def category(path):
    if any(token in path for token in ("q_cmd", "dq_cmd", "motors", "tau", "base_ref", "v_base_ref")):
        return "execution_or_actuator"
    if any(token in path for token in ("goal", "target", "contact_time", "capture_time", "locked", "planned")):
        return "task_and_target"
    if any(token in path for token in ("accepted", "applied", "reason", "active", "fallback", "reject", "qp_")):
        return "acceptance_and_fallback"
    return "state_and_other"


class Differences:
    def __init__(self, args):
        self.args = args
        self.total = 0
        self.categories = Counter()
        self.examples = []
        self.excluded = Counter()
        self.max_abs_error = 0.

    def add(self, path, baseline, candidate, reason):
        self.total += 1
        self.categories[category(path)] += 1
        if len(self.examples) < self.args.max_differences:
            self.examples.append({"path": path, "baseline": baseline, "candidate": candidate, "reason": reason})

    def compare(self, baseline, candidate, path, top_level=False):
        if isinstance(baseline, bool) or isinstance(candidate, bool):
            if type(baseline) is not type(candidate) or baseline != candidate:
                self.add(path, baseline, candidate, "boolean_or_type")
        elif isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
            error = abs(baseline - candidate)
            self.max_abs_error = max(self.max_abs_error, error)
            if not math.isclose(baseline, candidate, rel_tol=self.args.rtol, abs_tol=self.args.atol):
                self.add(path, baseline, candidate, "numeric_tolerance")
        elif isinstance(baseline, dict) and isinstance(candidate, dict):
            for key in sorted(set(baseline) | set(candidate)):
                if key in WALL_FIELDS:
                    self.excluded[key] += 1
                    continue
                if key not in baseline or key not in candidate:
                    if top_level and key in TRACE_EXTENSIONS:
                        # 增量schema已单独验证并报告；绝不豁免旧字段或runtime_identity内的变化。
                        continue
                    self.add(path + "." + key, baseline.get(key), candidate.get(key), "missing_field")
                else:
                    self.compare(baseline[key], candidate[key], path + "." + key)
        elif isinstance(baseline, list) and isinstance(candidate, list):
            if len(baseline) != len(candidate):
                self.add(path + ".length", len(baseline), len(candidate), "length")
            for index, (before, after) in enumerate(zip(baseline, candidate)):
                self.compare(before, after, f"{path}[{index}]")
        elif baseline != candidate:
            self.add(path, baseline, candidate, "value_or_type")

    def report(self):
        return {"differences": self.total, "categories": dict(self.categories),
                "examples": self.examples, "examples_truncated": self.total > len(self.examples),
                "maximum_absolute_numeric_error": self.max_abs_error,
                "excluded_wall_fields": dict(self.excluded)}


def clock_gate_changes(before, after, variant):
    """显式重建旧时钟门，仅诊断，不把由此发生的其他差异从比较中删除。"""
    changes = []
    coherent = variant not in ("baseline", "feasible")
    history = []
    for rows in (before, after):
        have = False
        gates = {}
        counts = Counter()
        for row in rows:
            if row["kind"] != "solve":
                continue
            identity = (row["kind"], round(row["t"] * 1_000_000_000))
            key = identity + (counts[identity],)
            counts[identity] += 1
            milliseconds = row.get("ms")
            if not isinstance(milliseconds, (int, float)):
                continue
            if row.get("mode") == "nmpc6":
                remaining = row.get("remaining_s", row.get("target_time", row["t"]) - row["t"])
                active = not coherent and remaining < .100
                gate = not active or (row.get("reason") == 0 and row.get("solver_ok") is True and milliseconds < 10.)
                gates[key] = {"mode": "nmpc6", "active": active, "threshold_ms": 10.,
                              "strict_less_than": True, "ms": milliseconds, "gate_pass": gate,
                              "accepted": row.get("accepted"), "applied": row.get("applied"),
                              "solver_ok": row.get("solver_ok"), "status": row.get("reason"), "remaining_s": remaining}
            elif row.get("mode") == "wbmpc":
                threshold = 60. if have else 100.
                active = row.get("reason") in ("solve_time_budget", "base_jump", "accepted")
                gates[key] = {"mode": "wbmpc", "active": active, "threshold_ms": threshold,
                              "strict_less_than": False, "ms": milliseconds,
                              "gate_pass": milliseconds <= threshold if active else None,
                              "accepted": row.get("accepted"), "reason": row.get("reason")}
                have = have or bool(row.get("accepted"))
        history.append(gates)
    for key in sorted(set(history[0]) & set(history[1])):
        a, b = history[0][key], history[1][key]
        decision_changed = any(a.get(field) != b.get(field) for field in ("accepted", "applied", "reason"))
        gate_changed = (a["gate_pass"] != b["gate_pass"] or a["active"] != b["active"] or
                        ((a["active"] or b["active"]) and a["threshold_ms"] != b["threshold_ms"]))
        timed_reason = a.get("reason") == "solve_time_budget" or b.get("reason") == "solve_time_budget"
        if gate_changed or (decision_changed and timed_reason):
            changes.append({"event_key": list(key), "baseline": a, "candidate": b,
                            "gate_changed": gate_changed, "decision_changed": decision_changed,
                            "policy": "reported_and_counted_as_non_equivalent; no downstream waiver"})
    return changes


def compare_pair(args, case_dir, baseline, candidate):
    report = {"schema": SCHEMA, "completed": baseline["completed"] and candidate["completed"],
              "baseline_run": resource(case_dir / "baseline/run.json"),
              "candidate_run": resource(case_dir / "candidate/run.json")}
    if not report["completed"]:
        report.update(equivalent=False, reason="incomplete_run", baseline_error=baseline.get("error"), candidate_error=candidate.get("error"))
        return report
    before = read_trace(case_dir / "baseline/diagnostics.jsonl")
    after = read_trace(case_dir / "candidate/diagnostics.jsonl")
    a, b = keyed(before), keyed(after)
    differences = Differences(args)
    for key in sorted(set(a) | set(b)):
        path = f"{key[0]}@{key[1] / 1e9:.9f}#{key[2]}"
        if key not in a or key not in b:
            differences.add(path, key in a, key in b, "missing_event")
        else:
            differences.compare(a[key], b[key], path, top_level=True)
    gates = clock_gate_changes(before, after, args.variant)
    report.update(differences.report())
    report["clock_gate_changes"] = gates
    report["clock_gate_note"] = "reconstructed from current legacy thresholds: nmpc6 <10ms in final <100ms unless coherent; wbmpc cold100/hot60ms. Not a wall-clock realtime acceptance."
    report["trace_counts"] = {"baseline": dict(Counter(row["kind"] for row in before)),
                              "candidate": dict(Counter(row["kind"] for row in after))}
    report["schema_extensions"] = {"allowed_additions": sorted(TRACE_EXTENSIONS),
        "baseline": baseline["metadata_validation"], "candidate": candidate["metadata_validation"],
        "new_only_fields": sorted({key for key in TRACE_EXTENSIONS
            if any(key in row for row in after) and not any(key in row for row in before)})}
    report["same_returncode"] = baseline["returncode"] == candidate["returncode"]
    report["config_consistent"] = baseline["config_validation"]["passed"] and candidate["config_validation"]["passed"]
    report["metadata_consistent"] = baseline["metadata_validation"]["passed"] and candidate["metadata_validation"]["passed"]
    report["equivalent"] = differences.total == 0 and not gates and report["same_returncode"] and report["config_consistent"] and report["metadata_consistent"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--team", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--predictor-python", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--samples", help="cached measured sample ids, comma separated; default first available")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scene", default="models/generated/x4_tennis_demo.mjcf")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--steer-rate-mode", choices=("limited", "direction_unlimited", "feedforward_unlimited", "both_unlimited"))
    parser.add_argument("--timeout", type=float, default=600.)
    parser.add_argument("--startup-timeout", type=float, default=15.)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--max-differences", type=int, default=100)
    args = parser.parse_args()
    for name in ("baseline", "candidate", "team", "prediction", "selection", "predictor_python"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve(strict=True))
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error(f"output already exists; refusing overwrite: {args.out}")
    if (args.selection or args.predictor_python) and not args.prediction:
        parser.error("--selection/--predictor-python require --prediction")
    if (any(not math.isfinite(value) for value in (args.timeout, args.startup_timeout, args.atol, args.rtol))
            or args.timeout <= 0 or args.startup_timeout <= 0 or args.atol < 0 or args.rtol < 0 or args.max_differences < 1):
        parser.error("timeouts/max-differences must be positive; numeric tolerances must be nonnegative")
    modes = args.modes.split(",")
    families = args.families.split(",") if args.families else []
    if any(mode not in MODES for mode in modes) or len(modes) != len(set(modes)):
        parser.error("--modes must contain unique pfly/nmpc6/wbmpc names")
    if any(family not in FAMILIES for family in families) or len(families) != len(set(families)):
        parser.error("--families must contain unique fixed/rand/gs names, or be empty for cache only")
    measured, measured_resources = cached_inputs(args)
    cases = [{"id": f"synthetic/{mode}/{family}", "mode": mode, "family": family,
              "rule": "xb" if family == "gs" else "v2"} for mode in modes for family in families]
    cases += [{"id": f"measured/{mode}/{item['sample_id']}", "mode": mode,
               "family": "measured", "rule": "xb", **item} for mode in modes for item in measured]
    if not cases:
        parser.error("no verification cases selected")
    resources = collect_resources(args, measured_resources)
    args.out.mkdir(parents=True, exist_ok=False)
    dump_new(args.out / "input_manifest.json", {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resources": resources, "cases": cases, "comparison_policy": {"excluded_wall_fields": sorted(WALL_FIELDS),
        "clock_gate_changes": "explicitly reported; never waived", "event_time_key_resolution_s": 1e-9,
        "allowed_schema_additions": sorted(TRACE_EXTENSIONS), "schema_addition_policy": "validate identity fields and report; preserve all legacy comparisons"}})
    reports = []
    for index, case in enumerate(cases, 1):
        case_dir = args.out / "cases" / case["id"]
        baseline = run_binary(args, args.baseline, "baseline", case, case_dir / "baseline")
        candidate = run_binary(args, args.candidate, "candidate", case, case_dir / "candidate")
        try:
            comparison = compare_pair(args, case_dir, baseline, candidate)
        except Exception as error:
            comparison = {"schema": SCHEMA, "equivalent": False, "error": f"{type(error).__name__}: {error}"}
        comparison["case"] = case
        dump_new(case_dir / "comparison.json", comparison)
        reports.append({"id": case["id"], "equivalent": comparison["equivalent"],
                        "comparison": resource(case_dir / "comparison.json")})
        print(f"{index}/{len(cases)} {case['id']}: {'equivalent' if comparison['equivalent'] else 'different_or_incomplete'}", flush=True)
    changed = []
    for label, original in resources.items():
        path = Path(original["path"])
        current = sha256(path) if path.is_file() else None
        if current != original["sha256"]:
            changed.append({"label": label, "path": str(path), "before": original["sha256"], "after": current})
    dump_new(args.out / "input_integrity.json", {"unchanged": not changed, "changed": changed})
    passed = all(item["equivalent"] for item in reports) and not changed
    dump_new(args.out / "summary.json", {"schema": SCHEMA, "equivalent": passed, "cases": reports,
        "inputs_unchanged": not changed, "note": "Serial paired simulation comparison. Return codes 0 and 1 are both legitimate trial outcomes; not realtime or hit-quality acceptance."})
    artifacts = {str(path.relative_to(args.out)): resource(path) for path in sorted(args.out.rglob("*")) if path.is_file()}
    dump_new(args.out / "result_manifest.json", {"schema": SCHEMA, "files": artifacts,
        "self_hash": "stored in result_manifest.sha256; manifest and sidecar intentionally not self-listed"})
    with (args.out / "result_manifest.sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha256(args.out / "result_manifest.json") + "  result_manifest.json\n")
    print(f"saved {args.out / 'summary.json'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
