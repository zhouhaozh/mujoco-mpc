"""在Ubuntu以独立进程比较旧预测源码、新legacy入口、新ModelBundle入口。

父进程只用标准库；三个worker使用prediction/.venv-tennis/bin/python和NumPy。
每个worker独立load_dataset，按原列表选择首/中/末3条，消费每个raw前缀。
每20seq及最后seq在初始化后查询forecast；只排除forecast_compute_ms，
公共update/forecast所有其他字段、内部state/covariance逐项比较。
hold/recovery仅用于覆盖记录，离线清洗掩码/反弹标签不作为观察器输入。
输出必须为新目录；不改源码、模型、数据或旧结果。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


SCHEMA = "PREDICTOR_BUNDLE_EQUIVALENCE/1"
WORKERS = ("old", "new_legacy", "new_bundle")
LEGACY_RESOURCES = {
    "collision": "artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json",
    "identified": "artifacts/tennis_identified/20261009_training_prior/selected_model.json",
    "noise": "artifacts/tennis_robust/20261013/validation_v2/noise_proxy.json",
    "collision_velocity_0": "artifacts/tennis_collision/20261009_frozen_pre/models/velocity_0",
    "collision_velocity_1": "artifacts/tennis_collision/20261009_frozen_pre/models/velocity_1",
    "collision_velocity_3": "artifacts/tennis_collision/20261009_frozen_pre/models/velocity_3",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a regular resource: {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def bundle_value(args, name):
    manifest = load_json(args.bundle)
    descriptor = manifest["resources"][name]
    if "inline" in descriptor:
        return descriptor["inline"]
    return load_json(args.bundle.parent / descriptor["path"])


def transform_value(args):
    frame = load_json(args.bundle)["settings"]["frame"]
    return {key: frame[key] for key in ("R_world_from_model", "translation_world_from_model_m")}


def add_tree(resources, label, directory, *, code_only=False):
    directory = Path(directory).resolve(strict=True)
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if code_only and path.suffix != ".py":
            continue
        resources[f"{label}/{path.relative_to(directory)}"] = file_record(path)


def collect_inputs(args):
    manifest = load_json(args.bundle)
    resources = {
        "verification_script": file_record(Path(__file__)),
        "bundle": file_record(args.bundle),
        "worker_python": file_record(args.python),
    }
    add_tree(resources, "source/old", args.baseline_package, code_only=True)
    add_tree(resources, "source/new", args.prediction / "tennis_calibration", code_only=True)
    add_tree(resources, "calibration", args.calibration)
    declared = {}
    for name, descriptor in manifest["resources"].items():
        if descriptor is None:
            declared[name] = None
        elif "path" in descriptor:
            record = file_record(args.bundle.parent / descriptor["path"])
            resources[f"bundle_resource/{name}"] = record
            declared[name] = {**record, "declared_sha256": descriptor["sha256"],
                              "hash_matches": record["sha256"] == descriptor["sha256"]}
        elif "inline" in descriptor:
            digest = hashlib.sha256(canonical(descriptor["inline"])).hexdigest()
            declared[name] = {"inline": True, "sha256": digest,
                              "declared_sha256": descriptor["sha256"], "hash_matches": digest == descriptor["sha256"]}
        else:
            raise ValueError(f"unsupported bundle resource descriptor: {name}")
    alignment = {}
    legacy = dict(LEGACY_RESOURCES)
    selection = bundle_value(args, "selection")
    if selection.get("head"):
        legacy["head"] = selection["head"]
    for name, relative in legacy.items():
        record = file_record(args.prediction / relative)
        resources[f"legacy_resource/{name}"] = record
        target = declared.get(name)
        alignment[name] = {"legacy": record, "bundle_sha256": target["sha256"] if target else None,
                           "same_content": bool(target and record["sha256"] == target["sha256"])}
    return resources, declared, alignment


def worker(args):
    # 只有worker路径导入NumPy及预测package；parent可使用系统纯标准库Python。
    import numpy as np

    package = args.baseline_package if args.worker == "old" else args.prediction / "tennis_calibration"
    package = package.resolve(strict=True)
    if any(name == "tennis_calibration" or name.startswith("tennis_calibration.") for name in sys.modules):
        raise RuntimeError("worker package was imported before source isolation")
    sys.path.insert(0, str(package.parent))
    service = importlib.import_module("tennis_calibration.controller_service")
    source = Path(service.__file__).resolve()
    if not source.is_relative_to(package):
        raise RuntimeError(f"wrong worker source imported: {source}, expected {package}")
    # 旧文件从隔离目录导入后仍读取prediction的真实资源，不从历史快照取模型。
    service.ROOT = args.prediction
    data_module = importlib.import_module("tennis_calibration.online_data")
    if not Path(data_module.__file__).resolve().is_relative_to(package):
        raise RuntimeError("load_dataset imported from a different source tree")
    data = data_module.load_dataset(args.calibration)
    if not data:
        raise ValueError("load_dataset returned no trajectories")
    indices = list(dict.fromkeys((0, len(data) // 2, len(data) - 1)))
    selected = [data[index] for index in indices]
    selection = bundle_value(args, "selection")
    transform = transform_value(args)
    rotation = np.asarray(transform["R_world_from_model"], dtype=float)
    shift = np.asarray(transform["translation_world_from_model_m"], dtype=float)
    if rotation.shape != (3, 3) or shift.shape != (3,):
        raise ValueError("invalid bundle transform shape")
    bundle = None
    if args.worker == "new_bundle":
        model_module = importlib.import_module("tennis_calibration.model_bundle")
        bundle = model_module.ModelBundle.load(args.bundle)
    destination = args.out / "workers" / args.worker
    destination.mkdir(parents=True, exist_ok=False)

    def jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    inputs = []
    for index, trace in zip(indices, selected):
        times = np.asarray(trace["raw_times"], dtype=float)
        raw = np.asarray(trace["raw_xyz"], dtype=float)
        if not len(times) or raw.shape != (len(times), 3):
            raise ValueError(f"empty/invalid raw trajectory: {trace['id']}")
        world = raw @ rotation.T + shift
        if not np.isfinite(times).all() or not np.isfinite(world).all() or not np.all(np.diff(times) > 0):
            raise ValueError(f"invalid raw chronology: {trace['id']}")
        inputs.append({"dataset_index": index, "trajectory_id": trace["id"],
                       "raw_times": times.tolist(), "raw_xyz": raw.tolist(), "world_xyz": world.tolist()})
    dump_new(destination / "selected_inputs.json", inputs)
    coverage = []
    recovery_window = float(load_json(args.bundle)["settings"]["observer"]["fixed_settings"]["recovery_window_s"])
    with (destination / "events.jsonl").open("x", encoding="utf-8") as stream:
        for trace in inputs:
            predictor = (service.ControllerPredictor(bundle=bundle) if bundle is not None
                         else service.ControllerPredictor(selection, transform))
            counts = Counter()
            reasons = Counter()
            hold_intervals = []
            hold_start = None
            resume_times = set()
            for seq, (stamp, point) in enumerate(zip(trace["raw_times"], trace["world_xyz"]), 1):
                update = predictor.update(seq, stamp, point)
                snapshot = predictor.snapshot
                if not isinstance(snapshot, dict):
                    raise ValueError("service did not retain an observer snapshot")
                diagnostics = snapshot.get("diagnostics", {})
                hold = bool(update.get("hold_active", snapshot.get("hold_active", False)))
                counts["updates"] += 1
                counts["initialized_updates"] += bool(update["initialized"])
                counts["hold_updates"] += hold
                reasons[str(update.get("reason"))] += 1
                if hold and hold_start is None:
                    hold_start = stamp
                elif not hold and hold_start is not None:
                    hold_intervals.append([hold_start, stamp])
                    hold_start = None
                resume = snapshot.get("resume_t_s")
                if resume is not None:
                    resume_times.add(float(resume))
                    counts["post_resume_updates"] += stamp >= resume
                    counts["recovery_window_updates"] += 0 <= stamp - resume < recovery_window
                counts["recovery_gain_protected_updates"] += bool(diagnostics.get("recovery_gain_protected", False))
                counts["recovery_budget_updates"] += "recovery_delta_v_budget_m_s" in diagnostics
                scheduled = seq % 20 == 0 or seq == len(trace["raw_times"])
                forecast = None
                if scheduled:
                    counts["scheduled_forecasts"] += 1
                    if update["initialized"]:
                        forecast = predictor.forecast(seq)
                        counts["forecast_queries"] += 1
                        counts["forecast_hold_queries"] += bool(forecast.get("hold_active", False))
                    else:
                        counts["uninitialized_forecast_slots"] += 1
                event = {
                    "trajectory_id": trace["trajectory_id"], "seq": seq, "capture_t_s": stamp,
                    "world_xyz": point, "update": update,
                    "observer_state": snapshot.get("state"),
                    "observer_covariance": snapshot.get("covariance"),
                    "forecast_scheduled": scheduled, "forecast": forecast,
                }
                # 保留公共forecast完整原值；只有parent比较时排除forecast_compute_ms。
                stream.write(json.dumps(jsonable(event), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            if hold_start is not None:
                hold_intervals.append([hold_start, trace["raw_times"][-1]])
            coverage.append({"trajectory_id": trace["trajectory_id"], "counts": dict(counts),
                             "update_reasons": dict(reasons), "hold_intervals_s": hold_intervals,
                             "resume_times_s": sorted(resume_times)})
            print(args.worker, trace["trajectory_id"], dict(counts), flush=True)
    result = {
        "schema": SCHEMA, "worker": args.worker, "source": str(source),
        "load_dataset_source": str(Path(data_module.__file__).resolve()),
        "dataset_size": len(data), "selected_indices": indices,
        "selected_ids": [trace["trajectory_id"] for trace in inputs],
        "python": sys.version, "numpy": np.__version__, "transform": transform,
        "public_payload_exclusion": ["forecast.forecast_compute_ms"],
        "observer_state_note": "Internal state/covariance checked; wall-latency diagnostic object is not part of the public update/forecast payload.",
        "bundle_fingerprint": getattr(predictor, "bundle_fingerprint", None),
        "coverage": coverage,
        "hold_observed": any(item["counts"].get("hold_updates", 0) for item in coverage),
        "recovery_observed": any(item["counts"].get("recovery_window_updates", 0) for item in coverage),
        "artifacts": {name: file_record(destination / name) for name in ("events.jsonl", "selected_inputs.json")},
    }
    dump_new(destination / "worker.json", result)
    return 0


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


def run_worker(args, mode):
    command = [
        str(args.python), str(Path(__file__).resolve()),
        "--baseline-package", str(args.baseline_package), "--prediction", str(args.prediction),
        "--bundle", str(args.bundle), "--out", str(args.out), "--calibration", str(args.calibration),
        "--python", str(args.python), "--worker", mode,
    ]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    result = {"mode": mode, "command": command, "completed": False}
    with (args.out / "workers" / (mode + ".log")).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=args.prediction, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result["returncode"] = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            result["error"] = "worker timeout"
        finally:
            stop_owned(process)
    path = args.out / "workers" / mode / "worker.json"
    if result.get("returncode") == 0 and path.is_file():
        result["completed"] = True
        result["report"] = file_record(path)
    result["wall_s"] = time.monotonic() - started
    dump_new(args.out / "workers" / (mode + ".process.json"), result)
    return result


class Comparison:
    def __init__(self, args):
        self.args = args
        self.count = 0
        self.examples = []
        self.numeric_comparisons = 0
        self.max_absolute_error = 0.
        self.excluded_forecast_times = 0

    def difference(self, path, left, right, reason):
        self.count += 1
        if len(self.examples) < self.args.max_differences:
            self.examples.append({"path": path, "baseline": left, "candidate": right, "reason": reason})

    def compare(self, left, right, path):
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                self.difference(path, left, right, "boolean_or_type")
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            self.numeric_comparisons += 1
            self.max_absolute_error = max(self.max_absolute_error, abs(left - right))
            if not math.isclose(left, right, abs_tol=self.args.atol, rel_tol=self.args.rtol):
                self.difference(path, left, right, "numeric_tolerance")
        elif isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key == "forecast_compute_ms" and path.endswith(".forecast"):
                    self.excluded_forecast_times += 1
                    continue
                if key not in left or key not in right:
                    self.difference(path + "." + key, left.get(key), right.get(key), "missing_field")
                else:
                    self.compare(left[key], right[key], path + "." + key)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                self.difference(path + ".length", len(left), len(right), "length")
            for index, (a, b) in enumerate(zip(left, right)):
                self.compare(a, b, f"{path}[{index}]")
        elif left != right:
            self.difference(path, left, right, "value_or_type")


def compare_workers(args, left_mode, right_mode, processes):
    result = {"left": left_mode, "right": right_mode, "equivalent": False}
    if not processes[left_mode]["completed"] or not processes[right_mode]["completed"]:
        result["reason"] = "incomplete_worker"
        return result
    left_dir = args.out / "workers" / left_mode
    right_dir = args.out / "workers" / right_mode
    left = load_json(left_dir / "worker.json")
    right = load_json(right_dir / "worker.json")
    comparison = Comparison(args)
    for field in ("dataset_size", "selected_indices", "selected_ids", "transform", "coverage"):
        comparison.compare(left[field], right[field], field)
    result["identical_input_content"] = (sha256(left_dir / "selected_inputs.json") == sha256(right_dir / "selected_inputs.json"))
    rows = 0
    with (left_dir / "events.jsonl").open(encoding="utf-8") as before, (right_dir / "events.jsonl").open(encoding="utf-8") as after:
        for index, (a, b) in enumerate(zip_longest(before, after), 1):
            rows += 1
            if a is None or b is None:
                comparison.difference(f"event[{index}]", a is not None, b is not None, "missing_event")
                continue
            # worker写出时allow_nan=False；再次解析时仍拒绝非标准浮点。
            def reject_constant(value):
                raise ValueError(f"nonfinite worker payload: {value}")
            comparison.compare(json.loads(a, parse_constant=reject_constant),
                               json.loads(b, parse_constant=reject_constant), f"event[{index}]")
    result.update({
        "event_rows": rows, "numeric_comparisons": comparison.numeric_comparisons,
        "differences": comparison.count, "examples": comparison.examples,
        "examples_truncated": comparison.count > len(comparison.examples),
        "maximum_absolute_numeric_error": comparison.max_absolute_error,
        "excluded_forecast_compute_ms_fields": comparison.excluded_forecast_times,
        "equivalent": rows > 0 and comparison.count == 0 and result["identical_input_content"],
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--python", type=Path, help="NumPy worker interpreter; default prediction/.venv-tennis/bin/python")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--max-differences", type=int, default=100)
    parser.add_argument("--worker", choices=WORKERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.baseline_package = args.baseline_package.resolve(strict=True)
    args.prediction = args.prediction.resolve(strict=True)
    args.bundle = args.bundle.resolve(strict=True)
    args.out = args.out.resolve()
    args.calibration = (args.calibration or args.prediction / "artifacts/new_ubuntu/tennis_calibration").resolve(strict=True)
    # 保留venv入口路径；resolve()会把symlink解到系统Python并丢失venv site-packages。
    args.python = Path(os.path.abspath(args.python or args.prediction / ".venv-tennis/bin/python"))
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error(f"worker Python is unavailable or not executable: {args.python}")
    if args.worker:
        return worker(args)
    if args.out.exists():
        parser.error(f"output already exists; refusing overwrite: {args.out}")
    if (not all(math.isfinite(value) for value in (args.timeout, args.atol, args.rtol))
            or args.timeout <= 0 or args.atol < 0 or args.rtol < 0 or args.max_differences < 1):
        parser.error("timeout/max-differences must be positive and tolerances finite nonnegative")
    resources, declared, alignment = collect_inputs(args)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "workers").mkdir(exist_ok=False)
    dump_new(args.out / "input_manifest.json", {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "host": platform.node(), "parent_python": sys.version,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resources": resources, "bundle_resources": declared, "legacy_resource_alignment": alignment,
        "protocol": {"trajectories": "load_dataset order: first/middle/last",
                     "input": "all finite strictly chronological raw observations, transformed model→world by bundle frame",
                     "forecast": "every20seq and finalseq when initialized", "only_public_exclusion": "forecast_compute_ms",
                     "isolation": "three fresh workers; import selected source tree; service.ROOT=prediction"},
    })
    processes = {}
    for mode in WORKERS:
        processes[mode] = run_worker(args, mode)
        print(mode, "complete" if processes[mode]["completed"] else "failed", flush=True)
    comparisons = []
    for left, right in (("old", "new_legacy"), ("old", "new_bundle"), ("new_legacy", "new_bundle")):
        result = compare_workers(args, left, right, processes)
        dump_new(args.out / f"{left}_vs_{right}.json", result)
        comparisons.append(result)
    changed = []
    for label, original in resources.items():
        path = Path(original["path"])
        current = sha256(path) if path.is_file() else None
        if current != original["sha256"]:
            changed.append({"label": label, "path": str(path), "before": original["sha256"], "after": current})
    dump_new(args.out / "input_integrity.json", {"unchanged": not changed, "changed": changed})
    hashes_valid = all(item is None or item["hash_matches"] for item in declared.values())
    aligned = all(item["same_content"] for item in alignment.values())
    coverage = {mode: load_json(args.out / "workers" / mode / "worker.json")["coverage"]
                for mode in WORKERS if processes[mode]["completed"]}
    passed = not changed and hashes_valid and aligned and all(result["equivalent"] for result in comparisons)
    dump_new(args.out / "summary.json", {
        "schema": SCHEMA, "equivalent": passed, "comparisons": comparisons,
        "inputs_unchanged": not changed, "bundle_hashes_valid": hashes_valid,
        "legacy_resources_match_bundle": aligned, "worker_processes": processes, "coverage": coverage,
        "limit": "Three historical raw trajectories; numerical refactor equivalence only, not fresh blind accuracy or realtime acceptance.",
    })
    artifacts = {str(path.relative_to(args.out)): file_record(path) for path in sorted(args.out.rglob("*")) if path.is_file()}
    dump_new(args.out / "result_manifest.json", {"schema": SCHEMA, "files": artifacts,
        "self_hash": "result_manifest.sha256; manifest and sidecar are intentionally not self-listed"})
    with (args.out / "result_manifest.sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha256(args.out / "result_manifest.json") + "  result_manifest.json\n")
    print("saved", args.out / "summary.json", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
