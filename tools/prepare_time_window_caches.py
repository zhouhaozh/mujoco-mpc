"""Ubuntu独立预测时距缓存：原始帧/模型/外参固定，各H重新forecast。

使用预测项目的 .venv-tennis，并通过 PYTHONPATH 选择预测源码。只在新输出
目录写入清单和缓存；完整同身份且全部输出SHA256一致时才复用。旧缓存仅供
可选1.5s逐数值对照，从不覆盖。第二次触地截断与<=0.7s残差域保持原样。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

from pipeline_experiment import (ExperimentSpec, RunAttempt, digest, file_resources,
                                 recorded_environment, write_json)


SCHEMA = "FORECAST_HORIZON_CACHES/1"


def predictor_components():
    # Deliberately lazy: this tool runs in the prediction venv; fixture tests
    # can exercise lifecycle/integrity without importing prediction libraries.
    from tennis_calibration import controller_service
    from tennis_calibration.model_bundle import ModelBundle
    return ModelBundle, controller_service.ControllerPredictor, Path(controller_service.__file__).resolve().parents[1]


def horizon_tag(horizon):
    if isinstance(horizon, bool) or not isinstance(horizon, (int, float)) or not math.isfinite(horizon):
        raise ValueError("forecast horizons must be finite numbers")
    return f"horizon_{round(horizon * 1000):04d}ms"


def _transform(value):
    return {key: value[key] for key in ("R_world_from_model", "translation_world_from_model_m")}


def load_source(source, expected_transform, samples=None):
    source = Path(source).resolve(strict=True)
    manifest_path, transform_path = source / "input_manifest.json", source / "transform.json"
    manifest = json.loads(manifest_path.read_text())
    transform = json.loads(transform_path.read_text())
    if _transform(transform) != expected_transform:
        raise ValueError("source/transform.json differs from the bundle; refusing an implicit frame change")
    if "transform" in manifest and _transform(manifest["transform"]) != expected_transform:
        raise ValueError("input_manifest transform differs from the bundle")
    requested = None if samples is None else set(samples)
    items, seen = [], set()
    resources = {"input_manifest": manifest_path, "source_transform": transform_path}
    for raw in manifest["items"]:
        sample = raw["sample_id"]
        if not isinstance(sample, str) or not re.fullmatch("[A-Za-z0-9_-]+", sample) or sample in seen:
            raise ValueError(f"invalid or duplicate sample identity: {sample}")
        seen.add(sample)
        if requested is not None and sample not in requested:
            continue
        path = Path(raw["csv"])
        path = (path if path.is_absolute() else source / path).resolve(strict=True)
        actual_sha = digest(path)
        if raw.get("source_sha256") is not None and raw["source_sha256"] != actual_sha:
            raise ValueError(f"raw source SHA256 differs from input_manifest: {sample}")
        item = {**raw, "csv": str(path), "source_sha256": actual_sha}
        items.append(item)
        resources["raw/" + sample] = path
    if not items or requested is not None and requested - {item["sample_id"] for item in items}:
        raise ValueError("no samples selected or requested samples are missing")
    return items, resources


def compare_cache_files(actual_path, reference_path, *, atol=1e-10, rtol=1e-12):
    """Compare every field/row, ignoring only forecast_compute_ms telemetry."""
    stats = {"rows": 0, "numeric_values": 0, "max_abs_delta": 0.,
             "absolute_tolerance": atol, "relative_tolerance": rtol,
             "ignored_fields": ["forecast_compute_ms"]}

    def compare(actual, reference, location):
        if isinstance(actual, bool) or isinstance(reference, bool):
            if type(actual) is not type(reference) or actual != reference:
                raise ValueError(f"baseline cache differs at {location}")
        elif isinstance(actual, (int, float)) and isinstance(reference, (int, float)):
            if not math.isfinite(actual) or not math.isfinite(reference):
                raise ValueError(f"nonfinite baseline cache number at {location}")
            delta = abs(actual-reference)
            stats["numeric_values"] += 1
            stats["max_abs_delta"] = max(stats["max_abs_delta"], delta)
            if delta > atol + rtol * abs(reference):
                raise ValueError(f"baseline cache differs at {location}: absolute delta={delta}")
        elif isinstance(actual, dict) and isinstance(reference, dict):
            left, right = set(actual)-{"forecast_compute_ms"}, set(reference)-{"forecast_compute_ms"}
            if left != right:
                raise ValueError(f"baseline cache field set differs at {location}")
            for key in sorted(left):
                compare(actual[key], reference[key], location + "." + key)
        elif isinstance(actual, list) and isinstance(reference, list):
            if len(actual) != len(reference):
                raise ValueError(f"baseline cache array length differs at {location}")
            for index, (left, right) in enumerate(zip(actual, reference)):
                compare(left, right, f"{location}[{index}]")
        elif type(actual) is not type(reference) or actual != reference:
            raise ValueError(f"baseline cache differs at {location}")

    missing = object()
    with Path(actual_path).open() as actual, Path(reference_path).open() as reference:
        for index, (left, right) in enumerate(zip_longest(actual, reference, fillvalue=missing), 1):
            if left is missing or right is missing:
                raise ValueError(f"baseline cache row count differs at row {index}")
            compare(json.loads(left), json.loads(right), f"row[{index}]")
            stats["rows"] += 1
    return stats


def _prepare_sample(job):
    item, domains, baseline_cache = job
    ModelBundle, ControllerPredictor, _ = predictor_components()
    path = Path(item["csv"])
    if digest(path) != item["source_sha256"]:
        raise ValueError(f"raw input changed before replay: {item['sample_id']}")
    with path.open(newline="") as stream:
        observations = []
        for row in csv.DictReader(stream):
            # Match control_ablation_cache.prepare's original column order and
            # numerical parsing; do not smooth, interpolate, drop or relabel.
            values = list(map(float, row.values()))
            if len(values) < 4:
                raise ValueError(f"t/XYZ columns missing: {path}")
            observations.append((values[0], values[1:4]))
    if not observations or item.get("input_frames", len(observations)) != len(observations):
        raise ValueError(f"raw input frame count differs from input_manifest: {item['sample_id']}")
    result = {"sample_id": item["sample_id"], "input_frames": len(observations), "horizons": {}}
    for domain in domains:
        bundle = ModelBundle.load(domain["bundle"])
        predictor = ControllerPredictor(bundle=bundle)  # Never share trackers across H/sample.
        cache = Path(domain["cache"]) / (item["sample_id"] + ".jsonl")
        initialized, empty_domain = 0, 0
        with cache.open("x") as stream:
            for seq, (stamp, xyz) in enumerate(observations, 1):
                update = predictor.update(seq, stamp, xyz)
                forecast = predictor.forecast(seq) if update["initialized"] else None
                if forecast is not None:
                    initialized += 1
                    empty_domain += not forecast["domain_valid"]
                    if forecast["max_horizon_s"] != domain["horizon_s"]:
                        raise ValueError("predictor did not consume the requested experimental horizon")
                row = {"seq": seq, "t_s": stamp, "xyz": xyz, "update": update, "forecast": forecast}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        record = {"file": cache.name, "sha256": digest(cache), "frames": len(observations),
                  "initialized_frames": initialized, "empty_domain_forecasts": empty_domain}
        if baseline_cache is not None and domain["horizon_s"] == 1.5:
            reference = Path(baseline_cache) / cache.name
            record["baseline_comparison"] = {"reference": str(reference), "reference_sha256": digest(reference),
                                              **compare_cache_files(cache, reference)}
        result["horizons"][domain["tag"]] = record
    return result


def prepare(bundle_path, prediction_project, source, out, horizons, *, workers=4,
            baseline_cache=None, samples=None):
    """Generate each H independently; a completed exact identity is read-only reusable."""
    ModelBundle, _, prediction_source = predictor_components()
    bundle_path = Path(bundle_path).resolve(strict=True)
    prediction_project, source, out = (Path(path).resolve() for path in (prediction_project, source, out))
    baseline_cache = None if baseline_cache is None else Path(baseline_cache).resolve(strict=True)
    if out.is_relative_to(source) or baseline_cache is not None and out.is_relative_to(baseline_cache):
        raise ValueError("output must be separate from the raw source and original cache directories")
    if workers < 1:
        raise ValueError("workers must be positive")
    base = ModelBundle.load(bundle_path)
    horizons = [float(value) for value in horizons]
    if len(set(horizons)) != len(horizons) or not horizons:
        raise ValueError("nonempty unique forecast horizons are required")
    experiments = {horizon_tag(h): base.with_forecast_horizon(h) for h in horizons}
    if len(experiments) != len(horizons):
        raise ValueError("forecast horizon output tags collide")
    if baseline_cache is not None and 1.5 not in horizons:
        raise ValueError("--baseline-cache requires a 1.5s experiment")
    items, source_files = load_source(source, base.effective_config["transform"], samples)
    paths = {**source_files, "base_model_bundle": bundle_path,
             "predictor_python": prediction_project / ".venv-tennis/bin/python",
             "predictor_code": prediction_source / "tennis_calibration",
             "runner": Path(__file__), "runner_common": Path(__file__).with_name("pipeline_experiment.py")}
    expected = {}
    for name, resource in base.resolved_manifest["resources"].items():
        if resource is not None and "resolved_path" in resource:
            label = "model/" + name
            paths[label] = resource["resolved_path"]
            expected[label] = resource["sha256"]
    if baseline_cache is not None:
        paths.update({"old_cache/" + item["sample_id"]: baseline_cache / (item["sample_id"] + ".jsonl") for item in items})
    resources = file_resources(paths)
    for label, expected_sha in expected.items():
        if resources[label]["sha256"] != expected_sha:
            raise RuntimeError(f"model resource changed after bundle inspection: {label}")
    spec = ExperimentSpec("prepare_time_window_caches", (sys.executable, str(Path(__file__).resolve())),
        str(Path.cwd()), {"schema": SCHEMA, "source": str(source), "prediction_project": str(prediction_project),
            "prediction_source": str(prediction_source), "forecast_horizons_s": horizons,
            "sample_ids": [item["sample_id"] for item in items], "workers": workers,
            "strategy": "independent_forecast_per_horizon_and_sample", "base_bundle_fingerprint": base.fingerprint,
            "experimental_bundle_fingerprints": {tag: bundle.fingerprint for tag, bundle in experiments.items()},
            "source_transform": base.effective_config["transform"], "baseline_cache": None if baseline_cache is None else str(baseline_cache)},
        recorded_environment(os.environ), resources)
    with RunAttempt(out, spec, completion="manifest.json") as attempt:
        if attempt.existing is not None:
            return attempt.existing
        domains = []
        for horizon in horizons:
            tag, bundle = horizon_tag(horizon), experiments[horizon_tag(horizon)]
            directory = out / tag
            cache = directory / "cache"
            cache.mkdir(parents=True)
            path = bundle.write(directory / "model_bundle.json")
            deployed = ModelBundle.load(path)
            write_json(directory / "resolved_manifest.json", deployed.resolved_manifest, exclusive=True)
            domains.append({"horizon_s": horizon, "tag": tag, "bundle": str(path), "cache": str(cache),
                            "bundle_fingerprint": deployed.fingerprint})
        jobs = [(item, domains, None if baseline_cache is None else str(baseline_cache)) for item in items]
        results = []
        if workers == 1:
            for job in jobs:
                results.append(_prepare_sample(job))
                print(f"cache {len(results)}/{len(items)} {results[-1]['sample_id']} ({len(domains)} independent horizons)", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(workers) as pool:
                futures = [pool.submit(_prepare_sample, job) for job in jobs]
                try:
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                        print(f"cache {len(results)}/{len(items)} {results[-1]['sample_id']} ({len(domains)} independent horizons)", flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        by_id = {result["sample_id"]: result for result in results}
        reports = []
        for domain in domains:
            tag = domain["tag"]
            records = [by_id[item["sample_id"]]["horizons"][tag] for item in items]
            cache_manifest = {"schema": SCHEMA, "source": str(source), "selection": base.selection,
                "model_bundle": "../model_bundle.json", "bundle_fingerprint": domain["bundle_fingerprint"],
                "forecast_max_horizon_s": domain["horizon_s"], "strategy": "independent_forecast_per_horizon_and_sample",
                "causal": "each raw frame updates a fresh per-sample/per-H tracker; forecast immediately follows that prefix only",
                "files": records}
            path = Path(domain["cache"]) / "manifest.json"
            write_json(path, cache_manifest, exclusive=True)
            reports.append({"horizon_s": domain["horizon_s"], "tag": tag,
                "bundle_fingerprint": domain["bundle_fingerprint"], "cache": str(path.parent.relative_to(out)),
                "cache_manifest": str(path.relative_to(out)), "cache_manifest_sha256": digest(path),
                "sample_count": len(records), "frames": sum(record["frames"] for record in records)})
        return attempt.complete({"schema": SCHEMA, "source": str(source), "sample_count": len(items),
            "strategy": "independent_forecast_per_horizon_and_sample", "horizons": reports,
            "base_bundle_fingerprint": base.fingerprint,
            "baseline_comparison": "all selected 1.5s rows and fields, excluding forecast_compute_ms" if baseline_cache is not None else "unavailable_no_reference_requested"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prediction-project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizons", default="0.8,1,1.2,1.5,1.8,2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--samples", help="optional comma-separated smoke subset; exact IDs recorded")
    args = parser.parse_args()
    if platform.system() != "Linux" or Path(sys.prefix).resolve() != (args.prediction_project / ".venv-tennis").resolve():
        parser.error("use Ubuntu and prediction-project/.venv-tennis/bin/python; select source with PYTHONPATH")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        horizons = [float(value) for value in args.horizons.split(",")]
        result = prepare(args.bundle, args.prediction_project, args.source, args.out, horizons,
                         workers=args.workers, baseline_cache=args.baseline_cache,
                         samples=None if args.samples is None else args.samples.split(","))
        print(json.dumps({"manifest": str(args.out.resolve() / "manifest.json"),
                          "sample_count": result["sample_count"], "horizons": len(result["horizons"])}), flush=True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"forecast cache error: {error}\n")


if __name__ == "__main__":
    main()
