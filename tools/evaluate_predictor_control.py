"""Ubuntu 预测器/控制器组合评估；隔离服务、完整参数及内容身份，保留原指标。"""
import argparse
import concurrent.futures
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import subprocess
import time

from pipeline_experiment import (ExperimentSpec, RunAttempt, STEER_RATE_MODES,
    clean_environment, file_resources, recorded_environment, resolved_runtime_config, run_logged, socket_path,
    socket_service, team_resources, write_json)


_BUNDLE_INSPECTOR = """import json,sys
from pathlib import Path
from tennis_calibration.model_bundle import ModelBundle
root=Path(sys.argv[1]);kind=sys.argv[2];source=Path(sys.argv[3])
bundle=(ModelBundle.load(source) if kind=='bundle' else
        ModelBundle.from_legacy(source,synthetic_world=True,root=root))
print(json.dumps(bundle.resolved_manifest,ensure_ascii=False,allow_nan=False))
"""


def prediction_source_path(prediction, prediction_source=None):
    source = Path(prediction_source if prediction_source is not None else prediction).resolve(strict=True)
    if not source.is_dir() or not (source / "tennis_calibration").is_dir():
        raise ValueError(f"prediction source must contain tennis_calibration/: {source}")
    return source


def prediction_environment(environment, source):
    result = dict(environment)
    inherited = result.get("PYTHONPATH", "")
    result["PYTHONPATH"] = str(source) + (os.pathsep + inherited if inherited else "")
    return result


def inspect_predictor_bundle(prediction, selection=None, *, model_bundle=None,
                             prediction_source=None, timeout=30):
    """在预测环境中解析配置；团队runner不导入预测器/NumPy/SciPy依赖。"""
    prediction = Path(prediction).resolve()
    source_root = prediction_source_path(prediction, prediction_source)
    source = model_bundle if model_bundle is not None else selection
    if source is None:
        raise ValueError("a selection or model bundle is required for external prediction")
    source = Path(source).resolve(strict=True)
    command = [str(prediction / ".venv-tennis/bin/python"), "-c", _BUNDLE_INSPECTOR,
               str(prediction), "bundle" if model_bundle is not None else "selection", str(source)]
    process = subprocess.run(command, cwd=source_root,
                             env=prediction_environment(clean_environment(), source_root),
                             capture_output=True, text=True, timeout=timeout)
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()[-4000:]
        raise RuntimeError(f"predictor bundle inspection failed ({process.returncode}): {detail}")
    try:
        resolved = json.loads(process.stdout)
    except (ValueError, TypeError) as error:
        raise ValueError("predictor inspector did not return a JSON manifest") from error
    if (not isinstance(resolved, dict) or resolved.get("schema") != "TENNIS_MODEL_BUNDLE/1"
            or not isinstance(resolved.get("resources"), dict)
            or not isinstance(resolved.get("fingerprint"), str)
            or not re.fullmatch("[0-9a-f]{64}", resolved["fingerprint"])):
        raise ValueError("predictor inspector returned an invalid resolved model bundle")
    return resolved


def predictor_resources(prediction, selection=None, *, model_bundle=None, resolved_bundle=None,
                        prediction_source=None):
    """只读取bundle显式列出的模型资源；同时保留运行时和源码内容身份。"""
    prediction = Path(prediction).resolve()
    source_root = prediction_source_path(prediction, prediction_source)
    resolved = resolved_bundle if resolved_bundle is not None else inspect_predictor_bundle(
        prediction, selection, model_bundle=model_bundle, prediction_source=source_root)
    paths = {"predictor_python": prediction / ".venv-tennis/bin/python",
             "predictor_code": source_root / "tennis_calibration"}
    labels = {"selection": "selection", "collision": "collision_selection",
              "identified": "filter_selection", "noise": "noise_proxy", "head": "residual_head",
              "collision_velocity_0": "collision_models/velocity_0",
              "collision_velocity_1": "collision_models/velocity_1",
              "collision_velocity_3": "collision_models/velocity_3", "transform": "predictor_transform"}
    expected = {}
    for name, resource in resolved["resources"].items():
        if resource is None:
            if name != "head":
                raise ValueError(f"required predictor resource is unavailable: {name}")
            continue
        if not isinstance(resource, dict) or not re.fullmatch("[0-9a-f]{64}", str(resource.get("sha256", ""))):
            raise ValueError(f"invalid predictor resource descriptor: {name}")
        if "path" in resource:
            raw = resource.get("resolved_path")
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise ValueError(f"predictor resource has no absolute resolved path: {name}")
            label = labels.get(name, "predictor_bundle/" + name)
            paths[label] = Path(raw)
            expected[label] = resource["sha256"]
        elif "inline" not in resource:
            raise ValueError(f"predictor resource is neither a file nor inline: {name}")
    if model_bundle is not None:
        paths["model_bundle"] = Path(model_bundle).resolve(strict=True)
    lock = prediction / "requirements.lock.txt"
    if lock.is_file():
        paths["predictor_requirements"] = lock
    resources = file_resources(paths)
    for label, expected_sha in expected.items():
        if resources[label]["sha256"] != expected_sha:
            raise RuntimeError(f"predictor resource changed after bundle inspection: {resources[label]['path']}")
    return resources


def predictor_service_command(prediction, selection, destination, *, model_bundle=None,
                              prediction_source=None):
    prediction, destination = Path(prediction), Path(destination)
    command = [str(prediction / ".venv-tennis/bin/python"), "-m", "tennis_calibration.controller_service"]
    if model_bundle is not None:
        command += ["--bundle", str(Path(model_bundle).resolve())]
    else:
        command += ["--selection", str(selection)]
        if prediction_source is not None and Path(prediction_source).resolve() != prediction.resolve():
            command += ["--project-root", str(prediction.resolve())]
    command += ["--socket", "{socket}"]
    if model_bundle is None:
        command += ["--synthetic-world"]
    return command + ["--telemetry", str(destination / "timing.jsonl"),
                      "--resolved-manifest", str(destination / "predictor_resolved_manifest.json")]


def verify_predictor_manifest(path, expected):
    actual = json.loads(Path(path).read_text())
    if actual != expected:
        raise RuntimeError("prediction service resolved manifest differs from preflight inspection")
    return actual


def parse_summary(log, result):
    """兼容当前 C++ stdout；结构化 episode 输出接通前，仅此处保留文本适配。"""
    match = re.search(r"命中 (\d+)/(\d+)\(.*?平均击球点偏差 (.*?)m", log)
    if not match or int(match[2]) > result["trials"]:
        raise RuntimeError("missing/invalid control summary; see control.log")
    result.update(hits=int(match[1]), mean_plan_error_m=None if match[3] == "n/a" else float(match[3]))
    match = re.search(r"随机发球族: 可达 (\d+)/(\d+)", log)
    result["feasible"] = int(match[1]) if match else result["trials"]
    match = re.search(r"旧机NMPC: (\d+) 解 (\d+) 拒收 .*?回退 (\d+) trial", log)
    if match:
        result.update(nmpc_accepted=int(match[1]), nmpc_rejected=int(match[2]), fallback_trials=int(match[3]))
    match = re.search(r"全身MPC: (\d+) 解 (\d+) 拒收 (\d+) 失败,最大单解 ([\d.]+)ms.*?回退 (\d+) trial", log)
    if match:
        result.update(wbmpc_accepted=int(match[1]), wbmpc_rejected=int(match[2]), wbmpc_failed=int(match[3]),
                      wbmpc_accepted_max_ms=float(match[4]), fallback_trials=int(match[5]))
    match = re.search(r"NMPC attempts=(\d+) mean_ms=([\d.]+) max_ms=([\d.]+)", log)
    if match:
        result.update(nmpc_attempts=int(match[1]), nmpc_attempt_mean_ms=float(match[2]), nmpc_attempt_max_ms=float(match[3]))
    return result


def run(job):
    a, variant, mode, family, seed = job
    if variant not in ("legacy", "dedup", "external"):
        raise ValueError(f"unknown predictor variant: {variant}")
    team = Path(a.team).resolve()
    prediction = Path(a.prediction).resolve()
    binary = Path(getattr(a, "binary", None) or team / "sckd_sim/build-predictor/m2_intercept").resolve(strict=True)
    dest = Path(a.out).resolve() / f"{variant}_{mode}_{family}_{seed}"
    sock = socket_path(team, "forecast")
    # Preserve the previous binary defaults, then make the effective value explicit.
    steering = getattr(a, "steer_rate_mode", None) or ("both_unlimited" if mode in ("pfly", "nmpc6") else "limited")
    env = clean_environment({"M2_CONTROL_VARIANT": "baseline", "M2_STEER_RATE_MODE": steering})
    if a.profile:
        env["M2_TIMING_PATH"] = str(dest / "stage_timing.json")
    video_trace = bool(getattr(a, "video_trace", False))
    if video_trace:
        env["M2_VIDEO_TRACE"] = str(dest / "frames.jsonl")
    if variant == "external":
        env["BALL_PREDICTOR_SOCKET"] = str(sock)
    elif variant == "dedup":
        env["BALL_DEDUP"] = "1"
    command = [str(binary), "models/generated/x4_tennis_demo.mjcf",
               str(a.trials), "qp", ".25", str(seed), mode, family, "xb" if family == "gs" else "v2"]
    resources = {**team_resources(team, mode, binary=binary), **file_resources({"runner": Path(__file__),
                 "runner_common": Path(__file__).with_name("pipeline_experiment.py")})}
    service = []
    bundle_parameters = {}
    resolved_bundle = None
    source_root = prediction
    if variant == "external":
        source_root = prediction_source_path(prediction, getattr(a, "prediction_source", None))
        env = prediction_environment(env, source_root)
        selection = prediction / "artifacts/tennis_controller/20261014/selected.json"
        model_bundle = getattr(a, "model_bundle", None)
        resolved_bundle = inspect_predictor_bundle(prediction, selection, model_bundle=model_bundle,
                                                   prediction_source=source_root)
        service = predictor_service_command(prediction, selection, dest, model_bundle=model_bundle,
                                              prediction_source=source_root)
        resources.update(predictor_resources(prediction, selection,
            model_bundle=model_bundle, resolved_bundle=resolved_bundle, prediction_source=source_root))
        bundle_parameters = {"predictor_bundle_fingerprint": resolved_bundle["fingerprint"],
                             "predictor_resolved_manifest": resolved_bundle,
                             "prediction_project": str(prediction), "prediction_source": str(source_root)}
    timeout = getattr(a, "timeout", 1800)
    startup_timeout = getattr(a, "startup_timeout", 10)
    spec = ExperimentSpec("predictor_control", tuple(command), str(team / "sckd_sim"),
        {"variant": variant, "controller_variant": "baseline", "mode": mode, "family": family,
         "binary": str(binary), "scene": command[1], "trials": a.trials, "swing": "qp", "init_delay_s": .25,
         "seed": seed, "rule": command[-1], "steer_rate_mode": steering, "profile": a.profile,
         "video_trace": video_trace,
         "timeout_s": timeout, "startup_timeout_s": startup_timeout, **bundle_parameters},
        recorded_environment(env, sock), resources, tuple(service))
    with RunAttempt(dest, spec, completion="summary.json",
                    retry_incomplete=getattr(a, "retry_incomplete", False)) as attempt:
        if attempt.existing is not None:
            return attempt.existing
        manager = socket_service([str(sock) if value == "{socket}" else value for value in service],
            socket=sock, cwd=source_root, env=env, log=dest / "service.log",
            startup_timeout=startup_timeout, stop_timeout=5) if service else nullcontext()
        started = time.monotonic()
        with manager:
            if resolved_bundle is not None:
                verify_predictor_manifest(dest / "predictor_resolved_manifest.json", resolved_bundle)
            process = run_logged(command, cwd=team / "sckd_sim", env=env,
                                 log=dest / "control.log", timeout=timeout)
        if process.returncode not in (0, 1):
            raise RuntimeError(f"control exited with {process.returncode}; see {dest / 'control.log'}")
        log = (dest / "control.log").read_text()
        runtime = resolved_runtime_config(log)
        result = parse_summary(log,
            dict(variant=variant, mode=mode, family=family, seed=seed, trials=a.trials,
                 returncode=process.returncode, wall_s=time.monotonic() - started, command=command,
                 binary=str(binary), binary_sha256=resources["binary"]["sha256"],
                 resolved_runtime_config=runtime, resolved_runtime_config_available=runtime is not None))
        if video_trace and not (dest / "frames.jsonl").is_file():
            raise RuntimeError("video trace requested but not recorded")
        if a.profile:
            result["stage_timing"] = json.loads((dest / "stage_timing.json").read_text())
        if variant == "external":
            result.update(bundle_parameters)
            import numpy as np
            timings = [json.loads(line) for line in (dest / "timing.jsonl").read_text().splitlines()]
            result["service_timing"] = {op: {"n": len(values), "mean_ms": float(np.mean(values)),
                "p95_ms": float(np.percentile(values, 95)), "max_ms": float(np.max(values))}
                for op in ("update", "forecast")
                if (values := [row["elapsed_ms"] for row in timings if row["op"] == op])}
        return attempt.complete(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--prediction-source", type=Path,
                        help="含tennis_calibration的源码根；默认prediction项目，解释器和资源仍在prediction")
    parser.add_argument("--binary", type=Path,
                        help="实际执行的m2_intercept；默认team/sckd_sim/build-predictor/m2_intercept")
    parser.add_argument("--model-bundle", type=Path,
                        help="external预测器使用的完整bundle；省略时保留原selection与synthetic-world默认")
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--variants", default="legacy,dedup,external")
    parser.add_argument("--seeds", default="2026,31,777")
    parser.add_argument("--modes", default="pfly,nmpc6,wbmpc")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--video-trace", action="store_true")
    parser.add_argument("--steer-rate-mode", choices=STEER_RATE_MODES)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--startup-timeout", type=float, default=10)
    parser.add_argument("--retry-incomplete", action="store_true")
    a = parser.parse_args()
    if a.trials < 1 or a.workers < 1:
        parser.error("--trials and --workers must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(a, variant, mode, family, int(seed)) for variant in a.variants.split(",")
            for mode in a.modes.split(",") for family in ("fixed", "rand", "gs") for seed in a.seeds.split(",")]
    key = lambda result: (result["variant"], result["mode"], result["family"], result["seed"])
    summary = out / "summary.json"
    records = {key(result): result for result in json.loads(summary.read_text())} if summary.exists() else {}
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, job) for job in jobs]):
            result = future.result()
            previous = records.get(key(result))
            if previous and previous != result:
                raise FileExistsError(f"aggregate contains a different/legacy result; preserved: {summary}; use a new output directory")
            records[key(result)] = result
            print(result["variant"], result["mode"], result["family"], result["seed"], result["hits"],
                  result.get("nmpc_accepted"), flush=True)
            write_json(summary, [records[k] for k in sorted(records)])


if __name__ == "__main__":
    main()
