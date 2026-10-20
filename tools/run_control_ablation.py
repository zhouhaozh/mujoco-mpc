"""nmpc6/pfly 配对消融；固定因果快照，完整内容身份匹配后才复用结果。"""
import argparse
import json
from pathlib import Path
import sys
import time

from pipeline_experiment import (ExperimentSpec, RunAttempt, STEER_RATE_MODES,
    clean_environment, digest, file_resources, recorded_environment, resolved_runtime_config, run_logged,
    socket_path, socket_service, team_resources, write_json)


def run_one(a, variant, mode, item, resources=None):
    team = a.team.resolve()
    source = a.source.resolve()
    root = a.root.resolve()
    binary = team / "sckd_sim/build-predictor/m2_intercept"
    dest = root / "runs" / variant / mode / item["sample_id"]
    diagnostics = dest / "diagnostics.jsonl"
    cache = root / "cache" / (item["sample_id"] + ".jsonl")
    identified = root / "identification/identified.json"
    sock = socket_path(team, "ablate")
    env = clean_environment({"M2_CONTROL_VARIANT": variant,
        "M2_CONTROL_DIAGNOSTICS": diagnostics, "M2_MEASURED_REPLAY": item["csv"],
        "BALL_PREDICTOR_SOCKET": sock, "M2_SERVO_IDENTIFICATION": identified,
        "M2_STEER_RATE_MODE": a.steer_rate_mode})
    if a.video_trace:
        env["M2_VIDEO_TRACE"] = str(dest / "frames.jsonl")
    command = [str(binary), "models/generated/x4_tennis_demo.mjcf", "1", "qp", ".25", "2026", mode, "measured", "xb"]
    service = [sys.executable, str(team / "tools/control_ablation_cache.py"),
               "--serve", str(cache), "--socket", "{socket}"]
    resources = {**(team_resources(team, mode) if resources is None else resources),
        **file_resources({"input": item["csv"], "cache": cache, "identified": identified,
            "input_manifest": source / "input_manifest.json", "cache_service": service[1],
            "runner": Path(__file__), "runner_common": Path(__file__).with_name("pipeline_experiment.py")})}
    timeout = getattr(a, "timeout", 600)
    startup_timeout = getattr(a, "startup_timeout", 2)
    spec = ExperimentSpec("control_ablation", tuple(command), str(team / "sckd_sim"),
        {"variant": variant, "mode": mode, "sample_id": item["sample_id"],
         "trajectory_id": item["trajectory_id"], "scene": command[1], "trials": 1,
         "swing": "qp", "init_delay_s": .25, "seed": 2026, "family": "measured",
         "rule": "xb", "steer_rate_mode": a.steer_rate_mode, "video_trace": a.video_trace,
         "timeout_s": timeout, "startup_timeout_s": startup_timeout},
        recorded_environment(env, sock), resources, tuple(service))
    with RunAttempt(dest, spec, retry_incomplete=getattr(a, "retry_incomplete", False)) as attempt:
        if attempt.existing is not None:
            return attempt.existing
        started = time.monotonic()
        with socket_service([str(sock) if value == "{socket}" else value for value in service],
                socket=sock, cwd=team, env=env, log=dest / "cache_service.log",
                startup_timeout=startup_timeout):
            result = run_logged(command, cwd=team / "sckd_sim", env=env,
                                log=dest / "control.log", timeout=timeout)
        if result.returncode not in (0, 1) or not diagnostics.is_file():
            raise RuntimeError(f"{dest} failed with {result.returncode} or has no diagnostics")
        runtime = resolved_runtime_config((dest / "control.log").read_text())
        record = {"steer_rate_mode": a.steer_rate_mode, "variant": variant, "mode": mode,
            "sample_id": item["sample_id"], "trajectory_id": item["trajectory_id"],
            "wall_s": time.monotonic() - started, "returncode": result.returncode,
            "binary_sha256": resources["binary"]["sha256"], "cache_sha256": resources["cache"]["sha256"],
            "input_sha256": resources["input"]["sha256"], "identified_sha256": resources["identified"]["sha256"],
            "resolved_runtime_config": runtime, "resolved_runtime_config_available": runtime is not None}
        if a.video_trace:
            record["video_trace_sha256"] = digest(dest / "frames.jsonl")
        if variant.startswith("chassis_"):
            paths = [team / "sckd_sim" / p for p in (
                "models/config/chassis_identified_20261015.json", "third_party/nmpc6/nlp_chassis.so",
                "third_party/nmpc6/chassis_rollout.so")]
            record["chassis_resources"] = {str(p.relative_to(team)): digest(p) for p in paths}
        return attempt.complete(record)


def main(a):
    team = a.team.resolve()
    source = a.source.resolve()
    root = a.root.resolve()
    items = json.loads((source / "input_manifest.json").read_text())["items"]
    if a.samples:
        selected = set(a.samples.split(","))
        items = [item for item in items if item["sample_id"] in selected]
        missing = selected - {item["sample_id"] for item in items}
        if missing:
            raise ValueError(f"unknown samples: {sorted(missing)}")
    if not items:
        raise ValueError("no input samples")
    variants = a.variants.split(",")
    modes = a.modes.split(",")
    jobs = [(variant, mode, item) for variant in variants for mode in modes
            if not (variant == "servo_mpc" and mode != "nmpc6") for item in items]
    resources = {mode: team_resources(team, mode) for mode in modes}
    results = []
    for number, (variant, mode, item) in enumerate(jobs, 1):
        result = run_one(a, variant, mode, item, resources[mode])
        results.append(result)
        print(f'{number}/{len(jobs)} {variant} {mode} {item["sample_id"]} {result["wall_s"]:.2f}s', flush=True)
        write_json(root / "progress.json", {"done": len(results), "expected_this_command": len(jobs),
                   "current": [variant, mode, item["sample_id"]]})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--variants", default="baseline,feasible,clock,feedforward,gains,servo_mpc")
    parser.add_argument("--modes", default="nmpc6,pfly")
    parser.add_argument("--samples")
    parser.add_argument("--video-trace", action="store_true")
    parser.add_argument("--steer-rate-mode", choices=STEER_RATE_MODES, default="limited")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=2)
    parser.add_argument("--retry-incomplete", action="store_true")
    main(parser.parse_args())
