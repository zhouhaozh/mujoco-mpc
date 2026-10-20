"""Ubuntu独立合成场景确认：固定实测发现集选出的设置，不按合成结果重选。

--settings为JSON数组，每项{id,mode,policy,bundle,nlp_path?}，文件路径相对
settings文件。policy使用已冻结的时间策略，bundle必须声明synthetic_world。
默认fixed/rand/gs × seeds2026,31 × 每组5球；4条设置共120球。每组独立预测
服务和RunAttempt，串行运行。stdout命中为合成解析碰撞判定，不是实机命中率。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from evaluate_predictor_control import (inspect_predictor_bundle, parse_summary,
    prediction_environment, prediction_source_path, predictor_resources,
    predictor_service_command, verify_predictor_manifest)
from pipeline_experiment import (ExperimentSpec, RunAttempt, clean_environment,
    file_resources, positive_timeout, recorded_environment, resolved_runtime_config,
    run_logged, socket_path, socket_service, team_resources, write_json)

SCHEMA = "TIME_WINDOWS_SYNTHETIC_CONFIRMATION/1"
INTERPRETATION = "synthetic analytic racket-ball collision hits; not real-robot validation"


def content_identity(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f"different frozen confirmation inputs preserved: {path}; use a new --out")
    else:
        write_json(path, value, exclusive=True)


def path_from_setting(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("setting paths must be nonempty strings")
    path = Path(value)
    path = (path if path.is_absolute() else base / path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"setting resource must be a regular file: {path}")
    return path


def load_settings(path):
    path = Path(path).resolve(strict=True)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("--settings must contain a nonempty JSON array of fixed finalists")
    settings, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or set(item) - {"id", "mode", "policy", "bundle", "nlp_path"}:
            raise ValueError("setting fields must be id/mode/policy/bundle/nlp_path?")
        if not {"id", "mode", "policy", "bundle"}.issubset(item):
            raise ValueError("every setting requires id, mode, policy and bundle")
        name = item["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name) or name in seen:
            raise ValueError("setting IDs must be unique safe path components")
        seen.add(name)
        if item["mode"] not in ("pfly", "nmpc6"):
            raise ValueError("synthetic time confirmation supports pfly and nmpc6")
        policy_path = path_from_setting(item["policy"], path.parent)
        policy = json.loads(policy_path.read_text())
        if not isinstance(policy, dict) or policy.get("schema") != "SCKD_TIME_WINDOWS/1":
            raise ValueError(f"invalid time policy: {policy_path}")
        delay = policy.get("pfly_init_delay_s")
        delay = .25 if delay is None else delay
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
            raise ValueError("pfly_init_delay_s must be finite and nonnegative, or null for legacy .25")
        nlp = path_from_setting(item["nlp_path"], path.parent) if item.get("nlp_path") is not None else None
        if item["mode"] == "nmpc6" and policy.get("nmpc6", {}).get("allow_nonuniform_dt") and nlp is None:
            raise ValueError("nonuniform nmpc6 setting requires an explicit independent nlp_path")
        settings.append({"id": name, "mode": item["mode"], "policy_path": str(policy_path),
            "policy": policy, "bundle_path": str(path_from_setting(item["bundle"], path.parent)),
            "nlp_path": None if nlp is None else str(nlp), "init_delay_s": float(delay)})
    return path, settings


def assert_requested_values(actual, requested, label):
    """允许策略省略默认字段，但显式请求的每个值必须由实际二进制报告。"""
    if isinstance(requested, dict):
        if not isinstance(actual, dict):
            raise RuntimeError(f"missing resolved object: {label}")
        for key, expected in requested.items():
            if key not in actual:
                raise RuntimeError(f"missing resolved value: {label}.{key}")
            assert_requested_values(actual[key], expected, label + "." + key)
    elif actual != requested:
        raise RuntimeError(f"binary did not consume {label}: {actual!r} != {requested!r}")


def verify_runtime(runtime, setting, *, socket, cwd, family, seed, trials):
    if not isinstance(runtime, dict) or not isinstance(runtime.get("values"), dict):
        raise RuntimeError("binary did not report a resolved runtime configuration")
    values = runtime["values"]
    expected = {"control_variant": "baseline", "steer_rate_mode": "both_unlimited",
        "base_mode": setting["mode"], "trials": trials, "swing_mode": "qp", "seed": seed,
        "serve_family": family, "fly_mode": "xb" if family == "gs" else "v2",
        "init_delay_s": setting["init_delay_s"], "time_windows_path": setting["policy_path"],
        "predictor_socket_path": str(socket), "measured_replay_path": None}
    assert_requested_values(values, expected, "values")
    assert_requested_values(values.get("time_windows"), setting["policy"], "values.time_windows")
    actual_library = Path(values.get("nmpc6_nlp_path", ""))
    actual_library = (actual_library if actual_library.is_absolute() else cwd / actual_library).resolve()
    expected_library = Path(setting["nlp_path"]) if setting["nlp_path"] else cwd / "third_party/nmpc6/nlp.so"
    if actual_library != expected_library.resolve():
        raise RuntimeError("binary did not consume the frozen solver library")
    if runtime.get("sources", {}).get("time_windows") != "file:" + setting["policy_path"]:
        raise RuntimeError("binary did not report consumption of the requested external policy file")
    return values["time_windows"]


def parse_group_result(log, result):
    matches = re.findall(r"命中 (\d+)/(\d+)\(.*?平均击球点偏差 (.*?)m", log)
    if len(matches) != 1:
        raise RuntimeError("control must report exactly one complete trial summary")
    hits, denominator, _ = matches[0]
    if int(denominator) != result["trials"] or not 0 <= int(hits) <= int(denominator):
        raise RuntimeError("control summary trial denominator differs from the fixed request")
    result = parse_summary(log, result)
    if not 0 <= result["feasible"] <= result["trials"] or result["hits"] > result["feasible"]:
        raise RuntimeError("invalid feasible-domain counts")
    result.update(denominator=result["trials"], hit_rate=result["hits"] / result["trials"],
        feasible_denominator=result["feasible"],
        feasible_hit_rate=result["hits"] / result["feasible"] if result["feasible"] else None,
        interpretation=INTERPRETATION)
    return result


class SyntheticConfirmation:
    def __init__(self, args):
        self.args = args
        self.team = Path(args.team).resolve(strict=True)
        self.cwd = self.team / "sckd_sim"
        self.prediction = Path(args.prediction).resolve(strict=True)
        self.source = prediction_source_path(self.prediction, args.prediction_source)
        self.binary = Path(args.binary).resolve(strict=True)
        self.out = Path(args.out).resolve()
        self.timeout = positive_timeout(args.timeout)
        self.startup_timeout = positive_timeout(args.startup_timeout)
        if args.trials < 1:
            raise ValueError("--trials must be positive")
        self.families = args.families.split(",")
        self.seeds = [int(seed) for seed in args.seeds.split(",")]
        if not self.families or set(self.families) - {"fixed", "rand", "gs"} or len(set(self.families)) != len(self.families):
            raise ValueError("--families must be unique fixed/rand/gs entries")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(not -(2**31) <= seed < 2**31 for seed in self.seeds):
            raise ValueError("--seeds must be unique 32-bit signed integers")
        self.settings_path, self.settings = load_settings(args.settings)
        self.out.mkdir(parents=True, exist_ok=True)
        common = file_resources({"runner": Path(__file__), "runner_common": Path(__file__).with_name("pipeline_experiment.py"),
            "predictor_adapter": Path(__file__).with_name("evaluate_predictor_control.py"), "settings": self.settings_path})
        by_mode = {mode: team_resources(self.team, mode, binary=self.binary) for mode in {s["mode"] for s in self.settings}}
        self.preflight = {}
        for setting in self.settings:
            bundle = inspect_predictor_bundle(self.prediction, model_bundle=setting["bundle_path"], prediction_source=self.source)
            if bundle.get("settings", {}).get("frame", {}).get("mode") != "synthetic_world":
                raise ValueError("synthetic confirmation requires a synthetic_world bundle; no implicit frame replacement")
            loaded_nlp = setting["nlp_path"] or self.cwd / "third_party/nmpc6/nlp.so"
            resources = {**common, **by_mode[setting["mode"]],
                **file_resources({"time_policy": setting["policy_path"], "loaded_nlp": loaded_nlp}),
                **predictor_resources(self.prediction, model_bundle=setting["bundle_path"],
                    resolved_bundle=bundle, prediction_source=self.source)}
            self.preflight[setting["id"]] = {"setting": setting, "predictor_resolved_manifest": bundle,
                "resources": resources, "resources_fingerprint": content_identity(resources)}
        self.frozen = {"schema": SCHEMA, "selection_protocol": "caller-supplied fixed finalists; no synthetic ranking or reselection",
            "requested_settings": self.settings, "families": self.families, "seeds": self.seeds,
            "trials_per_group": args.trials, "expected_groups": len(self.settings) * len(self.families) * len(self.seeds),
            "expected_balls": len(self.settings) * len(self.families) * len(self.seeds) * args.trials,
            "execution": "sequential; fresh prediction service/socket per setting-family-seed group",
            "interpretation": INTERPRETATION, "preflight": self.preflight}
        # 首球前冻结全部设置/资源身份；结果不会改变设置顺序、参数或候选集合。
        write_once(self.out / "frozen_settings.json", self.frozen)

    def run_group(self, setting, family, seed):
        dest = self.out / "runs" / setting["id"] / family / str(seed)
        sock = socket_path(self.team, "synthetic-confirm")
        env = clean_environment({"M2_CONTROL_VARIANT": "baseline", "M2_STEER_RATE_MODE": "both_unlimited",
            "M2_TIME_WINDOWS_PATH": setting["policy_path"], "BALL_PREDICTOR_SOCKET": str(sock)})
        # 旧独立验收实验的EXP_*会改变门槛；本次只改变显式时间策略。
        env = {key: value for key, value in env.items() if not key.startswith("EXP_")}
        if setting["nlp_path"]:
            env["M2_NMPC6_NLP"] = setting["nlp_path"]
        env = prediction_environment(env, self.source)
        command = [str(self.binary), "models/generated/x4_tennis_demo.mjcf", str(self.args.trials), "qp",
            repr(setting["init_delay_s"]), str(seed), setting["mode"], family, "xb" if family == "gs" else "v2"]
        service = predictor_service_command(self.prediction, None, dest,
            model_bundle=setting["bundle_path"], prediction_source=self.source)
        preflight = self.preflight[setting["id"]]
        bundle, resources = preflight["predictor_resolved_manifest"], preflight["resources"]
        parameters = {"setting_id": setting["id"], "mode": setting["mode"], "family": family, "seed": seed,
            "trials": self.args.trials, "controller_variant": "baseline", "steer_rate_mode": "both_unlimited",
            "swing": "qp", "init_delay_s": setting["init_delay_s"], "policy": setting["policy"],
            "policy_path": setting["policy_path"], "nlp_path": resources["loaded_nlp"]["path"],
            "prediction_project": str(self.prediction), "prediction_source": str(self.source),
            "predictor_bundle_fingerprint": bundle["fingerprint"], "predictor_resolved_manifest": bundle,
            "timeout_s": self.timeout, "startup_timeout_s": self.startup_timeout,
            "settings_fixed_before_synthetic_runs": True, "interpretation": INTERPRETATION}
        spec = ExperimentSpec("time_windows_synthetic_confirmation", tuple(command), str(self.cwd), parameters,
            recorded_environment(env, sock), resources, tuple(service))
        with RunAttempt(dest, spec, completion="summary.json", retry_incomplete=self.args.retry_incomplete) as attempt:
            if attempt.existing is not None:
                return attempt.existing
            started = time.monotonic()
            actual_service = [str(sock) if token == "{socket}" else token for token in service]
            with socket_service(actual_service, socket=sock, cwd=self.source, env=env,
                    log=dest / "service.log", startup_timeout=self.startup_timeout, stop_timeout=5):
                verify_predictor_manifest(dest / "predictor_resolved_manifest.json", bundle)
                process = run_logged(command, cwd=self.cwd, env=env, log=dest / "control.log", timeout=self.timeout)
            if process.returncode not in (0, 1):
                raise RuntimeError(f"control exited with {process.returncode}; see {dest / 'control.log'}")
            log = (dest / "control.log").read_text()
            runtime = resolved_runtime_config(log)
            effective_policy = verify_runtime(runtime, setting, socket=sock, cwd=self.cwd, family=family,
                seed=seed, trials=self.args.trials)
            write_json(dest / "resolved_runtime_config.json", runtime, exclusive=True)
            result = parse_group_result(log, {"schema": SCHEMA, "setting_id": setting["id"],
                "mode": setting["mode"], "family": family, "seed": seed, "trials": self.args.trials,
                "returncode": process.returncode, "wall_s": time.monotonic() - started, "command": command,
                "binary": str(self.binary), "binary_sha256": resources["binary"]["sha256"],
                "policy_sha256": resources["time_policy"]["sha256"], "effective_policy": effective_policy,
                "nlp_sha256": resources["loaded_nlp"]["sha256"], "predictor_bundle_fingerprint": bundle["fingerprint"],
                "resources_fingerprint": preflight["resources_fingerprint"], "resolved_runtime_config": runtime})
            # rc1表示阈值未过，含0命中；只要分母/配置/服务完整便保留为有效负结果。
            return attempt.complete(result)

    def summarize(self, records, failures):
        expected = self.frozen["expected_groups"]
        complete = not failures and len(records) == expected
        settings = []
        for setting in self.settings:
            selected = [r for r in records if r["setting_id"] == setting["id"]]
            denominator = len(self.families) * len(self.seeds) * self.args.trials
            observed = sum(r["trials"] for r in selected)
            hits, feasible = sum(r["hits"] for r in selected), sum(r["feasible"] for r in selected)
            settings.append({"setting_id": setting["id"], "mode": setting["mode"], "expected_trials": denominator,
                "completed_trials": observed, "hits": hits, "feasible": feasible,
                "hit_rate": hits / denominator if observed == denominator else None,
                "feasible_hit_rate": hits / feasible if observed == denominator and feasible else None})
        return {"schema": SCHEMA, "complete": complete, "expected_groups": expected,
            "completed_groups": len(records), "expected_balls": self.frozen["expected_balls"],
            "completed_balls": sum(r["trials"] for r in records), "settings": settings,
            "groups": records, "technical_failures": failures, "selection_changed": False,
            "interpretation": INTERPRETATION}

    def run(self):
        records, failures = [], []
        for setting in self.settings:
            for family in self.families:
                for seed in self.seeds:
                    try:
                        result = self.run_group(setting, family, seed)
                        records.append(result)
                        print(setting["id"], family, seed, f"hits={result['hits']}/{result['trials']}",
                              f"returncode={result['returncode']}", flush=True)
                    except Exception as error:
                        failures.append({"setting_id": setting["id"], "family": family, "seed": seed,
                            "type": type(error).__name__, "message": str(error)})
                        print(setting["id"], family, seed, "technical failure:", str(error), flush=True)
                    write_json(self.out / "summary.json", self.summarize(records, failures))
        return self.summarize(records, failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--team", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-source", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", default="fixed,rand,gs")
    parser.add_argument("--seeds", default="2026,31")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--startup-timeout", type=float, default=10)
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args()
    summary = SyntheticConfirmation(args).run()
    print("complete=" + str(summary["complete"]), "balls=" + str(summary["completed_balls"]), flush=True)
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
