#!/usr/bin/env python3
"""Ubuntu CLI拒绝路径检查；所有用例均禁止进入硬件启动函数。"""
import argparse
import json
import pathlib
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    binary, config = args.binary.resolve(), args.config.resolve()
    results = []
    with tempfile.TemporaryDirectory(prefix="x4-arm-cli-") as temporary:
        work = pathlib.Path(temporary)

        def run(name, options, expected, contains=None):
            result = subprocess.run(
                [str(binary), *options], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=10,
            )
            assert result.returncode == expected, (name, result.stdout, result.stderr)
            if contains:
                assert contains in result.stdout + result.stderr, (name, result.stdout, result.stderr)
            results.append({"name": name, "exit_code": result.returncode, "passed": True})
            return result

        run("help_no_config", ["--help"], 0)
        run("default_inspect", ["--config", str(config)], 0)
        inspected = run("explicit_inspect", ["--config", str(config), "--backend", "inspect"], 0)
        data = json.loads(inspected.stdout)
        assert data["backend"] == "inspect" and not data["simulation_profile_applied"]
        assert data["can_if"] == ["can4", "can5", "can6"]
        assert data["step_deg"] == 30 and data["excursion_deg"] == 45
        assert data["peak_speed_deg_s"] is None
        assert data["command_profile"] == "direct_position"
        assert data["test_mode"] == "direction"
        assert data["angular_deviation_policy"] == "diagnostic_only_with_fixed_excursion"
        assert data["snapshot_max_attempts"] == 3
        assert data["feedback_excursion_deg"] == 45 and data["excursion_deg"] == 45
        tracking = run("tracking_mode", ["--config", str(config), "--test-mode", "tracking"], 0)
        assert json.loads(tracking.stdout)["angular_deviation_policy"] == "tracking_and_other_axis_fault"
        assert json.loads(tracking.stdout)["feedback_excursion_deg"] == 45
        run("invalid_test_mode", ["--config", str(config), "--test-mode", "invalid"], 2, "direction or tracking")
        legacy = run("one_degree_option", ["--config", str(config), "--step-deg", "1"], 0)
        assert json.loads(legacy.stdout)["step_deg"] == 1
        for step in (10,30):
            option=run(str(step)+"_degree_option", ["--config", str(config), "--step-deg", str(step)], 0)
            assert json.loads(option.stdout)["step_deg"]==step
        for value in ("0", "-10", "11", "nan", "10junk"):
            run("invalid_step_"+value, ["--config", str(config), "--step-deg", value], 2, "must be 1, 10 or 30")
        destination = work / "must_not_be_created"
        run("unknown_backend", ["--config", str(config), "--backend", "invalid"], 2, "unknown backend")
        run("hardware_demo_rejected", ["--config", str(config), "--backend", "hardware", "--demo",
                                       "--output", str(destination)], 2, "only with sim")
        # DEVNULL + captured stdout确保无TTY；Parse阶段拒绝，尚未构造后端。
        run("hardware_noninteractive_rejected", ["--config", str(config), "--backend", "hardware",
                                                "--output", str(destination)], 2, "interactive terminal")
        assert not destination.exists()
        run("existing_output_rejected", ["--config", str(config), "--backend", "sim",
                                         "--output", str(work)], 2, "output exists")
        original = config.read_text()
        mutations = [
            ("missing_sign", r"(?m)^joint_sign\s*=.*$", "", "missing [Arm].joint_sign"),
            ("invalid_sign", r"(?m)^joint_sign\s*=.*$", "joint_sign = -1,0,1,1,1", "only +1/-1"),
            ("invalid_home", r"(?m)^home_deg\s*=.*$", "home_deg = nan,35,-22.5,-88,0", "finite float"),
            ("position_range", r"(?m)^home_deg\s*=.*$", "home_deg = 100000,35,-22.5,-88,0", "position_encoding_range"),
            ("invalid_gain", r"(?m)^kp\s*=.*$", "kp = 0,70,10,8,5", "nonpositive_hold_or_exit_gain"),
            ("invalid_can", r"(?m)^can_if\s*= can4.*$", "can_if = can0,can1,can2", "can4,can5,can6"),
            ("slow_period", r"(?m)^control_period_ms\s*= 4.*$", "control_period_ms = 100", "within 1..10 ms"),
        ]
        for name, pattern, replacement, error in mutations:
            content, count = re.subn(pattern, replacement, original)
            assert count > 0, name
            path = work / (name + ".ini")
            path.write_text(content)
            run(name, ["--config", str(path)], 2, error)
    args.output.write_text(json.dumps({"tests": results, "passed": len(results)}, ensure_ascii=False, indent=2))
    print(f"{len(results)} CLI checks passed; no hardware backend started")


if __name__ == "__main__":
    main()
