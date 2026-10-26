#!/usr/bin/env python3
"""Ubuntu 纯内存 SDK 命令接管回归；所有 CAN 入口拦截，socket 系统调用禁止。"""
import argparse
import json
import pathlib
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=pathlib.Path,
                        default=pathlib.Path("/home/ubuntu/Zhaohui/x4-sdk"))
    parser.add_argument("--backend-source", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expect-missing", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("仅 Ubuntu x86_64 执行，与权威 SDK ABI 一致")
    root = pathlib.Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.backend_source or root / "sckd_sim/cpp/runtime/x4_arm_test_backend.cpp"
    executable = args.output / "probe"
    command = ["c++", "-std=c++17", "-O0", "-Wall", "-Wextra",
               "-I" + str(args.sdk / "include"), "-I/usr/include/eigen3",
               "-I" + str(root / "sckd_sim/cpp"),
               str(root / "tests/probe_x4_arm_command_gate.cc"), str(source),
               "-L" + str(args.sdk / "lib"), "-Wl,-rpath," + str(args.sdk / "lib"),
               "-Wl,--export-dynamic", "-larm", "-lhal", "-lcore", "-ldl", "-pthread",
               "-o", str(executable)]
    subprocess.run(command, check=True)
    run = [str(executable), str(args.config or args.sdk / "config/robot_8w4s.ini")]
    if args.expect_missing:
        run.append("--expect-missing")
    result = subprocess.run(run, capture_output=True, text=True, timeout=10)
    (args.output / "stdout.txt").write_text(result.stdout)
    (args.output / "stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise SystemExit(result.stderr or f"probe failed: exit {result.returncode}")
    data = json.loads(result.stdout)
    (args.output / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
