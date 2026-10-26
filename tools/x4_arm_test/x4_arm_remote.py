#!/usr/bin/env python3
"""隔离控制包入口；inspect完全不配置CAN，prepare只配置can4..6，run须终端确认。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import termios
from datetime import datetime

ROOT = Path(__file__).resolve().parent
CHANNELS = ("can4", "can5", "can6")


def can_info(name):
    return json.loads(subprocess.check_output(["ip", "-details", "-j", "link", "show", name]))[0]


def configured(value):
    data = value.get("linkinfo", {}).get("info_data", {})
    modes = data.get("ctrlmode", [])
    return ("UP" in value.get("flags", []) and
            data.get("bittiming", {}).get("bitrate") == 1000000 and
            data.get("data_bittiming", {}).get("bitrate") == 2000000 and
            "FD" in modes and "ONE-SHOT" in modes and
            data.get("state") == "ERROR-ACTIVE")


def check_processes():
    # 只是已知旧入口的辅助检查；SocketCAN本身不提供跨工具独占保证。
    old = {"robotd", "court_run", "arm_test", "arm_pose", "arm_joint_tune", "x4_arm_axis_test"}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            name = (path / "comm").read_text().strip()
            exe = (path / "cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
            if name in old or Path(exe).name in old:
                raise RuntimeError("已有控制进程，先由操作者处理：" + path.name + " " + name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inspect", "prepare", "run"), default="inspect", nargs="?")
    p.add_argument("--step-deg", type=int, choices=(1, 10, 30), default=30)
    p.add_argument("--test-mode", choices=("direction", "tracking"), default="direction")
    args = p.parse_args()
    binary = ROOT / "bin/x4_arm_axis_test"
    config = ROOT / "config/robot_8w4s.ini"
    env = dict(os.environ, LD_LIBRARY_PATH=str(ROOT / "lib"))
    inspect_command = [str(binary), "--config", str(config), "--step-deg", str(args.step_deg),
                       "--test-mode", args.test_mode]
    if args.action == "inspect":
        subprocess.run(inspect_command, env=env, check=True)
        for name in CHANNELS:
            print(json.dumps(can_info(name)))
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("prepare/run须由交互终端启动")
    with (ROOT / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_processes()
        states = {name: can_info(name) for name in CHANNELS}
        if args.action == "prepare":
            subprocess.run(inspect_command, env=env, check=True)
            for name, state in states.items():
                if "UP" in state.get("flags", []) and not configured(state):
                    raise RuntimeError(name + "已UP但参数或状态不符；拒绝改动正在使用的通道")
            down = [name for name, state in states.items() if not configured(state)]
            if down:
                print("准备can4–6，CAN FD 1M/2M one-shot；尚不发送电机目标。", flush=True)
                subprocess.run(["sudo", "-v"], check=True)
                for name in down:
                    subprocess.run(["sudo", "-n", "ip", "link", "set", name, "type", "can", "bitrate", "1000000",
                                    "dbitrate", "2000000", "fd", "on", "one-shot", "on"], check=True)
                    subprocess.run(["sudo", "-n", "ip", "link", "set", name, "up"], check=True)
            if not all(configured(can_info(name)) for name in CHANNELS):
                raise RuntimeError("CAN配置后状态检查未通过")
            print("CAN已准备。下一步仍须现场输入START-HOME才会使能。", flush=True)
            return
        if not all(configured(value) for value in states.values()):
            raise RuntimeError("CAN未准备或状态异常；没有启动SDK")
        # 防止PTY回显心跳/动作。真正确认由C++读取，不把START-HOME写在脚本中。
        attrs = termios.tcgetattr(sys.stdin.fileno())
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, attrs)
        output = ROOT / "records" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.parent.mkdir(exist_ok=True)
        os.set_inheritable(lock.fileno(), True)
        os.execve(str(binary), [str(binary), "--config", str(config), "--backend", "hardware",
                               "--step-deg", str(args.step_deg), "--test-mode", args.test_mode,
                               "--ui", "--output", str(output)], env)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("拒绝/失败：" + str(error), file=sys.stderr)
        sys.exit(1)
