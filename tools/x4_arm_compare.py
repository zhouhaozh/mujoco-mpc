#!/usr/bin/env python3
"""Ubuntu统一入口：MuJoCo几何对照 + SDK本机sim/远端hardware控制。

硬件认证和CAN准备在启动子进程前通过正常SSH终端完成，不保存密码。
GUI主循环才可发heartbeat；读线程只排空遥测，不延续控制租约。
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "X4_ARM_COMPARE/1"
DEFAULT_MODEL = ROOT / "sckd_sim/models/generated/x4_tennis_demo.mjcf"
DEFAULT_MAPPING = ROOT / "tools/x4_arm_test/sdk_mujoco_display_mapping.json"


class Transport:
    def __init__(self, command, viewer, log, messages=None):
        self.viewer, self.log = viewer, log
        self.messages = messages
        self.phase = "connecting"
        self.last_seq = -1
        self.last_received = time.monotonic()
        self.failure = ""
        self.closed = False
        self.last_action = 0.0
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, bufsize=0)
        os.set_blocking(self.process.stdin.fileno(), False)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for raw in self.process.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                # PTY可能使用CRLF；前缀必须在行首，普通SDK日志不当作状态。
                if line.startswith(("X4_STATE ", "X4_UI_READY ")):
                    prefix, payload = line.split(" ", 1)
                    value = json.loads(payload)
                    if value.get("schema") != SCHEMA:
                        raise ValueError("unexpected state schema")
                    if prefix == "X4_UI_READY":
                        phase = value["phase"]
                        if phase not in ("awaiting_start_home", "startup_settling", "initialized"):
                            raise ValueError("unknown control phase")
                        self.last_received = time.monotonic()  # 首帧有界等待从初始化完成起计时。
                        self.phase = phase
                        print("\n控制端：" + phase, flush=True)
                        if phase == "awaiting_start_home":
                            print("请阅读上方home与启动说明；现场确认后在此终端输入 START-HOME。", flush=True)
                    else:
                        seq = value.get("seq")
                        if type(seq) is not int or seq <= self.last_seq:
                            raise ValueError("state sequence must increase")
                        self.last_seq = seq
                        if not self.viewer.update(value):
                            raise ValueError("viewer rejected invalid state; ending lease")
                        self.last_received = time.monotonic()
                        self.log.write(json.dumps({"received_monotonic": time.monotonic(),
                                                   "state": value}, allow_nan=False) + "\n")
                elif line:
                    # 认证在创建Transport前完成；保存控制端提示以追溯选轴/循环拒绝。
                    if self.messages is not None:
                        self.messages.write(json.dumps({"received_monotonic": time.monotonic(),
                                                        "message": line}, ensure_ascii=False) + "\n")
                    print(line, flush=True)
        except Exception as error:
            self.failure = str(error)
        finally:
            self.closed = True
            if not self.failure:
                self.failure = "control process closed"

    def send(self, command):
        # 先检查整条输入，再支持终端数字选轴；不把多条命令改成可执行队列。
        if len(command.encode()) > 1023 or "\n" in command or "\r" in command:
            raise ValueError("one command per input, at most 1024 bytes")
        command = command.strip()
        if command in ("1", "2", "3", "4", "5"):
            command = "select " + command
        elif command in ("select1", "select2", "select3", "select4", "select5"):
            command = "select " + command[-1]
        if self.closed or self.process.poll() is not None:
            raise RuntimeError(self.failure or "control process ended")
        now = time.monotonic()
        if command == "START-HOME":
            if self.phase != "awaiting_start_home":
                raise ValueError("START-HOME is only accepted at the on-site confirmation prompt")
        elif command not in ("quit", "stop", "heartbeat") and self.phase != "initialized":
            raise ValueError("controller initialization pending")
        if command not in ("quit", "stop", "heartbeat", "START-HOME", "status", "help"):
            blocked = self.viewer.blocked_reason()
            if blocked:
                raise ValueError("动作被锁定：" + blocked)
        if command not in ("quit", "stop", "pause", "heartbeat", "START-HOME"):
            if now - self.last_action < .12:
                print("输入过快，已丢弃本条；动作不会排队", flush=True)
                return False
            self.last_action = now
        payload = (command + "\n").encode()
        if len(payload) > 1024 or "\n" in command or "\r" in command:
            raise ValueError("one command per input, at most 1024 bytes")
        # 小于PIPE_BUF的单次非阻塞写；阻塞就终止会话，不排队旧动作。
        try:
            if os.write(self.process.stdin.fileno(), payload) != len(payload):
                raise RuntimeError("incomplete command write")
        except (OSError, RuntimeError) as error:
            self.failure = "command channel failed: " + str(error)
            self.closed = True
            raise
        return True

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    os.write(self.process.stdin.fileno(), b"quit\n")
                except (OSError, RuntimeError):
                    pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.terminate()  # SSH断开触发远端HUP；租约也停止刷新。
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=3)
        finally:
            self.reader.join(timeout=2)
            self.process.stdin.close()
            self.process.stdout.close()


def send_terminal(transport, command):
    """输入验证拒绝可纠正；通信失败仍交给主循环结束会话。"""
    try:
        return transport.send(command)
    except ValueError as error:
        print("输入被拒绝：" + str(error), flush=True)
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("sim", "hardware"), default="sim")
    p.add_argument("--step-deg", type=int, choices=(1, 10, 30), default=30,
                   help="single-step and loop amplitude in degrees; direct position target")
    p.add_argument("--test-mode", choices=("direction", "tracking"), default="direction",
                   help="direction ignores small angular residuals and compares motion from actual initial feedback")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--mapping", type=Path,
                   help="display-only mapping; standard model defaults to the SDK joint convention preset")
    p.add_argument("--binary", type=Path, default=ROOT / "sckd_sim/build-arm-axis/x4_arm_axis_test")
    p.add_argument("--config", type=Path, default=Path("/home/ubuntu/Zhaohui/x4-sdk/config/robot_8w4s.ini"))
    p.add_argument("--host", default="x4-t1@172.20.0.38")
    p.add_argument("--remote-dir", default="/home/x4-t1/Zhaohui/x4-arm-compare")
    p.add_argument("--known-hosts", type=Path, default=ROOT / "tools/x4_arm_test/robot_known_hosts")
    p.add_argument("--output", type=Path)
    p.add_argument("--smoke-seconds", type=float, help="sim only: render then quit without motion")
    p.add_argument("--snapshot", type=Path, help="sim smoke only: static verification image")
    args = p.parse_args()
    if args.mapping is None and args.model.resolve() == DEFAULT_MODEL.resolve():
        args.mapping = DEFAULT_MAPPING
    if args.backend == "hardware" and (args.smoke_seconds is not None or args.snapshot):
        p.error("smoke/snapshot options are sim only")
    if args.backend == "hardware" and not sys.stdin.isatty():
        p.error("hardware must be launched from an Ubuntu desktop terminal")
    if args.snapshot and args.smoke_seconds is None:
        p.error("--snapshot requires --smoke-seconds")
    output = args.output or ROOT / "sckd_sim/artifacts/arm_compare" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print("遥测记录：" + str(output), flush=True)
    print("仅显示坐标映射：" + (str(args.mapping) if args.mapping else "自定义模型：同名恒等候选，未标定"), flush=True)
    from x4_arm_compare_viewer import CompareViewer
    transport = None
    viewer = None
    ssh_base = None
    with tempfile.TemporaryDirectory(prefix="x4-compare-") as temp, \
            (output / "telemetry.jsonl").open("w", buffering=1) as log, \
            (output / "controller_messages.jsonl").open("w", buffering=1) as messages:
        try:
            # 先验证模型和显示可用；失败时根本不拉起远端控制程序。
            def send(command):
                if transport is None:
                    raise RuntimeError("control transport not ready")
                return transport.send(command)
            viewer = CompareViewer(str(args.model), send, args.backend,
                                   str(args.mapping) if args.mapping else None)
            if args.backend == "hardware":
                if not args.known_hosts.is_file():
                    raise ValueError("pinned robot known_hosts file missing")
                ssh_base = ["ssh", "-F", "/dev/null", "-o", "StrictHostKeyChecking=yes",
                            "-o", "UserKnownHostsFile=" + str(args.known_hosts),
                            "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=2",
                            "-o", "ServerAliveCountMax=2", "-o", "ControlMaster=auto",
                            "-o", "ControlPersist=60", "-S", str(Path(temp) / "ssh"), args.host]
                remote = shlex.quote(args.remote_dir + "/bin/x4_arm_remote")
                print("先认证并仅准备机械臂CAN。真正使能须随后手动输入 START-HOME。", flush=True)
                # 继承真实终端，SSH及sudo密码由系统读取，不经过遥测/日志。
                step_option = " --step-deg " + str(args.step_deg)
                step_option += " --test-mode " + args.test_mode
                subprocess.run(ssh_base[:-1] + ["-t", args.host, remote + " prepare" + step_option], check=True)
                command = ssh_base[:-1] + ["-o", "BatchMode=yes", "-tt", args.host, remote + " run" + step_option]
            else:
                command = [str(args.binary), "--config", str(args.config), "--backend", "sim",
                           "--step-deg", str(args.step_deg), "--test-mode", args.test_mode,
                           "--ui", "--output", str(output / "controller")]
            transport = Transport(command, viewer, log, messages)
            print(f"本次幅度 {args.step_deg}°，直接下发位置目标，速度由SDK/电机控制；位置稳定超时15秒。", flush=True)
            print("测试模式：" + args.test_mode + "；方向模式左图=初始反馈+命令增量，原始目标/反馈仍保留在表格。", flush=True)
            print("窗口：1–5选轴，+/-单步，L往返3轮，P完成当前往返后暂停，空格停止，Q退出。", flush=True)
            print("终端输入后按Enter：1–5或select N选轴；loop N（1..10）、status、record N same|opposite 说明。", flush=True)
            pending = b""
            heartbeat_at = 0.0
            started = time.monotonic()
            stdin_active = args.smoke_seconds is None
            while viewer.is_open():
                if transport.closed or transport.process.poll() is not None:
                    print("控制通道已结束：" + transport.failure, flush=True)
                    break
                if not viewer.render():
                    break
                now = time.monotonic()
                # 主线程渲染/输入卡住就停止租约刷新，不用后台心跳掩盖故障。
                if transport.phase in ("startup_settling", "initialized") and now - transport.last_received > .5:
                    raise RuntimeError("超过500ms未收到新遥测；结束会话")
                if transport.phase in ("startup_settling", "initialized") and now - heartbeat_at >= .10:
                    transport.send("heartbeat")
                    heartbeat_at = now
                if stdin_active and select.select([sys.stdin], [], [], 0)[0]:
                    raw = os.read(sys.stdin.fileno(), 2048)
                    if not raw:
                        break
                    pending += raw
                    if len(pending) > 4096:
                        raise ValueError("terminal input too long")
                    commands = []
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        command = line.decode().strip()
                        if command:
                            commands.append(command)
                    urgent = next((c for c in ("quit", "stop", "pause") if c in commands), None)
                    if urgent:
                        send_terminal(transport, urgent)
                    elif len(commands) == 1:
                        send_terminal(transport, commands[0])
                    elif commands:
                        print("拒绝整批输入：每次只输入一条命令", flush=True)
                if args.smoke_seconds is not None and now - started >= args.smoke_seconds:
                    if args.snapshot:
                        viewer.save_snapshot(str(args.snapshot))
                    break
                time.sleep(.003)
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            finally:
                try:
                    if transport is not None:
                        transport.close()
                finally:
                    if ssh_base:
                        subprocess.run(ssh_base[:-1] + ["-O", "exit", args.host],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    return 0 if transport and transport.process.returncode == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print("会话结束：" + str(error), file=sys.stderr)
        sys.exit(1)
