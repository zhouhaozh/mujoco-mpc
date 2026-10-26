#!/usr/bin/env python3
"""Ubuntu真实桌面首帧回归：占位→测试终端持焦→合成反馈自动展示。

不启动SDK，不连接设备；按键回调只写内存。只创建自己的两个测试终端。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "sckd_sim/models/generated/x4_tennis_demo.mjcf")
    parser.add_argument("--mapping", type=Path, default=ROOT / "tools/x4_arm_test/sdk_mujoco_display_mapping.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("仅在Ubuntu执行")
    args.output.mkdir(parents=True, exist_ok=False)
    for line in subprocess.check_output(["systemctl", "--user", "show-environment"], text=True).splitlines():
        name, _, value = line.partition("=")
        if name in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_SESSION_TYPE", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
            os.environ[name] = value
    sys.path.insert(0, str(ROOT / "tools"))
    from x4_arm_compare_viewer import CompareViewer
    import glfw
    import numpy as np

    commands, focus_calls, show_calls, first_received = [], [], [], []
    main_thread = threading.get_ident()
    reader_errors = []
    stop_reader = threading.Event()
    terminals = []
    reader = None
    viewer = None
    report = {"hardware_started": False, "sdk_started": False,
              "source": "synthetic telemetry; in-memory command spy only", "passed": False}
    original_focus, original_show = glfw.focus_window, glfw.show_window

    def focus(window):
        focus_calls.append({"at": time.monotonic(), "thread": threading.get_ident()})
        return original_focus(window)

    def show(window):
        show_calls.append({"at": time.monotonic(), "thread": threading.get_ident()})
        return original_show(window)

    def open_terminal(title):
        process = subprocess.Popen(["gnome-terminal", "--wait", "--title=" + title,
                                    "--", "bash", "-c", "sleep 8"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        terminals.append(process)

    def wait_for_background():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not viewer.render():
                raise RuntimeError("test viewer closed unexpectedly")
            if not glfw.get_window_attrib(viewer.window, glfw.FOCUSED):
                return
            time.sleep(.003)
        raise RuntimeError("test terminal did not take focus")

    try:
        viewer = CompareViewer(str(args.model), commands.append, "OFFLINE SIM FIRST POSE TEST",
                               str(args.mapping), headless=False)
        glfw.focus_window, glfw.show_window = focus, show
        viewer.update({"schema": "X4_ARM_COMPARE/1", "seq": 0, "backend": "sim",
                       "initialized": False, "q_ref": [None] * 5, "q_fb": [None] * 5,
                       "reason": "awaiting_synthetic_pose"})
        open_terminal("X4 autofocus regression: initial placeholder")
        wait_for_background()
        report["terminal_owned_focus_before_first_pose"] = True
        reference = np.radians([-1.4, 35., -22.5, -88., 0.]).tolist()
        feedback = np.radians([-1.664, 13.338, -27.066, -88.442, -2.725]).tolist()

        def receive():
            sequence = 1
            try:
                while not stop_reader.is_set():
                    packet = {"schema": "X4_ARM_COMPARE/1", "seq": sequence, "backend": "sim",
                              "initialized": True, "moving": False, "settled": True, "fault": False,
                              "selected_joint": 1, "loop_active": False, "loop_remaining": 0,
                              "session_stopped": False, "reason": "synthetic_holding_no_sdk",
                              "q_ref": reference, "q_fb": feedback}
                    received_at = time.monotonic()
                    if not viewer.update(packet):
                        raise RuntimeError("synthetic packet rejected")
                    if not first_received:
                        first_received.append(received_at)
                    sequence += 1
                    stop_reader.wait(.02)
            except Exception as error:
                reader_errors.append(str(error))

        reader = threading.Thread(target=receive, name="synthetic-pose-reader")
        reader.start()
        deadline = time.monotonic() + 5
        first_presented = None
        while time.monotonic() < deadline:
            viewer.render()
            if first_received and viewer._last_draw_at is not None and viewer._last_draw_at >= first_received[0]:
                first_presented = time.monotonic()
                break
            time.sleep(.003)
        if first_presented is None:
            raise RuntimeError("first valid pose never presented without user action")
        report["first_pose_presentation_latency_s"] = first_presented - first_received[0]
        report["first_pose_focused"] = bool(glfw.get_window_attrib(viewer.window, glfw.FOCUSED))
        report["reference_rendered_deg"] = np.degrees(viewer.reference.qpos[viewer._qpos_adr]).tolist()
        report["feedback_rendered_deg"] = np.degrees(viewer.feedback.qpos[viewer._qpos_adr]).tolist()
        viewer.save_snapshot(str(args.output / "first_pose.png"))
        open_terminal("X4 autofocus regression: later background")
        wait_for_background()
        start = time.monotonic()
        regained_focus = False
        while time.monotonic() - start < 2:
            viewer.render()
            regained_focus |= bool(glfw.get_window_attrib(viewer.window, glfw.FOCUSED))
            time.sleep(.003)
        report["later_background_refocused"] = regained_focus
        report["commands_before_close"] = list(commands)
        report["focus_requests"] = focus_calls
        report["show_requests"] = show_calls
        report["reader_errors"] = reader_errors
        report["passed"] = (report["first_pose_presentation_latency_s"] < .5 and
                            report["first_pose_focused"] and not regained_focus and not commands and
                            len(focus_calls) == len(show_calls) == 1 and not reader_errors and
                            all(call["thread"] == main_thread for call in focus_calls + show_calls))
        return 0 if report["passed"] else 1
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return 1
    finally:
        stop_reader.set()
        if reader is not None:
            reader.join(timeout=2)
        glfw.focus_window, glfw.show_window = original_focus, original_show
        if viewer is not None:
            viewer.close()
        for terminal in terminals:
            try:
                terminal.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminal.terminate()
        report["command_spy_after_close"] = commands
        (args.output / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
