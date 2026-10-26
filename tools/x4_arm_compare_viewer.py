#!/usr/bin/env python3
"""X4 机械臂目标/反馈几何对照；仅显示，不启动 SDK 或发送任何设备帧。

CompareViewer.update() 接收 q_ref/q_fb（各五项，SDK模型角，rad），以及
backend（hardware/sim）、initialized，以及selected_joint（1..5）、moving、
settled、fault、loop_active、loop_remaining、paused、session_stopped、reason。
test_mode=direction时还接收固定command_origin和feedback_anchor：左图显示
反馈起点+命令增量，右图显示原始反馈；原始角度表不变，未初始化时两图均为反馈。
省略test_mode或tracking保留直接显示q_ref的行为；本轮不允许切换模式或重设锚点。
接收线程可调用update；GLFW的
render/is_open/close必须在创建窗口的线程调用。send_command应为不阻塞回调。

两侧都只通过mj_forward重建几何，绝不是独立动力学预测。SDK和团队模型的
关节坐标尚未标定；默认同名identity只作候选。mapping JSON定义：
  {"permutation":[1,2,3,4,5], "sign":[1,1,1,1,1], "offset_rad":[0,0,0,0,0]}
即 MuJoCo Joint(i+1) = sign[i] * SDK[permutation[i]-1] + offset_rad[i]。
该映射只影响显示，不改SDK配置、控制指令或机械臂测试顺序。

Ubuntu无可见窗口截图（需要可用GLFW显示；可用xvfb-run创建虚拟显示）：
  xvfb-run -a /path/to/python x4_arm_compare_viewer.py --model MODEL \
    --state-json STATE.json --snapshot /tmp/arm_compare.png
这个独立入口只显示输入JSON，按键仅打印命令；不启动任何控制程序。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable

# Ubuntu Wayland桌面提供DISPLAY时使用XWayland。原生Wayland的缓冲呈现
# 可能在后台等待合成器，且无法可靠报告最小化/请求恢复；不改变桌面会话。
# Python GLFW在首次import时选择库，因此必须先设置。显式用户配置仍优先。
if sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE") == "wayland" and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import glfw
import mujoco
import numpy as np


def _five_numbers(value, field: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{field} must contain numbers, not strings/bools/null")
    result = np.asarray(value, dtype=float)
    if (result.shape != (5,) or not np.all(np.isfinite(result)) or
            np.any(np.abs(result) > np.finfo(float).max / 360.)):
        raise ValueError(f"{field} must contain five finite numbers")
    return result.copy()


def _backend_kind(label: str) -> str | None:
    words = str(label).lower().replace("|", " ").replace(":", " ").split()
    if any(word in ("hardware", "real") for word in words):
        return "hardware"
    if any(word in ("sim", "simulation") for word in words):
        return "sim"
    return None


def _flag(state: dict, name: str, default: bool = False) -> bool:
    value = state.get(name, default)
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean or 0/1")


class CompareViewer:
    """两个同步相机视口；全部真实控制由调用方的send_command负责。"""

    STALE_SECONDS = 0.5
    DRAW_INTERVAL_SECONDS = 1.0 / 30.0

    def __init__(self, model_path: str, send_command: Callable[[str], bool | None],
                 backend_label: str, mapping_path: str | None = None, *,
                 headless: bool = False, width: int = 1440, height: int = 900):
        self.send_command = send_command
        self.backend_label = str(backend_label)
        self._expected_backend = _backend_kind(self.backend_label)
        self._observed_backend: str | None = None
        self._observed_test_mode: str | None = None
        self._direction_anchors: tuple[np.ndarray, np.ndarray] | None = None
        self._backend_mismatch = False
        self._offline = "offline" in self.backend_label.lower()
        self._lock = threading.RLock()
        self._state: dict = {}
        self._q_ref = np.zeros(5)
        self._q_fb = np.zeros(5)
        self._received_at: float | None = None
        self._invalid = ""
        self._notice = "Waiting for SDK state; controls locked"
        self._transport_failed = False
        self._quit_sent = False
        self._closing = False
        self._session_stopped = False
        self._last_seq: int | None = None
        self._closed = False
        self._glfw_initialized = False
        self._headless = bool(headless)
        self._last_draw_at: float | None = None
        self._first_pose_focus_pending = False
        self.window = None
        self.context = None
        self._mouse_last = (0.0, 0.0)
        self._permutation = np.arange(5)
        self._sign = np.ones(5)
        self._offset = np.zeros(5)
        self.mapping_label = "IDENTITY CANDIDATE - NOT CALIBRATED"
        if mapping_path is not None:
            raw = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("mapping JSON must be an object")
            expected_digest = raw.get("model_sha256", raw.get("source_model_sha256"))
            if expected_digest is not None:
                model_digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
                if expected_digest != model_digest:
                    raise ValueError("display mapping model hash mismatch; verify the model and mapping together")
            permutation = raw.get("permutation")
            if (not isinstance(permutation, list) or len(permutation) != 5 or
                    any(type(x) is not int for x in permutation) or
                    sorted(permutation) != [1, 2, 3, 4, 5]):
                raise ValueError("mapping permutation must contain 1..5 exactly once")
            self._permutation = np.asarray(permutation, dtype=int) - 1
            self._sign = _five_numbers(raw.get("sign"), "mapping sign")
            self._offset = _five_numbers(raw.get("offset_rad"), "mapping offset_rad")
            if not np.all(np.isin(self._sign, [-1, 1])):
                raise ValueError("mapping signs must be +1 or -1")
            self.mapping_label = raw.get("label", "USER DISPLAY MAPPING - NOT PHYSICALLY VERIFIED")
            if not isinstance(self.mapping_label, str):
                raise ValueError("mapping label must be text")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        # 仅改善本地几何显示：提高头灯，避免黑色连杆在两视口内难以辨认。
        self.model.vis.headlight.ambient[:] = [.35, .35, .35]
        self.model.vis.headlight.diffuse[:] = [.85, .85, .85]
        self.reference = mujoco.MjData(self.model)
        self.feedback = mujoco.MjData(self.model)
        self._qpos_adr = []
        for number in range(1, 6):
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"Joint{number}")
            if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"model must contain scalar hinge Joint{number}")
            self._qpos_adr.append(int(self.model.jnt_qposadr[joint]))
        # 整个qpos0为固定基准，只覆盖五轴；不积分底盘，不运行mj_step。
        self._baseline = self.model.qpos0.copy()
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 1.0]
        self.camera.distance = 3.2
        self.camera.azimuth = 135
        self.camera.elevation = -18
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        self._scenes = [mujoco.MjvScene(self.model, maxgeom=10000) for _ in range(2)]
        try:
            if not glfw.init():
                raise RuntimeError("GLFW initialization failed; set DISPLAY or run under xvfb-run")
            self._glfw_initialized = True
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE if headless else glfw.TRUE)
            glfw.window_hint(glfw.SAMPLES, 4)
            self.window = glfw.create_window(width, height, "X4 arm | reference / feedback", None, None)
            if self.window is None:
                raise RuntimeError("cannot create GLFW window")
            glfw.set_window_size_limits(self.window, 1024, 720, glfw.DONT_CARE, glfw.DONT_CARE)
            glfw.make_context_current(self.window)
            # 交换缓冲不承担调度；失焦/遮挡时垂直同步可能阻塞主事件循环。
            glfw.swap_interval(0)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
            glfw.set_scroll_callback(self.window, self._on_scroll)
            glfw.set_window_close_callback(self.window, self._on_window_close)
            if not self._headless:
                # 先提交占位首帧，避免窗口系统在首次呈现前尚未映射/聚焦。
                # 此时仅显示等待反馈提示，调用方尚未启动任何SDK控制端。
                first_width, first_height = self._draw()
                if first_width > 0 and first_height > 0:
                    glfw.swap_buffers(self.window)
        except Exception:
            self.close()
            raise

    def update(self, state: dict) -> bool:
        """返回本包是否有效且被接受；重复/无效/关闭时False。

        合法null初始化目标属于生命周期消息，返回True但不刷新姿态年龄。
        收包时钟始终为本机单调钟，绝不相减两台机器的单调钟。
        """
        received_at = time.monotonic()
        try:
            if not isinstance(state, dict):
                raise ValueError("state must be an object")
            stopped = _flag(state, "session_stopped")
            # 终止包可能尚无有效目标角；先锁存终止，不因姿态解析失败漏掉它。
            with self._lock:
                self._session_stopped = self._session_stopped or stopped
                if self._closed or self._closing:
                    return False
            normalized = dict(state)
            for name in ("initialized", "moving", "settled", "fault", "loop_active", "paused"):
                normalized[name] = _flag(state, name)
            normalized["session_stopped"] = self._session_stopped
            test_mode = state.get("test_mode", "tracking")
            if test_mode not in ("direction", "tracking"):
                raise ValueError("test_mode must be direction or tracking")
            normalized["test_mode"] = test_mode
            with self._lock:
                if self._observed_test_mode is not None and test_mode != self._observed_test_mode:
                    raise ValueError("test_mode differs from previous source")
            backend = state.get("backend")
            if backend not in ("hardware", "sim"):
                raise ValueError("state backend must explicitly be hardware or sim")
            with self._lock:
                if (self._expected_backend is not None and backend != self._expected_backend) or (
                        self._observed_backend is not None and backend != self._observed_backend):
                    self._backend_mismatch = True
                    raise ValueError("backend differs from configured/previous source")
                self._observed_backend = backend
            sequence = state.get("seq")
            if sequence is not None:
                if type(sequence) is not int or sequence < 0:
                    raise ValueError("seq must be a nonnegative integer")
                with self._lock:
                    if self._last_seq is not None and sequence < self._last_seq:
                        raise ValueError("out-of-order state sequence")
                    if sequence == self._last_seq:
                        return False  # 重复包不刷新接收年龄，也不解锁之前的异常。
            for name, minimum, maximum in (("selected_joint", 1, 5), ("loop_remaining", 0, None)):
                if name in state:
                    value = state[name]
                    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
                        raise ValueError(f"invalid {name}")
            if "reason" in state and not isinstance(state["reason"], str):
                raise ValueError("reason must be text")
            if (not normalized["initialized"] and isinstance(state.get("q_ref"), list)
                    and state["q_ref"] == [None] * 5):
                # 控制端尚未提交第一条命令时主动发布null目标，属于正常等待。
                # 它只能更新生命周期，不能刷新有效姿态年龄或允许操作。
                with self._lock:
                    self._state = normalized
                    self._observed_test_mode = test_mode
                    self._invalid = ""
                    if sequence is not None:
                        self._last_seq = sequence
                return True
            q_ref = _five_numbers(state.get("q_ref"), "q_ref")
            q_fb = _five_numbers(state.get("q_fb"), "q_fb")
            anchors = None
            if test_mode == "direction" and normalized["initialized"]:
                command_origin = _five_numbers(state.get("command_origin"), "command_origin")
                feedback_anchor = _five_numbers(state.get("feedback_anchor"), "feedback_anchor")
                anchors = (command_origin, feedback_anchor)
                with self._lock:
                    if self._direction_anchors is not None and any(
                            not np.array_equal(old, new)
                            for old, new in zip(self._direction_anchors, anchors)):
                        raise ValueError("direction display anchors changed during session")
                normalized["command_origin"] = command_origin.tolist()
                normalized["feedback_anchor"] = feedback_anchor.tolist()
            # 映射/误差也必须可表示；有效输入相加仍可能溢出。
            with np.errstate(over="ignore", invalid="ignore"):
                displayed_reference = self._reference_angles(normalized, q_ref, q_fb)
                for angles in (q_ref, q_fb, displayed_reference):
                    if not np.all(np.isfinite(np.degrees(angles))):
                        raise ValueError("display reference overflow")
                    if not np.all(np.isfinite(self._sign * angles[self._permutation] + self._offset)):
                        raise ValueError("display mapping overflow")
                if not np.all(np.isfinite(np.degrees(q_fb - q_ref))):
                    raise ValueError("angle difference overflow")
        except (TypeError, ValueError, OverflowError) as error:
            with self._lock:
                self._invalid = str(error)
                # 无效包不替换最近有效状态；独立锁存的终止标志始终保留。
                self._state["session_stopped"] = self._session_stopped
                if self._session_stopped and isinstance(state, dict) and isinstance(state.get("reason"), str):
                    self._state["reason"] = state["reason"]
            return False
        with self._lock:
            if self._closed or self._closing:
                return False
            self._state = normalized
            self._observed_test_mode = test_mode
            if anchors is not None and self._direction_anchors is None:
                self._direction_anchors = tuple(value.copy() for value in anchors)
            self._q_ref = q_ref
            self._q_fb = q_fb
            if self._received_at is None and not self._headless:
                # update可能由读线程调用：只请求主线程聚焦，不在此调用GLFW。
                self._first_pose_focus_pending = True
            self._received_at = received_at
            if sequence is not None:
                self._last_seq = sequence
            self._invalid = ""
            if self._notice == "Waiting for SDK state; controls locked":
                self._notice = "Choose J1..J5, then +/- or L; controller checks readiness"
        return True

    def _snapshot(self):
        with self._lock:
            return (dict(self._state), self._q_ref.copy(), self._q_fb.copy(),
                    self._received_at, self._invalid)

    @staticmethod
    def _reference_angles(state: dict, q_ref: np.ndarray, q_fb: np.ndarray) -> np.ndarray:
        """只改变左侧几何：方向模式比较本轮反馈起点上的命令增量。"""
        if state.get("test_mode") != "direction":
            return q_ref
        if not state.get("initialized", False):
            return q_fb
        return (np.asarray(state["feedback_anchor"], dtype=float) +
                (q_ref - np.asarray(state["command_origin"], dtype=float)))

    def _blocked_reason(self) -> str:
        state, _, _, received, invalid = self._snapshot()
        if self._closed or self._closing:
            return "VIEWER CLOSING"
        if self._session_stopped or state.get("session_stopped", False):
            return "SESSION STOPPED - NEW ACTIONS LOCKED"
        if self._backend_mismatch:
            return "BACKEND MISMATCH - ACTIONS LOCKED"
        if self._offline:
            return "OFFLINE DATA - NO CONTROL CONNECTION"
        if self._transport_failed:
            return "COMMAND CHANNEL FAILED"
        if invalid:
            return "INVALID STATE"
        if received is None:
            return "WAITING FOR STATE"
        if self._observed_backend is None:
            return "BACKEND SOURCE UNCONFIRMED"
        if time.monotonic() - received > self.STALE_SECONDS:
            return "STALE STATE - ACTIONS LOCKED"
        if state.get("fault", False):
            return "CONTROLLER FAULT - ACTIONS LOCKED"
        if not state.get("initialized", False):
            return "INITIALIZATION PENDING"
        return ""

    def blocked_reason(self) -> str:
        """公开同一动作门供终端/GUI共用；非空表示只允许stop/quit。"""
        return self._blocked_reason()

    def _backend_display(self) -> str:
        with self._lock:
            source = {"hardware": "REAL HARDWARE", "sim": "SDK SIMULATION"}.get(
                self._observed_backend, "SOURCE UNCONFIRMED")
        if self._backend_mismatch:
            return f"BACKEND MISMATCH | configured: {self.backend_label}"
        if self._offline:
            return f"OFFLINE DATA | recorded source: {source}"
        return f"{source} | {self.backend_label}"

    def _dispatch(self, command: str) -> None:
        # 软件停止/退出始终可尝试；界面锁只禁止新的动作，不能充当硬件急停。
        blocked = self._blocked_reason()
        if command not in ("stop", "quit") and blocked:
            self._notice = f"Blocked: {blocked}"
            return
        if command == "quit" and self._quit_sent:
            return
        if self._closed:
            return
        if command == "stop":
            # 本轮停止是终止操作；在控制端确认包到达前也禁止继续排队动作。
            self._session_stopped = True
        if command == "quit":
            self._closing = True
        try:
            if self.send_command(command) is False:
                self._notice = f"Not sent: {command} (input rate limit; retry separately)"
                return
            if command == "quit":
                self._quit_sent = True
            self._notice = f"Sent: {command} (controller decides acceptance)"
        except ValueError as error:
            # 输入/状态拒绝不代表传输断开；允许用户修正后继续操作。
            self._notice = f"Rejected: {error}"
        except Exception as error:
            self._transport_failed = True
            self._notice = f"Command send failed: {error}"

    def _on_key(self, window, key, scancode, action, mods) -> None:
        del scancode, mods
        if action != glfw.PRESS:
            return
        if glfw.KEY_1 <= key <= glfw.KEY_5:
            self._dispatch(f"select {key - glfw.KEY_1 + 1}")
        elif glfw.KEY_KP_1 <= key <= glfw.KEY_KP_5:
            self._dispatch(f"select {key - glfw.KEY_KP_1 + 1}")
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self._dispatch("+")
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self._dispatch("-")
        elif key == glfw.KEY_L:
            self._dispatch("loop 3")
        elif key == glfw.KEY_P:
            self._dispatch("pause")
        elif key == glfw.KEY_SPACE:
            self._dispatch("stop")
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            self._dispatch("quit")
            glfw.set_window_should_close(window, True)

    def _on_window_close(self, window) -> None:
        del window
        self._dispatch("quit")

    def _on_mouse_button(self, window, button, action, mods) -> None:
        del button, action, mods
        self._mouse_last = glfw.get_cursor_pos(window)

    def _on_mouse_move(self, window, x, y) -> None:
        dx, dy = x - self._mouse_last[0], y - self._mouse_last[1]
        self._mouse_last = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or middle or right):
            return
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif middle:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        _, height = glfw.get_window_size(window)
        mujoco.mjv_moveCamera(self.model, action, dx / max(1, height), dy / max(1, height), self.camera)

    def _on_scroll(self, window, x, y) -> None:
        del window, x
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05 * y, self.camera)

    def _set_pose(self, data, sdk_angles: np.ndarray) -> None:
        data.qpos[:] = self._baseline
        data.qvel[:] = 0
        mapped = self._sign * sdk_angles[self._permutation] + self._offset
        data.qpos[self._qpos_adr] = mapped
        mujoco.mj_forward(self.model, data)

    def _text(self, rect, text: str, *, large: bool = False) -> None:
        # MuJoCo内建字体只保证ASCII；界面措辞与源码中文说明保持一致。
        ascii_text = str(text).encode("ascii", "replace").decode("ascii")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG if large else mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, rect, ascii_text, "", self.context)

    def _draw(self, framebuffer_size: tuple[int, int] | None = None) -> tuple[int, int]:
        if self._closed:
            raise RuntimeError("viewer is closed")
        glfw.make_context_current(self.window)
        width, height = framebuffer_size or glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return width, height
        state, q_ref, q_fb, received, invalid = self._snapshot()
        age = math.inf if received is None else max(0., time.monotonic() - received)
        blocked = self._blocked_reason()
        header = min(94, max(40, height // 9))
        # footer左右均有九行；100%字体配>=220px，避免J3..J5被底边裁掉。
        footer = min(260, max(220, height // 3))
        view_height = max(1, height - header - footer)
        half = width // 2
        direction_view = state.get("test_mode") == "direction"
        displayed_reference = self._reference_angles(state, q_ref, q_fb)
        self._set_pose(self.reference, displayed_reference)
        self._set_pose(self.feedback, q_fb)
        full = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_rectangle(full, .035, .045, .06, 1.)
        reference_title = ("DIRECTION EXPECTATION: feedback anchor + command delta"
                           if direction_view and state.get("initialized", False) else
                           "STARTUP SDK FEEDBACK: direction anchor pending" if direction_view else
                           "REFERENCE q_ref")
        for index, (data, title) in enumerate(((self.reference, reference_title),
                                              (self.feedback, "SDK FEEDBACK q_fb"))):
            viewport = mujoco.MjrRect(index * half, footer, half if index == 0 else width - half, view_height)
            scene = self._scenes[index]
            if received is not None:
                mujoco.mjv_updateScene(self.model, data, self.option, None, self.camera,
                                       mujoco.mjtCatBit.mjCAT_ALL, scene)
                mujoco.mjr_render(viewport, scene, self.context)
            else:
                # 尚无数据时不画零位机器人，避免将失焦后保留的首帧误认为实物姿态。
                mujoco.mjr_rectangle(viewport, .055, .065, .08, 1.)
            suffix = ""
            if received is None:
                suffix = " | WAITING FOR SDK POSE - WILL SHOW AUTOMATICALLY"
            elif invalid or age > self.STALE_SECONDS or state.get("session_stopped", False):
                suffix = " | LAST VALID POSE (NOT CURRENT)"
            elif not state.get("initialized", False):
                suffix = " | LIVE STARTUP FEEDBACK (CONTROLS LOCKED)"
            self._text(viewport, title + suffix + "\nGEOMETRY RECONSTRUCTION (mj_forward)")
        top = mujoco.MjrRect(0, height - header, width, header)
        if blocked:
            mujoco.mjr_rectangle(top, .55, .045, .045, 1.)
        else:
            mujoco.mjr_rectangle(top, .035, .19, .24, 1.)
        age_text = "no packet" if received is None else f"receive age {age:.3f}s"
        banner = blocked or "LOCAL STREAM FRESH | not hardware acceptance"
        mapping_text = ("STATIC OFFSET IGNORED FOR DIRECTION ONLY | " + self.mapping_label
                        if direction_view else self.mapping_label)
        self._text(top, f"{self._backend_display()} | {banner} | {age_text}\n"
                       f"{mapping_text}\n"
                       "Both panels use the same geometry; no independent dynamics prediction.")
        selected = state.get("selected_joint", "?")
        loop = (f"active={bool(state.get('loop_active', False))} "
                f"remaining={state.get('loop_remaining', '?')}")
        left_lines = [f"Selected J{selected} | moving={bool(state.get('moving', False))} "
                      f"settled={bool(state.get('settled', False))}",
                      f"Loop {loop}",
                      "RAW SDK MODEL ANGLES (deg), before display mapping",
                      "axis      reference      feedback      fb-ref"]
        for j in range(5):
            if received is None:
                left_lines.append(f"J{j + 1}             --             --            --")
            else:
                left_lines.append(f"J{j + 1}       {math.degrees(q_ref[j]):+9.3f}      "
                                  f"{math.degrees(q_fb[j]):+9.3f}      "
                                  f"{math.degrees(q_fb[j] - q_ref[j]):+8.3f}")
        self._text(mujoco.MjrRect(0, 0, half, footer), "\n".join(left_lines))
        controls = ("1..5: select axis   +/-: one step\n"
                    "L: 3 out-and-back cycles   P: pause after pair\n"
                    "SPACE: stop/hold   Q/ESC: quit\n"
                    "Mouse: left rotate / middle pan / wheel zoom\n"
                    "Base fixed at model qpos0; cameras synchronized\n"
                    "Age is local receipt age; not motor feedback age\n"
                    "Stop/quit are software actions, not emergency stop\n"
                    f"Status: {str(state.get('reason', 'waiting'))[:100]}\n"
                    f"{('Invalid state: ' + invalid) if invalid else self._notice}")
        self._text(mujoco.MjrRect(half, 0, width - half, footer), controls)
        return width, height

    def render(self) -> bool:
        """处理事件，并按30Hz上限绘制可见窗口；返回是否仍打开。

        首次有效姿态到达后由本线程显示/聚焦窗口一次，无需先点动。
        失焦但仍可见时持续绘制，便于终端操作与反馈并排观察；只有最小化、
        隐藏或零尺寸暂停呈现，不再抢回焦点。
        headless调用每次显式绘制，便于离线姿态验证；截图也不受节流影响。
        此处不续租，任何实际事件处理/绘图阻塞仍会中断调用方的心跳。
        """
        if not self.is_open():
            if not self._closed:
                self._dispatch("quit")
            return False
        try:
            with self._lock:
                focus_first_pose = self._first_pose_focus_pending and not self._headless
                if focus_first_pose:
                    self._first_pose_focus_pending = False
            if focus_first_pose:
                # START-HOME在终端输入，首个有效包到达时终端往往仍持焦。
                # 仅此一次请求展示；后续失焦不再抢焦点，也不在读线程续租。
                glfw.show_window(self.window)
                glfw.focus_window(self.window)
                self._last_draw_at = None
            # 必须先处理失焦、最小化和关闭事件，不能先等待一次GL呈现。
            glfw.poll_events()
            if not self.is_open():
                if not self._closed:
                    self._dispatch("quit")
                return False
            width, height = glfw.get_framebuffer_size(self.window)
            inactive = not self._headless and (
                glfw.get_window_attrib(self.window, glfw.ICONIFIED) or
                not glfw.get_window_attrib(self.window, glfw.VISIBLE))
            if inactive or width <= 0 or height <= 0:
                # 恢复后立刻使用最新遥测重绘，不沿用暂停前的绘制期限。
                self._last_draw_at = None
                return True
            now = time.monotonic()
            if (not self._headless and self._last_draw_at is not None and
                    now - self._last_draw_at < self.DRAW_INTERVAL_SECONDS):
                return True
            self._draw(framebuffer_size=(width, height))
            glfw.swap_buffers(self.window)
            self._last_draw_at = now
            return self.is_open()
        except Exception:
            # 图形出错不能留下无人操作的控制会话；退出请求仍非硬件急停。
            self.close()
            raise

    def is_open(self) -> bool:
        return bool(not self._closed and self.window is not None and
                    not glfw.window_should_close(self.window))

    def save_snapshot(self, path: str) -> None:
        """显式离屏FBO截图，避免隐藏窗口swap后默认back buffer不可回读。"""
        from PIL import Image
        if self._closed:
            raise RuntimeError("viewer is closed")
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            raise RuntimeError("cannot capture a minimized window")
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        # 隐藏GLFW窗口的window back buffer受窗口系统/交换影响；原实现可
        # 绘制成功却读出全黑。为截图明确分配、绑定、绘制、回读同一离屏FBO，
        # 不依赖窗口是否可见或先前swap次数。完成后恢复实时窗口绘制目标。
        mujoco.mjr_resizeOffscreen(width, height, self.context)
        try:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            if self.context.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
                raise RuntimeError("offscreen framebuffer unavailable")
            self._draw(framebuffer_size=(width, height))
            mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, width, height), self.context)
        finally:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        Image.fromarray(np.flipud(pixels)).save(path)

    def close(self) -> None:
        """最佳努力请求控制会话quit，再释放显示资源；多次调用不会重复发送。"""
        if self._closed:
            return
        # 发送失败也继续释放；控制端仍需自己的心跳超时/退出保护。
        if self.window is not None:
            self._dispatch("quit")
        self._closed = True
        try:
            if self.window is not None:
                glfw.make_context_current(self.window)
            if self.context is not None:
                self.context.free()
        finally:
            self.context = None
            # Python MjvScene由绑定的析构器释放，没有公开free()方法。
            self._scenes = []
            try:
                if self.window is not None:
                    glfw.destroy_window(self.window)
            finally:
                self.window = None
                if self._glfw_initialized:
                    glfw.terminate()
                    self._glfw_initialized = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend-label", default="OFFLINE INPUT | NO CONTROL CONNECTION")
    parser.add_argument("--mapping-json")
    parser.add_argument("--state-json", help="read one state packet; freshness expires normally")
    parser.add_argument("--snapshot", help="save PNG in a hidden GLFW window and exit")
    args = parser.parse_args()
    viewer = CompareViewer(args.model, lambda command: print(f"DISPLAY ONLY: {command}", flush=True),
                           args.backend_label, args.mapping_json, headless=bool(args.snapshot))
    try:
        if args.state_json:
            viewer.update(json.loads(Path(args.state_json).read_text(encoding="utf-8")))
        if args.snapshot:
            viewer.save_snapshot(args.snapshot)
        else:
            while viewer.render():
                time.sleep(.01)
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
