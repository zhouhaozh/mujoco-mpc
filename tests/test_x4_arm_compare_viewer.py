#!/usr/bin/env python3
"""Ubuntu真实GLFW/MuJoCo查看器验收；回调仅记录字符串，绝不启动SDK。

DISPLAY=:1 /home/ubuntu/Zhaohui/sim/.venv/bin/python \
  tests/test_x4_arm_compare_viewer.py --model MODEL.mjcf --output NEW_DIRECTORY

所有窗口headless=True，但仍需要可用DISPLAY/OpenGL。验证真实模型姿态、
动作入口的拒绝行为和资源退出；输出comparison.png与verification.json。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import x4_arm_compare_viewer as viewer_module
import glfw
import mujoco
import numpy as np
from PIL import Image


MODEL: Path | None = None
OUTPUT: Path | None = None


def state_packet(sequence: int = 1, **changes) -> dict:
    result = {
        "schema": "X4_ARM_COMPARE/1", "seq": sequence, "backend": "sim",
        "initialized": True, "moving": False, "settled": True, "fault": False,
        "selected_joint": 1, "loop_active": False, "loop_remaining": 0,
        "paused": False, "session_stopped": False, "reason": "holding",
        "q_ref": [-.0244346, .610865, -.392699, -1.53589, 0.0],
        "q_fb": [-.0244346, .610865, -.392699, -1.53589, 0.0],
    }
    result.update(changes)
    return result


class CompareViewerChecks(unittest.TestCase):
    def setUp(self):
        self.commands: list[str] = []
        self.viewer = viewer_module.CompareViewer(
            str(MODEL), self.commands.append, "SIM | VIEWER TEST", headless=True)

    def tearDown(self):
        self.viewer.close()

    def replace_viewer(self, *, backend_label="SIM | VIEWER TEST", mapping=None,
                       callback=None):
        self.viewer.close()
        self.commands.clear()
        self.viewer = viewer_module.CompareViewer(
            str(MODEL), callback or self.commands.append, backend_label,
            mapping_path=str(mapping) if mapping else None, headless=True)

    def assert_actions_locked(self):
        before = list(self.commands)
        for command in ("+", "-", "select 2", "loop 3", "pause"):
            self.viewer._dispatch(command)
        self.assertEqual(self.commands, before)
        self.assertTrue(self.viewer._blocked_reason())

    def test_initial_placeholder_and_null_target_stay_locked(self):
        self.assert_actions_locked()
        with mock.patch.object(mujoco, "mjv_updateScene") as draw_robot:
            self.assertTrue(self.viewer.render())
        draw_robot.assert_not_called()  # 没有反馈时不能画一个貌似有效的零位机器人。
        self.assertTrue(self.viewer.update(state_packet(initialized=False, q_ref=[None] * 5,
                                                        q_fb=[None] * 5, reason="initializing")))
        self.assert_actions_locked()
        self.assertIsNone(self.viewer._received_at)
        self.assertFalse(self.viewer._invalid)
        self.assertTrue(self.viewer.render())
        self.assertTrue(self.viewer.update(state_packet(2)))
        self.viewer._dispatch("+")
        self.assertEqual(self.commands, ["+"])

    def test_fresh_startup_pose_is_live_but_controls_stay_locked(self):
        packet=state_packet(initialized=False, settled=False, reason="startup_settling")
        self.assertTrue(self.viewer.update(packet))
        self.assert_actions_locked()
        with mock.patch.object(self.viewer, "_text") as text:
            self.viewer.render()
            labels="\n".join(str(call.args[1]) for call in text.call_args_list)
        self.assertIn("LIVE STARTUP FEEDBACK", labels)
        self.assertNotIn("NOT CURRENT", labels)
        np.testing.assert_allclose(self.viewer.feedback.qpos[self.viewer._qpos_adr], packet["q_fb"])

    def test_direction_view_anchors_initial_pose_and_preserves_raw_sdk_angles(self):
        origin = state_packet()["q_ref"]
        anchor = np.radians([-1.64, 13.34, -27.07, -88.44, -2.77]).tolist()
        packet = state_packet(test_mode="direction", command_origin=origin,
                              feedback_anchor=anchor, q_fb=anchor)
        self.assertTrue(self.viewer.update(packet))
        with mock.patch.object(self.viewer, "_text") as text:
            self.viewer.render()
            labels = "\n".join(str(call.args[1]) for call in text.call_args_list)
        np.testing.assert_allclose(self.viewer.reference.qpos[self.viewer._qpos_adr], anchor)
        np.testing.assert_allclose(self.viewer.feedback.qpos[self.viewer._qpos_adr], anchor)
        np.testing.assert_array_equal(self.viewer._q_ref, origin)
        np.testing.assert_array_equal(self.viewer._q_fb, anchor)
        self.assertEqual(self.viewer._state["q_ref"], origin)
        self.assertIn("DIRECTION EXPECTATION: feedback anchor + command delta", labels)
        self.assertIn("STATIC OFFSET IGNORED FOR DIRECTION ONLY", labels)
        self.assertIn("RAW SDK MODEL ANGLES", labels)
        self.assertIn("+35.000", labels)  # 原始目标表仍显示35°，没有被改写成13°。
        self.assertEqual(self.commands, [])

    def test_direction_view_uses_fixed_anchor_for_each_command_delta(self):
        origin = np.asarray(state_packet()["q_ref"])
        anchor = np.radians([-1.64, 13.34, -27.07, -88.44, -2.77])
        self.assertTrue(self.viewer.update(state_packet(test_mode="direction",
            command_origin=origin.tolist(), feedback_anchor=anchor.tolist(), q_fb=anchor.tolist())))
        target, feedback = origin.copy(), anchor.copy()
        target[1] += np.radians(10)
        feedback[1] += np.radians(8)
        self.assertTrue(self.viewer.update(state_packet(2, test_mode="direction",
            command_origin=origin.tolist(), feedback_anchor=anchor.tolist(),
            q_ref=target.tolist(), q_fb=feedback.tolist(), selected_joint=2, moving=True)))
        self.viewer.render()
        expected = anchor + target - origin
        np.testing.assert_allclose(self.viewer.reference.qpos[self.viewer._qpos_adr], expected)
        np.testing.assert_allclose(self.viewer.feedback.qpos[self.viewer._qpos_adr], feedback)
        target[1] = origin[1]
        target[2] += np.radians(10)
        self.assertTrue(self.viewer.update(state_packet(3, test_mode="direction",
            command_origin=origin.tolist(), feedback_anchor=anchor.tolist(),
            q_ref=target.tolist(), q_fb=feedback.tolist(), selected_joint=3)))
        self.viewer.render()
        np.testing.assert_allclose(self.viewer.reference.qpos[self.viewer._qpos_adr], anchor + target - origin)
        np.testing.assert_array_equal(self.viewer._direction_anchors[1], anchor)

    def test_direction_startup_without_anchor_shows_live_feedback_in_both_panels(self):
        feedback = np.radians([-1.64, -14., -36.6, -88.44, -2.77]).tolist()
        self.assertTrue(self.viewer.update(state_packet(test_mode="direction", initialized=False,
            settled=False, q_fb=feedback, reason="startup_settling")))
        self.assert_actions_locked()
        self.viewer.render()
        np.testing.assert_allclose(self.viewer.reference.qpos[self.viewer._qpos_adr], feedback)
        np.testing.assert_allclose(self.viewer.feedback.qpos[self.viewer._qpos_adr], feedback)
        self.assertIsNone(self.viewer._direction_anchors)

    def test_invalid_or_changed_direction_anchors_preserve_last_valid_packet(self):
        origin = state_packet()["q_ref"]
        anchor = np.radians([-1.64, 13.34, -27.07, -88.44, -2.77]).tolist()
        good = state_packet(test_mode="direction", command_origin=origin,
                            feedback_anchor=anchor, q_fb=anchor)
        mutations = [
            {"command_origin": None}, {"feedback_anchor": None},
            {"command_origin": [0.] * 4}, {"feedback_anchor": [False] * 5},
            {"feedback_anchor": [float("inf"), 0., 0., 0., 0.]},
            {"feedback_anchor": [1e307, 0., 0., 0., 0.]},
            {"command_origin": [0.] * 5},
            {"feedback_anchor": [0.] * 5},
            {"test_mode": "tracking"}, {"test_mode": "unknown"},
        ]
        self.assertTrue(self.viewer.update(good))
        received = self.viewer._received_at
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = dict(good, seq=2, **mutation)
                self.assertFalse(self.viewer.update(bad))
                self.assertEqual(self.viewer._received_at, received)
                np.testing.assert_array_equal(self.viewer._q_fb, anchor)
                np.testing.assert_array_equal(self.viewer._q_ref, origin)
        self.assertTrue(self.viewer.update(dict(good, seq=2)))
        self.assertEqual(self.viewer._blocked_reason(), "")

    def test_tracking_view_still_renders_raw_reference_with_static_error(self):
        packet = state_packet(test_mode="tracking", q_fb=[0.] * 5)
        self.assertTrue(self.viewer.update(packet))
        self.viewer.render()
        np.testing.assert_array_equal(self.viewer.reference.qpos[self.viewer._qpos_adr], packet["q_ref"])
        np.testing.assert_array_equal(self.viewer.feedback.qpos[self.viewer._qpos_adr], packet["q_fb"])

    def test_direction_view_applies_standard_sdk_mapping_after_anchoring(self):
        mapping = Path(__file__).resolve().parents[1] / "tools/x4_arm_test/sdk_mujoco_display_mapping.json"
        self.replace_viewer(mapping=mapping)
        origin = np.asarray(state_packet()["q_ref"])
        anchor = np.radians([-1.64, 13.34, -27.07, -88.44, -2.77])
        command = origin.copy()
        command[1] += np.radians(10)
        feedback = anchor.copy()
        feedback[1] += np.radians(8)
        packet = state_packet(test_mode="direction", command_origin=origin.tolist(),
            feedback_anchor=anchor.tolist(), q_ref=command.tolist(), q_fb=feedback.tolist())
        original = copy.deepcopy(packet)
        self.assertTrue(self.viewer.update(packet))
        self.viewer.render()
        sign = np.array([-1, -1, 1, 1, 1])
        offset = np.array([0, 0, 0, np.pi / 2, 0])
        np.testing.assert_allclose(self.viewer.reference.qpos[self.viewer._qpos_adr],
                                   sign * (anchor + command - origin) + offset)
        np.testing.assert_allclose(self.viewer.feedback.qpos[self.viewer._qpos_adr], sign * feedback + offset)
        self.assertEqual(packet, original)

    def test_terminal_packet_latches_even_without_pose(self):
        self.viewer.update(state_packet())
        self.viewer.update(state_packet(2, initialized=False, q_ref=[None] * 5,
                                        q_fb=[None] * 5, session_stopped=True,
                                        reason="session_finished"))
        self.assertTrue(self.viewer._session_stopped)
        self.assertIn("SESSION STOPPED", self.viewer._blocked_reason())
        self.assert_actions_locked()
        # 后续新包/重新声称initialized都不得复活已经终止的本轮会话。
        self.viewer.update(state_packet(3, session_stopped=False))
        self.assertTrue(self.viewer._session_stopped)
        self.assert_actions_locked()
        self.viewer._dispatch("stop")
        self.viewer._dispatch("quit")
        self.assertEqual(self.commands, ["stop", "quit"])

    def test_stop_request_locks_before_remote_ack(self):
        self.viewer.update(state_packet())
        self.viewer._dispatch("stop")
        self.assertEqual(self.commands, ["stop"])
        self.assert_actions_locked()
        self.viewer.update(state_packet(2))
        self.assert_actions_locked()

    def test_local_receive_age_and_duplicate_sequence(self):
        self.viewer.update(state_packet(t_monotonic=-123456789.0))
        received = self.viewer._received_at
        self.assertIsNotNone(received)
        with mock.patch.object(viewer_module.time, "monotonic", return_value=received + .2):
            self.assertEqual(self.viewer._blocked_reason(), "")
            self.assertFalse(self.viewer.update(state_packet(t_monotonic=10 ** 12)))
            self.assertEqual(self.viewer._received_at, received)
        with mock.patch.object(viewer_module.time, "monotonic", return_value=received + .501):
            self.assertIn("STALE", self.viewer._blocked_reason())
            self.assert_actions_locked()
            self.assertTrue(self.viewer.render())
            self.viewer._dispatch("stop")
            self.viewer._dispatch("quit")
        self.assertEqual(self.commands, ["stop", "quit"])

    def test_backend_labels_and_mismatch_latch(self):
        self.viewer.update(state_packet())
        self.assertIn("SDK SIMULATION", self.viewer._backend_display())
        self.viewer.update(state_packet(2, backend="hardware"))
        self.assertIn("BACKEND MISMATCH", self.viewer._backend_display())
        self.assert_actions_locked()
        self.viewer.update(state_packet(3))
        self.assert_actions_locked()
        # REAL只表示遥测标记；这里回调仍是list.append，没有设备控制连接。
        self.replace_viewer(backend_label="REAL HARDWARE | FAKE TELEMETRY TEST")
        self.viewer.update(state_packet(backend="hardware"))
        self.assertIn("REAL HARDWARE", self.viewer._backend_display())
        self.assertNotIn("SDK SIMULATION", self.viewer._backend_display())
        self.assertEqual(self.viewer._blocked_reason(), "")

    def test_invalid_packets_preserve_last_valid_pose_and_age(self):
        mutations = [
            ("nan", {"q_fb": [0., 0., float("nan"), 0., 0.]}),
            ("infinity", {"q_ref": [0., 0., 0., float("inf"), 0.]}),
            ("null", {"q_ref": [None] * 5}),
            ("missing_feedback", {"q_fb": None}),
            ("bad_boolean", {"initialized": "false"}),
            ("bad_fault_boolean", {"fault": "false"}),
            ("boolean_angles", {"q_ref": [False] * 5}),
            ("string_angles", {"q_fb": ["0"] * 5}),
            ("oversized_integer", {"q_ref": [10 ** 400, 0, 0, 0, 0]}),
            ("wrong_backend", {"backend": "unknown"}),
            ("wrong_joint", {"selected_joint": 6}),
        ]
        sequence = 1
        for name, mutation in mutations:
            with self.subTest(name=name):
                self.viewer.update(state_packet(sequence))
                self.assertEqual(self.viewer._blocked_reason(), "")
                reference = self.viewer._q_ref.copy()
                feedback = self.viewer._q_fb.copy()
                received = self.viewer._received_at
                invalid = state_packet(sequence + 1)
                invalid.update(mutation)
                self.assertFalse(self.viewer.update(invalid))
                self.assertTrue(self.viewer._invalid)
                self.assert_actions_locked()
                self.assertEqual(self.viewer.blocked_reason(), self.viewer._blocked_reason())
                self.assertEqual(self.viewer._received_at, received)
                np.testing.assert_array_equal(self.viewer._q_ref, reference)
                np.testing.assert_array_equal(self.viewer._q_fb, feedback)
                self.assertTrue(self.viewer.render())
                sequence += 2
        self.assertEqual(self.commands, [])

    def test_out_of_order_packet_does_not_refresh_age(self):
        self.viewer.update(state_packet(20))
        received = self.viewer._received_at
        self.viewer.update(state_packet(19))
        self.assertEqual(self.viewer._received_at, received)
        self.assertIn("out-of-order", self.viewer._invalid)
        self.assert_actions_locked()

    def test_display_mapping_only_changes_five_named_joints(self):
        mapping = {
            "permutation": [5, 3, 1, 4, 2],
            "sign": [-1, 1, -1, 1, -1],
            "offset_rad": [.01, -.02, .03, -.04, .05],
        }
        with tempfile.TemporaryDirectory(prefix="viewer-mapping-", dir=OUTPUT) as temporary:
            path = Path(temporary) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            self.replace_viewer(mapping=path)
            reference = [.10, -.20, .30, -.40, .50]
            feedback = [.11, -.22, .33, -.44, .55]
            packet = state_packet(q_ref=reference, q_fb=feedback)
            original = copy.deepcopy(packet)
            self.viewer.update(packet)
            self.assertTrue(self.viewer.render())
            for values, data in ((reference, self.viewer.reference), (feedback, self.viewer.feedback)):
                arm_addresses = []
                for index in range(5):
                    joint = mujoco.mj_name2id(self.viewer.model, mujoco.mjtObj.mjOBJ_JOINT,
                                              f"Joint{index + 1}")
                    address = int(self.viewer.model.jnt_qposadr[joint])
                    arm_addresses.append(address)
                    expected = mapping["sign"][index] * values[mapping["permutation"][index] - 1] + mapping["offset_rad"][index]
                    self.assertAlmostEqual(data.qpos[address], expected, places=12)
                other_addresses = [i for i in range(self.viewer.model.nq) if i not in arm_addresses]
                np.testing.assert_array_equal(data.qpos[other_addresses], self.viewer.model.qpos0[other_addresses])
                np.testing.assert_array_equal(data.qvel, np.zeros(self.viewer.model.nv))
            self.assertEqual(packet, original)
            self.assertIn("NOT PHYSICALLY VERIFIED", self.viewer.mapping_label)
            self.viewer._dispatch("select 2")
            self.assertEqual(self.commands, ["select 2"])
            self.assertEqual(json.loads(path.read_text()), mapping)

    def test_sdk_display_preset_keeps_raw_feedback_and_commands(self):
        mapping = Path(__file__).resolve().parents[1] / "tools/x4_arm_test/sdk_mujoco_display_mapping.json"
        self.replace_viewer(mapping=mapping)
        packet = state_packet(q_fb=np.radians([-1.6, 13.36, -27.066, -88.442, -2.725]).tolist())
        original = copy.deepcopy(packet)
        self.assertTrue(self.viewer.update(packet))
        self.assertTrue(self.viewer.render())
        sign = np.array([-1, -1, 1, 1, 1])
        offset = np.array([0, 0, 0, np.pi / 2, 0])
        for values, data in ((packet["q_ref"], self.viewer.reference),
                             (packet["q_fb"], self.viewer.feedback)):
            np.testing.assert_allclose(data.qpos[self.viewer._qpos_adr], sign * values + offset)
        self.assertEqual(packet, original)
        self.viewer._on_key(self.viewer.window, glfw.KEY_L, 0, glfw.PRESS, 0)
        self.assertEqual(self.commands, ["loop 3"])
        self.viewer.save_snapshot(str(OUTPUT / "sdk_display_preset.png"))

    def test_bad_mapping_rejected_before_creating_second_window(self):
        with tempfile.TemporaryDirectory(prefix="viewer-bad-mapping-", dir=OUTPUT) as temporary:
            path = Path(temporary) / "mapping.json"
            for mapping in (
                {"permutation": [1, 1, 3, 4, 5], "sign": [1] * 5, "offset_rad": [0] * 5},
                {"permutation": [1, 2, 3, 4, 5], "sign": [1, 1, 0, 1, 1], "offset_rad": [0] * 5},
                {"permutation": [1, 2, 3, 4, 5], "sign": [1] * 5, "offset_rad": [0] * 5,
                 "model_sha256": "0" * 64},
            ):
                path.write_text(json.dumps(mapping), encoding="utf-8")
                with self.assertRaises(ValueError):
                    viewer_module.CompareViewer(str(MODEL), self.commands.append, "SIM",
                                                str(path), headless=True)
        self.assertTrue(self.viewer.is_open())
        self.assertEqual(self.commands, [])

    def test_close_and_escape_send_quit_once(self):
        self.viewer.update(state_packet())
        self.viewer._on_key(self.viewer.window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
        self.viewer._on_window_close(self.viewer.window)
        self.viewer.close()
        self.viewer.close()
        self.assertEqual(self.commands, ["quit"])
        self.assertFalse(self.viewer.is_open())
        self.assertIsNone(self.viewer.window)
        self.assertIsNone(self.viewer.context)
        self.assertFalse(self.viewer.update(state_packet(10)))

    def test_number_row_and_keypad_select_all_five_axes(self):
        self.viewer.update(state_packet())
        for first_key in (glfw.KEY_1, glfw.KEY_KP_1):
            for index in range(5):
                self.viewer._on_key(self.viewer.window, first_key + index, 0, glfw.PRESS, 0)
                self.viewer._on_key(self.viewer.window, first_key + index, 0, glfw.REPEAT, 0)
                self.viewer._on_key(self.viewer.window, first_key + index, 0, glfw.RELEASE, 0)
        self.assertEqual(self.commands, [f"select {axis}" for axis in range(1, 6)] * 2)

    def test_input_rate_rejection_does_not_claim_sent_or_lock_channel(self):
        self.replace_viewer(callback=lambda command: False)
        self.viewer.update(state_packet())
        self.viewer._dispatch("select 3")
        self.assertIn("Not sent: select 3", self.viewer._notice)
        self.assertFalse(self.viewer._transport_failed)
        self.assertEqual(self.viewer.blocked_reason(), "")
        self.viewer.send_command = self.commands.append
        self.viewer._dispatch("select 3")
        self.assertEqual(self.commands, ["select 3"])

    def test_input_value_error_is_recoverable_but_io_failure_locks(self):
        def rejected(command):
            raise ValueError("controller initialization pending")
        self.replace_viewer(callback=rejected)
        self.viewer.update(state_packet())
        self.viewer._dispatch("select 3")
        self.assertIn("Rejected: controller initialization pending", self.viewer._notice)
        self.assertFalse(self.viewer._transport_failed)
        self.assertEqual(self.viewer.blocked_reason(), "")
        def broken_pipe(command):
            raise BrokenPipeError("test pipe closed")
        self.viewer.send_command = broken_pipe
        self.viewer._dispatch("select 3")
        self.assertTrue(self.viewer._transport_failed)
        self.assertEqual(self.viewer.blocked_reason(), "COMMAND CHANNEL FAILED")

    def test_render_exception_requests_quit_and_closes(self):
        self.viewer.update(state_packet())
        with mock.patch.object(self.viewer, "_draw", side_effect=RuntimeError("injected renderer failure")):
            with self.assertRaisesRegex(RuntimeError, "injected renderer failure"):
                self.viewer.render()
        self.assertEqual(self.commands, ["quit"])
        self.assertFalse(self.viewer.is_open())
        self.viewer.close()
        self.assertEqual(self.commands, ["quit"])

    def test_hidden_minimized_or_zero_size_polls_without_draw_or_swap(self):
        self.viewer.update(state_packet())
        self.viewer._headless = False  # 已有姿态；本项不测试首次展示。
        for visible, iconified, size in ((False, False, (1440, 900)),
                                         (True, True, (1440, 900)),
                                         (True, False, (0, 0))):
            with self.subTest(visible=visible, iconified=iconified, size=size):
                self.viewer._last_draw_at = 123.0
                attributes = {glfw.FOCUSED: False, glfw.VISIBLE: visible, glfw.ICONIFIED: iconified}
                with (mock.patch.object(glfw, "poll_events") as poll,
                      mock.patch.object(glfw, "get_window_attrib",
                                        side_effect=lambda window, attribute: attributes[attribute]),
                      mock.patch.object(glfw, "get_framebuffer_size", return_value=size),
                      mock.patch.object(self.viewer, "_draw") as draw,
                      mock.patch.object(glfw, "swap_buffers") as swap,
                      mock.patch.object(glfw, "make_context_current") as make_current):
                    self.assertTrue(self.viewer.render())
                poll.assert_called_once_with()
                draw.assert_not_called()
                swap.assert_not_called()
                make_current.assert_not_called()
                self.assertIsNone(self.viewer._last_draw_at)
        self.assertEqual(self.commands, [])

    def test_minimize_restore_immediately_renders_latest_received_pose(self):
        self.viewer.update(state_packet())
        self.viewer._headless = False  # 已有姿态；本项不测试首次展示。
        self.viewer._last_draw_at = 10.0
        iconified = True
        order = []
        real_draw = self.viewer._draw

        def attribute(window, key):
            return iconified if key == glfw.ICONIFIED else key == glfw.VISIBLE

        def draw(framebuffer_size=None):
            order.append("draw")
            return real_draw(framebuffer_size)

        with (mock.patch.object(glfw, "get_window_attrib", side_effect=attribute),
              mock.patch.object(glfw, "poll_events", side_effect=lambda: order.append("poll")),
              mock.patch.object(glfw, "swap_buffers", side_effect=lambda window: order.append("swap")),
              mock.patch.object(viewer_module.time, "monotonic", return_value=10.001),
              mock.patch.object(self.viewer, "_draw", side_effect=draw)):
            self.assertTrue(self.viewer.render())
            updated = state_packet(2, q_ref=[.1, .2, .3, .4, .5],
                                   q_fb=[.11, .22, .33, .44, .55])
            self.assertTrue(self.viewer.update(updated))
            iconified = False
            self.assertTrue(self.viewer.render())
        self.assertEqual(order, ["poll", "poll", "draw", "swap"])
        np.testing.assert_allclose(self.viewer.reference.qpos[self.viewer._qpos_adr], updated["q_ref"])
        np.testing.assert_allclose(self.viewer.feedback.qpos[self.viewer._qpos_adr], updated["q_fb"])
        self.assertEqual(self.commands, [])

    def test_first_valid_pose_requests_focus_in_render_thread_once(self):
        self.viewer._headless = False
        main_thread = threading.get_ident()
        focus_threads = []
        with (mock.patch.object(glfw, "show_window") as show,
              mock.patch.object(glfw, "focus_window", side_effect=lambda window:
                                focus_threads.append(threading.get_ident())) as focus,
              mock.patch.object(glfw, "poll_events"),
              mock.patch.object(glfw, "get_window_attrib",
                                side_effect=lambda window, key: key in (glfw.FOCUSED, glfw.VISIBLE)),
              mock.patch.object(self.viewer, "_draw"),
              mock.patch.object(glfw, "swap_buffers")):
            self.viewer.update(state_packet(initialized=False, q_ref=[None] * 5))
            self.viewer.render()
            show.assert_not_called()
            focus.assert_not_called()
            accepted = []
            reader = threading.Thread(target=lambda: accepted.append(self.viewer.update(
                state_packet(2, initialized=False, settled=False, reason="startup_settling"))))
            reader.start()
            reader.join(timeout=2)
            self.assertFalse(reader.is_alive())
            self.assertEqual(accepted, [True])
            show.assert_not_called()
            focus.assert_not_called()  # 接收线程只置标志，不能执行任何GLFW操作。
            self.viewer.render()
            show.assert_called_once_with(self.viewer.window)
            focus.assert_called_once_with(self.viewer.window)
            self.assertEqual(focus_threads, [main_thread])
            self.viewer.update(state_packet(3))
            self.viewer.render()
            show.assert_called_once()
            focus.assert_called_once()
        self.assertEqual(self.commands, [])

    def test_later_background_packets_do_not_take_focus_back(self):
        self.viewer._headless = False
        focused = True
        clock = 10.
        with (mock.patch.object(glfw, "show_window") as show,
              mock.patch.object(glfw, "focus_window") as focus,
              mock.patch.object(glfw, "poll_events"),
              mock.patch.object(viewer_module.time, "monotonic", side_effect=lambda: clock),
              mock.patch.object(glfw, "get_window_attrib",
                                side_effect=lambda window, key: focused if key == glfw.FOCUSED else key == glfw.VISIBLE),
              mock.patch.object(self.viewer, "_draw") as draw,
              mock.patch.object(glfw, "swap_buffers") as swap):
            self.viewer.update(state_packet())
            self.viewer.render()
            self.assertEqual((show.call_count, focus.call_count, draw.call_count, swap.call_count), (1, 1, 1, 1))
            focused = False
            clock += .04
            self.viewer.update(state_packet(2))
            self.viewer.render()
            self.assertEqual((show.call_count, focus.call_count, draw.call_count, swap.call_count), (1, 1, 2, 2))
            self.assertEqual(self.viewer._last_draw_at, clock)
        self.assertEqual(self.commands, [])

    def test_visible_unfocused_window_renders_each_new_pose_without_refocusing(self):
        self.viewer.update(state_packet())
        self.viewer._headless = False
        clock = 10.
        with (mock.patch.object(viewer_module.time, "monotonic", side_effect=lambda: clock),
              mock.patch.object(glfw, "get_window_attrib",
                                side_effect=lambda window, key: key == glfw.VISIBLE),
              mock.patch.object(glfw, "poll_events") as poll,
              mock.patch.object(glfw, "focus_window") as focus,
              mock.patch.object(glfw, "swap_buffers") as swap):
            for sequence in range(2, 5):
                clock += .04
                reference = [.01 * sequence, .2, .3, .4, .5]
                feedback = [.01 * sequence - .003, .19, .29, .39, .49]
                self.assertTrue(self.viewer.update(state_packet(sequence, q_ref=reference, q_fb=feedback)))
                self.assertTrue(self.viewer.render())
                np.testing.assert_allclose(self.viewer.reference.qpos[self.viewer._qpos_adr], reference)
                np.testing.assert_allclose(self.viewer.feedback.qpos[self.viewer._qpos_adr], feedback)
        self.assertEqual((poll.call_count, swap.call_count), (3, 3))
        focus.assert_not_called()
        self.assertEqual(self.commands, [])

    def test_headless_first_pose_never_requests_focus(self):
        with (mock.patch.object(glfw, "show_window") as show,
              mock.patch.object(glfw, "focus_window") as focus):
            self.viewer.update(state_packet())
            self.viewer.render()
            show.assert_not_called()
            focus.assert_not_called()

    def test_mouse_focus_and_l_preserve_nonzero_pose_through_render_and_timeout(self):
        # 真实日志的代表性数值；这是离线包，不连接SDK或发送设备指令。
        home = np.radians([-1.4, 35., -22.5, -88., 0.]).tolist()
        feedback = np.radians([-1.6, 13.36, -27.066, -88.442, -2.725]).tolist()
        self.viewer._headless = False
        focused = True
        clock = 10.
        queued_events = []
        draw_checks = []
        real_scene_update = mujoco.mjv_updateScene
        initial_geometry = mujoco.MjData(self.viewer.model)
        mujoco.mj_forward(self.viewer.model, initial_geometry)

        def poll_events():
            while queued_events:
                queued_events.pop(0)()

        def check_scene(model, data, option, perturb, camera, category, scene):
            expected = self.viewer._q_ref if data is self.viewer.reference else self.viewer._q_fb
            expected_pose = self.viewer._baseline.copy()
            expected_pose[self.viewer._qpos_adr] = expected
            # 检查实际渲染输入，不能仅断言遥测缓存仍然非零。
            np.testing.assert_allclose(data.qpos, expected_pose, rtol=0, atol=1e-12)
            before = data.qpos.copy()
            real_scene_update(model, data, option, perturb, camera, category, scene)
            np.testing.assert_array_equal(data.qpos, before)
            self.assertGreater(float(np.max(np.linalg.norm(
                data.geom_xpos - initial_geometry.geom_xpos, axis=1))), 1e-4)
            checked_geoms = 0
            for index in range(scene.ngeom):
                geom = scene.geoms[index]
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0:
                    np.testing.assert_allclose(geom.pos, data.geom_xpos[geom.objid],
                                               rtol=1e-5, atol=1e-6)
                    checked_geoms += 1
            self.assertGreater(checked_geoms, 0)
            draw_checks.append({"seq": self.viewer._state["seq"],
                                "panel": "reference" if data is self.viewer.reference else "feedback",
                                "arm_qpos": data.qpos[self.viewer._qpos_adr].tolist(),
                                "scene_geometries_checked": checked_geoms})

        def assert_poses(reference_angles, feedback_angles):
            for data, expected in ((self.viewer.reference, reference_angles),
                                   (self.viewer.feedback, feedback_angles)):
                np.testing.assert_allclose(data.qpos[self.viewer._qpos_adr], expected,
                                           rtol=0, atol=1e-12)

        with (mock.patch.object(viewer_module.time, "monotonic", side_effect=lambda: clock),
              mock.patch.object(glfw, "get_window_attrib",
                                side_effect=lambda window, key: focused if key == glfw.FOCUSED else key == glfw.VISIBLE),
              mock.patch.object(glfw, "get_cursor_pos", return_value=(400., 300.)),
              mock.patch.object(glfw, "show_window"),
              mock.patch.object(glfw, "focus_window"),
              mock.patch.object(glfw, "poll_events", side_effect=poll_events),
              mock.patch.object(glfw, "swap_buffers"),
              mock.patch.object(mujoco, "mjv_updateScene", side_effect=check_scene)):
            self.assertTrue(self.viewer.update(state_packet(q_ref=home, q_fb=feedback)))
            self.assertTrue(self.viewer.render())
            assert_poses(home, feedback)
            home_reference_geometry = self.viewer.reference.geom_xpos.copy()
            focused = False
            clock += .04
            self.assertTrue(self.viewer.update(state_packet(2, q_ref=home, q_fb=feedback)))
            before_background = len(draw_checks)
            self.assertTrue(self.viewer.render())
            self.assertEqual(len(draw_checks), before_background + 2)
            assert_poses(home, feedback)
            # 点击只改变相机鼠标起点；聚焦后的首帧仍应保留非零的两侧姿态。
            focused = True
            clock += .04
            queued_events.append(lambda: self.viewer._on_mouse_button(
                self.viewer.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0))
            self.assertTrue(self.viewer.render())
            assert_poses(home, feedback)
            self.assertEqual(self.commands, [])
            self.viewer.save_snapshot(str(OUTPUT / "focus_recovery_before_loop.png"))
            clock += .04
            queued_events.append(lambda: self.viewer._on_key(
                self.viewer.window, glfw.KEY_L, 0, glfw.PRESS, 0))
            self.assertTrue(self.viewer.render())
            self.assertEqual(self.commands, ["loop 3"])
            assert_poses(home, feedback)
            target = home.copy()
            target[0] += np.radians(10.)
            clock += .04
            self.assertTrue(self.viewer.update(state_packet(
                3, q_ref=target, q_fb=feedback, moving=True, settled=False,
                loop_active=True, loop_remaining=3, reason="moving")))
            self.assertTrue(self.viewer.render())
            assert_poses(target, feedback)
            self.assertGreater(float(np.max(np.linalg.norm(
                self.viewer.reference.geom_xpos - home_reference_geometry, axis=1))), 1e-4)
            self.viewer.save_snapshot(str(OUTPUT / "focus_recovery_loop_target.png"))
            clock += .04
            self.assertTrue(self.viewer.update(state_packet(
                4, q_ref=target, q_fb=feedback, settled=False, fault=True,
                session_stopped=True, reason="move_settle_timeout")))
            queued_events.append(lambda: self.viewer._on_key(
                self.viewer.window, glfw.KEY_L, 0, glfw.PRESS, 0))
            self.assertTrue(self.viewer.render())
            assert_poses(target, feedback)
            self.assertEqual(self.commands, ["loop 3"])
            self.assertIn("SESSION STOPPED", self.viewer._notice)
        (OUTPUT / "focus_recovery_render_checks.json").write_text(
            json.dumps({"source": "synthetic representative hardware angles; no SDK",
                        "draw_checks": draw_checks, "commands": self.commands}, indent=2),
            encoding="utf-8")

    def test_visible_render_rate_limit_keeps_polling(self):
        self.viewer._headless = False
        with (mock.patch.object(glfw, "get_window_attrib",
                                side_effect=lambda window, key: key in (glfw.FOCUSED, glfw.VISIBLE)),
              mock.patch.object(glfw, "poll_events") as poll,
              mock.patch.object(glfw, "swap_buffers") as swap,
              mock.patch.object(self.viewer, "_draw") as draw,
              mock.patch.object(viewer_module.time, "monotonic",
                                side_effect=[10., 10.01, 10.02, 10.04])):
            for _ in range(4):
                self.assertTrue(self.viewer.render())
        self.assertEqual(poll.call_count, 4)
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(swap.call_count, 2)
        self.assertEqual(self.commands, [])

    def test_headless_explicit_render_ignores_focus_and_rate_limit(self):
        with (mock.patch.object(glfw, "get_window_attrib") as attribute,
              mock.patch.object(glfw, "poll_events") as poll,
              mock.patch.object(glfw, "swap_buffers") as swap,
              mock.patch.object(self.viewer, "_draw") as draw,
              mock.patch.object(viewer_module.time, "monotonic", return_value=10.)):
            self.assertTrue(self.viewer.render())
            self.assertTrue(self.viewer.render())
        attribute.assert_not_called()
        self.assertEqual(poll.call_count, 2)
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(swap.call_count, 2)
        self.assertEqual(self.commands, [])

    def test_poll_close_prevents_draw_and_swap(self):
        with (mock.patch.object(glfw, "poll_events", side_effect=lambda: self.viewer._on_key(
                  self.viewer.window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)),
              mock.patch.object(self.viewer, "_draw") as draw,
              mock.patch.object(glfw, "swap_buffers") as swap):
            self.assertFalse(self.viewer.render())
        draw.assert_not_called()
        swap.assert_not_called()
        self.assertEqual(self.commands, ["quit"])

    def test_close_still_releases_after_command_channel_failure(self):
        attempts = []

        def broken_sender(command):
            attempts.append(command)
            raise BrokenPipeError("injected closed channel")

        self.replace_viewer(callback=broken_sender)
        self.viewer.update(state_packet())
        self.viewer.close()
        self.viewer.close()
        self.assertEqual(attempts, ["quit"])
        self.assertFalse(self.viewer.is_open())
        self.assertIsNone(self.viewer.context)

    def test_snapshot_renders_two_panels(self):
        packet = state_packet(selected_joint=3)
        packet["q_fb"][2] += .01
        self.viewer.update(packet)
        self.assertTrue(self.viewer.render())
        self.viewer.update(state_packet(2, selected_joint=3, q_fb=packet["q_fb"]))
        target = OUTPUT / "comparison.png"
        self.viewer.save_snapshot(str(target))
        self.assertEqual(self.viewer.context.currentBuffer, mujoco.mjtFramebuffer.mjFB_WINDOW)
        with Image.open(target) as screenshot:
            self.assertGreaterEqual(screenshot.width, 1000)
            self.assertGreaterEqual(screenshot.height, 700)
            pixels = np.asarray(screenshot.convert("RGB"))
            half = screenshot.width // 2
            # 基本像素检查仅排除空图；布局/映射文字仍由根代理目检。
            for panel in (pixels[:, :half], pixels[:, half:]):
                self.assertGreater(float(panel.std()), 5.)
        self.assertEqual(self.commands, [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    global MODEL, OUTPUT
    MODEL = args.model.resolve()
    OUTPUT = args.output.resolve()
    if not MODEL.is_file():
        parser.error(f"model does not exist: {MODEL}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CompareViewerChecks)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    report = {
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [{"test": str(test), "traceback": detail} for test, detail in result.failures],
        "errors": [{"test": str(test), "traceback": detail} for test, detail in result.errors],
        "platform": "Ubuntu GLFW/MuJoCo (run location must be verified by caller)",
        "model": str(MODEL), "headless_glfw": True,
        "sdk_started": False, "hardware_started": False,
        "commands": "in-memory spy callbacks only", "snapshot": str(OUTPUT / "comparison.png"),
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
