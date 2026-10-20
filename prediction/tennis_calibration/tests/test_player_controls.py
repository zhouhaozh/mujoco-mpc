"""Actual Tk command and layout checks; run in an Ubuntu virtual display."""
from __future__ import annotations

import os
import unittest

from tennis_calibration.player_controls import (BOUNCE_GAPS, BOUNCE_MODEL_LABELS,
    ControlsWindow, VIEW_LABELS, clamp_speed, trajectory_choices)


class ControlChoicesTests(unittest.TestCase):
    def test_sample_filter_preserves_full_identity_with_repeated_filenames(self):
        traces = [{'id': 'folder-a/1.txt', 'split': 'train'},
                  {'id': 'folder-b/1.txt', 'split': 'test'}]
        found = trajectory_choices(traces, 'test', '1.txt')
        self.assertEqual([x['trajectory_id'] for x in found], ['folder-b/1.txt'])
        self.assertEqual(found[0]['index'], 1)

    def test_low_speed_and_missing_segment_choices(self):
        self.assertEqual(clamp_speed(.01), .1)
        self.assertEqual(clamp_speed(8.), 4.)
        self.assertEqual(BOUNCE_GAPS, (0., .005, .01, .02, .04, .08))


@unittest.skipUnless(os.environ.get('DISPLAY'), 'requires Ubuntu Tk display')
class ControlsWindowTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.state = {'trajectory_id': 'a/1.txt', 'view_mode': 'online',
            'bounce_model': 'gp', 'bounce_gap_s': 0., 'bounce_compare': True,
            'speed': .5, 'duration_s': 1., 'elapsed_s': 0.}
        self.window = ControlsWindow([{'id': 'a/1.txt', 'split': 'train'}],
            self.commands.append, self.state, geometry='540x600+0+0')
        self.addCleanup(self.window.close)
        self.window.poll(self.state)
        self.commands.clear()

    def activate_bounce(self):
        self.state = {**self.state, 'view_mode': 'bounce'}
        self.window.poll(self.state)
        self.commands.clear()

    def test_observation_switch_only_queues_command(self):
        self.window.view_box.current(1)
        self.window.view_box.event_generate('<<ComboboxSelected>>')
        self.window.root.update()
        self.assertEqual(self.commands, [{'action': 'view_mode', 'value': 'bounce'}])
        self.assertEqual(self.window._state['view_mode'], 'online')

    def test_model_gap_and_comparison_emit_exact_protocol(self):
        self.activate_bounce()
        for index, key in enumerate(BOUNCE_MODEL_LABELS):
            self.window.bounce_model_box.current(index)
            self.window.bounce_model_box.event_generate('<<ComboboxSelected>>')
            self.window.root.update()
            self.assertEqual(self.commands[-1], {'action': 'bounce_model', 'value': key})
        for index, gap in enumerate(BOUNCE_GAPS):
            self.window.bounce_gap_box.current(index)
            self.window.bounce_gap_box.event_generate('<<ComboboxSelected>>')
            self.window.root.update()
            self.assertEqual(self.commands[-1], {'action': 'bounce_gap', 'seconds': gap})
        self.window.bounce_compare_button.invoke()
        self.assertEqual(self.commands[-1], {'action': 'bounce_compare', 'enabled': False})

    def test_online_disables_frozen_controls_and_poll_never_emits(self):
        self.assertIn('disabled', self.window.bounce_model_box.state())
        self.assertIn('disabled', self.window.bounce_gap_box.state())
        self.assertIn('disabled', self.window.bounce_compare_button.state())
        self.window.bounce_compare_button.invoke()
        self.assertEqual(self.commands, [])
        state = {**self.state, 'view_mode': 'bounce', 'bounce_model': 'ridge2',
            'bounce_gap_s': .08, 'bounce_compare': False,
            'bounce_status': '冻结前缀 123 点 · 缺失 80 ms'}
        self.window.poll(state)
        self.assertEqual(self.commands, [])
        self.assertIn('readonly', self.window.bounce_model_box.state())
        self.assertNotIn('disabled', self.window.bounce_compare_button.state())
        self.assertEqual(self.window.bounce_model_var.get(), BOUNCE_MODEL_LABELS['ridge2'])
        self.assertEqual(self.window.bounce_gap_var.get(), '80 ms')
        self.assertEqual(self.window.bounce_status_var.get(), state['bounce_status'])
        self.assertEqual(self.window.view_var.get(), VIEW_LABELS['bounce'])
        self.assertEqual(self.window.layer_buttons['estimate'].cget('text'), '蓝色 · 所选模型预测')
        self.assertIn('反弹后测量仅用于参考', self.window.view_help_var.get())
        self.window.poll(self.state)
        self.assertEqual(self.window.layer_buttons['estimate'].cget('text'), '蓝色 · 在线估计')

    def test_scroll_reaches_camera_controls_in_short_window(self):
        self.window.root.update_idletasks()
        self.assertGreater(self.window.content.winfo_height(), self.window.canvas.winfo_height())
        self.window.canvas.yview_moveto(1.)
        self.window.root.update_idletasks()
        button = self.window.camera_buttons['follow']
        y = button.winfo_rooty()-self.window.canvas.winfo_rooty()
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y+button.winfo_height(), self.window.canvas.winfo_height())
        button.invoke()
        self.assertEqual(self.commands[-1], {'action': 'camera', 'preset': 'follow'})


if __name__ == '__main__':
    unittest.main()
