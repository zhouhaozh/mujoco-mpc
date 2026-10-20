"""Independent replay-control and causal seek checks; execute on Ubuntu only.

The short fixtures are synthetic. Replay, estimator, forecast and free-joint
simulation are the real implementations, so backward seeks are checked against
a fresh replay rather than against duplicated controller logic.
"""
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from tennis_calibration.identified_filter import forecast_states_batch
from tennis_calibration.identified_mujoco import StandaloneBall
from tennis_calibration.identified_player import _draw, _save
from tennis_calibration.player_session import LAYERS, PlayerSession, SPEED_PRESETS
from tennis_calibration.player_scene import configure_scene, draw_environment


PARAMETERS = {'k_drag': .02, 'e_n': .83, 'alpha_t': .56,
              'impact_mode': 'effective'}
HEIGHT = .03
SELECTED = {'parameters': PARAMETERS, 'contact_height': HEIGHT,
            'filter': {'init_window_s': .08,
                       'initial_magnus_sigma': .04,
                       'magnus_drive_sigma': .012},
            'augmented': True, 'selected_name': 'synthetic_control_fixture'}


@unittest.skipUnless(importlib.util.find_spec('mujoco'),
                     'requires project Ubuntu MuJoCo environment')
class PlayerSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        local_times = np.linspace(0., .60, 121)
        cls.traces = []
        for identity, split, offset in [('folder-a/1.txt', 'train', 0.),
                                        ('folder-b/1.txt', 'validation', 2.),
                                        ('folder-c/2.txt', 'test', -1.)]:
            initial = np.array([offset, .1, .6, 7., .3, -.7,
                                .05, -.03, .02])
            truth = forecast_states_batch(initial[None], np.array([False]),
                local_times, PARAMETERS, HEIGHT)[0]
            times = local_times + 7.3
            cls.traces.append({'id': identity, 'split': split,
                'raw_times': times, 'raw_xyz': truth[:, :3],
                'eval_times': times, 'eval_xyz': truth[:, :3],
                'eval_scope_end_s': float(times[-1])})

    def setUp(self):
        import mujoco
        self.mj = mujoco
        self.sim = StandaloneBall(np.r_[self.traces[0]['raw_xyz'][0], np.zeros(6)],
            PARAMETERS, HEIGHT, timestep_s=.001)
        self.sim.__enter__()
        self.addCleanup(self.sim.__exit__, None, None, None)
        self.session = PlayerSession(self.sim, self.traces, SELECTED,
            self.traces[0]['id'], [t['id'] for t in self.traces], loop=False)

    def assert_reset(self, identity):
        replay = self.session.replay
        self.assertEqual(self.session.trajectory_id, identity)
        self.assertEqual(replay.cursor, 0)
        self.assertEqual(replay.elapsed, 0.)
        self.assertEqual(replay.forecast_updates, 0)
        self.assertEqual(replay.audit, [])
        self.assertEqual(replay.filter_trail, [])
        self.assertIsNone(replay.snapshot)
        self.assertIsNone(replay.forecast_t)
        self.assertEqual(replay.forecast.shape, (0, 3))
        self.assertEqual(self.sim.events, [])
        self.assertFalse(self.sim.bounced)
        self.assertFalse(replay.initialized)
        np.testing.assert_array_equal(self.sim.data.qvel, np.zeros(6))

    def test_speed_range_finite_and_keyboard_reaches_point_one(self):
        for speed in [.1, .123, .25, .5, 1., 2., 4.]:
            self.session.dispatch({'action': 'speed', 'value': speed})
            self.assertEqual(self.session.controls['speed'], speed)
        for invalid in [0., .099999, -1., 4.000001, np.nan, np.inf, -np.inf]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.session.dispatch({'action': 'speed', 'value': invalid})
        self.session.dispatch({'action': 'speed', 'value': 4.})
        reached = []
        for _ in range(8):
            self.session.key(45)
            reached.append(self.session.controls['speed'])
        self.assertEqual(reached[:5], [2., 1., .5, .25, .1])
        self.assertEqual(reached[-1], .1)
        for _ in range(8):
            self.session.key(61)
        self.assertEqual(self.session.controls['speed'], 4.)
        self.assertEqual(SPEED_PRESETS, (.1, .25, .5, 1., 2., 4.))

    def test_pause_stops_clock_raw_filter_and_real_physics(self):
        self.session.replay.advance(.15)
        before = self.session.state()
        physics = self.sim.state().copy()
        snapshot = self.session.replay.snapshot['state'].copy()
        self.session.key(32)
        self.session.advance(1.)
        self.assertEqual(self.session.replay.elapsed, before['elapsed_s'])
        self.assertEqual(self.session.replay.cursor, before['raw_rows_received'])
        self.assertEqual(self.session.replay.forecast_updates, before['forecast_updates'])
        np.testing.assert_array_equal(self.sim.state(), physics)
        np.testing.assert_array_equal(self.session.replay.snapshot['state'], snapshot)
        self.session.key(32)
        self.session.advance(.02)
        self.assertAlmostEqual(self.session.replay.elapsed, .16)

    def test_wall_delta_respects_speed_and_negative_delta_does_not_rewind(self):
        self.session.dispatch({'action': 'speed', 'value': .1})
        self.session.advance(.1)
        self.assertAlmostEqual(self.session.replay.elapsed, .01)
        self.session.advance(-.1)
        self.assertAlmostEqual(self.session.replay.elapsed, .01)
        self.session.dispatch({'action': 'speed', 'value': 4.})
        self.session.advance(.01)
        self.assertAlmostEqual(self.session.replay.elapsed, .05)

    def test_next_previous_and_duplicate_filename_use_full_identity(self):
        self.session.replay.advance(.15)
        self.session.key(78)
        self.assert_reset('folder-b/1.txt')
        self.session.key(80)
        self.assert_reset('folder-a/1.txt')
        self.session.key(80)
        self.assert_reset('folder-c/2.txt')
        self.session.dispatch({'action': 'select', 'trajectory_id': 'folder-b/1.txt'})
        self.assert_reset('folder-b/1.txt')
        with self.assertRaises(ValueError):
            self.session.dispatch({'action': 'select', 'trajectory_id': '1.txt'})

    def test_filtered_pool_auto_selects_deduplicates_and_handles_empty(self):
        self.session.replay.advance(.15)
        self.session.dispatch({'action': 'trajectory_pool',
            'trajectory_ids': ['folder-b/1.txt', 'folder-c/2.txt', 'folder-b/1.txt']})
        self.assertEqual(self.session.pool_ids, ['folder-b/1.txt', 'folder-c/2.txt'])
        self.assert_reset('folder-b/1.txt')
        self.session.dispatch({'action': 'next'})
        self.assert_reset('folder-c/2.txt')
        self.session.dispatch({'action': 'next'})
        self.assert_reset('folder-b/1.txt')
        self.session.dispatch({'action': 'trajectory_pool', 'trajectory_ids': []})
        replay = self.session.replay
        self.session.key(78)
        self.session.key(80)
        self.assertIs(self.session.replay, replay)
        self.assertIsNone(self.session.state()['pool_index'])
        with self.assertRaises(ValueError):
            self.session.dispatch({'action': 'trajectory_pool', 'trajectory_ids': ['unknown']})

    def test_backward_seek_removes_future_state_and_matches_fresh_prefix(self):
        self.session.replay.advance(.5)
        old_replay = self.session.replay
        self.assertTrue(self.sim.bounced)
        self.assertGreater(old_replay.forecast_updates, 10)
        self.session.dispatch({'action': 'seek', 'fraction': .25})
        sought = self.session.replay
        self.assertIsNot(sought, old_replay)
        self.assertTrue(self.session.controls['paused'])
        self.assertAlmostEqual(sought.elapsed, .15)
        self.assertFalse(self.sim.bounced)
        self.assertEqual(self.sim.events, [])
        self.assertEqual(len(sought.audit), 1)
        self.assertLess(sought.cursor, old_replay.cursor)
        self.assertLess(sought.forecast_updates, old_replay.forecast_updates)
        received = sought.times[:sought.cursor]
        self.assertTrue(np.all(received <= sought.origin+sought.elapsed+1e-12))
        self.assertLessEqual(sought.forecast_t, sought.origin+sought.elapsed+1e-12)
        self.assertTrue(all(t <= sought.origin+sought.elapsed+1e-12
                            for t, _ in sought.tracker._recent))
        got_physics, got_filter = self.sim.state().copy(), sought.snapshot['state'].copy()
        got_forecast = sought.forecast.copy()
        got_covariance = sought.snapshot['covariance'].copy()
        got_trail = np.array(sought.filter_trail)
        self.session.select(self.session.trajectory_id)
        self.session.replay.advance(.15)
        np.testing.assert_allclose(self.sim.state(), got_physics, atol=2e-12, rtol=0)
        np.testing.assert_array_equal(self.session.replay.snapshot['state'], got_filter)
        np.testing.assert_array_equal(self.session.replay.snapshot['covariance'], got_covariance)
        np.testing.assert_array_equal(self.session.replay.forecast, got_forecast)
        np.testing.assert_array_equal(self.session.replay.filter_trail, got_trail)

    def test_seek_zero_before_initialization_clears_all_future(self):
        self.session.replay.advance(.5)
        self.session.seek(0.)
        self.assertEqual(self.session.replay.elapsed, 0.)
        self.assertEqual(self.session.replay.cursor, 1)
        self.assertFalse(self.session.replay.initialized)
        self.assertEqual(self.session.replay.forecast_updates, 0)
        self.assertEqual(self.session.replay.audit, [])
        self.assertEqual(self.session.replay.filter_trail, [])
        self.assertIsNone(self.session.replay.forecast_t)
        self.assertEqual(self.sim.events, [])
        self.assertEqual(self.sim.data.time, 0.)
        np.testing.assert_array_equal(self.sim.data.qpos[:3], self.traces[0]['raw_xyz'][0])
        for invalid in [-.001, 1.001, np.nan, np.inf]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.session.seek(invalid)

    def test_restart_recovers_playback_from_pause(self):
        self.session.replay.advance(.3)
        self.session.dispatch({'action': 'pause', 'paused': True})
        self.session.key(82)
        self.assert_reset('folder-a/1.txt')
        self.assertFalse(self.session.controls['paused'])
        self.session.advance(.02)
        self.assertGreater(self.session.replay.elapsed, 0.)

    def test_keyboard_step_pauses_and_clamps_endpoints(self):
        self.session.key(262)
        self.assertAlmostEqual(self.session.replay.elapsed, .02)
        self.assertTrue(self.session.controls['paused'])
        self.session.key(263)
        self.assertAlmostEqual(self.session.replay.elapsed, 0.)
        self.session.key(263)
        self.assertAlmostEqual(self.session.replay.elapsed, 0.)
        self.session.dispatch({'action': 'step', 'seconds': 10.})
        self.assertAlmostEqual(self.session.replay.elapsed, self.session.replay.duration)

    def test_nonloop_end_pauses_at_final_sample(self):
        self.session.advance(10.)
        self.assertTrue(self.session.controls['paused'])
        self.assertAlmostEqual(self.session.replay.elapsed, self.session.replay.duration)
        self.assertEqual(self.session.replay.cursor, self.session.replay.end_index)

    def test_loop_end_restarts_same_trajectory(self):
        self.session.dispatch({'action': 'loop', 'enabled': True})
        self.session.advance(10.)
        self.assert_reset('folder-a/1.txt')
        self.assertFalse(self.session.controls['paused'])

    def test_auto_next_end_stops_and_loop_wraps_pool(self):
        self.session.dispatch({'action': 'auto_next', 'enabled': True})
        self.session.advance(10.)
        self.assert_reset('folder-b/1.txt')
        self.session.advance(10.)
        self.assert_reset('folder-c/2.txt')
        self.session.advance(10.)
        self.assertEqual(self.session.trajectory_id, 'folder-c/2.txt')
        self.assertTrue(self.session.controls['paused'])
        self.session.dispatch({'action': 'loop', 'enabled': True})
        self.session.dispatch({'action': 'pause', 'paused': False})
        self.session.advance(.01)
        self.assert_reset('folder-a/1.txt')
        self.assertFalse(self.session.controls['paused'])

    def test_saved_finished_replay_keeps_real_engine_evidence_after_shared_reset(self):
        for target in [.1, .2, .3, .4, .5, .6]:
            self.session.replay.advance(target)
        old_replay = self.session.replay
        expected_steps = self.sim.engine_steps
        expected_callbacks = self.sim.callback_calls
        expected_events = json.loads(json.dumps(self.sim.events))
        self.assertGreater(expected_steps, 0)
        self.assertGreater(expected_callbacks, expected_steps)
        self.assertEqual(len(expected_events), 1)
        self.session.select('folder-b/1.txt')
        self.assertEqual(self.sim.engine_steps, 0)
        self.assertEqual(self.sim.events, [])
        evidence = old_replay.simulation_evidence
        self.assertEqual(evidence['engine_steps'], expected_steps)
        self.assertEqual(evidence['callback_calls'], expected_callbacks)
        self.assertEqual(evidence['events'], expected_events)
        # A later real collision in the reused engine must not overwrite the
        # finished interval's deep-copied incident state or event counters.
        self.session.replay.advance(.5)
        self.assertEqual(len(self.sim.events), 1)
        self.assertNotEqual(self.sim.events[0]['incoming_state'][0],
                            expected_events[0]['incoming_state'][0])
        self.assertEqual(evidence['events'], expected_events)
        with tempfile.TemporaryDirectory(prefix='player-evidence-', dir='.tmp') as temporary:
            destination = Path(temporary)
            with redirect_stdout(io.StringIO()):
                _save(destination, old_replay, SELECTED, time.perf_counter(), True)
            summary = json.loads((destination/'summary.json').read_text())
            self.assertEqual(summary['trajectory_id'], 'folder-a/1.txt')
            self.assertEqual(summary['engine_steps'], expected_steps)
            self.assertEqual(summary['callback_calls'], expected_callbacks)
            self.assertEqual(summary['events'], expected_events)
            self.assertTrue(summary['assertions']['actual_qpos_moved'])
            self.assertTrue(summary['assertions']['future_prediction_repeatedly_updated'])

    def test_camera_and_layer_commands_do_not_touch_observation_or_state(self):
        self.session.replay.advance(.15)
        replay = self.session.replay
        physics, posterior = self.sim.state().copy(), replay.snapshot['state'].copy()
        forecasts, cursor = replay.forecast.copy(), replay.cursor
        for layer in LAYERS:
            self.session.dispatch({'action': 'visibility', 'layer': layer, 'enabled': False})
        for key, preset in [(79, 'overview'), (84, 'top'), (83, 'side'), (70, 'follow')]:
            self.session.key(key)
            self.assertEqual(self.session.controls['camera'], preset)
        self.assertIs(self.session.replay, replay)
        self.assertEqual(replay.cursor, cursor)
        np.testing.assert_array_equal(self.sim.state(), physics)
        np.testing.assert_array_equal(replay.snapshot['state'], posterior)
        np.testing.assert_array_equal(replay.forecast, forecasts)
        for command in [{'action': 'visibility', 'layer': 'invalid', 'enabled': True},
                        {'action': 'camera', 'preset': 'invalid'}, {'action': 'invalid'}]:
            with self.assertRaises(ValueError):
                self.session.dispatch(command)

    def test_decorations_draw_outside_collision_model_and_leave_physics_unchanged(self):
        self.session.replay.advance(.15)
        replay = self.session.replay
        physics = self.sim.state().copy()
        original_count = self.sim.model.ngeom
        gravity = self.sim.model.opt.gravity.copy()
        positions = self.sim.model.geom_pos.copy()
        scene = self.mj.MjvScene(self.sim.model, maxgeom=4096)
        style = configure_scene(self.sim, self.traces[1], HEIGHT)
        draw_environment(self.mj, scene, style)
        self.assertGreater(scene.ngeom, 20)
        self.assertLess(scene.ngeom, 100)
        self.assertTrue(all(scene.geoms[i].category == int(self.mj.mjtCatBit.mjCAT_DECOR)
                            for i in range(scene.ngeom)))
        self.assertEqual(self.sim.model.ngeom, original_count)
        np.testing.assert_array_equal(self.sim.model.geom_contype, np.zeros(original_count))
        np.testing.assert_array_equal(self.sim.model.geom_conaffinity, np.zeros(original_count))
        np.testing.assert_array_equal(self.sim.model.opt.gravity, gravity)
        # The disabled-contact backdrop may move in XY to center the court;
        # the physical ball geom and effective surface height must stay fixed.
        ground = self.sim.model.geom('effective_ground_visual').id
        ball = self.sim.model.geom('ball_visual').id
        self.assertEqual(self.sim.model.geom_pos[ground, 2], positions[ground, 2])
        np.testing.assert_array_equal(self.sim.model.geom_pos[ball], positions[ball])
        np.testing.assert_array_equal(self.sim.state(), physics)
        self.assertIs(self.session.replay, replay)
        self.assertEqual(self.sim.data.ncon, 0)

    def test_renderer_layer_toggles_do_not_change_future_physics_or_predictor(self):
        # A minimal real MjvScene viewer exercises _draw without a GUI or renderer.
        class Viewer:
            def __init__(self, scene):
                self.user_scn = scene
                self.overlay_calls = 0
            def lock(self):
                return nullcontext()
            def set_texts(self, *args):
                self.overlay_calls += 1
            def sync(self):
                pass

        self.session.replay.advance(.15)
        viewer = Viewer(self.mj.MjvScene(self.sim.model, maxgeom=4096))
        state_before = self.sim.state().copy()
        posterior_before = self.session.replay.snapshot['state'].copy()
        cursor_before = self.session.replay.cursor
        _draw(viewer, self.session)
        self.assertGreater(viewer.user_scn.ngeom, 20)
        for layer in LAYERS:
            self.session.dispatch({'action': 'visibility', 'layer': layer, 'enabled': False})
        _draw(viewer, self.session)
        self.assertEqual(viewer.user_scn.ngeom, 0)
        self.assertEqual(self.session.replay.cursor, cursor_before)
        np.testing.assert_array_equal(self.sim.state(), state_before)
        np.testing.assert_array_equal(self.session.replay.snapshot['state'], posterior_before)
        self.session.replay.advance(.2)
        rendered_physics = self.sim.state().copy()
        rendered_posterior = self.session.replay.snapshot['state'].copy()
        rendered_forecast = self.session.replay.forecast.copy()
        self.session.select(self.session.trajectory_id)
        self.session.replay.advance(.15)
        self.session.replay.advance(.2)
        np.testing.assert_allclose(self.sim.state(), rendered_physics, atol=2e-12, rtol=0)
        np.testing.assert_array_equal(self.session.replay.snapshot['state'], rendered_posterior)
        np.testing.assert_array_equal(self.session.replay.forecast, rendered_forecast)


if __name__ == '__main__':
    unittest.main()
