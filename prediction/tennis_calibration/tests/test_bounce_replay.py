"""Independent native frozen-rebound replay checks; execute on Ubuntu only.

Uses the deployed frozen heads, real prefix fitter and real MuJoCo free joint.
Changing a playback head must change its impulse, never its observations or
clock. Synthetic references are scenario labels; future data is perturbed in
one check to prove that the replay does not assimilate it.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from tennis_calibration.bounce_replay import BounceReplay
from tennis_calibration.collision_prefix import predict_contact
from tennis_calibration.identified_filter import forecast_states_batch
from tennis_calibration.identified_player import Replay
from tennis_calibration.online_data import load_dataset
from tennis_calibration.player_bounce import BounceModelBank, SelectableCollisionBall
from tennis_calibration.player_session import PlayerSession


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json'
ONLINE = ROOT / 'artifacts/tennis_identified/20261009_training_prior/selected_model.json'
CALIBRATION = ROOT / 'artifacts/tennis_calibration/20261008_calibration'


@unittest.skipUnless(importlib.util.find_spec('mujoco') and CHECKPOINT.exists(),
                     'requires Ubuntu project MuJoCo and frozen collision artifacts')
class BounceReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = BounceModelBank(CHECKPOINT)
        cls.selected = json.loads(ONLINE.read_text())
        cls.clock_offset = 7.3
        initial = np.array([.2, .1, 1.2, 14., 1.3, -1.2, .025, -.018, .012])
        contact = predict_contact({'state': initial, 'capture_t_s': cls.clock_offset,
                                   'converged': True}, cls.bank.parameters, cls.bank.height)
        if not contact['valid']:
            raise AssertionError(contact['reason'])
        times = np.linspace(0., 1., 401)
        truth = forecast_states_batch(initial[None], np.array([False]), times,
            cls.bank.parameters, cls.bank.height)[0]
        cls.trace = {'id': 'synthetic/rebound.txt', 'split': 'validation',
            'raw_times': times + cls.clock_offset, 'raw_xyz': truth[:, :3],
            'eval_times': times + cls.clock_offset, 'eval_xyz': truth[:, :3],
            'eval_scope_end_s': float(times[-1] + cls.clock_offset),
            'bounce_reference_s': contact['contact_t_s']}

    def setUp(self):
        self.sim = SelectableCollisionBall(np.r_[self.trace['raw_xyz'][0], np.zeros(6)],
            self.bank.parameters, self.bank.height, timestep_s=.001)
        self.sim.__enter__()
        self.addCleanup(self.sim.__exit__, None, None, None)
        self.session = PlayerSession(self.sim, [deepcopy(self.trace)], self.selected,
            self.trace['id'], [self.trace['id']], loop=False, bounce_bank=self.bank)

    def enter_bounce(self, gap=.04, model='gp'):
        self.session.dispatch({'action': 'bounce_gap', 'seconds': gap})
        self.session.dispatch({'action': 'bounce_model', 'value': model})
        self.session.dispatch({'action': 'view_mode', 'value': 'bounce'})
        self.assertIsInstance(self.session.replay, BounceReplay)
        self.assertTrue(self.session.replay.valid, self.session.replay.status)
        return self.session.replay

    def test_postcontact_playback_keeps_exact_same_frozen_input(self):
        replay = self.enter_bounce()
        prepared = replay.prepared
        expected_times = self.trace['raw_times'][self.trace['raw_times'] <= prepared['cut_t_s']]
        np.testing.assert_array_equal(prepared['raw_prefix_times'], expected_times)
        self.assertLessEqual(expected_times[-1], self.trace['bounce_reference_s']-.04)
        count, fingerprint = replay.cursor, replay.input_sha256
        forecast = replay.forecast.copy()
        for fraction in (0., .1, .5, .9, 1.):
            replay.advance(fraction*replay.duration)
            self.assertEqual(replay.cursor, count)
            self.assertEqual(replay.forecast_updates, 1)
            self.assertEqual(replay.input_sha256, fingerprint)
            self.assertEqual(replay.forecast_t, replay.origin)
            np.testing.assert_array_equal(replay.forecast, forecast)
        self.assertTrue(self.sim.bounced)
        self.assertTrue(all(row['raw_rows_received'] == count and
                            not row['uses_post_contact_observations']
                            for row in replay.audit))
        self.assertGreater(self.trace['raw_times'][-1], replay.origin+replay.elapsed)

    def test_changing_future_observations_leaves_real_playback_unchanged(self):
        replay = self.enter_bounce()
        replay.advance(.8*replay.duration)
        expected = self.sim.state().copy()
        forecast = replay.forecast.copy()
        fingerprint = replay.input_sha256
        changed = deepcopy(self.trace)
        future = changed['raw_times'] > replay.prepared['cut_t_s']
        changed['raw_xyz'][future] = np.array([1e8, -1e8, 1e9])
        changed['keep'] = np.zeros(len(changed['raw_times']), dtype=bool)
        changed['eval_xyz'] = np.full_like(changed['eval_xyz'], -1e9)
        altered = BounceReplay(changed, self.selected, self.sim, self.bank, 'gp', .04)
        altered.advance(.8*altered.duration)
        self.assertEqual(altered.input_sha256, fingerprint)
        np.testing.assert_array_equal(altered.forecast, forecast)
        np.testing.assert_allclose(self.sim.state(), expected, atol=2e-12, rtol=0)

    def test_four_heads_share_input_contact_and_clock_but_change_impulse(self):
        replay = self.enter_bounce()
        fingerprint = replay.input_sha256
        contact_time = replay.prepared['contact']['contact_t_s']
        clock = replay.target_times.copy()
        outputs = {}
        for name in self.bank.model_ids:
            self.session.dispatch({'action': 'bounce_model', 'value': name})
            current = self.session.replay
            self.assertEqual(current.input_sha256, fingerprint)
            self.assertEqual(current.prepared['contact']['contact_t_s'], contact_time)
            np.testing.assert_array_equal(current.target_times, clock)
            self.assertEqual(current.elapsed, 0.)
            self.assertEqual(self.sim.events, [])
            current.advance(current.duration)
            self.assertEqual(len(self.sim.events), 1)
            outputs[name] = {'event': deepcopy(self.sim.events[0]),
                             'path': current.forecast.copy()}
            self.assertEqual(self.sim.events[0]['collision_model_id'], name)
            self.assertAlmostEqual(current.origin+self.sim.events[0]['t_s'], contact_time, places=6)
            np.testing.assert_allclose(self.sim.state()[:3], current.snapshot['state'][:3],
                                       atol=3e-6, rtol=1e-7)
        fixed = outputs['fixed']
        for name in ('ridge1', 'ridge2', 'gp'):
            with self.subTest(model=name):
                event = outputs[name]['event']
                np.testing.assert_array_equal(event['incoming_state'], fixed['event']['incoming_state'])
                self.assertEqual(event['t_s'], fixed['event']['t_s'])
                self.assertGreater(np.linalg.norm(np.array(event['outgoing_state'])[3:6]-
                    np.array(fixed['event']['outgoing_state'])[3:6]), 1e-4)
                before = clock < contact_time
                np.testing.assert_array_equal(outputs[name]['path'][before], fixed['path'][before])
                self.assertGreater(np.linalg.norm(outputs[name]['path'][-1]-fixed['path'][-1]), 1e-4)

    def test_six_missing_tail_scenarios_rebuild_a_shorter_prefix(self):
        self.enter_bounce()
        previous_count, previous_capture = np.inf, np.inf
        fingerprints = set()
        for gap in (0., .005, .01, .02, .04, .08):
            self.session.dispatch({'action': 'bounce_gap', 'seconds': gap})
            replay = self.session.replay
            self.assertTrue(replay.valid, replay.status)
            self.assertAlmostEqual(replay.prepared['cut_t_s'], self.trace['bounce_reference_s']-gap)
            self.assertLessEqual(replay.cursor, previous_count)
            self.assertLessEqual(replay.origin, previous_capture)
            self.assertLessEqual(replay.origin, replay.prepared['cut_t_s'])
            self.assertEqual(replay.elapsed, 0.)
            previous_count, previous_capture = replay.cursor, replay.origin
            fingerprints.add(replay.input_sha256)
        self.assertEqual(len(fingerprints), 6)

    def test_backward_seek_matches_fresh_physics_and_prediction(self):
        replay = self.enter_bounce()
        replay.advance(.9*replay.duration)
        self.assertTrue(self.sim.bounced)
        self.session.dispatch({'action': 'seek', 'fraction': .1})
        sought = self.session.replay
        self.assertIsNot(sought, replay)
        self.assertTrue(self.session.controls['paused'])
        self.assertEqual(sought.input_sha256, replay.input_sha256)
        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        prediction = sought.snapshot['state'].copy()
        events, target = deepcopy(self.sim.events), sought.elapsed
        self.session.select(self.trace['id'])
        fresh = self.session.replay
        fresh.advance(target)
        np.testing.assert_array_equal(fresh.target_times, sought.target_times)
        np.testing.assert_array_equal(fresh.forecast, sought.forecast)
        np.testing.assert_array_equal(fresh.snapshot['state'], prediction)
        np.testing.assert_allclose(self.sim.data.qpos, qpos, atol=2e-12, rtol=0)
        np.testing.assert_allclose(self.sim.data.qvel, qvel, atol=2e-12, rtol=0)
        self.assertEqual(self.sim.events, events)

    def test_switches_preserve_controls_and_old_real_engine_evidence(self):
        self.session.dispatch({'action': 'speed', 'value': .1})
        self.session.dispatch({'action': 'pause', 'paused': True})
        self.session.dispatch({'action': 'camera', 'preset': 'top'})
        self.session.dispatch({'action': 'bounce_compare', 'enabled': False})
        replay = self.enter_bounce()
        replay.advance(replay.duration)
        engine_steps = self.sim.engine_steps
        events = deepcopy(self.sim.events)
        self.assertGreater(engine_steps, 0)
        self.session.dispatch({'action': 'bounce_model', 'value': 'fixed'})
        self.assertEqual(replay.simulation_evidence['engine_steps'], engine_steps)
        self.assertEqual(replay.simulation_evidence['events'], events)
        self.assertEqual(self.sim.engine_steps, 0)
        self.assertEqual(self.sim.events, [])
        self.session.replay.advance(self.session.replay.duration)
        self.assertEqual(replay.simulation_evidence['events'], events)
        for key, value in {'speed': .1, 'paused': True, 'camera': 'top',
                           'bounce_compare': False}.items():
            self.assertEqual(self.session.controls[key], value)

    def test_comparison_toggle_does_not_change_model_or_state(self):
        replay = self.enter_bounce()
        replay.advance(.5*replay.duration)
        physics, prediction = self.sim.state().copy(), replay.snapshot['state'].copy()
        expected = self.bank.predict(replay.prepared, 'fixed', replay.target_times)[:, :3]
        np.testing.assert_array_equal(replay.comparison, expected)
        for visible in (False, True, False):
            self.session.dispatch({'action': 'bounce_compare', 'enabled': visible})
            self.assertIs(self.session.replay, replay)
            np.testing.assert_array_equal(self.sim.state(), physics)
            np.testing.assert_array_equal(replay.snapshot['state'], prediction)
            np.testing.assert_array_equal(replay.comparison, expected)

    def test_returning_to_online_clears_selected_impulse_and_resumes_stream(self):
        bounce = self.enter_bounce()
        bounce.advance(bounce.duration)
        self.session.dispatch({'action': 'view_mode', 'value': 'online'})
        replay = self.session.replay
        self.assertIsInstance(replay, Replay)
        self.assertIsNone(self.sim.collision_response)
        self.assertIsNone(self.sim.collision_model_id)
        self.assertEqual(replay.cursor, 0)
        self.assertEqual(replay.forecast_updates, 0)
        self.assertEqual(self.sim.events, [])
        replay.advance(.15)
        count, updates = replay.cursor, replay.forecast_updates
        replay.advance(.7)
        self.assertGreater(replay.cursor, count)
        self.assertGreater(replay.forecast_updates, updates)
        self.assertGreater(replay.forecast_t, self.trace['bounce_reference_s'])
        self.assertNotEqual(replay.cursor, bounce.cursor)
        self.assertEqual(len(self.sim.events), 1)
        self.assertNotEqual(self.sim.events[0]['impact_mode'], 'selectable_velocity_map')

    @unittest.skipUnless(CALIBRATION.exists(), 'requires frozen measured dataset')
    def test_all_three_missing_reference_samples_remain_navigable(self):
        missing = [trace for trace in load_dataset(CALIBRATION)
                   if trace['bounce_reference_s'] is None]
        self.assertEqual(len(missing), 3)
        traces = [deepcopy(self.trace), *missing]
        session = PlayerSession(self.sim, traces, self.selected, self.trace['id'],
            [trace['id'] for trace in traces], loop=False, bounce_bank=self.bank)
        session.dispatch({'action': 'view_mode', 'value': 'bounce'})
        for trace in missing:
            session.select(trace['id'])
            state = session.state()
            self.assertFalse(state['bounce_info']['valid'])
            self.assertTrue(state['paused'])
            self.assertTrue(state['bounce_status'])
            self.assertIsNone(self.sim.collision_response)
            session.advance(.1)
            self.assertEqual(session.replay.elapsed, 0.)
            session.dispatch({'action': 'view_mode', 'value': 'online'})
            session.replay.advance(min(.1, session.replay.duration))
            self.assertGreater(session.replay.cursor, 0)
            session.dispatch({'action': 'view_mode', 'value': 'bounce'})
        session.select(self.trace['id'])
        self.assertTrue(session.replay.valid)
        session.dispatch({'action': 'pause', 'paused': False})
        session.advance(.04)
        self.assertGreater(session.replay.elapsed, 0.)

    def test_invalid_switch_commands_do_not_replace_a_valid_replay(self):
        replay = self.enter_bounce()
        before = dict(self.session.controls)
        for command in ({'action': 'view_mode', 'value': 'unknown'},
                        {'action': 'bounce_model', 'value': 'unknown'},
                        {'action': 'bounce_gap', 'seconds': .015},
                        {'action': 'bounce_gap', 'seconds': np.nan}):
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.session.dispatch(command)
            self.assertIs(self.session.replay, replay)
            self.assertEqual(self.session.controls, before)


if __name__ == '__main__':
    unittest.main()
