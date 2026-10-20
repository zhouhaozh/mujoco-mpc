"""Frozen collision playback invariants, executed in Ubuntu .venv-tennis."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.collision_mapping import load_model, predict_velocity
from tennis_calibration.collision_prefix import predict_contact
from tennis_calibration.identified_dynamics import impact_map, simulate
from tennis_calibration.identified_mujoco import StandaloneBall
from tennis_calibration.player_bounce import BounceModelBank, SelectableCollisionBall


ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / 'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json'


@unittest.skipUnless(SELECTED.exists(), 'requires deployed frozen collision artifacts')
class BounceBankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = BounceModelBank(SELECTED)
        cls.initial = np.array([0., 0., 1.2, 14., 1., -1., .02, -.01, .01])
        times = np.linspace(0., .6, 241)
        cls.trace = {'id': 'synthetic/prefix', 'raw_times': times,
                     'raw_xyz': simulate(times, cls.initial, cls.bank.parameters['k_drag'])[:, :3],
                     'bounce_reference_s': .4, 'keep': np.zeros(len(times), bool)}

    def test_uses_existing_selected_physics_and_maps(self):
        selected = json.loads((ROOT / 'artifacts/tennis_identified/20261009_training_prior/selected_model.json').read_text())
        for key in ('k_drag', 'e_n', 'alpha_t'):
            self.assertEqual(self.bank.parameters[key], selected['parameters'][key])
        self.assertEqual(self.bank.height, selected['contact_height'])
        self.assertEqual(self.bank.selected_model_id, 'gp')
        incoming = np.array([4., 2., self.bank.height, 17., 1.8, -4.5, .02, -.01, .01])
        for name in self.bank.model_ids:
            predicted = self.bank.apply_state(incoming, name)
            if name == 'fixed':
                expected = impact_map(incoming, self.bank.parameters)
                np.testing.assert_allclose(predicted, expected, atol=1e-14)
            else:
                checkpoint = load_model(SELECTED.parent / 'models' / self.bank.checkpoint_names[name])
                expected = predict_velocity(checkpoint, incoming[None, 3:6], incoming[None, 6:9],
                    self.bank.parameters['e_n'], self.bank.parameters['alpha_t'])[0]
                np.testing.assert_allclose(predicted[3:6], expected, atol=1e-14)
            np.testing.assert_array_equal(predicted[:3], incoming[:3])
            np.testing.assert_array_equal(predicted[6:], incoming[6:])

    def test_future_and_offline_labels_cannot_change_prefix(self):
        trace = deepcopy(self.trace)
        before = self.bank.prepare_prefix(trace, .1)
        self.assertTrue(before['valid'], before['reason'])
        future = trace['raw_times'] > before['cut_t_s']
        trace['raw_xyz'][future] = np.nan
        trace['keep'] = np.ones(len(trace['raw_times']), bool)
        trace['impact_reference'] = {'v_minus_m_s': [1e9]*3, 'v_plus_m_s': [-1e9]*3}
        trace['eval_xyz'] = np.full((23, 3), -1e8)
        with patch('tennis_calibration.collision_model_bank.estimate_prefix', side_effect=AssertionError('cache missed')):
            after = self.bank.prepare_prefix(trace, .1)
        np.testing.assert_array_equal(before['fit']['state'], after['fit']['state'])
        np.testing.assert_array_equal(before['contact']['state_minus'], after['contact']['state_minus'])
        self.assertEqual(before['input_sha256'], after['input_sha256'])
        self.assertLessEqual(after['raw_prefix_times'][-1], after['cut_t_s'])
        after['fit']['state'][0] = 12345
        again = self.bank.prepare_prefix(self.trace, .1)
        np.testing.assert_array_equal(before['fit']['state'], again['fit']['state'])

    def test_event_reference_only_selects_cutoff_and_is_not_a_model_feature(self):
        a = self.bank.prepare_prefix(self.trace, .1)
        shifted = deepcopy(self.trace)
        shifted['bounce_reference_s'] = .42
        # Same cutoff and same raw data; changing scenario event/gap labels must
        # leave fit, contact, and output unchanged.
        b = self.bank.prepare_prefix(shifted, .12)
        np.testing.assert_allclose(a['fit']['state'], b['fit']['state'], atol=1e-13)
        target = np.array([.45, .5, .6])
        np.testing.assert_allclose(self.bank.predict(a, 'gp', target), self.bank.predict(b, 'gp', target), atol=1e-13)

    def test_missing_reference_and_insufficient_prefix_are_explicit(self):
        trace = deepcopy(self.trace); trace['bounce_reference_s'] = None
        self.assertEqual(self.bank.prepare_prefix(trace)['reason'], 'no_firstcontact_reference_for_replay_cutoff')
        result = self.bank.prepare_prefix(self.trace, 1.)
        self.assertFalse(result['valid'])
        with self.assertRaises(ValueError): self.bank.predict(result, 'gp', [.5])

    def test_shared_absolute_clock_and_no_input_mutation(self):
        prepared = self.bank.prepare_prefix(self.trace, .1)
        contact = prepared['contact']['contact_t_s']
        target = np.array([prepared['fit']['capture_t_s'], contact-.02, contact+.2])
        before = target.copy()
        fixed = self.bank.predict(prepared, 'fixed', target)
        gp = self.bank.predict(prepared, 'gp', target)
        np.testing.assert_array_equal(target, before)
        np.testing.assert_array_equal(fixed[:2], gp[:2])
        self.assertLess(fixed[1, 5], 0.)
        self.assertGreater(np.linalg.norm(fixed[2, :3]-gp[2, :3]), 1e-4)

    def test_validation(self):
        for bad in ('unknown', None):
            with self.assertRaises(ValueError): self.bank.apply_state(self.initial, bad)
        for state in (self.initial[:6], np.full(9, np.nan), self.initial * -1):
            with self.assertRaises(ValueError): self.bank.apply_state(state, 'fixed')
        for gap in (-1., np.nan):
            with self.assertRaises(ValueError): self.bank.prepare_prefix(self.trace, gap)


@unittest.skipUnless(SELECTED.exists() and importlib.util.find_spec('mujoco'), 'requires Ubuntu MuJoCo/artifacts')
class SelectableEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = BounceModelBank(SELECTED)
        cls.initial = np.array([1., 2., 1.4, 16., 1.5, -2., .02, -.01, .01])

    def test_none_response_preserves_original_engine_behavior(self):
        times = np.linspace(0., .8, 81)
        with StandaloneBall(self.initial, self.bank.parameters, self.bank.height) as sim:
            baseline = sim.replay(times)
        with SelectableCollisionBall(self.initial, self.bank.parameters, self.bank.height) as sim:
            selected = sim.replay(times)
        np.testing.assert_array_equal(selected['states'], baseline['states'])
        np.testing.assert_array_equal(selected['qpos'], baseline['qpos'])
        np.testing.assert_array_equal(selected['qvel'], baseline['qvel'])
        self.assertEqual(selected['events'], baseline['events'])

    def test_actual_engine_incoming_shared_but_outgoing_changes(self):
        outputs = {}
        times = np.linspace(0., .8, 81)
        for name in ('fixed', 'gp'):
            seen = []
            def response(state, model=name):
                seen.append(state.copy())
                return self.bank.apply_state(state, model)
            with SelectableCollisionBall(self.initial, self.bank.parameters, self.bank.height,
                    collision_response=response, collision_model_id=name) as sim:
                outputs[name] = sim.replay(times)
            self.assertEqual(len(seen), 1)
            event = outputs[name]['events'][0]
            np.testing.assert_array_equal(seen[0], event['incoming_state'])
            np.testing.assert_allclose(event['outgoing_state'][3:], self.bank.apply_state(seen[0], name)[3:], atol=1e-12)
            self.assertEqual(event['collision_model_id'], name)
            self.assertEqual(event['native_contact_count'], 0)
        a, b = outputs['fixed']['events'][0], outputs['gp']['events'][0]
        np.testing.assert_array_equal(a['incoming_state'], b['incoming_state'])
        self.assertEqual(a['t_s'], b['t_s'])
        self.assertGreater(np.linalg.norm(np.array(a['outgoing_state'])[3:6]-np.array(b['outgoing_state'])[3:6]), 1e-3)

    def test_forecast_and_native_selected_impulse_are_consistent(self):
        capture = 6.25
        fit = {'state': self.initial, 'capture_t_s': capture, 'converged': True}
        contact = predict_contact(fit, self.bank.parameters, self.bank.height)
        self.assertTrue(contact['valid'])
        prepared = {'valid': True, 'fit': fit, 'contact': contact}
        times = np.array([capture, capture+.08, contact['contact_t_s']+.02, contact['contact_t_s']+.3])
        for name in self.bank.model_ids:
            with self.subTest(model=name):
                forecast = self.bank.predict(prepared, name, times)
                with SelectableCollisionBall(self.initial, self.bank.parameters, self.bank.height,
                        collision_response=lambda state: self.bank.apply_state(state, name), collision_model_id=name) as sim:
                    sim.data.time = capture
                    output = sim.replay(times)
                np.testing.assert_allclose(output['states'], forecast, atol=2e-6, rtol=1e-7)
                self.assertAlmostEqual(output['events'][0]['t_s'], contact['contact_t_s'], places=6)

    def test_response_cannot_move_position_or_return_invalid_state(self):
        incoming = self.initial.copy(); incoming[2] = self.bank.height
        for response in (lambda state: state[:6], lambda state: np.full(9, np.nan),
                         lambda state: state+np.r_[1., np.zeros(8)]):
            with self.subTest(response=response):
                with SelectableCollisionBall(incoming, self.bank.parameters, self.bank.height,
                        collision_response=response) as sim:
                    with self.assertRaises(ValueError): sim.advance_to(.001)


if __name__ == '__main__':
    unittest.main()
