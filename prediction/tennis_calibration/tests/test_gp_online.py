"""Causal method-6 adapter invariants; numerical execution on Ubuntu only."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.gp_online import GPOnlinePredictor
from tennis_calibration.identified_dynamics import impact_map
from tennis_calibration.identified_filter import IdentifiedFilter, forecast_states_batch


PARAMETERS = {"k_drag": 0., "e_n": .83, "alpha_t": .56, "impact_mode": "effective"}
HEIGHT = .03
ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / "artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json"


class SyntheticBank:
    parameters = PARAMETERS
    height = HEIGHT

    def __init__(self):
        self.calls = []

    def apply_state(self, incoming, model_id):
        if model_id != "gp":
            raise AssertionError("all first-impact paths must select method 6")
        self.calls.append(np.asarray(incoming).copy())
        result = impact_map(incoming, self.parameters)
        result[3] += .8
        result[4] -= .1
        result[5] += .2
        return result


def prefix_fixture(state=None, capture=.1):
    state = np.array([0., 0., .5, 8., .4, -2., 0., 0., 0.]) if state is None else np.asarray(state).copy()
    times = np.array([capture-.04, capture-.02, capture])
    dt = times-capture
    xyz = state[None, :3]+dt[:, None]*state[None, 3:6]
    fit = {"converged": True, "capture_t_s": capture, "state": state,
           "covariance": np.eye(9)*1e-5}
    return fit, times, xyz


class GPOnlineTests(unittest.TestCase):
    def make(self, state=None, **kwargs):
        bank = SyntheticBank()
        tracker = GPOnlinePredictor(bank, **kwargs)
        fit, times, xyz = prefix_fixture(state)
        tracker.seed_from_prefix(fit, times, xyz)
        return bank, tracker, fit, times, xyz

    def test_prefix_seed_copies_same_state_and_fixed_covariance_inflation(self):
        _, tracker, fit, times, xyz = self.make()
        result = tracker.snapshot()
        np.testing.assert_array_equal(result["state"], fit["state"])
        floor = np.repeat(tracker.covariance_floor_sigmas, 3)
        np.testing.assert_allclose(result["covariance"], fit["covariance"]+np.diag(floor**2), atol=1e-14)
        self.assertFalse(result["bounced"])
        self.assertFalse(result["bounce_confirmed"])
        self.assertEqual(result["t_s"], times[-1])
        self.assertTrue(all(t <= times[-1] for t, _ in tracker._recent))
        fit["state"][:] = 1e6
        fit["covariance"][:] = 1e8
        xyz[:] = 1e9
        self.assertLess(np.linalg.norm(tracker.state), 100)
        self.assertGreaterEqual(np.linalg.eigvalsh(tracker.covariance).min(), 0)
        self.assertFalse(result["seed_metadata"]["reference_contact_used_as_input"])

    def test_prefix_rejects_future_rows_old_clock_and_nonconvergence(self):
        _, tracker, fit, times, xyz = self.make()
        with self.assertRaisesRegex(ValueError, "future observations"):
            tracker.seed_from_prefix(fit, np.r_[times, times[-1]+.001], np.vstack([xyz, xyz[-1]]))
        with self.assertRaises(ValueError):
            tracker.seed_from_prefix(fit, times[:-1], xyz[:-1])
        with self.assertRaises(ValueError):
            tracker.seed_from_prefix({**fit, "converged": False}, times, xyz)
        with self.assertRaises(ValueError):
            tracker.seed_from_prefix(fit, times[::-1], xyz[::-1])

    def test_posterior_covariance_validation_and_projection_restore_isolation(self):
        _, tracker, _, _, _ = self.make()
        posterior = tracker.snapshot()
        projector = GPOnlinePredictor(SyntheticBank())
        projector.restore_snapshot(posterior)
        self.assertEqual(len(projector._recent), 0)
        self.assertTrue(projector.snapshot()["projection_only"])
        with self.assertRaises(RuntimeError):
            projector.update(.2, [1, 2, 3])
        posterior["state"][:] = 1e7
        self.assertLess(np.linalg.norm(projector.state), 100)
        invalid = tracker.snapshot()
        invalid["covariance"][0, 0] = -1
        with self.assertRaises(ValueError):
            projector.restore_snapshot(invalid)
        invalid = tracker.snapshot(); invalid["covariance"][0, 1] += .1
        with self.assertRaises(ValueError):
            projector.restore_snapshot(invalid)
        invalid = tracker.snapshot(); invalid["bounce_confirmed"] = True
        with self.assertRaises(ValueError):
            projector.restore_snapshot(invalid)
        invalid = tracker.snapshot(); invalid["bounced"] = 1
        with self.assertRaises(ValueError):
            projector.restore_snapshot(invalid)

    def test_absolute_unsorted_duplicate_forecast_uses_gp_and_never_changes_posterior(self):
        bank, tracker, _, _, _ = self.make()
        before = tracker.snapshot()
        targets = np.array([.6, .1, .25, .6])
        output = tracker.predict_target_times(targets)
        np.testing.assert_array_equal(output[1], before["state"])
        np.testing.assert_array_equal(output[0], output[3])
        self.assertEqual(len(bank.calls), 1)
        after = tracker.snapshot()
        for key in ("state", "covariance"):
            np.testing.assert_array_equal(after[key], before[key])
        self.assertEqual(after["t_s"], before["t_s"])
        self.assertEqual(after["bounced"], before["bounced"])
        # Segmented target clocks partition RK4 steps differently; the inherited
        # 14-round event root search allows a few micro-m/s difference.
        np.testing.assert_allclose(tracker.predict(.6)["state"], output[0], atol=5e-6, rtol=0)
        fixed = forecast_states_batch(before["state"][None], np.array([False]), [.5], PARAMETERS, HEIGHT)[0, 0]
        self.assertGreater(np.linalg.norm(output[0, :3]-fixed[:3]), .1)
        self.assertIsNone(tracker.predict(.6)["covariance"])

    def test_forecast_target_validation_empty_targets_and_late_contact(self):
        _, tracker, _, _, _ = self.make()
        self.assertEqual(tracker.predict_target_times([]).shape, (0, 9))
        for invalid in ([.099], [np.nan], [[.2]], [np.inf]):
            with self.assertRaises(ValueError):
                tracker.predict_target_times(invalid)
        forecast = tracker.predict(.12)
        self.assertFalse(forecast["bounced"])
        self.assertLess(forecast["v"][2], 0)
        uninitialized = GPOnlinePredictor(SyntheticBank())
        with self.assertRaises(ValueError):
            uninitialized.predict_target_times([0])

    def test_only_first_impact_and_degenerate_vertical_guard(self):
        at_contact = np.array([0., 0., HEIGHT, 8., .4, -2., 0., 0., 0.])
        bank, tracker, _, _, _ = self.make(at_contact)
        projected = tracker.predict_target_times([.100001, 1.1])
        self.assertEqual(len(bank.calls), 1)
        self.assertGreater(projected[0, 5], 0)
        self.assertLess(projected[1, 5], 0)
        self.assertLess(projected[1, 2], HEIGHT)
        vertical = at_contact.copy(); vertical[3:5] = 0
        bank, tracker, _, _, _ = self.make(vertical)
        output = tracker.predict_target_times([.100000001])[0]
        self.assertEqual(len(bank.calls), 0)
        np.testing.assert_allclose(output[3:6], [0, 0, 1.66], atol=2e-8)

    def test_actual_filter_advance_uses_gp_collision_transition(self):
        state = np.array([0., 0., HEIGHT, 8., .4, -2., 0., 0., 0.])
        bank, tracker, _, _, _ = self.make(state)
        expected = tracker.predict(.100001)["state"]
        tracker._advance(.000001)
        self.assertTrue(tracker.bounced)
        np.testing.assert_allclose(tracker.state, expected, atol=1e-12)
        self.assertGreater(len(bank.calls), 1)  # Includes GP transition Jacobian.
        self.assertGreaterEqual(np.linalg.eigvalsh(tracker.covariance).min(), -1e-12)

    def test_late_observed_contact_recovery_prior_also_uses_gp(self):
        bank, tracker, _, _, _ = self.make()
        tracker._recent.clear()
        times = np.linspace(-.05, .06, 45)
        for t in times:
            tracker._recent.append((float(t), np.array([4*t, .2*t, HEIGHT+3*abs(t)])))
        good_fit = {"velocity": np.array([4., .2, 3.]), "position": np.array([.24, .012, .18])}
        pre_fit = {**good_fit, "velocity": np.array([8., .4, -3.])}
        seen = []
        def fuse(_):
            seen.append(tracker.state.copy())
            return True
        with patch("tennis_calibration.identified_filter._local_fit", side_effect=[pre_fit, good_fit]), \
             patch.object(tracker, "_valid_fit", return_value=(True, "test")), \
             patch.object(tracker, "_fuse_local", side_effect=fuse):
            self.assertTrue(tracker._confirm_bounce())
        self.assertEqual(len(bank.calls), 1)
        self.assertAlmostEqual(seen[0][3], .56*8+.8)
        self.assertTrue(tracker.bounced)
        self.assertTrue(tracker.bounce_confirmed)

    def test_same_received_stream_same_posterior_and_old_observations_rejected(self):
        _, one, _, _, _ = self.make()
        _, two, _, _, _ = self.make()
        for t in np.linspace(.105, .2, 20):
            point = one.predict(float(t))["p"]
            a, b = one.update(float(t), point), two.update(float(t), point)
            np.testing.assert_array_equal(a["state"], b["state"])
            np.testing.assert_array_equal(a["covariance"], b["covariance"])
        before = one.snapshot()
        result = one.update(.1, [1e6, 1e6, 1e6])
        self.assertFalse(result["accepted"])
        np.testing.assert_array_equal(result["state"], before["state"])
        self.assertEqual(result["t_s"], before["t_s"])

    def test_default_identified_hook_is_unchanged_and_callback_is_validated(self):
        original = IdentifiedFilter(PARAMETERS, HEIGHT)
        explicit = IdentifiedFilter(PARAMETERS, HEIGHT, impact_response=None)
        state = np.array([0., 0., HEIGHT, 8., .4, -2., 0., 0., 0.])
        np.testing.assert_array_equal(original._impact_state(state), explicit._impact_state(state))
        np.testing.assert_array_equal(original._impact_transition(state), explicit._impact_transition(state))
        with self.assertRaises(ValueError):
            IdentifiedFilter(PARAMETERS, HEIGHT, impact_response=3)
        bad = IdentifiedFilter(PARAMETERS, HEIGHT, impact_response=lambda x: x+1)
        with self.assertRaisesRegex(ValueError, "preserve position"):
            bad._impact_state(state)


@unittest.skipUnless(SELECTED.exists(), "requires frozen GP collision checkpoint")
class RealGPCheckpointTest(unittest.TestCase):
    def test_real_frozen_map_is_used_in_immediate_forecast_and_preserves_b(self):
        from tennis_calibration.player_bounce import BounceModelBank
        bank = BounceModelBank(SELECTED)
        tracker = GPOnlinePredictor(bank)
        state = np.array([3., 2., bank.height, 17., 1.8, -4.5, .02, -.01, .01])
        fit, times, xyz = prefix_fixture(state)
        tracker.seed_from_prefix(fit, times, xyz)
        expected = bank.apply_state(state, "gp")
        forecast = tracker.predict_target_times([.100000001])[0]
        np.testing.assert_allclose(forecast[3:6], expected[3:6], atol=2e-8)
        np.testing.assert_array_equal(forecast[6:9], state[6:9])


if __name__ == "__main__":
    unittest.main()
