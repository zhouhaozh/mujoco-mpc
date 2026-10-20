"""Causal policy invariants; numerical execution is performed on Ubuntu."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.gp_online import GPOnlinePredictor
from tennis_calibration.height_policy import HeightPolicyPredictor
from tennis_calibration.identified_dynamics import impact_map


PARAMETERS = {"k_drag": 0., "e_n": .83, "alpha_t": .56, "impact_mode": "effective"}
HEIGHT = .03


class SyntheticBank:
    parameters = PARAMETERS
    height = HEIGHT

    def apply_state(self, incoming, model_id):
        result = impact_map(incoming, self.parameters)
        if model_id == "gp":
            result[3] += .2
        elif model_id == "ridge1":
            result[3] += .1
        elif model_id == "ridge2":
            result[3] += .15
        elif model_id != "fixed":
            raise ValueError("unsupported model")
        return result


def seeded(policy, state=None):
    tracker = HeightPolicyPredictor(SyntheticBank(), "gp", policy)
    state = (np.array([0., 0., HEIGHT + .01, 3., .2, -2., 0., 0., 0.])
             if state is None else np.asarray(state, dtype=float))
    times = np.array([.06, .08, .1])
    xyz = state[:3] + (times - .1)[:, None] * state[3:6]
    tracker.seed_from_prefix({"state": state, "covariance": np.eye(9) * 1e-4,
                              "capture_t_s": .1, "converged": True}, times, xyz)
    return tracker


def flight_xyz(timestamp, state=(0., 0., .8, 3., .2, -1.)):
    initial = np.asarray(state)
    return initial[:3] + timestamp * initial[3:6] + np.array([0., 0., -4.905 * timestamp**2])


class HeightPolicyTests(unittest.TestCase):
    def test_continuous_identical_to_original_gp_filter(self):
        original = GPOnlinePredictor(SyntheticBank())
        policy = HeightPolicyPredictor(SyntheticBank(), "gp", "continuous")
        for timestamp in np.arange(0., .501, .005):
            # Includes warmup, impact and rejected/accepted later observations.
            xyz = flight_xyz(timestamp)
            old, new = original.update(timestamp, xyz), policy.update(timestamp, xyz)
            self.assertEqual(old["initialized"], new["initialized"])
            self.assertEqual(old["accepted"], new["accepted"])
            self.assertEqual(old["reason"], new["reason"])
            self.assertEqual(old["bounced"], new["bounced"])
            if old["initialized"]:
                np.testing.assert_array_equal(old["state"], new["state"])
                np.testing.assert_array_equal(old["covariance"], new["covariance"])

    def test_height_trigger_excludes_crossing_frame_before_ordinary_fusion(self):
        tracker = seeded("hold100")
        pure = tracker.predict_target_times([.12])[0]
        with patch.object(tracker, "_confirm_bounce", side_effect=AssertionError("recovery forbidden in hold")):
            result = tracker.update(.12, [1., .1, .1])
        self.assertTrue(result["height_triggered"])
        self.assertTrue(result["hold_active"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["t_s"], .12)
        self.assertGreater(result["height_contact_t_s"], .1)
        self.assertLess(result["height_contact_t_s"], .12)
        self.assertEqual(result["height_discovery_t_s"], .12)
        np.testing.assert_allclose(result["state"], pure, atol=1e-10, rtol=0)

    def test_hold_observations_do_not_change_state_covariance_or_forecast(self):
        left, right = seeded("hold100"), seeded("hold100")
        for timestamp in (.12, .14, .16, .18, .20):
            reference = left.predict_target_times([timestamp])[0, :3]
            a = left.update(timestamp, reference)
            b = right.update(timestamp, reference + [.03, -.02, .01])
            self.assertTrue(a["hold_active"])
            np.testing.assert_array_equal(a["state"], b["state"])
            np.testing.assert_array_equal(a["covariance"], b["covariance"])
            np.testing.assert_array_equal(left.predict_target_times([.25, .4]),
                                          right.predict_target_times([.25, .4]))
        self.assertGreater(len(left._recent), 3)
        self.assertGreater(len(right._recent), 3)
        self.assertFalse(np.array_equal(left._recent[-1][1], right._recent[-1][1]))

    def test_resume_uses_received_history_and_does_not_drop_resume_frame(self):
        tracker = seeded("hold100")
        tracker.update(.12, tracker.predict_target_times([.12])[0, :3])
        stop = tracker.hold_until_t_s
        for timestamp in np.arange(.13, stop, .005):
            tracker.update(float(timestamp), tracker.predict_target_times([timestamp])[0, :3])
        history_before = len(tracker._recent)
        reference = tracker.predict_target_times([stop])[0, :3]
        result = tracker.update(stop, reference + [.005, 0., 0.])
        self.assertFalse(result["hold_active"])
        self.assertEqual(result["resume_t_s"], stop)
        self.assertNotEqual(result["reason"], "duplicate_or_old_timestamp")
        self.assertNotEqual(result["reason"], "height_policy_prediction_only")
        self.assertGreaterEqual(len(tracker._recent), history_before)
        self.assertEqual(tracker._recent[-1][0], stop)
        self.assertGreater(np.linalg.norm(result["p"] - reference), 1e-6)

    def test_permanent_hold_ignores_observations_after_100ms(self):
        left, right = seeded("permanent_hold"), seeded("permanent_hold")
        for timestamp in (.12, .2, .3, .4):
            p = left.predict_target_times([timestamp])[0, :3]
            a, b = left.update(timestamp, p), right.update(timestamp, p + [.5, .2, .2])
            np.testing.assert_array_equal(a["state"], b["state"])
            np.testing.assert_array_equal(a["covariance"], b["covariance"])
        self.assertIsNone(a["resume_t_s"])
        self.assertTrue(a["hold_active"])

    def test_freeze_after_init_preserves_warmup_then_observation_free_flight(self):
        left = HeightPolicyPredictor(SyntheticBank(), "gp", "freeze_after_init")
        right = HeightPolicyPredictor(SyntheticBank(), "gp", "freeze_after_init")
        for timestamp in np.arange(0., .081, .005):
            a, b = left.update(timestamp, flight_xyz(timestamp)), right.update(timestamp, flight_xyz(timestamp))
            if timestamp < .08 - 1e-12:
                self.assertFalse(a["initialized"])
        self.assertTrue(a["initialized"])
        self.assertAlmostEqual(a["initialization_t_s"], .08)
        for timestamp in (.1, .2, .3, .4):
            p = flight_xyz(timestamp)
            a, b = left.update(timestamp, p), right.update(timestamp, p + [.1, .1, .1])
            np.testing.assert_array_equal(a["state"], b["state"])
            np.testing.assert_array_equal(a["covariance"], b["covariance"])
        self.assertTrue(a["height_triggered"])
        self.assertTrue(a["hold_active"])

    def test_strict_height_modes_defer_bounce_confirmation_before_modeled_event(self):
        for policy in ("height_online", "hold100", "permanent_hold", "freeze_after_init", "init_hold100"):
            tracker = seeded(policy, [0., 0., 1., 3., .2, -1., 0., 0., 0.])
            before = tracker.snapshot()
            with patch("tennis_calibration.identified_filter.IdentifiedFilter._confirm_bounce",
                       side_effect=AssertionError("early observed reset forbidden")):
                self.assertFalse(tracker._confirm_bounce())
            np.testing.assert_array_equal(before["state"], tracker.state)
            self.assertFalse(tracker.bounced)

    def test_ordinary_precontact_reacquisition_is_preserved_without_false_bounce(self):
        for policy in ("height_online", "hold100", "permanent_hold"):
            tracker = seeded(policy, [0., 0., 1., 3., .2, -1., 0., 0., 0.])
            tracker._recent.clear()
            for timestamp in np.linspace(.062, .1, 18):
                delta = timestamp - .1
                point = tracker.state[:3] + delta * tracker.state[3:6]
                point += [.05, 0., -4.905 * delta**2]
                tracker._record_recent(float(timestamp), point)
            tracker._consecutive_rejections = tracker.config["reacquire_after_rejections"]
            before = tracker.state.copy()
            self.assertFalse(tracker.height_triggered)
            self.assertTrue(tracker._reacquire_from_history())
            self.assertGreater(np.linalg.norm(tracker.state[:3] - before[:3]), .04)
            self.assertFalse(tracker.bounced)
            self.assertFalse(tracker.bounce_confirmed)
            self.assertFalse(tracker.height_triggered)

    def test_init_hold100_ignores_all_later_precontact_data_and_resumes_after_height(self):
        left = HeightPolicyPredictor(SyntheticBank(), "gp", "init_hold100")
        right = HeightPolicyPredictor(SyntheticBank(), "gp", "init_hold100")
        for timestamp in np.arange(0., .081, .005):
            left.update(timestamp, flight_xyz(timestamp))
            right.update(timestamp, flight_xyz(timestamp))
        self.assertTrue(left.snapshot()["hold_active"])
        self.assertFalse(left.height_triggered)
        for timestamp in np.arange(.09, .401, .005):
            xyz = left.predict_target_times([timestamp])[0, :3]
            a = left.update(float(timestamp), xyz)
            b = right.update(float(timestamp), xyz + [.02, -.01, .005])
            self.assertTrue(a["hold_active"])
            np.testing.assert_array_equal(a["state"], b["state"])
            np.testing.assert_array_equal(a["covariance"], b["covariance"])
        self.assertTrue(left.height_triggered)
        self.assertGreater(left.height_contact_t_s, .3)
        stop = left.hold_until_t_s
        self.assertAlmostEqual(stop - left.height_contact_t_s, .1)
        xyz = left.predict_target_times([stop])[0, :3]
        a = left.update(stop, xyz + [.003, 0., 0.])
        b = right.update(stop, xyz + [-.003, 0., 0.])
        self.assertEqual(a["resume_t_s"], stop)
        self.assertFalse(a["hold_active"])
        self.assertNotEqual(a["reason"], "height_policy_prediction_only")
        self.assertGreater(np.linalg.norm(a["state"] - b["state"]), 1e-6)

    def test_pure_projection_and_future_changes_preserve_prior_snapshot(self):
        tracker = seeded("hold100")
        tracker.update(.12, [1., .1, .1])
        earlier = tracker.snapshot()
        predicted = tracker.predict_target_times([.18, .4])
        np.testing.assert_array_equal(earlier["state"], tracker.state)
        np.testing.assert_array_equal(earlier["covariance"], tracker.covariance)
        alternate = deepcopy(tracker)
        tracker.update(.16, [.1, .05, .15])
        alternate.update(.16, [.5, -.05, .18])
        # Forecasts partition RK4 steps at the newly advanced posterior clock;
        # only numerical integration/root-search roundoff may differ.
        np.testing.assert_allclose(predicted, tracker.predict_target_times([.18, .4]), atol=2e-6, rtol=0)
        np.testing.assert_allclose(predicted, alternate.predict_target_times([.18, .4]), atol=2e-6, rtol=0)
        self.assertEqual(earlier["t_s"], .12)

    def test_invalid_or_old_data_cannot_advance_hold_clock(self):
        tracker = seeded("hold100")
        tracker.update(.12, [1., .1, .1])
        earlier = tracker.snapshot()
        for timestamp, xyz in ((.12, [0, 0, 0]), (.11, [0, 0, 0]), (.2, [np.nan, 0, 0])):
            now = tracker.update(timestamp, xyz)
            np.testing.assert_array_equal(now["state"], earlier["state"])
            np.testing.assert_array_equal(now["covariance"], earlier["covariance"])
            self.assertEqual(now["t_s"], earlier["t_s"])
        with self.assertRaises(TypeError):
            tracker.update(.14, [0, 0, 0], reference_contact_t_s=.13)

    def test_different_maps_share_height_trigger_before_first_impact(self):
        results = []
        for model_id in HeightPolicyPredictor.model_ids:
            tracker = seeded("hold100")
            tracker.model_id = model_id
            results.append(tracker.update(.12, [.1, .01, .04]))
        self.assertEqual(len({r["height_contact_t_s"] for r in results}), 1)
        self.assertEqual(len({r["height_discovery_t_s"] for r in results}), 1)
        self.assertGreater(np.linalg.norm(results[-1]["v"] - results[0]["v"]), .1)


if __name__ == "__main__":
    unittest.main()
