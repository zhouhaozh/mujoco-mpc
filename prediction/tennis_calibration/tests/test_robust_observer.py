"""Causal/gain invariants for the isolated observer; execute on Ubuntu."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.height_policy import HeightPolicyPredictor
from tennis_calibration.identified_dynamics import impact_map
from tennis_calibration.robust_observer import RobustHeightPredictor, EXPERIMENTS


class SyntheticBank:
    parameters = {"k_drag": 0., "e_n": .83, "alpha_t": .56, "impact_mode": "effective"}
    height = .03

    def apply_state(self, incoming, model_id):
        result = impact_map(incoming, self.parameters)
        result[3] += .2
        return result


def seeded(experiment="adaptive_protected_smooth", near=False):
    tracker = RobustHeightPredictor(SyntheticBank(), experiment=experiment)
    state = np.array([0., 0., .04 if near else 1., 3., .2, -2., 0., 0., 0.])
    times = np.array([.06, .08, .1])
    xyz = state[:3] + (times - .1)[:, None] * state[3:6]
    tracker.seed_from_prefix({"state": state, "covariance": np.eye(9) * 1e-4,
                              "capture_t_s": .1, "converged": True}, times, xyz)
    return tracker


class RobustObserverTests(unittest.TestCase):
    def test_baseline_delegates_exact_state_covariance_and_event_clock(self):
        original = HeightPolicyPredictor(SyntheticBank(), "gp", "hold100")
        experimental = RobustHeightPredictor(SyntheticBank(), experiment="baseline")
        for t in np.arange(0., .601, .0025):
            xyz = np.array([3. * t, .2 * t, .8 - t - 4.905 * t * t])
            if t > .25:
                xyz[0] += .05
            a, b = original.update(float(t), xyz), experimental.update(float(t), xyz)
            for key in ("initialized", "accepted", "reason", "bounced", "height_contact_t_s", "resume_t_s"):
                self.assertEqual(a[key], b[key])
            if a["initialized"]:
                np.testing.assert_array_equal(a["state"], b["state"])
                np.testing.assert_array_equal(a["covariance"], b["covariance"])

    def test_all_candidates_hold_raw_data_without_assimilating(self):
        for experiment in EXPERIMENTS:
            a, b = seeded(experiment, near=True), seeded(experiment, near=True)
            for t in (.12, .14, .16, .18, .20):
                p = a.predict_target_times([t])[0, :3]
                with patch.object(a, "_confirm_bounce", side_effect=AssertionError("fusion in hold")):
                    x = a.update(t, p)
                y = b.update(t, p + [.02, -.02, .01])
                self.assertTrue(x["hold_active"])
                np.testing.assert_array_equal(x["state"], y["state"])
                np.testing.assert_array_equal(x["covariance"], y["covariance"])
            self.assertFalse(np.array_equal(a._recent[-1][1], b._recent[-1][1]))

    def test_protection_only_unchanged_outside_recovery_window(self):
        a, b = seeded("baseline"), seeded("protected_only")
        for t in np.arange(.1025, .2, .0025):
            p = a.predict_target_times([t])[0, :3] + [.001, -.002, .001]
            x, y = a.update(float(t), p), b.update(float(t), p)
            np.testing.assert_array_equal(x["state"], y["state"])
            np.testing.assert_array_equal(x["covariance"], y["covariance"])

    def test_axis_huber_weight_downweights_only_outlying_direction(self):
        tracker = seeded("huber_only")
        tracker.covariance = np.eye(9) * 1e-6
        measurement = tracker._measurement_covariance(np.array([.08, .001, .001]))
        weights = tracker._diagnostics["huber_axis_weights"]
        self.assertLess(weights[0], .25)
        self.assertEqual(weights[1:], [1., 1.])
        self.assertGreater(measurement[0, 0], tracker.config["measurement_sigma_xyz"][0] ** 2)
        self.assertEqual(measurement[1, 1], tracker.config["measurement_sigma_xyz"][1] ** 2)

    def test_protected_gain_budget_spans_both_corrections_and_joseph_is_actual(self):
        tracker = seeded("protected_only")
        tracker.resume_t_s = tracker.last_t
        tracker._start_recovery_budget()
        initial_state, prior = tracker.state.copy(), tracker.covariance.copy()
        gain = np.zeros((9, 3))
        gain[:3] = np.eye(3) * .5
        gain[3:6] = np.eye(3) * 100.
        innovation = np.array([.2, 0., 0.])
        protected = tracker._protected_gain(gain, innovation)
        self.assertLessEqual(np.linalg.norm(protected[3:6] @ innovation), .2 + 1e-12)
        noise = np.eye(3) * .001
        tracker._apply_gain(protected, innovation, noise)
        ikh = np.eye(9)
        ikh[:, :3] -= protected
        expected = ikh @ prior @ ikh.T + protected @ noise @ protected.T
        np.testing.assert_allclose(tracker.covariance, expected, atol=1e-14)
        np.testing.assert_allclose(tracker.state, initial_state + protected @ innovation)
        self.assertGreaterEqual(np.linalg.eigvalsh(tracker.covariance).min(), -1e-12)
        second = tracker._protected_gain(gain, -innovation)
        self.assertLess(np.linalg.norm(second[3:6] @ innovation), 1e-12)
        self.assertAlmostEqual(tracker._diagnostics["recovery_delta_v_spent_m_s"], .2)

    def test_local_fusion_respects_shared_frame_velocity_budget(self):
        tracker = seeded("protected_only")
        tracker.resume_t_s = tracker.last_t
        tracker._start_recovery_budget()
        tracker.covariance = np.eye(9)
        before = tracker.state[3:6].copy()
        fit = {"position": tracker.state[:3].copy(), "velocity": tracker.state[3:6] + [2., 1., .5],
               "covariance": np.eye(6) * .01, "span_s": .04, "samples": 18,
               "velocity_sigma": np.ones(3) * .1}
        self.assertTrue(tracker._fuse_local(fit))
        self.assertLessEqual(np.linalg.norm(tracker.state[3:6] - before), .2 + 1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(tracker.covariance).min(), -1e-12)

    def test_endpoint_smoothing_is_causal_and_retains_raw_history(self):
        tracker = seeded("smooth_only")
        tracker._recent.clear()
        for t in np.arange(.1, .14001, .0025):
            p = np.array([3. * t, .2 * t, 1. - 2. * t - 4.905 * t * t])
            if abs(t - .14) < 1e-9:
                p += [.009, 0., 0.]
            result = tracker.update(float(t), p)
        self.assertTrue(result["diagnostics"]["smoothing_applied"])
        np.testing.assert_array_equal(tracker._recent[-1][1], p)
        self.assertLessEqual(result["diagnostics"]["smoothing_history_end_s"], result["t_s"])
        old_state, old_cov = result["state"].copy(), result["covariance"].copy()
        future = deepcopy(tracker)
        future.update(.15, [4., 4., 4.])
        np.testing.assert_array_equal(result["state"], old_state)
        np.testing.assert_array_equal(result["covariance"], old_cov)
        np.testing.assert_array_equal(tracker.state, old_state)

    def test_smoothing_never_uses_precontact_points_after_height_event(self):
        tracker = seeded("smooth_only")
        tracker.height_contact_t_s = .095
        tracker.last_t = .12
        tracker._recent.clear()
        for t in np.arange(.080, .121, .0025):
            tracker._recent.append((float(t), np.array([t, t, .2])))
        history = tracker._eligible_history(.1, include_current=True)
        self.assertTrue(all(t >= .1 - 1e-12 for t, _ in history))
        tracker.height_contact_t_s = None
        raw = np.array([0., 0., .04])
        np.testing.assert_array_equal(tracker._smoothed_endpoint(raw), raw)
        self.assertEqual(tracker._diagnostics["smoothing_bypassed"], "height_event_guard")

    def test_height_guard_sees_recent_low_point_even_when_current_endpoint_high(self):
        tracker = seeded("smooth_only")
        tracker.height_contact_t_s = .060
        tracker.last_t = .120
        tracker.state[2], tracker.state[5] = .2, 10.
        tracker._recent.clear()
        tracker._recent.append((.110, np.array([0., 0., .04])))
        raw = np.array([0., 0., .20])
        np.testing.assert_array_equal(tracker._smoothed_endpoint(raw), raw)
        self.assertEqual(tracker._diagnostics["smoothing_bypassed"], "height_event_guard")
        tracker.height_contact_t_s = None
        tracker._recent.clear()
        tracker.state[2], tracker.state[5] = .15, -10.
        raw = np.array([0., 0., .20])
        np.testing.assert_array_equal(tracker._smoothed_endpoint(raw), raw)
        self.assertEqual(tracker._diagnostics["smoothing_bypassed"], "height_event_guard")

    def test_invalid_observation_does_not_move_clock_or_change_state(self):
        tracker = seeded()
        before = tracker.snapshot()
        for t, p in ((.11, [np.nan, 0., 0.]), (.1, [0., 0., 0.]), (.09, [0., 0., 0.])):
            result = tracker.update(t, p)
            self.assertEqual(before["t_s"], result["t_s"])
            np.testing.assert_array_equal(before["state"], result["state"])
            np.testing.assert_array_equal(before["covariance"], result["covariance"])

    def test_noise_proxy_is_copied_bounded_and_quality_excludes_current(self):
        sigma = np.array([1., .0001, .008])
        tracker = RobustHeightPredictor(SyntheticBank(), experiment="adaptive_huber", noise_sigma_xyz=sigma)
        sigma[:] = 7.
        np.testing.assert_array_equal(tracker.noise_proxy_sigma_xyz, [1., .0001, .008])
        tracker.state, tracker.covariance, tracker.last_t = np.zeros(9), np.eye(9), .1
        tracker.state[2] = 1.
        before = tracker._measurement_covariance(np.zeros(3))
        tracker._recent.append((.1, np.ones(3) * 1000.))
        after = tracker._measurement_covariance(np.zeros(3))
        np.testing.assert_array_equal(before, after)
        np.testing.assert_allclose(np.sqrt(np.diag(after)), [.012, .006, .008])


if __name__ == "__main__":
    unittest.main()
