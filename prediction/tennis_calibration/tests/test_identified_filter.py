import unittest

import numpy as np

from tennis_calibration import identified_dynamics as physics
from tennis_calibration.identified_filter import (
    IdentifiedFilter, _local_fit, _step_with_transition,
    dynamics, dynamics_jacobian, forecast_states_batch, forecast_batch, landing_batch,
)

PARAMETERS = {"k_drag": .02, "e_n": .83, "alpha_t": .56, "impact_mode": "effective"}
HEIGHT = .03


class IdentifiedFilterTests(unittest.TestCase):
    def make_filter(self, augmented=True, **config):
        return IdentifiedFilter(PARAMETERS, HEIGHT, config, augmented)

    def truth(self, times, start):
        return forecast_states_batch(np.asarray(start)[None], np.array([False]), times, PARAMETERS, HEIGHT)[0]

    def test_continuous_and_same_rk4_transition_jacobians(self):
        for size in (6, 9):
            state = np.array([1, 2, 3, 12, -4, 2, .4, -.3, .6])[:size]
            analytic = dynamics_jacobian(state, PARAMETERS)
            _, transition = _step_with_transition(state, .013, PARAMETERS)
            for axis in range(size):
                plus, minus = state.copy(), state.copy()
                plus[axis] += 1e-6
                minus[axis] -= 1e-6
                finite = (dynamics(plus, PARAMETERS) - dynamics(minus, PARAMETERS)) / 2e-6
                np.testing.assert_allclose(analytic[:, axis], finite, atol=1e-8, rtol=1e-7)
                finite = (_step_with_transition(plus, .013, PARAMETERS)[0]
                          - _step_with_transition(minus, .013, PARAMETERS)[0]) / 2e-6
                np.testing.assert_allclose(transition[:, axis], finite, atol=1e-8, rtol=1e-7)

    def test_prefix_causality_and_no_offline_contact_labels(self):
        times = np.linspace(0, .5, 201)
        states = self.truth(times, [0, 0, 3, 10, 1, 1, .1, .2, -.1])
        one, two = self.make_filter(), self.make_filter()
        for timestamp, state in zip(times, states):
            changed = state[:3].copy()
            if timestamp > .25:
                changed += [3, 5, -7]
            a, b = one.update(timestamp, state[:3]), two.update(timestamp, changed)
            if timestamp <= .25 and a["initialized"]:
                np.testing.assert_array_equal(a["state"], b["state"])
                np.testing.assert_array_equal(a["covariance"], b["covariance"])

    def test_zero_magnus_irregular_timestamp_and_covariance_psd(self):
        rng = np.random.default_rng(14)
        times = np.r_[0, np.cumsum(rng.uniform(.001, .007, 100))]
        states = self.truth(times, [0, .4, 4, 12, -2, 1, 0, 0, 0])
        for augmented in (False, True):
            estimator = self.make_filter(augmented)
            for timestamp, state in zip(times, states):
                result = estimator.update(timestamp, state[:3])
            self.assertTrue(result["accepted"])
            np.testing.assert_allclose(result["state"][:6], states[-1, :6], atol=5e-5, rtol=0)
            self.assertGreaterEqual(np.linalg.eigvalsh(result["covariance"]).min(), -1e-12)
            self.assertEqual(result["b_eff_units"], "1/s")

    def test_effective_magnus_tracking_forecast_without_true_spin_claim(self):
        times = np.linspace(0, .7, 281)
        states = self.truth(times, [0, 0, 4, 12, -1, 2, .2, -.15, .1])
        estimator = self.make_filter()
        for timestamp, state in zip(times, states):
            result = estimator.update(timestamp, state[:3])
        forecast = estimator.predict(.8)
        target = self.truth(np.array([.8]), [0, 0, 4, 12, -1, 2, .2, -.15, .1])[0]
        self.assertLess(np.linalg.norm(forecast["p"] - target[:3]), .025)
        self.assertEqual(result["state_definition"], "p_v_effective_magnus")
        np.testing.assert_allclose(result["a_res"], np.cross(result["b_eff"], result["v"]))

    def test_duplicate_invalid_and_prediction_do_not_mutate_posterior(self):
        times = np.linspace(0, .2, 81)
        states = self.truth(times, [0, 0, 4, 12, 1, 1, 0, 0, 0])
        estimator = self.make_filter()
        for timestamp, state in zip(times, states):
            estimator.update(timestamp, state[:3])
        before = estimator.snapshot()
        for timestamp, point in [(.2, [9, 9, 9]), (.1, [0, 0, 0]), (.21, [np.nan, 0, 0]), (np.nan, [1, 2, 3])]:
            result = estimator.update(timestamp, point)
            self.assertFalse(result["accepted"])
            np.testing.assert_array_equal(result["state"], before["state"])
            self.assertEqual(result["t_s"], before["t_s"])
        predicted = estimator.predict(.4)
        self.assertIsNone(predicted["covariance"])
        self.assertFalse(predicted["diagnostics"]["synthetic_measurement"])
        np.testing.assert_array_equal(estimator.snapshot()["state"], before["state"])

    def test_outlier_gate_and_stable_camera_offset_recovery(self):
        times = np.linspace(0, .65, 261)
        states = self.truth(times, [0, 0, 4, 12, 1, 1, 0, 0, 0])
        estimator = self.make_filter(augmented=False)
        recovered = False
        for timestamp, state in zip(times, states):
            point = state[:3].copy()
            if abs(timestamp - .20) < 1e-9:
                point += [3, -4, 2]
            if timestamp >= .25:
                point += [.4, -.2, 0]
            result = estimator.update(timestamp, point)
            if abs(timestamp - .20) < 1e-9:
                self.assertFalse(result["accepted"])
            recovered |= result["reason"] == "reacquired_from_causal_history"
        self.assertTrue(recovered)
        self.assertTrue(result["accepted"])
        self.assertLess(np.linalg.norm(result["p"] - point), .01)
        self.assertLess(np.linalg.norm(result["v"] - states[-1, 3:6]), .15)

    def test_short_four_point_postbounce_window_never_installs_negative_26_velocity(self):
        estimator = self.make_filter(augmented=False)
        estimator.state = np.array([0, 0, .08, 6, .3, 3])
        estimator.covariance = np.eye(6) * .01
        estimator.bounced = True
        estimator.last_t = .01
        # A channel discontinuity over 7.4 ms resembles a -26 m/s slope.
        before_times = np.linspace(-.035, -.002, 14)
        after_times = np.linspace(.001, .0084, 4)
        for timestamp in before_times:
            estimator._recent.append((float(timestamp), np.array([6*timestamp, .3*timestamp, HEIGHT - 4*timestamp])))
        estimator._recent.append((0., np.array([0, 0, HEIGHT])))
        for timestamp in after_times:
            estimator._recent.append((float(timestamp), np.array([-.026/.001*timestamp, .3*timestamp, HEIGHT + 3*timestamp])))
        before = estimator.state.copy()
        self.assertFalse(estimator._confirm_bounce())
        np.testing.assert_array_equal(estimator.state, before)
        self.assertFalse(estimator.bounce_confirmed)
        self.assertGreater(estimator.state[3], 0)

    def test_postbounce_channel_glitch_does_not_reverse_velocity_or_lock_gate(self):
        times = np.linspace(0, .65, 261)
        states = self.truth(times, [0, 0, .8, 9, .7, -1, 0, 0, 0])
        landing = landing_batch(np.array([[0, 0, .8, 9, .7, -1, 0, 0, 0]]), np.array([False]), PARAMETERS, HEIGHT)
        contact = float(landing["time_to_contact"][0])
        for augmented in (False, True):
            estimator = self.make_filter(augmented)
            minimum_vx = np.inf
            saw_glitch = False
            for timestamp, state in zip(times, states):
                point = state[:3].copy()
                if .001 < timestamp - contact < .010:
                    point[0] -= .08 + 26 * (timestamp - contact)
                    saw_glitch = True
                result = estimator.update(timestamp, point)
                if result["initialized"]:
                    minimum_vx = min(minimum_vx, result["v"][0])
            self.assertTrue(saw_glitch)
            self.assertGreater(minimum_vx, 0.)
            self.assertTrue(result["accepted"])
            self.assertTrue(result["bounce_confirmed"])
            self.assertLess(np.linalg.norm(result["v"] - states[-1, 3:6]), .4)
            self.assertLess(np.linalg.norm(result["p"] - states[-1, :3]), .02)

    def test_uncertain_inconsistent_fit_retains_physical_prior(self):
        estimator = self.make_filter(augmented=False)
        estimator.state = np.array([1, 0, .1, 6, .5, 3])
        estimator.covariance = np.eye(6) * .01
        estimator.bounced = True
        fit = {"position": np.array([1., 0., .1]), "velocity": np.array([-26., .5, 3.]),
               "covariance": np.eye(6)*.001, "span_s": .035, "samples": 16, "velocity_sigma": np.ones(3)*.2}
        before = estimator.state.copy()
        self.assertFalse(estimator._fuse_local(fit))
        np.testing.assert_array_equal(estimator.state, before)
        self.assertEqual(estimator._diagnostics["recovery_rejected"], "velocity_inconsistent_with_physical_prior")

    def test_good_observed_bounce_fuses_gradually_after_sufficient_span(self):
        times = np.linspace(0, .6, 241)
        states = self.truth(times, [0, 0, .8, 7, .7, -1, 0, 0, 0])
        estimator = self.make_filter()
        confirmed_at, fused_span = None, None
        for timestamp, state in zip(times, states):
            result = estimator.update(timestamp, state[:3])
            if result["diagnostics"].get("observed_bounce_confirmed_this_frame"):
                confirmed_at = timestamp
                fused_span = result["diagnostics"]["recovery_fit_span_s"]
                self.assertFalse(result["diagnostics"]["observed_bounce_reset"])
                self.assertLessEqual(result["diagnostics"]["recovery_velocity_gain_norm"], .6 + 1e-8)
        self.assertIsNotNone(confirmed_at)
        self.assertGreaterEqual(fused_span, .03 - 1e-12)
        self.assertTrue(result["accepted"])
        self.assertLess(np.linalg.norm(result["v"] - states[-1, 3:6]), .15)

    def test_batch_matches_single_landing_and_one_impact_only(self):
        states = np.array([[0, 0, 2, 12, 1, -3, .2, -.1, .3], [0, 0, .1, 3, 1, -2, 0, 0, 0]])
        horizons = np.array([0, .013, .15, .5])
        batch = forecast_states_batch(states, np.array([False, False]), horizons, PARAMETERS, HEIGHT)
        np.testing.assert_array_equal(forecast_batch(states, np.array([False, False]), horizons, PARAMETERS, HEIGHT), batch[..., :3])
        for index in range(2):
            one = forecast_states_batch(states[index:index+1], np.array([False]), horizons, PARAMETERS, HEIGHT)
            np.testing.assert_allclose(batch[index], one[0], atol=2e-12, rtol=0)
        ballistic = {**PARAMETERS, "k_drag": 0}
        contact = landing_batch(np.array([[0, 0, 2, 4, 1, 0]]), np.array([False]), ballistic, HEIGHT)
        expected = np.sqrt(2*(2-HEIGHT)/9.81)
        self.assertAlmostEqual(contact["time_to_contact"][0], expected, delta=3e-7)
        start = np.array([[0, 0, HEIGHT, 10, 2, -3, .2, -.1, .3]])
        outgoing = forecast_states_batch(start, np.array([False]), [1e-7], ballistic, HEIGHT)[0, 0]
        np.testing.assert_allclose(outgoing[3:6], [5.6, 1.12, 2.49], atol=3e-6, rtol=0)
        np.testing.assert_array_equal(outgoing[6:9], start[0, 6:9])
        no_second = forecast_states_batch(start, np.array([True]), [1e-7], ballistic, HEIGHT)[0, 0]
        self.assertLess(no_second[5], 0)

    def test_local_acceleration_compensation_and_uncertainty_not_zero(self):
        times = np.linspace(1., 1.04, 17)
        dt = times - times[-1]
        velocity, position, acceleration = np.array([7, .3, 2]), np.array([1, 2, .1]), np.array([-.5, -.02, -9.81])
        points = position + dt[:, None]*velocity + .5*dt[:, None]**2*acceleration
        fit = _local_fit(times, points, np.array([.006, .006, .008]), acceleration)
        np.testing.assert_allclose(fit["velocity"], velocity, atol=1e-12)
        self.assertGreater(fit["velocity_sigma"].min(), 0)
        self.assertLess(fit["consistency_m_s"], 1e-10)


if __name__ == "__main__":
    unittest.main()
