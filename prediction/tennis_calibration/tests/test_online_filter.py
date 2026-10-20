import unittest

import numpy as np

from tennis_calibration.flight import simulate
from tennis_calibration.online_filter import (
    OnlineFilter, _rk4_with_transition, dynamics, dynamics_jacobian,
    forecast_batch, forecast_states_batch, landing_batch,
)


class OnlineFilterTests(unittest.TestCase):
    def make_filter(self, augmented=True, **config):
        return OnlineFilter(0.02, 0.83, 0.56, 0.03, config=config, augmented=augmented)

    def test_continuous_and_discrete_jacobians(self):
        for dimension in (6, 9):
            state = np.array([1, 2, 3, 12, -4, 2, 0.4, -0.3, 0.6])[:dimension]
            analytic = dynamics_jacobian(state, 0.02, 0.6)
            _, transition = _rk4_with_transition(state, 0.013, 0.02, 0.6)
            for j in range(dimension):
                plus, minus = state.copy(), state.copy()
                plus[j] += 1e-6
                minus[j] -= 1e-6
                finite = (dynamics(plus, 0.02, 0.6) - dynamics(minus, 0.02, 0.6)) / 2e-6
                np.testing.assert_allclose(analytic[:, j], finite, atol=1e-8, rtol=1e-7)
                finite_transition = (_rk4_with_transition(plus, 0.013, 0.02, 0.6)[0]
                                     - _rk4_with_transition(minus, 0.013, 0.02, 0.6)[0]) / 2e-6
                np.testing.assert_allclose(transition[:, j], finite_transition, atol=1e-8, rtol=1e-7)

    def test_causal_prefix_does_not_depend_on_future(self):
        times = np.linspace(0, 0.5, 201)
        positions = simulate(times, np.array([0, 0, 2.5, 10, -1, 1]), 0.02)[:, :3]
        corrupted = positions.copy()
        corrupted[times > 0.25] += [3, 8, -4]
        one, two = self.make_filter(), self.make_filter()
        snapshots = []
        for t, p, q in zip(times, positions, corrupted):
            first, second = one.update(t, p), two.update(t, q)
            if t <= 0.25 and first["initialized"]:
                np.testing.assert_array_equal(first["state"], second["state"])
                np.testing.assert_array_equal(first["covariance"], second["covariance"])
                snapshots.append(first)
        self.assertGreater(len(snapshots), 20)

    def test_nonuniform_capture_timestamps_recover_drag_state(self):
        rng = np.random.default_rng(14)
        times = np.r_[0, np.cumsum(rng.uniform(0.001, 0.007, 140))]
        truth = simulate(times, np.array([0.2, -0.7, 4.5, 14, -3, 0.5]), 0.02)
        for augmented in (False, True):
            estimator = self.make_filter(augmented)
            for t, state in zip(times, truth):
                result = estimator.update(t, state[:3])
            self.assertTrue(result["accepted"])
            np.testing.assert_allclose(result["state"][:6], truth[-1], atol=3e-5, rtol=0)
            self.assertGreaterEqual(np.linalg.eigvalsh(result["covariance"]).min(), -1e-12)

    def test_duplicate_old_and_invalid_observations_do_not_mutate_state(self):
        estimator = self.make_filter()
        for t in np.linspace(0, 0.2, 81):
            estimator.update(t, [10 * t, 0, 3 - 4.905 * t * t])
        snapshot = estimator.snapshot()
        for timestamp, point in [(0.2, [50, 50, 50]), (0.1, [1, 1, 1]), (np.nan, [0, 0, 0]),
                                 (0.22, [np.nan, 0, 0]), ("bad", [0, 0, 0]), (0.23, ["bad", 0, 0])]:
            result = estimator.update(timestamp, point)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["t_s"], snapshot["t_s"])
            np.testing.assert_array_equal(result["state"], snapshot["state"])

    def test_large_outlier_is_rejected_and_next_valid_measurement_recovers(self):
        estimator = self.make_filter()
        times = np.linspace(0, 0.4, 161)
        states = simulate(times, np.array([0, 0, 3, 12, 0.3, 1]), 0.02)
        for i, (t, state) in enumerate(zip(times, states)):
            point = state[:3].copy()
            if i == 100:
                point += [3, -4, 2]
            result = estimator.update(t, point)
            if i == 100:
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "mahalanobis_outlier")
        np.testing.assert_allclose(result["state"][:6], states[-1], atol=1e-4, rtol=0)

    def test_robust_initialization_ignores_one_position_spike(self):
        times = np.linspace(0, 0.12, 49)
        states = simulate(times, np.array([0, 0, 2, 15, -2, 0.5]), 0.02)
        estimator = self.make_filter()
        for i, (t, state) in enumerate(zip(times, states)):
            position = state[:3].copy()
            if i == 12:
                position += [0.25, -0.2, 0.15]
            result = estimator.update(t, position)
        self.assertLess(np.linalg.norm(result["p"] - states[-1, :3]), 0.005)
        self.assertLess(np.linalg.norm(result["v"] - states[-1, 3:]), 0.05)

    def test_sustained_tracking_change_reacquires_instead_of_gate_lock(self):
        estimator = self.make_filter(augmented=False)
        times = np.linspace(0, 0.65, 261)
        truth = simulate(times, np.array([0, 0, 4, 12, 1, 1]), 0.02)
        reacquired = False
        for t, state in zip(times, truth):
            observed = state[:3].copy()
            if t >= 0.25:
                observed += [0.4, -0.2, 0]
            result = estimator.update(t, observed)
            reacquired |= result["reason"] == "reacquired_from_causal_history"
        self.assertTrue(reacquired)
        self.assertTrue(result["accepted"])
        self.assertLess(np.linalg.norm(result["p"] - observed), 0.01)
        self.assertLess(np.linalg.norm(result["v"] - truth[-1, 3:6]), 0.05)

    def test_batch_matches_single_trajectory_and_shape(self):
        states = np.array([[0, 0, 2, 15, -2, -3, 0.8, 0.2, -0.1],
                           [0, 0, 0.1, 4, 1, -2, 0.3, -0.1, 0.5],
                           [1, 1, 0.2, 3, 1, 2, 0, 0, 0]])
        flags = np.array([False, False, True])
        horizons = np.array([0, 0.015, 0.14, 0.5])
        batch = forecast_states_batch(states, flags, horizons, 0.02, 0.83, 0.56, 0.03)
        positions = forecast_batch(states, flags, horizons, 0.02, 0.83, 0.56, 0.03)
        self.assertEqual(batch.shape, (3, 4, 9))
        np.testing.assert_array_equal(batch[..., :3], positions)
        for i in range(3):
            single = forecast_states_batch(states[i:i + 1], flags[i:i + 1], horizons, 0.02, 0.83, 0.56, 0.03)
            np.testing.assert_allclose(batch[i], single[0], atol=2e-12, rtol=0)

    def test_first_bounce_resets_velocity_once_and_residual(self):
        start = np.array([[0, 0, 0.03, 10, 2, -3, 4, -1, 2]])
        outgoing = forecast_states_batch(start, np.array([False]), [1e-7], 0, 0.8, 0.5, 0.03)[0, 0]
        np.testing.assert_allclose(outgoing[3:6], [5, 1, 2.4], atol=2e-6, rtol=0)
        np.testing.assert_array_equal(outgoing[6:9], [0, 0, 0])
        # The already-bounced flag prevents a second artificial reset at plane.
        no_second = forecast_states_batch(start, np.array([True]), [1e-7], 0, 0.8, 0.5, 0.03)[0, 0]
        self.assertLess(no_second[5], 0)

    def test_online_observed_bounce_and_no_smoothing_across_jump(self):
        times = np.linspace(0, 0.55, 221)
        start = np.array([[0, 0, 0.8, 7, 0.7, -1, 0, 0, 0]])
        states = forecast_states_batch(start, np.array([False]), times, 0.02, 0.83, 0.56, 0.03)[0]
        estimator = self.make_filter()
        for t, state in zip(times, states):
            result = estimator.update(t, state[:3])
        self.assertTrue(result["bounced"])
        self.assertTrue(result["bounce_confirmed"])
        self.assertTrue(result["accepted"])
        self.assertLess(np.linalg.norm(result["p"] - states[-1, :3]), 0.006)
        self.assertLess(np.linalg.norm(result["v"] - states[-1, 3:6]), 0.15)

    def test_projection_does_not_insert_measurement_or_change_posterior(self):
        estimator = self.make_filter()
        states = simulate(np.linspace(0, 0.15, 61), np.array([0, 0, 3, 12, 1, 1]), 0.02)
        for t, state in zip(np.linspace(0, 0.15, 61), states):
            estimator.update(t, state[:3])
        before = estimator.snapshot()
        projection = estimator.predict(0.3)
        self.assertFalse(projection["diagnostics"]["synthetic_measurement"])
        self.assertEqual(projection["t_s"], 0.3)
        self.assertIsNone(projection["covariance"])
        np.testing.assert_array_equal(estimator.snapshot()["state"], before["state"])
        self.assertEqual(estimator.last_t, 0.15)

    def test_landing_batch_matches_analytic_ballistic_contact(self):
        states = np.array([[0, 0, 2, 4, 1, 0], [1, 1, 0.5, 3, 0, 1], [0, 0, 1, 0, 0, 1]])
        result = landing_batch(states, np.array([False, False, True]), 0, 0.83, 0.56, 0.03)
        expected = np.sqrt(2 * (2 - 0.03) / 9.81)
        self.assertTrue(result["valid"][0])
        self.assertAlmostEqual(result["time_to_contact"][0], expected, delta=3e-7)
        np.testing.assert_allclose(result["position"][0], [4 * expected, expected, 0.03], atol=2e-6)
        self.assertFalse(result["valid"][2])
        self.assertEqual(result["reason"][2], "already_bounced")


if __name__ == "__main__":
    unittest.main()
