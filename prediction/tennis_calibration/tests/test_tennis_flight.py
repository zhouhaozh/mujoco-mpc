import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tennis_calibration.flight import (
    _split_map, fit_shared_drag, forecast_from_initial_window,
    predicted_contact, simulate, simulate_batch_with_jacobian, state_at,
)


class TennisFlightTests(unittest.TestCase):
    def test_recovers_shared_drag_from_distinct_flights(self):
        coefficient = 0.022
        times = np.linspace(0, 0.65, 83)
        segments = []
        for i in range(5):
            state = np.array([i * 0.2, 0.3, 2.4, 8 + 2 * i, -1 + i * 0.7, 2 - i * 0.8])
            positions = simulate(times, state, coefficient, max_step_s=0.0007)[:, :3]
            segments.append((times + 3, positions))
        fit = fit_shared_drag(segments)
        self.assertTrue(fit["converged"], fit["message"])
        self.assertAlmostEqual(fit["k_drag_per_m"], coefficient, delta=0.0002)

    def test_heldout_initial_state_ignores_future_observations(self):
        times = np.linspace(0, 0.7, 281)
        state = np.array([0, 0, 2.1, 13, -2, 3.2])
        positions = simulate(times, state, 0.019)[:, :3]
        corrupt = positions.copy()
        corrupt[times > 0.08] += np.array([2.0, -1.0, 4.0])
        first = forecast_from_initial_window(times, positions, 0.019)
        second = forecast_from_initial_window(times, corrupt, 0.019)
        np.testing.assert_allclose(first["initial_state"], second["initial_state"], atol=0, rtol=0)
        np.testing.assert_allclose(first["prediction"], second["prediction"], atol=0, rtol=0)
        self.assertLess(np.max(first["position_errors_m"]), 1e-5)
        self.assertGreater(np.median(second["position_errors_m"]), 4)

    def test_raw_initial_window_ignores_future_derived_keep_flags(self):
        times = np.linspace(0, 0.7, 281)
        state = np.array([0, 0, 2.1, 13, -2, 3.2])
        positions = simulate(times, state, 0.019)[:, :3]
        retained_one = np.ones(len(times), dtype=bool)
        retained_two = retained_one.copy()
        retained_two[15:23] = False  # Offline future context might alter early keep flags.
        first = forecast_from_initial_window(
            times[retained_one], positions[retained_one], 0.019,
            initialization_observations=(times, positions))
        second = forecast_from_initial_window(
            times[retained_two], positions[retained_two], 0.019,
            initialization_observations=(times, positions))
        np.testing.assert_allclose(first["initial_state"], second["initial_state"], atol=0, rtol=0)
        self.assertEqual(first["t0_s"], second["t0_s"])

    def test_nonuniform_rk4_and_backward_endpoint_match(self):
        times = np.array([0, 0.0007, 0.011, 0.076, 0.23, 0.67])
        state = np.array([1.2, -0.4, 2, 22, -4, 2])
        predicted = simulate(times, state, 0.033)
        endpoint = state_at(times[-1], state, 0.033)
        np.testing.assert_allclose(predicted[-1], endpoint, atol=1e-7, rtol=0)
        recovered = state_at(-times[-1], endpoint, 0.033)
        np.testing.assert_allclose(recovered, state, atol=1e-7, rtol=0)

    def test_exact_discrete_sensitivities_match_central_differences(self):
        times = np.array([[0, 0.017, 0.13, 0.54], [0, 0.011, 0.27, 0.71]])
        initial = np.array([[1.0, -0.5, 2.0, 17.0, -3.0, 4.0],
                            [0.2, 0.1, 1.7, 9.0, 1.0, -1.5]])
        coefficient = 0.021
        positions, jacobian = simulate_batch_with_jacobian(times, initial, coefficient)
        for i in range(2):
            np.testing.assert_allclose(positions[i], simulate(times[i], initial[i], coefficient)[:, :3], atol=2e-8)
        step = 1e-6
        for parameter in range(7):
            plus, minus = initial.copy(), initial.copy()
            k_plus, k_minus = coefficient, coefficient
            if parameter < 6:
                plus[:, parameter] += step
                minus[:, parameter] -= step
            else:
                k_plus += step
                k_minus -= step
            p_plus, _ = simulate_batch_with_jacobian(times, plus, k_plus)
            p_minus, _ = simulate_batch_with_jacobian(times, minus, k_minus)
            finite_difference = (p_plus - p_minus) / (2 * step)
            np.testing.assert_allclose(jacobian[:, :, :, parameter], finite_difference,
                                       atol=2e-7, rtol=2e-6)

    def test_frozen_split_rejects_duplicate_full_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            path.write_text(json.dumps({"train": ["a/t.txt"], "test": ["a/t.txt"]}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                _split_map(path)

    def test_ground_contact_is_descending_ball_center_threshold(self):
        state = np.array([0, 0, 1.0, 2, 0, 0])
        result = predicted_contact(state, 0, ground_height_m=0.1, ball_radius_m=0.0335)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["position_m"][2], 0.1335, places=8)
        self.assertLess(result["velocity_m_s"][2], 0)


if __name__ == "__main__":
    unittest.main()
