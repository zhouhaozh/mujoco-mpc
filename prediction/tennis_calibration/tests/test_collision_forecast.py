"""Synthetic absolute-clock contact prediction tests, executed on Ubuntu."""
import unittest

import numpy as np

from tennis_calibration.collision_forecast import hybrid_forecast
from tennis_calibration import identified_dynamics as d


class HybridForecastTest(unittest.TestCase):
    def setUp(self):
        self.p = {"k_drag": 0.0}
        self.capture = np.array([7.0, 8.0])
        self.contact = np.array([7.20, 8.30])
        self.initial = np.array([[1., 2., 1., 10., 2., -2., 0., 0., 0.],
                                 [-1., 3., 2., -5., 1., -3., 0., 0., 0.]])
        self.minus = np.stack([d.step(x, t-c, 0) for x, t, c in
                                zip(self.initial, self.contact, self.capture)])
        self.plus = self.minus.copy()
        self.plus[:, 3:5] *= .55
        self.plus[:, 5] *= -.8
        self.targets = np.array([[7., 7.1, 7.2, 7.25, 7.5],
                                 [8., 8.2, 8.3, 8.37, 8.6]])

    def call(self, **kwargs):
        inputs = {"capture_states": self.initial, "capture_times": self.capture,
                  "contact_states_minus": self.minus, "contact_times": self.contact,
                  "states_plus": self.plus, "target_times": self.targets, "parameters": self.p}
        inputs.update(kwargs)
        return hybrid_forecast(**inputs)

    def analytical(self, state, delta):
        result = state.copy()
        result[:3] += delta * state[3:6] + .5 * d.GRAVITY * delta**2
        result[3:6] += d.GRAVITY * delta
        return result

    def test_exact_gravity_before_and_after_event(self):
        pred = self.call()
        for row in range(2):
            for column, target in enumerate(self.targets[row]):
                initial, origin = ((self.initial[row], self.capture[row])
                    if target < self.contact[row] else (self.plus[row], self.contact[row]))
                np.testing.assert_allclose(pred[row, column],
                    self.analytical(initial, target-origin), atol=1e-12)

    def test_late_predicted_impact_remains_preimpact_at_requested_clock(self):
        # Evaluation asks at 7.15; predicted impact is 7.20. Do not manufacture
        # an after-impact state merely because the requested clock came from an
        # externally annotated true-contact-plus-h evaluation timestamp.
        target = np.array([[7.15], [8.25]])
        pred = self.call(target_times=target)
        self.assertLess(pred[0, 0, 5], 0)
        self.assertLess(pred[1, 0, 5], 0)
        np.testing.assert_allclose(pred[0, 0], self.analytical(self.initial[0], .15), atol=1e-12)

    def test_at_event_is_postimpact_and_position_continuous(self):
        pred = self.call(target_times=self.contact[:, None])
        np.testing.assert_allclose(pred[:, 0], self.plus, atol=1e-12)
        np.testing.assert_allclose(pred[:, 0, :3], self.minus[:, :3], atol=1e-12)

    def test_batch_matches_independent_cases_with_irregular_clock(self):
        parameters = {"k_drag": .02043211081206929}
        pred = self.call(parameters=parameters)
        for row in range(2):
            single = hybrid_forecast(self.initial[row:row+1], self.capture[row:row+1],
                self.minus[row:row+1], self.contact[row:row+1], self.plus[row:row+1],
                self.targets[row:row+1], parameters)
            np.testing.assert_allclose(pred[row], single[0], atol=1e-9)

    def test_no_second_impact_is_introduced(self):
        target = self.contact[:, None] + 2.0
        pred = self.call(target_times=target)
        for row in range(2):
            np.testing.assert_allclose(pred[row, 0], self.analytical(self.plus[row], 2), atol=1e-10)
            self.assertLess(pred[row, 0, 5], 0)

    def test_zero_time_immediate_event_and_repeated_targets(self):
        pred = self.call(contact_times=self.capture.copy(),
                         target_times=np.repeat(self.capture[:, None], 3, axis=1))
        np.testing.assert_array_equal(pred, np.repeat(self.plus[:, None], 3, axis=1))

    def test_inputs_are_not_mutated(self):
        arrays = [self.initial, self.capture, self.minus, self.contact, self.plus, self.targets]
        copies = [array.copy() for array in arrays]
        self.call()
        for array, before in zip(arrays, copies):
            np.testing.assert_array_equal(array, before)

    def test_nonfinite_values_are_rejected(self):
        for key, value in (("capture_states", self.initial), ("capture_times", self.capture),
                           ("contact_states_minus", self.minus), ("contact_times", self.contact),
                           ("states_plus", self.plus), ("target_times", self.targets)):
            broken = value.copy(); broken.flat[0] = np.nan
            with self.assertRaises(ValueError): self.call(**{key: broken})

    def test_time_and_shape_validation(self):
        invalid = [
            {"capture_states": self.initial[:, :6]},
            {"contact_states_minus": self.minus[:1]},
            {"capture_times": self.capture[:, None]},
            {"contact_times": self.capture-.1},
            {"target_times": self.capture[:, None]-.1},
            {"target_times": self.targets[:, ::-1]},
            {"target_times": np.empty((2, 0))},
            {"parameters": {"k_drag": -1}},
            {"max_step_s": 0},
        ]
        for kwargs in invalid:
            with self.assertRaises(ValueError): self.call(**kwargs)


if __name__ == "__main__":
    unittest.main()
