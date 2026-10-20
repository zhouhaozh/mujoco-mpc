"""Legacy observation-hold invariants; numerical execution is on Ubuntu."""
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.height_legacy_baseline import update_filter_only
from tennis_calibration.legacy_hold_adapter import LegacyHoldFilter, create_held_legacy
from tennis_calibration.online_features import FEATURE_SCHEMA
from tennis_calibration.online_filter import DEFAULT_CONFIG, OnlineFilter, forecast_states_batch


PARAMETERS = {"k_drag": .02, "e_n": .83, "alpha_t": .56, "contact_height": .03}


def seeded():
    tracker = LegacyHoldFilter(**PARAMETERS)
    tracker.state = np.array([0., 0., .04, 3., .2, -2.])
    tracker.covariance = np.eye(6) * 1e-4
    tracker.last_t = .1
    return tracker


class LegacyHoldAdapterTests(unittest.TestCase):
    def test_normal_updates_match_original_including_contact_and_outliers(self):
        original = OnlineFilter(**PARAMETERS, augmented=False)
        adapter = LegacyHoldFilter(**PARAMETERS)
        times = np.arange(0., .60001, .005)
        truth = forecast_states_batch(np.array([[0., 0., .8, 7., .7, -1.]]),
                                      np.array([False]), times, **PARAMETERS)[0]
        for index, (timestamp, state) in enumerate(zip(times, truth)):
            point = state[:3].copy()
            if index in (31, 82):
                point += [.5, -.2, .15]
            left, right = original.update(timestamp, point), adapter.update(timestamp, point)
            for field in ("initialized", "accepted", "reason", "t_s", "bounced", "bounce_confirmed"):
                self.assertEqual(left[field], right[field])
            if left["initialized"]:
                np.testing.assert_array_equal(left["state"], right["state"])
                np.testing.assert_array_equal(left["covariance"], right["covariance"])

    def test_held_measurements_cannot_change_state_covariance_or_physics_projection(self):
        left, right = seeded(), seeded()
        left.prediction_only = right.prediction_only = True
        with patch.object(left, "_confirm_bounce", side_effect=AssertionError("hold recovery")), \
                patch.object(left, "_reacquire_from_history", side_effect=AssertionError("hold recovery")):
            for timestamp in (.12, .14, .16, .18, .20):
                point = left.predict(timestamp)["p"]
                a, b = left.update(timestamp, point), right.update(timestamp, point + [.02, -.01, .005])
                self.assertFalse(a["accepted"])
                self.assertEqual(a["reason"], "legacy_height_policy_prediction_only")
                self.assertEqual(a["t_s"], timestamp)
                np.testing.assert_array_equal(a["state"], b["state"])
                np.testing.assert_array_equal(a["covariance"], b["covariance"])
                np.testing.assert_array_equal(left.predict(.4)["state"], right.predict(.4)["state"])
        self.assertTrue(a["bounced"])
        self.assertGreater(len(left._recent), 3)
        self.assertFalse(np.array_equal(left._recent[-1][1], right._recent[-1][1]))

    def test_duplicate_invalid_and_future_points_cannot_change_earlier_hold_posterior(self):
        tracker = seeded()
        tracker.update(.12, [.1, 0., .08], prediction_only=True)
        earlier = tracker.snapshot()
        for timestamp, point in ((.12, [0., 0., 0.]), (.11, [0., 0., 0.]),
                                 (.2, [np.nan, 0., 0.]), (np.nan, [0., 0., 0.])):
            now = tracker.update(timestamp, point, prediction_only=True)
            self.assertEqual(now["t_s"], earlier["t_s"])
            np.testing.assert_array_equal(now["state"], earlier["state"])
            np.testing.assert_array_equal(now["covariance"], earlier["covariance"])
        changed = deepcopy(tracker)
        frozen_state, frozen_covariance = earlier["state"].copy(), earlier["covariance"].copy()
        tracker.predict(.4)
        self.assertEqual(tracker.last_t, .12)
        np.testing.assert_array_equal(tracker.state, frozen_state)
        np.testing.assert_array_equal(tracker.covariance, frozen_covariance)
        tracker.update(.16, [.2, 0., .12], prediction_only=True)
        changed.update(.16, [.8, -.2, .3], prediction_only=True)
        np.testing.assert_array_equal(earlier["state"], frozen_state)
        np.testing.assert_array_equal(earlier["covariance"], frozen_covariance)
        self.assertEqual(earlier["t_s"], .12)
        np.testing.assert_array_equal(tracker.state, changed.state)
        np.testing.assert_array_equal(tracker.covariance, changed.covariance)

    def test_resume_fuses_current_frame_once_and_retains_received_mlp_history(self):
        predictor = SimpleNamespace(filter=seeded(), _history=deque())
        predictor.filter.prediction_only = True
        for timestamp in (.12, .14, .16, .18, .20):
            point = predictor.filter.predict(timestamp)["p"]
            update_filter_only(predictor, timestamp, point)
        self.assertEqual([t for t, _ in predictor._history], [.12, .14, .16, .18, .20])
        expected = deepcopy(predictor.filter)
        predictor.filter.prediction_only = expected.prediction_only = False
        timestamp = .22
        point = expected.predict(timestamp)["p"] + [.001, 0., 0.]
        expected_snapshot = OnlineFilter.update(expected, timestamp, point)
        with patch.object(predictor.filter, "_advance", wraps=predictor.filter._advance) as advance:
            result = update_filter_only(predictor, timestamp, point)
        self.assertEqual(advance.call_count, 1)
        self.assertAlmostEqual(advance.call_args.args[0], .02)
        self.assertTrue(result["accepted"])
        np.testing.assert_array_equal(result["state"], expected_snapshot["state"])
        np.testing.assert_array_equal(result["covariance"], expected_snapshot["covariance"])
        self.assertEqual(sum(t == timestamp for t, _ in predictor._history), 1)
        self.assertEqual(predictor.filter._recent[-1][0], timestamp)

    def test_helper_keeps_bundle_contract_and_warmup_hold_is_explicitly_unsupported(self):
        config = {"physical": PARAMETERS, "filter": DEFAULT_CONFIG, "augmented": False,
                  "selected_filter": "physics6", "position_mlp_enabled": False,
                  "landing_mlp_enabled": False, "supported_horizons_s": [.1, .2, .5],
                  "trained_delays_s": [0.], "feature_schema": FEATURE_SCHEMA}
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(config))
            predictor = create_held_legacy(directory)
        self.assertIsInstance(predictor.filter, LegacyHoldFilter)
        self.assertEqual(predictor.filter.dimension, 6)
        self.assertEqual(predictor.config, config)
        with self.assertRaisesRegex(ValueError, "prior initialization"):
            predictor.filter.update(0., [0., 0., 1.], prediction_only=True)
        self.assertFalse(predictor.filter.prediction_only)
        self.assertIsNone(predictor.filter.state)
        self.assertEqual(len(predictor.filter._initial), 0)


if __name__ == "__main__":
    unittest.main()
