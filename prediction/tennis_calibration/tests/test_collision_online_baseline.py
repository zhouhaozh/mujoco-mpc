import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.collision_online_baseline import predict_frozen, snapshot_prefixes


class CollisionOnlineBaselineTests(unittest.TestCase):
    def selected(self):
        return {"parameters": {"k_drag": 0., "e_n": .8, "alpha_t": .6, "impact_mode": "effective"},
                "contact_height": 0., "filter": {}, "augmented": True}

    def test_future_observation_changes_do_not_change_frozen_prefixes(self):
        times = np.arange(0, .401, .005)
        xyz = np.column_stack((times, np.zeros(len(times)), 2.+times-4.905*times**2))
        cuts = [times[44], times[24]]
        first, _ = snapshot_prefixes(times, xyz, cuts, self.selected())
        altered = xyz.copy()
        altered[times>max(cuts)] += 100.
        second, _ = snapshot_prefixes(times, altered, cuts, self.selected())
        for a, b, cut in zip(first, second, cuts):
            self.assertLessEqual(a["input_end_t_s"], cut)
            self.assertEqual(a["input_sha256"], b["input_sha256"])
            np.testing.assert_array_equal(a["snapshot"]["state"], b["snapshot"]["state"])

    def test_already_bounced_does_not_search_second_impact(self):
        snapshot = {"initialized": True, "state": [0., 0., .1, 1., 0., 3., 0., 0., 0.],
                    "t_s": 1., "bounced": True}
        with patch("tennis_calibration.collision_online_baseline.landing_batch", side_effect=AssertionError("second impact queried")):
            prediction = predict_frozen(snapshot, [1.02, 1.05], self.selected())
        self.assertEqual(prediction["event_reason"], "filter_already_bounced_first_event_state_not_exposed")
        self.assertIsNone(prediction["predicted_v_plus"])
        self.assertTrue(np.isfinite(prediction["states"]).all())
        self.assertGreater(prediction["states"][0,2], .1)

    def test_before_contact_target_remains_before_contact(self):
        snapshot = {"initialized": True, "state": [0., 0., .1, 1., 0., -2., 0., 0., 0.],
                    "t_s": 1., "bounced": False}
        prediction = predict_frozen(snapshot, [1.02, 1.10], self.selected())
        self.assertLess(prediction["states"][0,5], 0.)
        self.assertGreater(prediction["states"][1,5], 0.)
        self.assertGreater(prediction["predicted_tb"], 1.02)
        self.assertGreater(prediction["predicted_v_plus"][2], 0.)


if __name__ == "__main__":
    unittest.main()
