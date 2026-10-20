"""Causality and diagnostic-integrity checks; numerical execution is on Ubuntu."""
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.measurement_noise_probe import (
    _robust_extrapolation, probe_measurement_noise,
)
from tennis_calibration.online_data import causal_anchors


def quadratic_trace(identity="quadratic", n=240, times=None, events=(5.,)):
    if times is None:
        # Strictly chronological, unequal real capture intervals.
        times = np.r_[0., np.cumsum(.002 + .0007 * (1 + np.sin(np.arange(n-1) * .31)))]
    times = np.asarray(times, dtype=float)
    xyz = np.column_stack((2 + .5*times + .3*times**2,
                           -.1 + .2*times - .4*times**2,
                           1.3 + 2*times - .2*times**2))
    return {"id": identity, "split": "test", "raw_times": times, "raw_xyz": xyz,
            "bounce_event_times_s": list(events),
            "bounce_reference_s": events[0] if len(events) else None}


class MeasurementNoiseProbeTests(unittest.TestCase):
    def test_irregular_timestamp_quadratic_has_negligible_discrepancy(self):
        result = probe_measurement_noise([quadratic_trace()])
        for window in ("40ms", "80ms"):
            cell = result["by_split"]["test"][window]
            self.assertEqual(cell["summary"]["n_trajectories_scored"], 1)
            self.assertGreater(cell["summary"]["counts"]["scored"], 10)
            stats = cell["trajectories"][0]["statistics"]
            np.testing.assert_allclose(stats["signed_axis_median_m"], 0., atol=1e-10)
            self.assertLess(stats["norm_max_m"], 1e-9)

    def test_current_and_future_changes_cannot_change_past_extrapolations(self):
        trace = quadratic_trace()
        cutoff = next(a for a in causal_anchors(trace, initialization_s=0.) if a["t_s"] >= .24)
        changed = {**trace, "raw_xyz": trace["raw_xyz"].copy()}
        # The current observation and all not-yet-received observations differ.
        changed["raw_xyz"][cutoff["current_index"]:] += [3., -4., 2.]

        def capture_predictions(source):
            predictions = {}

            def record(*args, **kwargs):
                prediction = _robust_extrapolation(*args, **kwargs)
                current_t, window = args[2], args[3]
                if current_t <= cutoff["t_s"]:
                    predictions[(current_t, window)] = prediction.copy()
                return prediction

            with patch("tennis_calibration.measurement_noise_probe._robust_extrapolation",
                       side_effect=record):
                probe_measurement_noise([source])
            return predictions

        before, after = capture_predictions(trace), capture_predictions(changed)
        self.assertEqual(set(before), set(after))
        self.assertIn((cutoff["t_s"], .04), before)
        self.assertIn((cutoff["t_s"], .08), before)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_event_and_gap_exclusions_preserve_complete_disposition_counts(self):
        times = np.arange(0., .651, .0025)
        # A 12.5ms intersample gap exceeds the configured limit while leaving
        # enough history for both windows, so the gap check is actually reached.
        times = times[(times < .41) | (times >= .42)]
        labelled = quadratic_trace(times=times, events=(.22,))
        unknown = quadratic_trace(identity="unknown_event", times=times, events=())
        result = probe_measurement_noise([labelled, unknown])
        for cell in result["by_split"]["test"].values():
            labelled_row, unknown_row = cell["trajectories"]
            counts = labelled_row["counts"]
            self.assertGreater(counts["skip_event_crossing"], 0)
            self.assertGreater(counts["skip_event_clearance"], 0)
            self.assertGreater(counts["skip_gap_over_limit"], 0)
            self.assertGreater(counts["scored"], 0)
            for row in (labelled_row, unknown_row):
                counts = row["counts"]
                self.assertEqual(counts["anchors_total"], counts["scored"] + sum(
                    value for key, value in counts.items() if key.startswith("skip_")))
            self.assertIsNone(unknown_row["statistics"])
            self.assertEqual(unknown_row["counts"]["skip_no_reference_events"],
                             unknown_row["counts"]["anchors_total"])
            self.assertEqual(cell["summary"]["n_trajectories_scored"], 1)

    def test_current_outlier_is_not_absorbed_into_its_own_fit(self):
        trace = quadratic_trace()
        anchor = next(a for a in causal_anchors(trace, initialization_s=0.) if a["t_s"] >= .24)
        end = anchor["current_index"] + 1
        trace["raw_times"] = trace["raw_times"][:end].copy()
        trace["raw_xyz"] = trace["raw_xyz"][:end].copy()
        offset = np.array([.03, -.04, .02])
        trace["raw_xyz"][-1] += offset
        result = probe_measurement_noise([trace])
        for cell in result["by_split"]["test"].values():
            stats = cell["trajectories"][0]["statistics"]
            self.assertAlmostEqual(stats["norm_max_m"], float(np.linalg.norm(offset)), places=9)
            self.assertLess(np.linalg.norm(stats["signed_axis_median_m"]), 1e-9)


if __name__ == "__main__":
    unittest.main()
