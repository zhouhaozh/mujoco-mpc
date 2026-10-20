"""Independent small-data checks for experiment scoring; run on Ubuntu."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_calibration.residual_head import FEATURE_NAMES
from tennis_calibration.robust_optimization import (
    summarize, describe, cache_arrays, check_baseline,
)


def sample(candidate="baseline", identity="a", stamp=1.12, error=.1,
           *, anchor=True, dense=True, valid=True, failure=""):
    return {"trajectory_id": identity, "split": "validation", "groupid": "group_a",
            "candidate": candidate, "capture_t_s": stamp, "elapsed_capture_s": stamp - .5,
            "relative_capture_s": stamp - 1., "target_t_s": stamp + .5, "horizon_s": .5,
            "twenty_ms_anchor": anchor, "dense_recovery_frame": dense, "valid": valid,
            "reason": "" if valid else "missing_future_reference", "prediction_reason": failure,
            "predicted_xyz": None if failure else [error, 0., 0.], "target_xyz": [0., 0., 0.],
            "initialized": not bool(failure), "hold_active": False, "residual_gate": 0.}


def event(candidate="baseline", identity="a", failure=""):
    return {"trajectory_id": identity, "split": "validation", "groupid": "group_a",
            "candidate": candidate, "diagnostic_resume_or_event_t_s": 1.10,
            "tracking_failure": failure}


class RobustOptimizationTests(unittest.TestCase):
    def test_dense_large_peak_does_not_change_common_anchor_ranking_or_fixed_window(self):
        regular = sample()
        peak = sample(stamp=1.121, error=100., anchor=False)
        result = summarize([regular, peak], [event()])
        self.assertAlmostEqual(result["ranking"][0]["balanced_future_rmse_m"], .1)
        self.assertAlmostEqual(result["first200ms_h500_metrics"][0]["mean_m"], .1)
        self.assertEqual(result["absolute_labeled_coverage"][0]["requested"], 1)
        dense = result["dense_resume_first50ms_metrics"][0]
        self.assertEqual(dense["requested"], 2)
        self.assertAlmostEqual(dense["max_m"], 100.)
        self.assertAlmostEqual(dense["mean_m"], 50.05)

    def test_total_and_partial_failure_arms_remain_visible_and_cannot_win(self):
        rows = [sample(stamp=1.12), sample(stamp=1.14)]
        for stamp in (1.12, 1.14):
            rows.append(sample("all_failed", stamp=stamp, failure="synthetic_tracking_failure"))
        rows += [sample("partial", stamp=1.12, error=.0001),
                 sample("partial", stamp=1.14, failure="synthetic_forecast_failure")]
        result = summarize(rows, [event(), event("all_failed", failure="failed"), event("partial")])
        coverage = {row["candidate"]: row for row in result["absolute_labeled_coverage"]}
        self.assertEqual(coverage["all_failed"]["labeled"], 2)
        self.assertEqual(coverage["all_failed"]["failed_labeled"], 2)
        self.assertEqual(coverage["all_failed"]["successful_keys"], 0)
        self.assertEqual(coverage["all_failed"]["missing_vs_reference"], 2)
        self.assertEqual(coverage["partial"]["failed_labeled"], 1)
        ranks = {row["candidate"]: row for row in result["ranking"]}
        self.assertTrue(ranks["baseline"]["eligible_for_ranking"])
        self.assertFalse(ranks["partial"]["eligible_for_ranking"])
        self.assertNotIn("all_failed", ranks)

    def test_trace_equal_mean_and_frame_p95_ignore_unequal_trace_sampling_density(self):
        a, b = [0.] * 99 + [1.], [2.]
        metric = describe({"a": a, "b": b})
        self.assertAlmostEqual(metric["mean_m"], 1.005)
        self.assertAlmostEqual(metric["rmse_m"], np.sqrt(2.005))
        self.assertEqual(metric["p95_frame_m"], 2.)
        self.assertEqual(metric["N"], 2)
        self.assertEqual(metric["n_frames"], 101)
        denser = describe({"a": a * 3, "b": b * 17})
        for key in ("mean_m", "rmse_m", "p95_frame_m"):
            self.assertAlmostEqual(metric[key], denser[key])

    def test_cache_preserves_row_identity_features_and_failure_validity_separately(self):
        rows = [sample(identity="train_a", stamp=1.1),
                sample(identity="train_b", stamp=1.16, anchor=False),
                sample(identity="train_c", stamp=1.2, failure="failed")]
        for row, gate in zip(rows, (0., .2, 1.)):
            row["residual_gate"] = gate
        features = np.arange(3 * len(FEATURE_NAMES), dtype=float).reshape(3, -1)
        features[-1] = np.nan
        cached = cache_arrays(rows, features)
        np.testing.assert_array_equal(cached["ids"], ["train_a", "train_b", "train_c"])
        np.testing.assert_array_equal(cached["X"], features)
        np.testing.assert_array_equal(cached["gate"], [0., .2, 1.])
        np.testing.assert_array_equal(cached["valid"], [True, True, True])
        np.testing.assert_array_equal(cached["prediction_ok"], [True, True, False])
        self.assertTrue(np.isnan(cached["pred"][-1]).all())
        self.assertFalse(any(value.dtype == object for value in cached.values()))

    def test_baseline_regression_accepts_exact_reference_and_rejects_changed_error(self):
        summary = summarize([sample()], [event()])
        reference = {"phase_metrics": [], "candidate_ranking": []}
        for row in summary["phase_metrics"]:
            reference["phase_metrics"].append({**row, "candidate": "gp:hold100"})
        for row in summary["ranking"]:
            reference["candidate_ranking"].append({**row, "candidate": "gp:hold100"})
        # Historical source also scores H0 current-state estimation; the new
        # optimization compares only positive future H100/H200/H500 horizons.
        reference["phase_metrics"].append({**reference["phase_metrics"][0], "horizon_s": 0.})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(reference), encoding="utf-8")
            self.assertEqual(check_baseline(summary, path, complete=True)["status"], "passed")
            altered = deepcopy(summary)
            altered["phase_metrics"][0]["rmse_m"] += .01
            with self.assertRaises(AssertionError):
                check_baseline(altered, path, complete=True)
            missing_phase = deepcopy(summary)
            missing_phase["phase_metrics"] = []
            with self.assertRaises(AssertionError):
                check_baseline(missing_phase, path, complete=True)


if __name__ == "__main__":
    unittest.main()
