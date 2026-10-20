import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_calibration.flight import simulate
from tennis_calibration.height_legacy_baseline import (
    HORIZONS, forecast_legacy, run_legacy_baselines, update_filter_only,
    verify_reference_prediction)
from tennis_calibration.online_features import FEATURE_SCHEMA
from tennis_calibration.online_filter import DEFAULT_CONFIG
from tennis_calibration.online_mlp import save_residual_mlp
from tennis_calibration.online_predictor import OnlinePredictor
from tennis_calibration.tests.test_online_predictor import constant_model


class HeightLegacyBaselineTests(unittest.TestCase):
    def make_bundle(self, path):
        config = {"physical": {"k_drag": .02, "e_n": .83, "alpha_t": .56,
                               "contact_height": .03},
                  "filter": DEFAULT_CONFIG, "augmented": False,
                  "selected_filter": "physics6", "position_mlp_enabled": True,
                  "landing_mlp_enabled": False,
                  "supported_horizons_s": [.1, .2, .5], "trained_delays_s": [0.],
                  "feature_schema": FEATURE_SCHEMA}
        Path(path, "config.json").write_text(json.dumps(config))
        save_residual_mlp(constant_model(98, [.03, -.02, .01],
                                       "future_position_residual_m"),
                          Path(path, "models/mlp_position"))

    def trace(self):
        times = np.arange(0., .30001, .0025)
        states = simulate(times, np.array([0., 0., 3., 12., 1., 1.]), .02)
        return {"id": "synthetic", "split": "validation", "raw_times": times,
                "raw_xyz": states[:, :3].copy(), "eval_times": times,
                "eval_xyz": states[:, :3].copy(), "bounce_reference_s": .2,
                "bounce_event_times_s": [.2], "eval_scope_end_s": .3}

    def test_original_forecasts_match_and_zero_horizon_has_no_mlp(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_bundle(directory)
            predictor = OnlinePredictor(directory)
            trace = self.trace()
            for stamp, xyz in zip(trace["raw_times"][:61], trace["raw_xyz"][:61]):
                snapshot = update_filter_only(predictor, float(stamp), xyz)
            predictions, _ = forecast_legacy(predictor, snapshot)
            check = verify_reference_prediction(predictor, predictions)
            self.assertEqual(check["position_difference_m"], 0.)
            np.testing.assert_array_equal(predictions["legacy_ekf6_mlp"][0], snapshot["p"])
            np.testing.assert_array_equal(predictions["legacy_ekf6"][0], snapshot["p"])
            for index in range(1, len(HORIZONS)):
                np.testing.assert_allclose(np.asarray(predictions["legacy_ekf6_mlp"][index])
                                           - predictions["legacy_ekf6"][index],
                                           [.03, -.02, .01])

    def test_first_frame_grid_retains_unavailable_warmup_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory, "bundle")
            bundle.mkdir()
            self.make_bundle(bundle)
            rows = run_legacy_baselines([self.trace()], Path(directory, "out"), run_dir=bundle)
            early = [r for r in rows if r["elapsed_capture_s"] < .08-1e-12]
            self.assertEqual(len(early), 4 * 4 * 2)
            self.assertTrue(all(r["predicted_xyz"] is None for r in early))
            self.assertTrue(all(r["prediction_reason"] == "not_initialized" for r in early))
            available = [r for r in rows if r["predicted_xyz"] is not None]
            self.assertTrue(all(r["prediction_reason"] == "" for r in available))
            self.assertTrue(all(r["inference_note"] != "" for r in available))
            self.assertEqual(min(r["elapsed_capture_s"] for r in rows), 0.)
            self.assertTrue(all(r["capture_t_s"] <= .3+1e-12 for r in rows))
            summary = json.loads(Path(directory, "out/legacy_baseline_summary.json").read_text())
            self.assertTrue(summary["checkpoint_unchanged"])
            self.assertFalse(summary["protocol"]["height_hold_policy_evaluated"])

    def test_later_observations_cannot_change_earlier_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory, "bundle")
            bundle.mkdir()
            self.make_bundle(bundle)
            original = self.trace()
            changed = dict(original)
            changed["raw_xyz"] = original["raw_xyz"].copy()
            changed["raw_xyz"][original["raw_times"] > .15] += [1., -1., .5]
            first = run_legacy_baselines([original], Path(directory, "first"), run_dir=bundle)
            second = run_legacy_baselines([changed], Path(directory, "second"), run_dir=bundle)
            for left, right in zip(first, second):
                if left["capture_t_s"] <= .15:
                    self.assertEqual(left["predicted_xyz"], right["predicted_xyz"])


if __name__ == "__main__":
    unittest.main()
