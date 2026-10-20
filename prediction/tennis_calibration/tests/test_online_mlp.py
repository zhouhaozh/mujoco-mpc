"""Synthetic residual-learning and checkpoint/split-isolation checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_calibration.online_mlp import (
    fit_residual_mlp, load_residual_mlp, predict_residual, save_residual_mlp,
)


class OnlineMlpTests(unittest.TestCase):
    def synthetic(self, count, seed):
        x = np.random.default_rng(seed).uniform(-1, 1, (count, 4))
        y = 0.05 * np.column_stack((0.7 * x[:, 1] + 0.1 * np.sin(x[:, 0]),
                                    -0.4 * x[:, 2], 0.3 * x[:, 0] * x[:, 3]))
        return x, y

    def test_known_residual_reduces_validation_error(self):
        x, y = self.synthetic(600, 3)
        vx, vy = self.synthetic(200, 4)
        with tempfile.TemporaryDirectory() as directory:
            model, report = fit_residual_mlp(x, y, vx, vy, directory, max_epochs=160)
            self.assertLess(report["val_position_rmse_m"],
                            0.65 * report["baseline_zero_correction_val_position_rmse_m"])
            self.assertEqual(predict_residual(model, vx).shape, (200, 3))
            self.assertLessEqual(report["best_epoch"], report["actual_epochs"])
            self.assertEqual(len(report["history"]), report["actual_epochs"] + 1)
            self.assertEqual(report["selection_source"], "validation_rows_only")
            self.assertTrue(Path(directory, "training_report.json").is_file())

    def test_normalizer_uses_train_only_despite_extreme_validation(self):
        x, y = self.synthetic(70, 10)
        vx, vy = self.synthetic(20, 11)
        vx += 1e6
        vy += 1e4
        with tempfile.TemporaryDirectory() as directory:
            model, _ = fit_residual_mlp(x, y, vx, vy, directory, max_epochs=2)
            np.testing.assert_allclose(model["x_mean"], x.mean(axis=0), atol=1e-14)
            np.testing.assert_allclose(model["x_scale"], x.std(axis=0), atol=1e-14)
            np.testing.assert_allclose(model["y_mean"], y.mean(axis=0), atol=1e-14)
            np.testing.assert_allclose(model["y_scale"], y.std(axis=0), atol=1e-14)
            # Later test inference cannot adapt a training normalizer either.
            before = {key: model[key].copy() for key in
                      ("x_mean", "x_scale", "y_mean", "y_scale")}
            predict_residual(model, np.full((30, 4), -1e8))
            for key, value in before.items():
                np.testing.assert_array_equal(model[key], value)

    def test_numpy_json_checkpoint_roundtrip(self):
        x, y = self.synthetic(90, 15)
        vx, vy = self.synthetic(30, 16)
        with tempfile.TemporaryDirectory() as directory:
            model, _ = fit_residual_mlp(x, y, vx, vy, directory, max_epochs=5)
            paths = save_residual_mlp(model, directory)
            with np.load(paths["parameters"], allow_pickle=False) as archive:
                self.assertEqual(len(archive.files), 6)
            loaded = load_residual_mlp(directory)
            np.testing.assert_array_equal(predict_residual(model, vx), predict_residual(loaded, vx))
            self.assertEqual(predict_residual(loaded, np.empty((0, 4))).shape, (0, 3))

    def test_mixed_unit_targets_are_not_reported_as_position_metres(self):
        x, y = self.synthetic(60, 35)
        vx, vy = self.synthetic(20, 36)
        with tempfile.TemporaryDirectory() as directory:
            model, report = fit_residual_mlp(x, y, vx, vy, directory,
                                              max_epochs=2, target_kind="landing_correction_m_m_s")
            self.assertIn("val_output_rmse", report)
            self.assertNotIn("val_position_rmse_m", report)
            self.assertNotIn("val_position_rmse_m", report["history"][0])
            self.assertEqual(load_residual_mlp(directory)["target_kind"], model["target_kind"])

    def test_trajectory_weights_and_constant_features_are_finite(self):
        x, y = self.synthetic(90, 25)
        vx, vy = self.synthetic(30, 26)
        x[:, 0] = vx[:, 0] = 2.0
        y[:, 2] = vy[:, 2] = 0.0
        weights = np.r_[np.zeros(10), np.ones(70), np.full(10, 7.0)]
        with tempfile.TemporaryDirectory() as directory:
            model, report = fit_residual_mlp(x, y, vx, vy, directory,
                                              sample_weights=weights, max_epochs=10)
            np.testing.assert_allclose(model["x_mean"], np.average(x, axis=0, weights=weights))
            self.assertTrue(np.all(np.isfinite(predict_residual(model, vx))))
            self.assertTrue(np.isfinite(report["train_position_rmse_m"]))
            for invalid in (np.zeros(90), -np.ones(90), np.full(90, np.nan)):
                with self.assertRaises(ValueError):
                    fit_residual_mlp(x, y, vx, vy, directory, sample_weights=invalid, max_epochs=1)

    def test_validation_trajectory_weights_prevent_duration_dominance(self):
        x, y = self.synthetic(70, 45)
        short_x, _ = self.synthetic(3, 46)
        long_x, _ = self.synthetic(600, 47)
        vx = np.vstack((short_x, long_x))
        # A short trajectory has one-metre residual; a long one has zero.
        vy = np.zeros((603, 3))
        vy[:3, 0] = 1.0
        val_weights = np.r_[np.full(3, 1 / 3), np.full(600, 1 / 600)]
        with tempfile.TemporaryDirectory() as directory:
            model, report = fit_residual_mlp(
                x, y, vx, vy, directory, max_epochs=3, val_sample_weights=val_weights)
            # Equal total trajectory weights yield sqrt((1**2 + 0**2) / 2),
            # independent of the long trajectory's 200x greater row count.
            self.assertAlmostEqual(report["baseline_zero_correction_val_output_rmse"],
                                   np.sqrt(0.5), places=12)
            error2 = np.sum((predict_residual(model, vx) - vy) ** 2, axis=1)
            expected = np.sqrt(np.average(error2, weights=val_weights))
            self.assertAlmostEqual(report["val_output_rmse"], expected, places=12)
            np.testing.assert_allclose(model["x_mean"], x.mean(axis=0), atol=1e-14)
            self.assertTrue(report["validation_weighting"]["sample_weights_supplied"])
            self.assertEqual(report["selection_metric"], "weighted_rms_euclidean_position_error_m")
            for invalid in (np.zeros(603), -np.ones(603), np.full(603, np.nan)):
                with self.assertRaises(ValueError):
                    fit_residual_mlp(x, y, vx, vy, directory,
                                     max_epochs=1, val_sample_weights=invalid)


if __name__ == "__main__":
    unittest.main()
