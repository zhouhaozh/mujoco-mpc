import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_calibration.flight import simulate
from tennis_calibration.online_data import history_features
from tennis_calibration.online_features import FEATURE_SCHEMA, base_features, landing_features, position_features
from tennis_calibration.online_mlp import save_residual_mlp
from tennis_calibration.online_predictor import OnlinePredictor


def constant_model(input_dim, correction, target_kind):
    return {"format_version": 1, "input_dim": input_dim, "output_dim": 3,
            "hidden": (2, 2), "activation": "tanh", "seed": 1,
            "target_kind": target_kind, "x_mean": np.zeros(input_dim), "x_scale": np.ones(input_dim),
            "y_mean": np.zeros(3), "y_scale": np.ones(3),
            "weights": [np.zeros((input_dim, 2)), np.zeros((2, 2)), np.zeros((2, 3))],
            "biases": [np.zeros(2), np.zeros(2), np.asarray(correction, dtype=float)]}


class OnlinePredictorTests(unittest.TestCase):
    def make_bundle(self, path, position=False, landing=False, landing_correction=(0.1, -0.2, 0.01)):
        config = {"physical": {"k_drag": 0.02, "e_n": 0.83, "alpha_t": 0.56, "contact_height": 0.03},
                  "filter": {}, "augmented": True, "position_mlp_enabled": position,
                  "landing_mlp_enabled": landing, "supported_horizons_s": [0.1, 0.2, 0.5],
                  "trained_delays_s": [0.0], "feature_schema": FEATURE_SCHEMA}
        Path(path, "config.json").write_text(json.dumps(config))
        if position:
            save_residual_mlp(constant_model(98, [0.03, -0.02, 0.01], "future_position_residual_m"),
                              Path(path, "models/mlp_position"))
        if landing:
            save_residual_mlp(constant_model(103, landing_correction, "landing_correction_m_m_s"),
                              Path(path, "models/mlp_landing"))
        return config

    def fill(self, predictor, finish=0.15):
        times = np.linspace(0, finish, int(round(finish / 0.0025)) + 1)
        states = simulate(times, np.array([0, 0, 3, 12, 1, 1]), 0.02)
        for t, state in zip(times, states):
            result = predictor.update(t, state[:3])
        return result

    def test_shared_feature_order_and_batch_match(self):
        times = np.linspace(0, 0.1, 41)
        positions = np.column_stack((times, times * 2, times * 3))
        history = history_features(times, positions, 40)
        snapshot = {"p": np.array([1, 2, 3]), "v": np.array([4, 5, 6]), "a_res": np.array([7, 8, 9]),
                    "covariance": np.diag(np.arange(1, 10)), "bounced": True}
        base = base_features(history, snapshot, 0.02)
        self.assertEqual(base.shape, (97,))
        np.testing.assert_array_equal(base[:82], history["features"])
        np.testing.assert_array_equal(base[82:89], [4, 5, 6, 7, 8, 9, 3])
        np.testing.assert_array_equal(base[89:95], np.sqrt(np.arange(1, 7)))
        np.testing.assert_array_equal(base[95:], [1, 0.02])
        self.assertEqual(position_features(base, 0.1).shape, (98,))
        landing = {"time_to_contact": 0.4, "position": [5, 6, 0.03], "velocity": [7, 8, -3]}
        feature = landing_features(base, snapshot, landing)
        np.testing.assert_array_equal(feature[-6:], [0.4, 4, 4, 7, 8, -3])
        batch = base_features({"features": np.stack([history["features"]] * 2)},
                              {k: np.stack([v] * 2) if isinstance(v, np.ndarray) else [v, v]
                               for k, v in snapshot.items()}, [0.02, 0.02])
        np.testing.assert_array_equal(batch, np.stack([base, base]))

    def test_bundle_baseline_only_and_capture_clock_projection(self):
        with tempfile.TemporaryDirectory() as path:
            self.make_bundle(path)
            predictor = OnlinePredictor(path)
            result = self.fill(predictor)
            self.assertTrue(result["initialized"])
            for row in result["forecasts"]:
                self.assertFalse(row["mlp_used"])
                np.testing.assert_array_equal(row["position_xyz_m"], row["physics_position_xyz_m"])
            before = predictor.filter.snapshot()
            delayed = predictor.predict(now_t_s=0.17, horizons=[0.2, 0.1])
            self.assertAlmostEqual(delayed["diagnostics"]["delay_s"], 0.02)
            self.assertAlmostEqual(delayed["forecasts"][0]["target_t_s"], 0.37)
            self.assertEqual(predictor.filter.last_t, 0.15)
            np.testing.assert_array_equal(predictor.filter.state, before["state"])
            with self.assertRaises(ValueError):
                predictor.predict(now_t_s=0.14)

    def test_validation_flags_and_supported_delay_horizon_govern_mlp(self):
        with tempfile.TemporaryDirectory() as path:
            self.make_bundle(path, position=True, landing=True)
            predictor = OnlinePredictor(path)
            result = self.fill(predictor)
            for row in result["forecasts"]:
                self.assertTrue(row["mlp_used"])
                np.testing.assert_allclose(row["position_xyz_m"] - row["physics_position_xyz_m"], [0.03, -0.02, 0.01])
            self.assertTrue(result["landing"]["mlp_used"])
            self.assertAlmostEqual(result["landing"]["contact_t_s"] - result["landing"]["physics_contact_t_s"], 0.01)
            unsupported = predictor.predict(horizons=[0.15])
            self.assertFalse(unsupported["forecasts"][0]["mlp_used"])
            self.assertEqual(unsupported["forecasts"][0]["mlp_reason"], "horizon_not_validated")
            delayed = predictor.predict(now_t_s=0.17)
            self.assertFalse(delayed["landing"]["mlp_used"])
            self.assertEqual(delayed["landing"]["mlp_reason"], "nonzero_delay_not_validated")
            for row in delayed["forecasts"]:
                self.assertFalse(row["mlp_used"])
                self.assertEqual(row["mlp_reason"], "nonzero_delay_not_validated")

    def test_corrected_past_contact_is_invalid_without_clipping(self):
        with tempfile.TemporaryDirectory() as path:
            self.make_bundle(path, landing=True, landing_correction=(0.1, -0.2, -10.0))
            result = self.fill(OnlinePredictor(path))
            self.assertFalse(result["landing"]["valid"])
            self.assertEqual(result["landing"]["reason"], "predicted_contact_not_in_future")
            self.assertLess(result["landing"]["remaining_from_now_s"], 0)
            self.assertAlmostEqual(result["landing"]["contact_t_s"] - result["landing"]["physics_contact_t_s"], -10)

    def test_invalid_timestamp_does_not_enter_history_and_reset_starts_new_ball(self):
        with tempfile.TemporaryDirectory() as path:
            self.make_bundle(path)
            predictor = OnlinePredictor(path)
            self.fill(predictor)
            length = len(predictor._history)
            old_state = predictor.filter.state.copy()
            result = predictor.update(0.1, [1, 2, 3])
            self.assertFalse(result["estimate"]["accepted"])
            self.assertEqual(len(predictor._history), length)
            np.testing.assert_array_equal(predictor.filter.state, old_state)
            predictor.reset()
            self.assertEqual(len(predictor._history), 0)
            self.assertFalse(predictor.update(5, [0, 0, 3])["initialized"])

    def test_bundle_mismatches_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as path:
            config = self.make_bundle(path)
            config["feature_schema"] = {"version": 0}
            Path(path, "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "schema"):
                OnlinePredictor(path)
        with tempfile.TemporaryDirectory() as path:
            self.make_bundle(path, position=True)
            save_residual_mlp(constant_model(98, [0, 0, 0], "wrong_target_kind"), Path(path, "models/mlp_position"))
            with self.assertRaisesRegex(ValueError, "semantics"):
                OnlinePredictor(path)


if __name__ == "__main__":
    unittest.main()
