"""Collision residual invariance, independent-event weights and offline labels."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_calibration import collision_mapping as c
from tennis_calibration.identified_dynamics import simulate


class CollisionMappingTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(31)
        self.vm = self.rng.normal(size=(40, 3))
        self.vm[:, :2] = self.vm[:, :2]*2 + [8, 3]
        self.vm[:, 2] = -abs(self.vm[:, 2])-3
        self.bm = self.rng.normal(size=(40, 3))*.05
        self.X, self.frames = c.pre_features(self.vm, self.bm)
        self.Y = np.c_[.02*self.X[:, 0], -.03*self.X[:, 3], .05*self.X[:, 1]]

    def test_feature_contract_and_frame(self):
        self.assertEqual(self.X.shape, (40, 5))
        np.testing.assert_allclose(self.frames@np.swapaxes(self.frames, 1, 2),
                                   np.broadcast_to(np.eye(3), (40, 3, 3)), atol=1e-14)
        self.assertEqual(c.FEATURE_NAMES[0], "incident_tangent_speed_m_s")

    def test_equivariance_about_ground_normal(self):
        angle = 1.1
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        xr, _ = c.pre_features(self.vm@R.T, self.bm@R.T)
        np.testing.assert_allclose(self.X, xr, atol=1e-14)
        model, _ = c.fit_model(self.X, self.Y, config={"kind": "ridge", "ridge": .01})
        vp = c.predict_velocity(model, self.vm, self.bm)
        vpr = c.predict_velocity(model, self.vm@R.T, self.bm@R.T)
        np.testing.assert_allclose(vpr, vp@R.T, atol=1e-14)

    def test_velocity_target_roundtrip(self):
        delta = np.einsum("nji,nj->ni", self.frames, self.Y)
        vp = c.baseline_velocity(self.vm)+delta
        X, Y = c.velocity_targets(self.vm, self.bm, vp)
        np.testing.assert_allclose(X, self.X)
        np.testing.assert_allclose(Y, self.Y, atol=1e-14)

    def test_effective_b_target_roundtrip_and_prior_modes(self):
        delta = np.einsum("nji,nj->ni", self.frames, self.Y)
        for mode in ("carry", "zero"):
            prior = self.bm if mode == "carry" else np.zeros_like(self.bm)
            X, Y = c.effective_targets(self.vm, self.bm, prior+delta, prior_mode=mode)
            np.testing.assert_allclose(Y, self.Y, atol=1e-14)
            model, _ = c.fit_model(X, Y, config={"kind": "ridge", "ridge": 1e-5})
            np.testing.assert_allclose(c.predict_bplus(model, self.vm, self.bm, prior_mode=mode),
                                       prior+delta, atol=1e-6)

    def test_scalar_post_lift_uses_complete_covariance(self):
        bp = self.bm+self.frames[:, 1]*.05
        labels = [{"available": True, "quality_reasons": [],
                   "b_plus_per_s": b.tolist(),
                   "conditional_b_covariance_per_s2": (np.eye(3)*.0025).tolist()}
                  for b in bp]
        X, Y, sigma = c.post_lift_targets(self.vm, self.bm, labels)
        np.testing.assert_allclose(Y, .05, atol=1e-14)
        np.testing.assert_allclose(sigma, .05, atol=1e-14)
        model, _ = c.fit_model(X, Y, config={"kind": "ridge", "ridge": .01})
        np.testing.assert_allclose(c.predict_post_lift(model, self.vm, self.bm), bp, atol=1e-12)
        # Older diagonal-only diagnostics cannot justify a rotated covariance.
        for row in labels:
            row.pop("conditional_b_covariance_per_s2")
            row["conditional_b_sigma_per_s"] = [.01, .01, .01]
        _, _, rejected = c.post_lift_targets(self.vm, self.bm, labels)
        self.assertTrue(np.isinf(rejected).all())

    def test_trajectory_weights_equal_total(self):
        ids = ["a", "a", "a", "b", "c", "c"]
        w = c.trajectory_weights(ids)
        for identity in set(ids):
            self.assertAlmostEqual(float(w[np.asarray(ids) == identity].sum()), 1.)

    def test_split_weight_duplicate_invariance_ridge_and_gp(self):
        for config in [{"kind": "ridge", "ridge": 1.},
                       {"kind": "gp", "noise": .3, "length_scale": 1.}]:
            model, _ = c.fit_model(self.X, self.Y, np.ones(40), config)
            doubled, _ = c.fit_model(np.repeat(self.X, 2, axis=0),
                                     np.repeat(self.Y, 2, axis=0), np.full(80, .5), config)
            np.testing.assert_allclose(c.predict_model(model, self.X),
                                       c.predict_model(doubled, self.X), atol=1e-9)

    def test_ridge_fits_lowdimensional_signal(self):
        model, report = c.fit_model(self.X, self.Y,
                                    config={"kind": "ridge", "degree": 1, "ridge": 1e-5})
        np.testing.assert_allclose(c.predict_model(model, self.X), self.Y, atol=1e-6)
        self.assertEqual(report["rows"], 40)

    def test_gp_shapes_std_and_far_prior(self):
        model, _ = c.fit_model(self.X, self.Y,
                               config={"kind": "gp", "noise": .1, "length_scale": 1.})
        prediction, std = c.predict_model(model, self.X, return_std=True)
        self.assertEqual(prediction.shape, (40, 3))
        self.assertEqual(std.shape, (40, 3))
        self.assertTrue(np.all(std >= 0))
        far = np.ones((1, 5))*1e6
        np.testing.assert_allclose(c.predict_model(model, far)[0], model["y_mean"], atol=1e-12)

    def test_training_normalization_only_and_serialization(self):
        for config in [{"kind": "ridge", "degree": 2}, {"kind": "gp"}]:
            model, _ = c.fit_model(self.X, self.Y, config=config)
            np.testing.assert_allclose(model["x_mean"], self.X.mean(axis=0))
            with tempfile.TemporaryDirectory() as path:
                checkpoint = Path(path)/"model.json"
                c.save_model(model, checkpoint)
                loaded = c.load_model(checkpoint)
                np.testing.assert_allclose(c.predict_model(loaded, self.X), c.predict_model(model, self.X))
                data = json.loads(checkpoint.read_text()); data["x_scale"][0] = 0
                checkpoint.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    c.load_model(checkpoint)

    def test_empty_inference_and_bad_inputs(self):
        model, _ = c.fit_model(self.X, self.Y)
        self.assertEqual(c.predict_model(model, np.empty((0, 5))).shape, (0, 3))
        with self.assertRaises(ValueError): c.pre_features([[0, 0, -3]], [[0, 0, 0]])
        with self.assertRaises(ValueError): c.pre_features([[3, 1, 3]], [[0, 0, 0]])
        with self.assertRaises(ValueError): c.fit_model(self.X, self.Y, np.zeros(40))
        with self.assertRaises(ValueError): c.fit_model(self.X, self.Y, config={"post_time": .02})
        with self.assertRaises(ValueError): c.fit_model(self.X, self.Y, config={"kind": "gp", "noise": 0})

    def test_labels_use_long_post_and_not_pre_observations(self):
        state = np.array([0, 0, .014, 8, 2, 4, .03, -.04, .02])
        tb = 1.
        times = np.arange(.025, .401, .005)
        positions = simulate(times, state, .02)[:, :3]
        label = c.fit_post_labels(times+tb, positions, tb, .02,
                                  measurement_sigma_m=.0001, b_prior_sigma=.2)
        self.assertTrue(label["converged"])
        self.assertFalse(label["quality_reasons"])
        np.testing.assert_allclose(label["v_plus_m_s"], state[3:6], atol=.01)
        np.testing.assert_allclose(label["b_plus_per_s"], state[6:9], atol=.01)
        combined_t = np.r_[.8, .9, times+tb]
        combined_p = np.vstack([[1e4, 2e4, 3e4], [-1e4, 0, 0], positions])
        unchanged = c.fit_post_labels(combined_t, combined_p, tb, .02,
                                      measurement_sigma_m=.0001, b_prior_sigma=.2)
        np.testing.assert_allclose(label["state_plus"], unchanged["state_plus"], atol=1e-14)
        with self.assertRaises(ValueError):
            c.fit_post_labels(times[:10]+tb, positions[:10], tb, .02)


if __name__ == "__main__":
    unittest.main()
