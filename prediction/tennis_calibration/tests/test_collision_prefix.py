"""Synthetic causal prefix tests; run formal tests on Ubuntu only."""
import unittest

import numpy as np

from tennis_calibration import identified_dynamics as d
from tennis_calibration.collision_prefix import estimate_prefix, predict_contact


class PrefixTest(unittest.TestCase):
    def setUp(self):
        self.parameters = {"k_drag": .02043211081206929,
                           "e_n": .8339975, "alpha_t": .5576036,
                           "impact_mode": "effective"}
        self.height = .014122684
        self.times = 7 + np.linspace(0, .35, 100)
        self.state = np.array([0., 1., 2., 12., -2., -1., 0., 0., 0.])
        self.xyz = d.simulate(self.times-self.times[0], self.state,
                              self.parameters["k_drag"])[..., :3]

    def fit(self, times=None, xyz=None, cut=None, config=None):
        return estimate_prefix(self.times if times is None else times,
             self.xyz if xyz is None else xyz,
             self.times[-1] if cut is None else cut,
             self.parameters, self.height, config)

    def test_noiseless_flight_and_contact(self):
        fit = self.fit()
        self.assertTrue(fit["converged"])
        true = d.step(self.state, .35, self.parameters["k_drag"])
        np.testing.assert_allclose(fit["state"], true, atol=2e-5)
        contact = predict_contact(fit, self.parameters, self.height)
        self.assertTrue(contact["valid"])
        self.assertGreater(contact["contact_t_s"], self.times[-1])
        self.assertAlmostEqual(contact["position"][2], self.height)
        self.assertLess(contact["v_minus"][2], 0)

    def test_future_observation_mutation_does_not_change_inference(self):
        cut = self.times[70]
        base = self.fit(cut=cut)
        changed = self.xyz.copy(); changed[71:] = np.array([1e8, -1e8, -1e8])
        changed[90] = np.nan
        future = self.fit(xyz=changed, cut=cut)
        np.testing.assert_array_equal(base["state"], future["state"])
        np.testing.assert_array_equal(base["covariance"], future["covariance"])

    def test_explicit_truncation_matches_cut(self):
        index = 75
        a = self.fit(cut=self.times[index])
        b = self.fit(times=self.times[:index+1], xyz=self.xyz[:index+1])
        np.testing.assert_array_equal(a["state"], b["state"])
        self.assertEqual(a["capture_t_s"], self.times[index])

    def test_cut_between_observations_uses_last_capture(self):
        cut = (self.times[60]+self.times[61])/2
        fit = self.fit(cut=cut)
        self.assertEqual(fit["capture_t_s"], self.times[60])
        self.assertLess(fit["capture_t_s"], cut)

    def test_recent_window_is_a_prefix_subset(self):
        fit = self.fit(config={"window_s": .12})
        self.assertTrue(fit["converged"])
        self.assertLessEqual(fit["diagnostics"]["fit_span_s"], .12)
        np.testing.assert_allclose(fit["state"][3:6],
            d.step(self.state, .35, self.parameters["k_drag"])[3:6], atol=1e-4)

    def test_covariance_is_finite_symmetric_psd(self):
        fit = self.fit()
        cov = fit["covariance"]
        self.assertTrue(np.isfinite(cov).all())
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(cov).min()), -1e-10)
        self.assertIn("sensitivity_without_prior", fit["diagnostics"])

    def test_sparse_noise_and_gross_outlier_are_robust(self):
        rng = np.random.default_rng(831)
        observed = self.xyz + rng.normal(0, .002, self.xyz.shape)
        observed[50] += [1., -1., .7]
        fit = self.fit(xyz=observed)
        self.assertTrue(fit["converged"])
        truth = d.step(self.state, .35, self.parameters["k_drag"])
        self.assertLess(np.linalg.norm(fit["state"][3:6]-truth[3:6]), .15)
        self.assertLess(fit["diagnostics"]["robust_row_weight_min"], .02)

    def test_missing_duplicate_and_nonfinite_input_rows(self):
        retained = np.r_[np.arange(20), np.arange(30, 100)]
        times, xyz = self.times[retained].copy(), self.xyz[retained].copy()
        times[11] = times[10]; xyz[14] = np.nan
        fit = self.fit(times=times, xyz=xyz)
        self.assertTrue(fit["converged"])
        self.assertEqual(fit["diagnostics"]["nonfinite_prefix_rows"], 1)
        self.assertEqual(fit["diagnostics"]["nonincreasing_prefix_rows"], 1)
        self.assertLess(np.linalg.norm(fit["state"][3:6]-
            d.step(self.state, .35, self.parameters["k_drag"])[3:6]), .001)

    def test_insufficient_prefix_never_invents_state(self):
        fit = self.fit(times=self.times[:3], xyz=self.xyz[:3])
        self.assertFalse(fit["converged"])
        self.assertIsNone(fit["state"])
        self.assertEqual(predict_contact(fit, self.parameters, self.height)["reason"], "prefix_not_converged")

    def test_bad_configuration_rejected(self):
        for config in ({"future_contact_time": 1}, {"window_s": -1},
                       {"representative_points": 3}, {"max_nfev": 1.5}):
            with self.assertRaises(ValueError): self.fit(config=config)
        with self.assertRaises(ValueError):
            self.fit(times=np.ones((3, 2)))

    def test_no_contact_within_horizon_is_explicit(self):
        fit = self.fit()
        contact = predict_contact(fit, self.parameters, self.height, max_horizon_s=.01)
        self.assertFalse(contact["valid"])
        self.assertEqual(contact["reason"], "no_contact_within_horizon")


if __name__ == "__main__":
    unittest.main()
