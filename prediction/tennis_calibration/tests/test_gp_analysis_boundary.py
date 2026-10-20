"""Tennis input boundaries for immutable method-6 full-precontact analysis.

Run formal tests on Ubuntu. The fit spy verifies the policy independently of
the numerical optimizer so future data and the old 72-point cap cannot leak in.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration import gp_analysis_run as runner


class TestGPAnalysisInputBoundary(unittest.TestCase):
    def setUp(self):
        times = np.arange(0., 1.51, .01)
        self.trace = {"bounce_reference_s": 1., "raw_times": times,
                      "raw_xyz": np.column_stack((times, times**2, 2.-times)),
                      "eval_times": times.copy(), "eval_xyz": np.zeros((len(times), 3)),
                      "offline_incident_velocity": [999., 999., 999.]}
        self.bank = SimpleNamespace(parameters={"k_drag": .02}, height=.014)
        self.calls = []

    def fake_fit(self, times, xyz, cutoff, parameters, height, config):
        self.calls.append({"times": times.copy(), "xyz": xyz.copy(),
                           "cutoff": cutoff, "parameters": parameters,
                           "height": height, "config": dict(config)})
        return {"converged": True, "capture_t_s": float(times[-1]),
                "state": np.r_[xyz[-1], [1., 1., -1.], np.zeros(3)],
                "covariance": np.eye(9), "diagnostics": {"sampled_rows": len(times)}}

    def fake_contact(self, fit, parameters, height):
        return {"valid": True, "reason": "predicted_contact",
                "state_minus": fit["state"].copy(),
                "contact_t_s": fit["capture_t_s"]+.005}

    def prepared(self, trace=None):
        with patch.object(runner, "estimate_prefix", self.fake_fit), \
             patch.object(runner, "predict_contact", self.fake_contact):
            return runner.all_pre_fit(self.trace if trace is None else trace, self.bank)

    def test_every_precontact_row_used_and_exact_contact_excluded(self):
        prepared = self.prepared()
        received = self.calls[-1]
        self.assertEqual(len(received["times"]), 100)
        self.assertTrue(np.all(received["times"] < 1.))
        self.assertEqual(received["config"]["window_s"], 0.)
        self.assertGreaterEqual(received["config"]["representative_points"], 100)
        self.assertTrue(prepared["all_precontact_observations_used"])
        self.assertFalse(prepared["uses_post_contact_observations"])
        self.assertEqual(prepared["fit"]["capture_t_s"], .99)

    def test_future_xyz_and_offline_labels_do_not_change_fit_or_input_hash(self):
        first = self.prepared()
        changed = dict(self.trace)
        changed["raw_xyz"] = self.trace["raw_xyz"].copy()
        changed["raw_xyz"][self.trace["raw_times"] >= 1.] = np.nan
        changed["eval_times"] = np.full(10, 999.)
        changed["eval_xyz"] = np.full((10, 3), np.nan)
        changed["offline_incident_velocity"] = [-999., -999., -999.]
        second = self.prepared(changed)
        self.assertEqual(first["input_sha256"], second["input_sha256"])
        np.testing.assert_array_equal(first["fit"]["state"], second["fit"]["state"])
        self.assertEqual(first["contact"]["contact_t_s"], second["contact"]["contact_t_s"])

    def test_reference_is_only_a_cutoff_and_not_a_contact_feature(self):
        first = self.prepared()
        # Two scenario boundaries in the same unobserved interval have the
        # same raw prefix. They must not force different predicted contacts.
        changed = dict(self.trace, bounce_reference_s=.995)
        second = self.prepared(changed)
        self.assertEqual(first["input_sha256"], second["input_sha256"])
        self.assertEqual(first["contact"]["contact_t_s"], second["contact"]["contact_t_s"])
        self.assertIs(self.calls[-1]["parameters"], self.bank.parameters)

    def test_subsampled_converged_fit_is_rejected(self):
        def reduced(*args):
            fit = self.fake_fit(*args)
            fit["diagnostics"]["sampled_rows"] = 72
            return fit
        with patch.object(runner, "estimate_prefix", reduced), \
             patch.object(runner, "predict_contact", self.fake_contact):
            with self.assertRaisesRegex(AssertionError, "no representative reduction"):
                runner.all_pre_fit(self.trace, self.bank)

    def test_nonconvergence_remains_explicit(self):
        fit = {"converged": False, "capture_t_s": .99, "state": None,
               "covariance": None, "diagnostics": {"reason": "optimizer_not_converged"}}
        with patch.object(runner, "estimate_prefix", return_value=fit), \
             patch.object(runner, "predict_contact", return_value={"valid": False,
                                                "reason": "prefix_not_converged"}):
            result = runner.all_pre_fit(self.trace, self.bank)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "prefix_not_converged")
        self.assertEqual(len(result["raw_prefix_times"]), 100)


if __name__ == "__main__":
    unittest.main()
