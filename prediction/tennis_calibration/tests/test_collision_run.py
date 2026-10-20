"""Integration audit of frozen-prefix input/label/clock boundaries.

Formal numerical execution is Ubuntu-only. These tests exercise public pipeline
functions, including real prefix ODE fitting and hybrid propagation. Supervised
outgoing labels may change targets, but joint incoming labels may not change
causal features, forecasts or event states.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tennis_calibration import collision_run as run
from tennis_calibration import identified_dynamics as d
from tennis_calibration.collision_mapping import fit_model, predict_model


class PipelineBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.parameters = {"k_drag": 0., "e_n": .8, "alpha_t": .55,
                           "impact_mode": "effective"}
        self.initial = np.array([0., 0., 1., 10., 1., -2., 0., 0., 0.])

    def trace(self, identity="synthetic/A", reference=.1):
        times = np.arange(161)*.005
        xyz = d.simulate(times, self.initial, 0)[..., :3]
        return {"id": identity, "split": "train", "groupid": 1,
                "raw_times": times, "raw_xyz": xyz,
                "eval_times": times, "eval_xyz": xyz.copy(),
                "bounce_reference_s": reference,
                "bounce_event_times_s": [reference],
                "eval_scope_end_s": .75,
                "impact_reference": {"p_b_m": [0., 0., .014],
                     "v_minus_m_s": [500., 1000., -300.],
                     "v_plus_m_s": [400., 800., 250.]}}

    def pack(self, trace=None, gap=0.):
        trace = self.trace() if trace is None else trace
        predicted_event = .2
        state_minus = d.step(self.initial, predicted_event, 0)
        return {"trace": trace, "gap": gap,
                "fit": {"converged": True, "capture_t_s": 0., "state": self.initial.tolist()},
                "base": {"input_count": 10, "last_input_t_s": 0.},
                "contact": {"valid": True, "reason": "predicted_contact",
                    "contact_t_s": predicted_event, "state_minus": state_minus.tolist(),
                    "position": state_minus[:3].tolist(), "v_minus": state_minus[3:6].tolist(),
                    "b_minus": state_minus[6:].tolist()}}

    def test_fit_case_future_and_clean_flags_cannot_change_fitted_state(self):
        trace = self.trace(reference=.35)
        a = run.fit_case(trace, .04, self.parameters, .014, 0.)
        changed = copy.deepcopy(trace)
        cutoff = .35-.04
        changed["raw_xyz"][changed["raw_times"] > cutoff] = [1e9, -1e9, -1e9]
        changed["eval_xyz"][:] = np.nan
        changed["keep"] = np.zeros(len(changed["raw_times"]), dtype=bool)
        changed["impact_reference"]["v_minus_m_s"] = [1e9, 1e9, -1e9]
        b = run.fit_case(changed, .04, self.parameters, .014, 0.)
        self.assertTrue(a["fit"]["converged"])
        self.assertEqual(a["input_sha256"], b["input_sha256"])
        np.testing.assert_array_equal(a["fit"]["state"], b["fit"]["state"])
        self.assertLessEqual(a["last_input_t_s"], cutoff)
        self.assertEqual(a["input_count"], int(np.sum(trace["raw_times"] <= cutoff)))

    def test_estimator_receives_only_cut_raw_arrays_and_fixed_parameters(self):
        trace = self.trace(reference=.35)
        captured = {}
        def spy(times, xyz, cutoff, parameters, height, config):
            captured.update(times=times.copy(), xyz=xyz.copy(), cutoff=cutoff,
                            parameters=dict(parameters), height=height, config=dict(config))
            return {"converged": False, "capture_t_s": None, "state": None, "diagnostics": {}}
        with patch.object(run, "estimate_prefix", side_effect=spy):
            run.fit_case(trace, .08, self.parameters, .014, .4)
        self.assertLessEqual(float(captured["times"].max()), .35-.08)
        np.testing.assert_array_equal(captured["xyz"], trace["raw_xyz"][trace["raw_times"] <= .35-.08])
        self.assertEqual(captured["parameters"], self.parameters)
        self.assertEqual(captured["config"], {"window_s": .4})
        self.assertNotIn("reference_contact", captured["parameters"])
        self.assertNotIn("gap", captured["config"])

    def test_joint_incident_label_changes_neither_features_nor_targets(self):
        pack = [self.pack()]
        labels = {pack[0]["trace"]["id"]: {"used_for_fit": True,
                    "v_minus_m_s": [1e8, 2e8, -3e8], "v_plus_m_s": [5.5, .55, 3.1]}}
        a = run.learning_arrays(pack, labels, self.parameters)
        labels[pack[0]["trace"]["id"]]["v_minus_m_s"] = [-1e9, -1e9, -1e9]
        b = run.learning_arrays(pack, labels, self.parameters)
        for left, right in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(a[1].shape[1], 5)
        np.testing.assert_allclose(a[4], [pack[0]["contact"]["v_minus"]])

    def test_outgoing_label_is_supervision_not_input_feature(self):
        pack = [self.pack()]
        identity = pack[0]["trace"]["id"]
        labels = {identity: {"used_for_fit": True, "v_plus_m_s": [5.5, .55, 3.1]}}
        a = run.learning_arrays(pack, labels, self.parameters)
        labels[identity]["v_plus_m_s"] = [6.5, .55, 4.1]
        b = run.learning_arrays(pack, labels, self.parameters)
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[4], b[4])
        np.testing.assert_array_equal(a[5], b[5])
        self.assertGreater(np.linalg.norm(a[2]-b[2]), 1.)

    def test_repeated_gap_variants_have_one_total_weight_per_event(self):
        ta, tb = self.trace("synthetic/A"), self.trace("synthetic/B")
        pack = [self.pack(ta, gap) for gap in [0., .005, .04]] + [self.pack(tb, .01)]
        impacts = {t["id"]: {"used_for_fit": True, "v_plus_m_s": [5.5, .55, 3.]} for t in [ta, tb]}
        active, _, _, weights, _, _ = run.learning_arrays(pack, impacts, self.parameters)
        for identity in [ta["id"], tb["id"]]:
            self.assertAlmostEqual(sum(w for c, w in zip(active, weights)
                if c["trace"]["id"] == identity), 1.)
        self.assertAlmostEqual(float(weights.sum()), 2.)

    def test_weighted_duplicate_prefixes_do_not_weaken_regularization(self):
        rng = np.random.default_rng(663)
        X, Y = rng.normal(size=(8, 5)), rng.normal(size=(8, 3))
        probe = rng.normal(size=(7, 5))
        for config in ({"kind": "ridge", "degree": 1, "ridge": .1},
                       {"kind": "gp", "length_scale": 2., "noise": .3}):
            first, _ = fit_model(X, Y, np.ones(8), config)
            repeated, _ = fit_model(np.repeat(X, 3, axis=0), np.repeat(Y, 3, axis=0),
                                    np.full(24, 1/3), config)
            np.testing.assert_allclose(predict_model(first, probe),
                                       predict_model(repeated, probe), atol=1e-9)

    def test_score_pack_uses_reference_absolute_targets_without_shifting_event(self):
        pack = [self.pack()]
        rows, events = run.score_pack(pack, self.parameters, .014, "fixed_carry")
        for row, horizon in zip(rows, run.HORIZONS):
            self.assertAlmostEqual(row["target_t_s"], .1+horizon)
        # First target .12 remains before the predicted .20 event. Its velocity
        # must be descending and its position is the capture-state free flight.
        expected = d.step(self.initial, .12, 0)
        np.testing.assert_allclose(rows[0]["predicted_xyz"], expected[:3], atol=1e-12)
        self.assertLess(rows[0]["velocity_predicted"][2], 0)
        self.assertEqual(events[0]["predicted_tb"], .2)

    def test_joint_event_labels_do_not_change_frozen_forecasts(self):
        a = self.pack()
        b = copy.deepcopy(a)
        b["trace"]["impact_reference"].update(v_minus_m_s=[1e9, 1e9, -1e9],
              v_plus_m_s=[-1e9, -1e9, 1e9], p_b_m=[1e9, 1e9, 1e9])
        rows_a, events_a = run.score_pack([a], self.parameters, .014, "fixed_carry")
        rows_b, events_b = run.score_pack([b], self.parameters, .014, "fixed_carry")
        for left, right in zip(rows_a, rows_b):
            np.testing.assert_array_equal(left["predicted_xyz"], right["predicted_xyz"])
            np.testing.assert_array_equal(left["velocity_predicted"], right["velocity_predicted"])
        for key in ("predicted_tb", "predicted_pb", "predicted_v_minus", "predicted_v_plus"):
            np.testing.assert_array_equal(events_a[0][key], events_b[0][key])

    def test_failed_contact_keeps_valid_targets_and_failure_reason(self):
        pack = [self.pack()]
        pack[0]["contact"] = {"valid": False, "reason": "no_contact_within_horizon"}
        rows, events = run.score_pack(pack, self.parameters, .014, "failed")
        self.assertTrue(any(row["valid"] for row in rows))
        self.assertTrue(all(row["predicted_xyz"] is None for row in rows))
        self.assertTrue(all(row["prediction_reason"] == "no_contact_within_horizon" for row in rows))
        self.assertEqual(run.selection_score(rows)[0], sum(row["valid"] for row in rows))
        self.assertIsNone(events[0]["predicted_tb"])

    def test_selection_is_macro_per_trajectory_not_row_frequency(self):
        row_a = {"valid": True, "trajectory_id": "A", "target_xyz": [0., 0., 0.],
                 "predicted_xyz": [1., 0., 0.]}
        row_b = {"valid": True, "trajectory_id": "B", "target_xyz": [0., 0., 0.],
                 "predicted_xyz": [3., 0., 0.]}
        score = run.selection_score([row_a]*100 + [row_b])
        self.assertEqual(score, (0, 2.))

    def test_cache_audit_metadata_never_runs_estimator(self):
        trace = self.trace(reference=.35)
        with patch.object(run, "estimate_prefix", side_effect=AssertionError("audit refitted")):
            audit = run.fit_case(trace, .04, self.parameters, .014, 0., audit_only=True)
        self.assertNotIn("fit", audit)
        self.assertEqual(len(audit["context_sha256"]), 64)
        self.assertEqual(len(audit["input_sha256"]), 64)
        context = audit["context_payload"]
        self.assertEqual(context["parameters"], self.parameters)
        self.assertEqual(context["contact_height_m"], .014)
        self.assertEqual(context["window_s"], 0.)
        self.assertEqual(set(context["source_sha256"]),
            {"tennis_calibration/collision_prefix.py", "tennis_calibration/identified_dynamics.py",
             "tennis_calibration/flight.py"})
        self.assertIn("b_prior_sigma_per_s", context["prefix_defaults"])

    def cache_seed(self, out):
        trace = self.trace(reference=.35)
        audit = run.fit_case(trace, 0., self.parameters, .014, 0., audit_only=True)
        audit["fit"] = {"state": [0.]*9, "converged": False}
        run.dump(out/"prefix_cache.json", {"full": {run.identity(trace, 0.): audit}, "recent": {}})
        return trace

    def test_matching_cache_reuse_never_refits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); trace = self.cache_seed(out)
            with patch.object(run, "estimate_prefix", side_effect=AssertionError("cache refitted")):
                cache = run.cache_cases([trace], self.parameters, .014, out, [("full", trace, [0.])])
            self.assertIn(run.identity(trace, 0.), cache["full"])

    def test_changed_parameter_or_height_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); trace = self.cache_seed(out)
            changed = {**self.parameters, "k_drag": .03}
            for parameters, height in [(changed, .014), (self.parameters, .025)]:
                with self.assertRaisesRegex(RuntimeError, "context mismatch"):
                    run.cache_cases([trace], parameters, height, out, [("full", trace, [0.])])

    def test_changed_raw_prefix_cache_is_rejected_but_future_mutation_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); trace = self.cache_seed(out)
            future = copy.deepcopy(trace)
            future["raw_xyz"][future["raw_times"] > .35] = 1e9
            run.cache_cases([future], self.parameters, .014, out, [("full", future, [0.])])
            changed = copy.deepcopy(trace); changed["raw_xyz"][10, 0] += .001
            with self.assertRaisesRegex(RuntimeError, "raw-prefix fingerprint mismatch"):
                run.cache_cases([changed], self.parameters, .014, out, [("full", changed, [0.])])

    def test_legacy_cache_without_context_requires_explicit_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); trace = self.cache_seed(out)
            import json
            cache = json.loads((out/"prefix_cache.json").read_text())
            del cache["full"][run.identity(trace, 0.)]["context_sha256"]
            run.dump(out/"prefix_cache.json", cache)
            with self.assertRaisesRegex(RuntimeError, "explicit verified migration"):
                run.cache_cases([trace], self.parameters, .014, out, [("full", trace, [0.])])


if __name__ == "__main__":
    unittest.main()
