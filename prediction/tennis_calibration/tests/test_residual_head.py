"""Causality, training isolation and bounded small-head invariants."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_calibration.residual_head import (
    FEATURE_NAMES, ResidualHead, apply_residual, causal_features,
    fit_residual_head, residual_gate,
)


def snapshot(t=.30):
    return {"state": np.array([1., .1, .4, 3., .2, 1., .1, -.02, .01]),
            "t_s": t, "height_contact_t_s": .1, "height_discovery_t_s": .102,
            "resume_t_s": .2, "bounced": True, "hold_active": False}


def dataset():
    generator = np.random.default_rng(88)
    x = generator.normal(size=(24, len(FEATURE_NAMES)))
    y = np.column_stack((.02*x[:, 0]+.01, -.01*x[:, 1], .005*x[:, 2]))
    ids = ["a"]*8+["b"]*16
    return x, y, ids


class ResidualHeadTests(unittest.TestCase):
    def test_feature_uses_only_received_and_does_not_mix_precontact_history(self):
        current = snapshot(.15)
        current["resume_t_s"] = None
        current["hold_active"] = True
        times = np.arange(.05, .1501, .005)
        points = np.column_stack((3*times, .2*times, times))
        forecast = current["state"][:3]+[.3, .02, .1]
        original = causal_features(current, forecast, .1, times, points)
        changed = points.copy()
        changed[times < .1-1e-12] += 100.
        np.testing.assert_array_equal(original, causal_features(current, forecast, .1, times, changed))
        self.assertEqual(original.shape, (len(FEATURE_NAMES),))
        self.assertAlmostEqual(original[FEATURE_NAMES.index("local_vx")], 3.)
        with self.assertRaisesRegex(ValueError, "future observations"):
            causal_features(current, forecast, .1, np.r_[times, .16], np.vstack((points, points[-1])))
        future = deepcopy(current)
        future["height_discovery_t_s"] = .2
        with self.assertRaisesRegex(ValueError, "future height-event"):
            causal_features(future, forecast, .1, times, points)

    def test_gate_is_zero_before_contact_during_hold_and_initial_resume(self):
        current = snapshot()
        for stamp, expected in ((.20, 0.), (.25, 0.), (.275, .5), (.30, 1.)):
            current["t_s"] = stamp
            self.assertAlmostEqual(residual_gate(current), expected)
        current["hold_active"] = True
        self.assertEqual(residual_gate(current), 0.)
        current["hold_active"] = False
        current["bounced"] = False
        self.assertEqual(residual_gate(current), 0.)
        current["bounced"] = True
        current["resume_t_s"] = None
        self.assertEqual(residual_gate(current), 0.)

    def test_explicit_training_membership_rejects_heldout_rows(self):
        x, y, ids = dataset()
        with self.assertRaisesRegex(ValueError, "non-training trajectory"):
            fit_residual_head("ridge", x, y, ids, allowed_train_ids={"a"})
        with self.assertRaisesRegex(ValueError, "nonempty training"):
            fit_residual_head("ridge", x[:0], y[:0], [], allowed_train_ids={"a"})

    def test_trace_equal_weighting_is_invariant_to_duplicating_one_trace(self):
        x, y, ids = dataset()
        one = fit_residual_head("ridge", x, y, ids, allowed_train_ids={"a", "b"})
        two = fit_residual_head("ridge", np.vstack((x, x[:8])), np.vstack((y, y[:8])),
                                ids+["a"]*8, allowed_train_ids={"a", "b"})
        np.testing.assert_allclose(one.mean, two.mean, atol=1e-14)
        np.testing.assert_allclose(one.scale, two.scale, atol=1e-14)
        np.testing.assert_allclose(one.predict(x), two.predict(x), atol=1e-12)
        expected_mean = (np.mean(x[:8], axis=0)+np.mean(x[8:], axis=0))/2
        np.testing.assert_allclose(one.mean, expected_mean, atol=1e-14)

    def test_serialization_predicts_identically(self):
        x, y, ids = dataset()
        head = fit_residual_head("ridge", x, y, ids, allowed_train_ids={"a", "b"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "head.json")
            head.save(path)
            restored = ResidualHead.load(path)
        np.testing.assert_array_equal(head.predict(x), restored.predict(x))
        self.assertEqual(restored.metadata["observed_training_ids"], ["a", "b"])

    def test_correction_is_capped_and_does_not_mutate_state_or_input(self):
        x, y, ids = dataset()
        head = fit_residual_head("ridge", x, y+10., ids, allowed_train_ids={"a", "b"})
        current, forecast = snapshot(), np.array([1., .2, .5])
        prior = deepcopy(current)
        before = forecast.copy()
        for horizon, bound in ((0., 0.), (.1, .03), (.5, .15)):
            result = apply_residual(head, x[0], current, horizon, forecast)
            self.assertLessEqual(np.linalg.norm(result-forecast), bound+1e-12)
        batch = head.predict_bounded(x[:3], np.array([0., .1, .5]))
        np.testing.assert_allclose(np.linalg.norm(batch, axis=1), [0., .03, .15], atol=1e-12)
        np.testing.assert_array_equal(current["state"], prior["state"])
        np.testing.assert_array_equal(forecast, before)
        current["hold_active"] = True
        np.testing.assert_array_equal(apply_residual(head, x[0], current, .5, forecast), forecast)

    def test_tiny_mlp_is_deterministic_and_learns_train_only_signal(self):
        x, y, ids = dataset()
        one = fit_residual_head("mlp", x, y, ids, allowed_train_ids={"a", "b"}, epochs=100)
        two = fit_residual_head("mlp", x, y, ids, allowed_train_ids={"a", "b"}, epochs=100)
        np.testing.assert_array_equal(one.predict(x), two.predict(x))
        self.assertLess(np.mean((one.predict(x)-y)**2), np.mean(y*y))


if __name__ == "__main__":
    unittest.main()
