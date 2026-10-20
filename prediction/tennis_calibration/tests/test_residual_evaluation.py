"""Residual replay isolation and unchanged parent-label/clock checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_calibration.residual_evaluation import (
    apply_head_to_rows, read_cache, training_arrays,
)
from tennis_calibration.residual_head import FEATURE_NAMES, fit_residual_head


def cache_fixture():
    n = 12
    x = np.random.default_rng(19).normal(size=(n, len(FEATURE_NAMES)))
    capture = np.arange(n)*.02+.1
    pred = np.column_stack((capture, capture*.2, capture+1.))
    reference = pred+np.column_stack((.01*x[:, 0], .02+0*x[:, 1], -.01*x[:, 2]))
    return {"X": x, "pred": pred, "ref": reference, "H": np.full(n, .5),
        "gate": np.array([0., .5, 1., 1., 1., 1., 0., .5, 1., 1., 1., 1.]),
        "ids": np.array(["train"]*6+["heldout"]*6), "capture_t_s": capture,
        "elapsed_capture_s": capture, "relative_capture_s": capture-.1,
        "target_t_s": capture+.5, "anchor": np.ones(n, dtype=bool),
        "dense": np.zeros(n, dtype=bool), "valid": np.ones(n, dtype=bool),
        "initialized": np.ones(n, dtype=bool), "prediction_ok": np.ones(n, dtype=bool),
        "split": np.array(["train"]*6+["validation"]*6),
        "groupid": np.array(["0"]*6+["1"]*6), "candidate": np.array(["baseline"]*n)}


def rows_from(cache):
    rows = []
    for i in range(len(cache["ids"])):
        rows.append({"trajectory_id": str(cache["ids"][i]), "candidate": "baseline",
            "split": str(cache["split"][i]), "valid": True,
            "capture_t_s": float(cache["capture_t_s"][i]), "horizon_s": .5,
            "target_t_s": float(cache["target_t_s"][i]), "prediction_reason": "",
            "predicted_xyz": cache["pred"][i].tolist(), "target_xyz": cache["ref"][i].tolist()})
    return rows


class ResidualEvaluationTests(unittest.TestCase):
    def test_training_selects_only_explicit_ids_fully_open_gate_and_anchors(self):
        cache = cache_fixture()
        cache["anchor"][3] = False
        x, y, ids, audit = training_arrays(cache, {"train"})
        self.assertEqual(ids, ["train"]*3)
        self.assertEqual(audit["eligible_training_rows"], 3)
        np.testing.assert_array_equal(x, cache["X"][[2, 4, 5]])
        changed = deepcopy(cache)
        changed["X"][6:] += 1e6
        changed["ref"][6:] += 1e6
        x2, y2, _, _ = training_arrays(changed, {"train"})
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(y, y2)

    def test_residual_preserves_closed_gate_labels_and_times(self):
        cache = cache_fixture()
        x, y, ids, _ = training_arrays(cache, {"train"})
        head = fit_residual_head("ridge", x, y, ids, allowed_train_ids={"train"})
        source = rows_from(cache)[6:]
        frozen = deepcopy(source)
        output, audit = apply_head_to_rows(head, cache, source, "baseline", "baseline__ridge")
        self.assertTrue(audit["zero_gate_rows_unchanged"])
        self.assertEqual(output[0]["predicted_xyz"], source[0]["predicted_xyz"])
        for before, after in zip(source, output):
            self.assertEqual(before["target_xyz"], after["target_xyz"])
            self.assertEqual(before["capture_t_s"], after["capture_t_s"])
            self.assertEqual(before["target_t_s"], after["target_t_s"])
            self.assertLessEqual(np.linalg.norm(np.asarray(after["predicted_xyz"])-before["predicted_xyz"]), .15+1e-12)
        self.assertEqual(source, frozen)

    def test_changed_cache_parent_predictions_are_rejected(self):
        cache = cache_fixture()
        x, y, ids, _ = training_arrays(cache, {"train"})
        head = fit_residual_head("ridge", x, y, ids, allowed_train_ids={"train"})
        source = rows_from(cache)[6:]
        source[0]["predicted_xyz"][0] += .01
        with self.assertRaises(AssertionError):
            apply_head_to_rows(head, cache, source, "baseline", "baseline__ridge")

    def test_npz_schema_and_parent_are_checked(self):
        cache = cache_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "cache_baseline.npz")
            np.savez_compressed(path, **cache)
            restored = read_cache(path, "baseline")
            np.testing.assert_array_equal(restored["X"], cache["X"])
            with self.assertRaisesRegex(ValueError, "requested observer"):
                read_cache(path, "another")


if __name__ == "__main__":
    unittest.main()
