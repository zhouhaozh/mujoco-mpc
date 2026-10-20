"""Serialization and OOF scoring checks independent of model execution."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tennis_calibration.group_fold_evaluation import read_json_csv, summarize


class FoldEvaluationTests(unittest.TestCase):
    def test_compressed_rows_preserve_null_boolean_array_and_original_split(self):
        expected = {"trajectory_id": "abc", "split": "oof", "original_split": "train", "fold": 2,
                    "height_contact_t_s": None, "valid": True, "predicted_xyz": [.1, .2, .3],
                    "prediction_reason": "", "hold_ms": 40, "capture_t_s": .02}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"rows.csv.gz"
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(expected)); writer.writeheader()
                writer.writerow({k: json.dumps(v) for k, v in expected.items()})
            self.assertEqual(read_json_csv(path), [expected])

    def test_oof_combines_original_splits_and_first200_uses_trajectory_weights(self):
        events, rows = [], []
        for identity, original, values in (("a", "train", [.1, .3]), ("b", "test", [.5])):
            events.append({"trajectory_id": identity, "split": "oof", "candidate": "gp_hold100ms",
                "diagnostic_resume_or_event_t_s": .5, "tracking_failure": ""})
            for index, value in enumerate(values):
                rows.append({"trajectory_id": identity, "split": "oof", "original_split": original,
                    "groupid": 1 if identity == "a" else 2, "candidate": "gp_hold100ms", "horizon_s": .5,
                    "capture_t_s": 1.+index*.02, "target_t_s": 1.5+index*.02,
                    "elapsed_capture_s": 1., "relative_capture_s": .1+index*.02,
                    "valid": True, "prediction_reason": "", "target_xyz": [0.,0.,0.],
                    "predicted_xyz": [value,0.,0.]})
        result = summarize(rows, events)
        self.assertEqual({r["split"] for r in result["ranking"]}, {"oof"})
        metric = result["first200ms_h500_metrics"][0]
        self.assertEqual(metric["N"], 2)
        self.assertAlmostEqual(metric["mean_m"], .35)
        self.assertAlmostEqual(metric["p95_frame_m"], .5)
        self.assertEqual(result["absolute_labeled_coverage"][0]["prediction_coverage_given_label"], 1.)


if __name__ == "__main__":
    unittest.main()
