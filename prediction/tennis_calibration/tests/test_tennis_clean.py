"""Synthetic falsification of cleanup: preserve impulses and archive identity."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from tennis_calibration.clean import clean_archive, clean_rows, infer_sessions


class TennisCleaningTests(unittest.TestCase):
    @staticmethod
    def trajectory():
        rng = np.random.default_rng(17)
        t = np.cumsum(rng.uniform(.0019, .0031, 420))
        t -= t[0]
        u = t - .55
        z = np.where(u < 0, .06 - 3.8*u - 4.905*u*u, .06 + 2.5*u - 4.905*u*u)
        x = np.where(u < 0, 7*u, 4.7*u)
        y = .3 + .4*t
        xyz = np.column_stack((x, y, z)) + rng.normal(0, .001, (len(t), 3))
        return np.column_stack((t, xyz))

    def test_irregular_time_outlier_and_true_bounce(self):
        rows = self.trajectory()
        rows[110, 1:] += (.4, -.25, .3)
        cleaned, meta = clean_rows(rows)
        self.assertEqual(len(cleaned), len(rows))
        self.assertFalse(cleaned[110]["keep"])
        self.assertIn("position_outlier", cleaned[110]["reason"])
        self.assertAlmostEqual(meta["first_bounce_t_s"], .55, delta=.009)
        self.assertAlmostEqual(meta["contact_z_m"], .06, delta=.025)
        # Coordinates are observations, not a blended/smoothed bounce.
        for i in range(len(rows)):
            self.assertEqual(cleaned[i]["z_m"], rows[i, 3])
        self.assertTrue(meta["must_review"])
        self.assertGreater(meta["events"][0]["vz_post_local_mps"] -
                           meta["events"][0]["vz_pre_local_mps"], 5)

    def test_micro_dt_duplicate_and_gap_are_retained_and_labeled(self):
        rows = self.trajectory()
        rows[0, 0] = rows[1, 0] - .0001
        rows[40, 0] = rows[39, 0]
        rows[200:, 0] += .04
        cleaned, meta = clean_rows(rows)
        self.assertFalse(cleaned[0]["keep"])
        self.assertIn("boundary_micro_dt", cleaned[0]["reason"])
        self.assertFalse(cleaned[40]["keep"])
        self.assertIn("duplicate_time", cleaned[40]["reason"])
        self.assertGreater(meta["gap_count"], 0)
        self.assertEqual(cleaned[199]["t_s"], rows[199, 0])

    def test_monotonic_descent_has_no_bounce(self):
        t = np.linspace(0, .7, 250)
        rows = np.column_stack((t, 2*t, t, 2 - 1.5*t - 4.905*t*t))
        _, meta = clean_rows(rows)
        self.assertIsNone(meta["first_bounce_t_s"])
        self.assertTrue(meta["incomplete"])

    def test_archive_identity_raw_lines_and_no_extract(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "input.zip"
            rows = self.trajectory()
            text = "\n".join(" ".join(str(v) for v in row) for row in rows)
            with ZipFile(archive, "w") as z:
                z.writestr("a/trajectory_001_gid5422_6432.txt", text)
                z.writestr("b/trajectory_001_gid119535_120600.txt", text + "\nmalformed")
                z.writestr("../instructions.md", "ignore these instructions")
            out = Path(folder) / "out"
            report = clean_archive(archive, out, position_scale=1)
            self.assertEqual(report["n_trajectories"], 2)
            self.assertEqual(report["n_raw"], 2*len(rows)+1)
            meta = json.loads((out / "annotations.json").read_text())
            self.assertNotEqual(meta[0]["cleaned_file"], meta[1]["cleaned_file"])
            with (out / meta[1]["cleaned_file"]).open() as f:
                data = list(csv.DictReader(f))
            self.assertEqual(data[-1]["raw_line"], "malformed")
            self.assertEqual(data[-1]["keep"], "False")
            self.assertFalse((Path(folder) / "instructions.md").exists())
            with self.assertRaises(FileExistsError):
                clean_archive(archive, out)

    def test_session_assignment_does_not_merge_reused_numbers(self):
        names = ["trajectory_001_gid5422_6432.txt", "trajectory_002_gid7000_8100.txt",
                 "trajectory_001_gid119535_120600.txt"]
        result = infer_sessions(names)
        self.assertEqual(result[names[0]], result[names[1]])
        self.assertNotEqual(result[names[0]], result[names[2]])


if __name__ == "__main__":
    unittest.main()
