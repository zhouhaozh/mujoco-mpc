"""Data-isolation checks; no model fitting or scientific benchmark runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.group_fold_training import (
    FOLD_GROUPS, assert_training_ids, fold_plan, prior_configs, training_bundle)


class FoldIsolationTests(unittest.TestCase):
    def split(self):
        ids = ["trajectory_"+str(g) for g in range(11)]
        return {"train": ids[:7], "validation": ids[7:9], "test": ids[9:],
                "trajectory_to_group": dict(zip(ids, range(11)))}

    def test_each_group_held_out_once_and_never_crosses_a_fold(self):
        split = self.split()
        evaluations = []
        for fold in range(5):
            plan = fold_plan(split, fold)
            self.assertFalse(set(plan["train_ids"]) & set(plan["evaluation_ids"]))
            self.assertFalse(set(plan["train_groups"]) & set(plan["evaluation_groups"]))
            self.assertEqual(set(plan["train_ids"]+plan["evaluation_ids"]), set(split["trajectory_to_group"]))
            evaluations.extend(plan["evaluation_ids"])
            with self.assertRaises(ValueError):
                assert_training_ids(plan["train_ids"]+plan["evaluation_ids"][:1], plan)
        self.assertCountEqual(evaluations, list(split["trajectory_to_group"]))

    def test_training_loader_never_opens_heldout_data_even_if_paths_are_poisoned(self):
        plan = fold_plan(self.split(), 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"clean").mkdir()
            annotations = [{"trajectory_id": i, "cleaned_file": "POISON" if i in plan["evaluation_ids"] else i+".csv"}
                           for i in plan["train_ids"]+plan["evaluation_ids"]]
            (root/"clean/annotations.json").write_text(json.dumps(annotations))
            for identity in plan["train_ids"]:
                (root/"clean"/(identity+".csv")).write_text("training-only-source")
            opened = []
            def loader(path, **kwargs):
                self.assertNotEqual(path.name, "POISON")
                opened.append(path.name)
                return np.arange(8, dtype=float), np.zeros((8, 3))
            with patch("tennis_calibration.group_fold_training._load_clean", side_effect=loader):
                bundle = training_bundle(root, plan)
            self.assertEqual({r["id"] for r in bundle}, set(plan["train_ids"]))
            self.assertEqual(set(opened), {i+".csv" for i in plan["train_ids"]})

    def test_fold_prior_replaces_both_filter_and_prefix_historical_values(self):
        states = np.zeros((2, 9)); states[:, 6] = .3
        template = {"initial_magnus_sigma": 99., "magnus_drive_sigma": 99., "measurement_sigma_xyz": [.006, .006, .008]}
        cfg, prefix = prior_configs(states, template)
        self.assertAlmostEqual(cfg["initial_magnus_sigma"], np.sqrt(.03))
        self.assertAlmostEqual(cfg["magnus_drive_sigma"], .3*np.sqrt(.03))
        self.assertEqual(prefix["b_prior_sigma_per_s"], cfg["initial_magnus_sigma"])
        self.assertEqual(template["initial_magnus_sigma"], 99.)
        floor, pref = prior_configs(np.zeros((2, 9)), template)
        self.assertEqual(floor["initial_magnus_sigma"], .02)
        self.assertEqual(floor["magnus_drive_sigma"], .006)
        self.assertEqual(pref["b_prior_sigma_per_s"], .02)


if __name__ == "__main__":
    unittest.main()
