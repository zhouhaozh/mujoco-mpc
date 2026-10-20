"""Ubuntu纯stdlib fixture：独立H缓存、完整性、旧缓存数值对照；不启动仿真。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_time_window_caches as preparation
from pipeline_experiment import digest


class FakeBundle:
    def __init__(self, payload, path=None):
        self.payload = deepcopy(payload)
        self.path = path
        identity = {"horizon": payload["horizon"], "experiment": payload.get("experiment", False),
                    "weights_sha256": payload["resources"]["weights"]["sha256"], "transform": payload["transform"]}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text())
        resource = payload["resources"]["weights"]
        if digest(resource["path"]) != resource["sha256"]:
            raise ValueError("fixture model resource SHA256 mismatch")
        return cls(payload, path)

    def with_forecast_horizon(self, horizon):
        if not .8 <= horizon <= 2. or abs(horizon/.02-round(horizon/.02)) > 1e-9:
            raise ValueError("fixture experimental horizon outside domain")
        return type(self)({**self.payload, "horizon": horizon, "experiment": True})

    @property
    def effective_config(self):
        return {"transform": deepcopy(self.payload["transform"])}

    @property
    def resolved_manifest(self):
        resources = deepcopy(self.payload["resources"])
        resources["weights"]["resolved_path"] = resources["weights"]["path"]
        return {"fingerprint": self.fingerprint, "resources": resources,
                "manifest_path": None if self.path is None else str(self.path),
                "settings": {"forecast": {"max_horizon_s": self.payload["horizon"]},
                             "residual": {"maximum_trained_horizon_s": .7}}}

    @property
    def selection(self):
        return {"candidate": "fixture", "max_horizon_s": 1.5}

    def write(self, path):
        path = Path(path)
        with path.open("x") as stream:
            json.dump(self.payload, stream)
        return path


class FakePredictor:
    instances = []

    def __init__(self, *, bundle):
        self.bundle = bundle
        self.instances.append(self)

    def update(self, seq, stamp, xyz):
        self.seq, self.stamp, self.xyz = seq, stamp, list(xyz)
        return {"initialized": seq >= 2, "seq": seq,
                "state": [*xyz, 1., 2., 3.] if seq >= 2 else None,
                "hold_active": seq == 2, "reason": "fixture_received"}

    def forecast(self, seq):
        assert seq == self.seq
        horizon = self.bundle.payload["horizon"]
        grid = [[self.stamp+i*.02, self.xyz[0]+i*.02, self.xyz[1], self.xyz[2],
                 1., 0., 0., False] for i in range(round(horizon/.02)+1)]
        return {"seq": seq, "last_capture_t_s": self.stamp, "model_version": "fixture",
                "max_horizon_s": horizon, "valid_until": grid[-1][0], "domain_valid": True,
                "next_bounce_t_s": -1., "height_contact_t_s": None,
                "hold_active": seq == 2, "resume_t_s": None, "future_covariance": None,
                "grid": grid, "forecast_compute_ms": 10.+seq}


class TimeWindowCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="horizon-cache-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.transform = {"R_world_from_model": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          "translation_world_from_model_m": [0, 0, .1]}
        (self.source / "transform.json").write_text(json.dumps(self.transform))
        self.csv = self.source / "test.csv"
        self.csv.write_text("t_s,x_m,y_m,z_m\n0,1,2,3\n0.02,2,3,4\n0.04,3,4,5\n")
        self.item = {"sample_id": "test_01", "trajectory_id": "fixture_1", "csv": str(self.csv),
                     "source_sha256": digest(self.csv), "input_frames": 3}
        (self.source / "input_manifest.json").write_text(json.dumps({"items": [self.item], "transform": self.transform}))
        self.prediction = self.root / "prediction"
        (self.prediction / ".venv-tennis/bin").mkdir(parents=True)
        (self.prediction / ".venv-tennis/bin/python").write_text("fixture prediction interpreter")
        self.code = self.root / "prediction_source"
        (self.code / "tennis_calibration").mkdir(parents=True)
        (self.code / "tennis_calibration/fixture.py").write_text("# fixed prediction source\n")
        self.weights = self.root / "weights.fixture"
        self.weights.write_text("immutable fixture weights\n")
        self.bundle = self.root / "base_bundle.json"
        self.bundle.write_text(json.dumps({"horizon": 1.5, "transform": self.transform,
            "resources": {"weights": {"path": str(self.weights), "sha256": digest(self.weights)}}}))
        FakePredictor.instances = []
        self.components = patch.object(preparation, "predictor_components", return_value=(FakeBundle, FakePredictor, self.code))
        self.components.start()
        self.addCleanup(self.components.stop)

    def prepare(self, out, horizons=(.8, 1.5, 2.), **kwargs):
        return preparation.prepare(self.bundle, self.prediction, self.source, out, horizons,
                                   workers=1, **kwargs)

    def test_every_horizon_uses_fresh_tracker_and_preserves_original_inputs(self):
        protected = {path: path.read_bytes() for path in (self.csv, self.bundle, self.weights,
                     self.source / "input_manifest.json", self.source / "transform.json")}
        out = self.root / "new_caches"
        result = self.prepare(out)
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(len(FakePredictor.instances), 3)
        self.assertEqual(len({id(predictor) for predictor in FakePredictor.instances}), 3)
        for horizon in (.8, 1.5, 2.):
            directory = out / preparation.horizon_tag(horizon)
            cache = directory / "cache/test_01.jsonl"
            rows = [json.loads(row) for row in cache.read_text().splitlines()]
            self.assertEqual([row["seq"] for row in rows], [1, 2, 3])
            self.assertIsNone(rows[0]["forecast"])
            self.assertEqual(rows[1]["forecast"]["max_horizon_s"], horizon)
            self.assertTrue(rows[1]["forecast"]["hold_active"])
            self.assertAlmostEqual(rows[1]["forecast"]["valid_until"], .02+horizon)
            self.assertEqual(len(rows[1]["forecast"]["grid"]), round(horizon/.02)+1)
            manifest = json.loads((directory / "cache/manifest.json").read_text())
            self.assertEqual(manifest["files"][0]["sha256"], digest(cache))
            resolved = json.loads((directory / "resolved_manifest.json").read_text())
            self.assertEqual(resolved["settings"]["residual"]["maximum_trained_horizon_s"], .7)
        for path, original in protected.items():
            self.assertEqual(path.read_bytes(), original)

    def test_only_complete_matching_identity_and_unchanged_outputs_can_be_skipped(self):
        out = self.root / "caches"
        first = self.prepare(out)
        calls = len(FakePredictor.instances)
        before = (out / "manifest.json").read_bytes()
        self.assertEqual(self.prepare(out), first)
        self.assertEqual(len(FakePredictor.instances), calls)
        self.assertEqual((out / "manifest.json").read_bytes(), before)
        with self.assertRaisesRegex(FileExistsError, "configuration"):
            self.prepare(out, horizons=(1.5,))
        cache = out / "horizon_0800ms/cache/test_01.jsonl"
        cache.write_text("tampered output\n")
        with self.assertRaisesRegex(ValueError, "output missing/changed"):
            self.prepare(out)
        self.assertEqual(cache.read_text(), "tampered output\n")

    def test_incomplete_output_is_preserved_instead_of_skipped_or_overwritten(self):
        out = self.root / "incomplete"
        out.mkdir()
        original = out / "partial.jsonl"
        original.write_text("incomplete prior work\n")
        with self.assertRaisesRegex(FileExistsError, "incomplete run preserved"):
            self.prepare(out)
        self.assertEqual(original.read_text(), "incomplete prior work\n")
        self.assertFalse((out / "manifest.json").exists())

    def test_original_cache_numerical_comparison_ignores_only_compute_time(self):
        first = self.root / "reference_generation"
        self.prepare(first, horizons=(1.5,))
        reference = self.root / "old_cache"
        reference.mkdir()
        rows = [json.loads(row) for row in (first / "horizon_1500ms/cache/test_01.jsonl").read_text().splitlines()]
        rows[1]["forecast"]["forecast_compute_ms"] += 999.
        reference_file = reference / "test_01.jsonl"
        reference_file.write_text("".join(json.dumps(row)+"\n" for row in rows))
        original = reference_file.read_bytes()
        out = self.root / "comparison"
        self.prepare(out, horizons=(1.5,), baseline_cache=reference)
        record = json.loads((out / "horizon_1500ms/cache/manifest.json").read_text())["files"][0]
        self.assertEqual(record["baseline_comparison"]["rows"], 3)
        self.assertEqual(record["baseline_comparison"]["max_abs_delta"], 0.)
        self.assertEqual(reference_file.read_bytes(), original)
        rows[1]["forecast"]["grid"][0][1] += .1
        bad = self.root / "bad_reference.jsonl"
        bad.write_text("".join(json.dumps(row)+"\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "baseline cache differs"):
            preparation.compare_cache_files(out / "horizon_1500ms/cache/test_01.jsonl", bad)

    def test_frame_mismatch_changed_raw_and_protected_output_locations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "separate from the raw source"):
            self.prepare(self.source / "cache")
        original = self.csv.read_bytes()
        self.csv.write_bytes(original+b"0.06,4,5,6\n")
        with self.assertRaisesRegex(ValueError, "raw source SHA256"):
            self.prepare(self.root / "changed_source")
        self.csv.write_bytes(original)
        bad_transform = {**self.transform, "translation_world_from_model_m": [0, 0, .2]}
        (self.source / "transform.json").write_text(json.dumps(bad_transform))
        with self.assertRaisesRegex(ValueError, "differs from the bundle"):
            self.prepare(self.root / "bad_frame")


if __name__ == "__main__":
    unittest.main()
