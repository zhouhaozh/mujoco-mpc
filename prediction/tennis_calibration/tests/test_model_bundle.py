"""Bundle integrity/relocation tests with temporary fake resources; run on Ubuntu.

These tests never load recorded trajectories or real model weights. Numerical
legacy/bundle equivalence remains in test_controller_integration on Ubuntu.
"""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tennis_calibration.model_bundle import (
    COLLISION_RESOURCES, LEGACY_PATHS, ModelBundle, bundle_fingerprint, main,
)


class ModelBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.selection_path = self.root / "controller/selected.json"
        self.frame = {"R_world_from_model": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                      "translation_world_from_model_m": [2., -1., .1]}
        self.selected = {
            "schema": "CONTROLLER_PREDICTOR/1", "candidate": "gp__adaptive_protected__mlp",
            "model_id": "gp", "experiment": "adaptive_protected", "hold_s": .1,
            "max_horizon_s": 1.5, "head": "controller/models/head.json",
        }
        self.write_json(self.selection_path, self.selected)
        self.write_json(self.root / self.selected["head"],
                        {"metadata": {"maximum_trained_horizon_s": .7}})
        self.write_json(self.root / LEGACY_PATHS["collision"], {
            "schema": "FROZEN_PRECONTACT/1", "upstream": {"height": .012},
            "parameters": {"k_drag": .02, "e_n": .8, "alpha_t": .5},
            "prefix_config": {},
            "selected_model": {"b_policy": "carry", "lift_model": None, "velocity_model": "velocity_3"},
        })
        for filename in ("velocity_0", "velocity_1", "velocity_3"):
            path = (self.root / LEGACY_PATHS["collision"]).parent / "models" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("temporary fake " + filename).encode())
        self.write_json(self.root / LEGACY_PATHS["identified"], {
            "schema": "TENNIS_IDENTIFIED/1", "parameters": {"k_drag": 999.},
            "filter": {"measurement_sigma_xyz": [.01, .02, .03]},
        })
        self.write_json(self.root / LEGACY_PATHS["noise"], {
            "schema": "TRAIN_ONLY_NOISE_PROXY/1", "sigma_xyz_m": [.01, .02, .03],
        })

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def make(self, **kwargs):
        return ModelBundle.from_legacy(self.selection_path, root=self.root,
                                       synthetic_world=True, **kwargs)

    def file_hashes(self):
        return {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.root.rglob("*") if path.is_file()}

    def test_manifest_is_complete_and_all_file_paths_are_relative(self):
        bundle = self.make()
        path = bundle.write(self.root / "deployment/model_bundle.json")
        manifest = json.loads(path.read_text())
        self.assertEqual(set(manifest["resources"]), {
            "selection", "collision", "identified", "noise", "head", "transform",
            *COLLISION_RESOURCES.values(),
        })
        for descriptor in manifest["resources"].values():
            self.assertEqual(len(descriptor["sha256"]), 64)
            if "path" in descriptor:
                self.assertFalse(Path(descriptor["path"]).is_absolute())
        loaded = ModelBundle.load(path)
        self.assertEqual(loaded.fingerprint, bundle.fingerprint)
        self.assertEqual(loaded.settings["observer"]["hold_s"], .1)
        self.assertTrue(loaded.settings["observer"]["experiment_flags"]["protection"])
        self.assertEqual(loaded.settings["observer"]["fixed_settings"]["recovery_window_s"], .080)
        self.assertIn("max_step_s", loaded.settings["observer"]["filter_config"])
        self.assertEqual(loaded.settings["residual"]["maximum_trained_horizon_s"], .7)
        self.assertEqual(loaded.settings["frame"]["translation_world_from_model_m"], [0., 0., .033-.012])

    def test_every_file_resource_including_unselected_collision_heads_is_checked(self):
        path = self.make().write(self.root / "bundle.json")
        pristine = ModelBundle.load(path)
        for name, descriptor in pristine.manifest["resources"].items():
            if "path" not in descriptor:
                continue
            with self.subTest(resource=name):
                resource = pristine.resource_path(name)
                original = resource.read_bytes()
                try:
                    resource.write_bytes(original + b"tampered")
                    with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                        ModelBundle.load(path)
                finally:
                    resource.write_bytes(original)

    def test_missing_collision_child_fails_without_implicit_fallback(self):
        path = self.make().write(self.root / "bundle.json")
        ModelBundle.load(path).resource_path("collision_velocity_1").unlink()
        with self.assertRaises(FileNotFoundError):
            ModelBundle.load(path)

    def test_resolution_does_not_depend_on_working_directory(self):
        path = self.make().write(self.root / "deployment/bundle.json")
        previous = Path.cwd()
        try:
            os.chdir(self.root / "controller")
            loaded = ModelBundle.load(path)
        finally:
            os.chdir(previous)
        self.assertEqual(loaded.resource_path("selection"), self.selection_path.resolve())

    def test_rebasing_preserves_content_fingerprint_and_does_not_modify_resources(self):
        before = self.file_hashes()
        bundle = self.make()
        first = bundle.write(self.root / "deployment/first.json")
        second = ModelBundle.load(first).write(self.root / "other/nested/second.json")
        self.assertEqual(ModelBundle.load(first).fingerprint, ModelBundle.load(second).fingerprint)
        self.assertNotEqual(json.loads(first.read_text())["resources"]["selection"]["path"],
                            json.loads(second.read_text())["resources"]["selection"]["path"])
        after = self.file_hashes()
        self.assertEqual({name: after[name] for name in before}, before)
        self.assertEqual(set(after)-set(before), {"deployment/first.json", "other/nested/second.json"})
        with self.assertRaises(FileExistsError):
            bundle.write(first)

    def test_legacy_dict_call_is_explicitly_hashed_and_returns_copies(self):
        selected = deepcopy(self.selected)
        selected["hold_s"] = .4  # Legacy service ignored this field and used .1.
        bundle = ModelBundle.from_legacy(selected, self.frame, root=self.root)
        self.assertIn("inline", bundle.manifest["resources"]["selection"])
        self.assertEqual(bundle.settings["observer"]["hold_s"], .1)
        selected["candidate"] = "caller-mutated"
        copy = bundle.settings
        copy["observer"]["hold_s"] = 9.
        self.assertEqual(bundle.selection["candidate"], self.selected["candidate"])
        self.assertEqual(bundle.settings["observer"]["hold_s"], .1)

    def test_resolved_config_retains_collision_physics_and_identified_filter_authority(self):
        bundle = self.make()
        self.assertEqual(bundle.effective_config["physics_parameters"]["k_drag"], .02)
        self.assertEqual(bundle.effective_config["filter"]["measurement_sigma_xyz"], [.01, .02, .03])
        resolved = bundle.resolved_manifest
        self.assertEqual(resolved["resources"]["selection"]["resolved_path"], str(self.selection_path))
        self.assertEqual(resolved["fingerprint"], bundle.fingerprint)

    def test_explicit_transform_file_is_hashed_and_bad_rotation_is_rejected(self):
        frame_path = self.root / "transforms/world.json"
        self.write_json(frame_path, self.frame)
        bundle = self.make(transform_path=frame_path)
        self.assertEqual(bundle.settings["frame"]["mode"], "explicit_transform")
        self.assertEqual(bundle.effective_config["transform"], self.frame)
        self.assertEqual(bundle.resource_path("transform"), frame_path)
        bad = deepcopy(self.frame)
        bad["R_world_from_model"][2][2] = -1.
        with self.assertRaisesRegex(ValueError, "preserving gravity"):
            ModelBundle.from_legacy(self.selection_path, bad, root=self.root)

    def test_fingerprint_settings_and_selection_disagreement_fail_closed(self):
        bundle = self.make()
        manifest = bundle.manifest
        manifest["settings"]["forecast"]["max_horizon_s"] = 1.2
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            ModelBundle(manifest, self.root)
        manifest["fingerprint"] = bundle_fingerprint(manifest)
        with self.assertRaisesRegex(ValueError, "disagree with the frozen selection"):
            ModelBundle(manifest, self.root)
        manifest = bundle.manifest
        manifest["settings"]["observer"]["fixed_settings"]["recovery_window_s"] = .9
        manifest["fingerprint"] = bundle_fingerprint(manifest)
        with self.assertRaisesRegex(ValueError, "observer recipe disagrees"):
            ModelBundle(manifest, self.root)
        manifest = bundle.manifest
        manifest["resources"]["selection"]["path"] = str(self.selection_path)
        with self.assertRaisesRegex(ValueError, "relative to the manifest"):
            ModelBundle(manifest, self.root)

    def test_optional_head_is_declared_and_unvalidated_horizon_is_rejected(self):
        selected = deepcopy(self.selected)
        selected["head"] = None
        bundle = ModelBundle.from_legacy(selected, self.frame, root=self.root)
        self.assertIsNone(bundle.manifest["resources"]["head"])
        self.assertFalse(bundle.settings["residual"]["enabled"])
        self.assertIsNone(bundle.settings["residual"]["maximum_trained_horizon_s"])
        selected["max_horizon_s"] = 2.5
        with self.assertRaisesRegex(ValueError, "unvalidated forecast horizon"):
            ModelBundle.from_legacy(selected, self.frame, root=self.root)

    def test_physics_horizon_experiments_preserve_resources_residual_and_base_identity(self):
        base = self.make()
        original = base.manifest
        fingerprints = set()
        for horizon in (.8, 1., 1.2, 1.5, 1.8, 2.):
            with self.subTest(horizon=horizon):
                derived = base.with_forecast_horizon(horizon)
                fingerprints.add(derived.fingerprint)
                self.assertEqual(derived.manifest["resources"], original["resources"])
                self.assertEqual(derived.settings["residual"], base.settings["residual"])
                forecast = derived.settings["forecast"]
                self.assertEqual(forecast["max_horizon_s"], horizon)
                self.assertEqual(forecast["horizon_experiment"]["base_bundle_fingerprint"], base.fingerprint)
                self.assertEqual(forecast["horizon_experiment"]["validation_status"], "experimental_not_validated")
                path = derived.write(self.root / f"experiments/h{round(horizon*1000)}/bundle.json")
                self.assertEqual(ModelBundle.load(path).fingerprint, derived.fingerprint)
                self.assertEqual(derived.with_forecast_horizon(1.).fingerprint,
                                 base.with_forecast_horizon(1.).fingerprint)
        self.assertEqual(len(fingerprints), 6)
        self.assertEqual(base.manifest, original)
        self.assertNotIn("horizon_experiment", base.settings["forecast"])

    def test_experimental_horizon_range_grid_and_declaration_are_enforced(self):
        base = self.make()
        for horizon in (0., .7, 2.02, 1.01, float("nan"), True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                base.with_forecast_horizon(horizon)
        declared = base.with_forecast_horizon(1.8).manifest
        declared["settings"]["forecast"]["horizon_experiment"]["validation_status"] = "validated"
        declared["fingerprint"] = bundle_fingerprint(declared)
        with self.assertRaisesRegex(ValueError, "horizon experiment declaration"):
            ModelBundle(declared, self.root)
        undeclared = base.manifest
        undeclared["settings"]["forecast"]["max_horizon_s"] = 1.8
        undeclared["fingerprint"] = bundle_fingerprint(undeclared)
        with self.assertRaisesRegex(ValueError, "frozen selection"):
            ModelBundle(undeclared, self.root)

    def test_horizon_experiment_cannot_extend_the_residual_training_domain(self):
        self.write_json(self.root / self.selected["head"],
                        {"metadata": {"maximum_trained_horizon_s": .9}})
        with self.assertRaisesRegex(ValueError, "residual training domain must not exceed 0.7"):
            self.make(forecast_max_horizon_s=2.)

    def test_forecast_horizon_command_derives_only_new_manifest(self):
        base_path = self.make().write(self.root / "deployment/base.json")
        before = self.file_hashes()
        output = self.root / "experiments/h2000/bundle.json"
        argv = ["model_bundle", "forecast-horizon", "--bundle", str(base_path),
                "--forecast-max-horizon-s", "2.0", "--output", str(output)]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            main()
        after = self.file_hashes()
        self.assertEqual({name: after[name] for name in before}, before)
        self.assertEqual(set(after)-set(before), {"experiments/h2000/bundle.json"})
        self.assertEqual(ModelBundle.load(output).settings["forecast"]["max_horizon_s"], 2.)

    def test_create_command_writes_only_new_manifest_and_reports_identity(self):
        before = self.file_hashes()
        output = self.root / "deployment/model_bundle.json"
        stdout = io.StringIO()
        argv = ["model_bundle", "create", "--selection", str(self.selection_path),
                "--project-root", str(self.root), "--output", str(output), "--synthetic-world"]
        with patch.object(sys, "argv", argv), redirect_stdout(stdout):
            main()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["operation"], "manifest_only_no_model_changes")
        self.assertEqual(result["bundle_fingerprint"], ModelBundle.load(output).fingerprint)
        after = self.file_hashes()
        self.assertEqual({name: after[name] for name in before}, before)
        self.assertEqual(set(after)-set(before), {"deployment/model_bundle.json"})

    def test_controller_import_does_not_load_player_or_mujoco_adapter(self):
        project = Path(__file__).resolve().parents[2]
        program = ("import sys; import tennis_calibration.controller_service; "
                   "assert 'tennis_calibration.player_bounce' not in sys.modules; "
                   "assert 'tennis_calibration.identified_mujoco' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", program], cwd=project,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_service_accepts_explicit_resource_root_for_separate_source_tree(self):
        from tennis_calibration.controller_service import service_model_bundle
        args = SimpleNamespace(bundle=None, selection=self.selection_path, transform=None,
                               synthetic_world=True, project_root=self.root)
        bundle = service_model_bundle(args)
        self.assertEqual(bundle.resource_path("identified"), self.root / LEGACY_PATHS["identified"])
        self.assertEqual(bundle.effective_config["physics_parameters"]["k_drag"], .02)

    def test_service_horizon_override_is_explicit_and_does_not_change_source_bundle(self):
        from tennis_calibration.controller_service import service_model_bundle
        path = self.make().write(self.root / "deployment/base.json")
        before = path.read_bytes()
        args = SimpleNamespace(bundle=path, selection=None, transform=None, synthetic_world=False,
                               project_root=None, forecast_max_horizon_s=1.8)
        bundle = service_model_bundle(args)
        self.assertEqual(bundle.settings["forecast"]["max_horizon_s"], 1.8)
        self.assertEqual(bundle.settings["residual"]["maximum_trained_horizon_s"], .7)
        self.assertEqual(path.read_bytes(), before)

    def test_create_missing_resource_exits_cleanly_without_partial_manifest(self):
        child = (self.root / LEGACY_PATHS["collision"]).parent / "models/velocity_3"
        child.unlink()
        output = self.root / "deployment/model_bundle.json"
        stderr = io.StringIO()
        argv = ["model_bundle", "create", "--selection", str(self.selection_path),
                "--project-root", str(self.root), "--output", str(output), "--synthetic-world"]
        with patch.object(sys, "argv", argv), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("model bundle error:", stderr.getvalue())
        self.assertIn("velocity_3", stderr.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
