"""Closed, hash-checked predictor resources and resolved runtime settings.

``create`` writes only a new manifest; it never fits, copies or changes models.
All file references are relative to that manifest. The fingerprint identifies
resource contents and settings, independently of their filesystem locations.
The legacy selection factory is the sole home of historical resource defaults.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "TENNIS_MODEL_BUNDLE/1"
HORIZON_EXPERIMENT_SCHEMA = "PHYSICS_FORECAST_HORIZON_EXPERIMENT/1"
MAX_RESIDUAL_HORIZON_S = .7
COLLISION_RESOURCES = {
    "ridge1": "collision_velocity_0",
    "ridge2": "collision_velocity_1",
    "gp": "collision_velocity_3",
}
COLLISION_FILENAMES = {"ridge1": "velocity_0", "ridge2": "velocity_1", "gp": "velocity_3"}
LEGACY_PATHS = {
    "collision": "artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json",
    "identified": "artifacts/tennis_identified/20261009_training_prior/selected_model.json",
    "noise": "artifacts/tennis_robust/20261013/validation_v2/noise_proxy.json",
}
CONTRACT = {
    "protocol": "BALL_FORECAST/1",
    "position_unit": "m",
    "velocity_unit": "m/s",
    "time_unit": "absolute_capture_seconds",
    "state_layout": "p_xyz/v_xyz/effective_b_xyz",
    "physics_parameter_source": "collision.parameters",
    "contact_height_source": "collision.upstream.height",
    "filter_config_source": "identified.filter",
    "noise_source": "noise.sigma_xyz_m",
    "collision_scope": "first_ground_impact_only",
    "future_covariance": "unavailable_uncalibrated",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def bundle_fingerprint(manifest):
    """Content/config identity; paths and the recorded fingerprint are excluded."""
    identity = deepcopy(manifest)
    identity.pop("fingerprint", None)
    identity["resources"] = {
        name: None if resource is None else {"sha256": resource["sha256"]}
        for name, resource in manifest["resources"].items()
    }
    return _sha256(_canonical(identity))


def _json(data, name):
    try:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        _canonical(value)  # Reject nonstandard NaN/Infinity, including metadata.
        return value
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid {name} JSON: {error}") from error


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _transform(value):
    """Retain the service's original gravity-preserving transform validation."""
    try:
        rotation = np.asarray(value["R_world_from_model"], float)
        shift = np.asarray(value["translation_world_from_model_m"], float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid world/model transform") from error
    if (rotation.shape != (3, 3) or shift.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(shift).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1.)
            or not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-8)):
        raise ValueError("requires a calibrated rigid transform preserving gravity direction")
    return {"R_world_from_model": rotation.tolist(),
            "translation_world_from_model_m": shift.tolist()}


def _file_resource(path, base_dir):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"model resource is not a file: {path}")
    return {"path": os.path.relpath(path, base_dir), "sha256": _sha256(path.read_bytes())}


def _inline_resource(value):
    return {"inline": deepcopy(value), "sha256": _sha256(_canonical(value))}


def _resolved_filter(config):
    # Resolve defaults through the existing validator, never a copied recipe.
    from .identified_filter import _configuration
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in _configuration(config).items()}


def _observer_recipe(experiment):
    # Freeze the named observer's flags/recovery constants as audit settings.
    # Runtime still uses the existing implementation; reject a stale recipe
    # rather than silently changing a bundle when these constants change.
    from .robust_observer import EXPERIMENTS, FIXED_SETTINGS
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown observer experiment: {experiment}")
    return {"experiment_flags": deepcopy(EXPERIMENTS[experiment]),
            "fixed_settings": deepcopy(FIXED_SETTINGS)}


def _experiment_horizon(value, grid_step_s):
    horizon = _positive(value, "forecast_max_horizon_s")
    if not .8 <= horizon <= 2.:
        raise ValueError("experimental forecast_max_horizon_s must be within 0.8--2.0 seconds")
    if grid_step_s != .02 or abs(horizon / grid_step_s - round(horizon / grid_step_s)) > 1e-9:
        raise ValueError("experimental forecast horizon must align with the unchanged 20ms grid")
    return horizon


def _horizon_experiment(horizon, selected_horizon, base_fingerprint):
    return {"schema": HORIZON_EXPERIMENT_SCHEMA,
            "forecast_max_horizon_s": horizon,
            "selected_max_horizon_s": selected_horizon,
            "base_bundle_fingerprint": base_fingerprint,
            "scope": "physics_first_ground_impact_domain_only",
            "validation_status": "experimental_not_validated",
            "residual_maximum_allowed_horizon_s": MAX_RESIDUAL_HORIZON_S}


class ModelBundle:
    """Validated, side-effect-free deployment description, not a model fitter.

    JSON settings and resolved manifests are returned as copies. Binary model
    contents are hash checked here and parsed by their existing model loaders.
    An identified checkpoint supplies the filter only; collision.parameters is
    the authoritative physical law, exactly as in the legacy service.
    """

    def __init__(self, manifest, base_dir, *, manifest_path=None):
        if not isinstance(manifest, dict):
            raise ValueError("model bundle manifest must be an object")
        self._manifest = deepcopy(manifest)
        self.base_dir = Path(base_dir).resolve()
        self.manifest_path = None if manifest_path is None else Path(manifest_path).resolve()
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"a {SCHEMA} manifest is required")
        resources = manifest.get("resources")
        required = {"selection", "collision", "identified", "noise", "head", "transform",
                    *COLLISION_RESOURCES.values()}
        if not isinstance(resources, dict) or set(resources) != required:
            raise ValueError("bundle must declare every selection/collision/filter/noise/head/transform resource")
        if manifest.get("contract") != CONTRACT:
            raise ValueError("unsupported model bundle units, scope or parameter-source contract")
        self._paths, self._data = {}, {}
        for name, resource in resources.items():
            if resource is None:
                if name != "head":
                    raise ValueError(f"required bundle resource is unavailable: {name}")
                continue
            if not isinstance(resource, dict):
                raise ValueError(f"invalid resource descriptor: {name}")
            expected = resource.get("sha256", "")
            if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
                raise ValueError(f"invalid resource SHA256: {name}")
            if set(resource) == {"path", "sha256"}:
                relative = resource["path"]
                if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                    raise ValueError(f"resource paths must be relative to the manifest: {name}")
                path = (self.base_dir / relative).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"model resource is not a file: {name}")
                data = path.read_bytes()
                self._paths[name] = path
            elif set(resource) == {"inline", "sha256"} and name in {"selection", "transform"}:
                data = _canonical(resource["inline"])
            else:
                raise ValueError(f"unsupported resource descriptor: {name}")
            if _sha256(data) != expected:
                raise ValueError(f"model resource SHA256 mismatch: {name}")
            self._data[name] = data
        # Validate descriptors before computing identity, so malformed inputs
        # fail with a resource error rather than an incidental KeyError.
        self.fingerprint = bundle_fingerprint(manifest)
        if manifest.get("fingerprint") != self.fingerprint:
            raise ValueError("model bundle fingerprint mismatch")
        self._selection = self._json_resource("selection")
        collision = self._json_resource("collision")
        identified = self._json_resource("identified")
        noise = self._json_resource("noise")
        self._validate_settings(collision, identified, noise)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve(strict=True)
        return cls(_json(path.read_bytes(), "model bundle"), path.parent, manifest_path=path)

    @classmethod
    def from_legacy(cls, selection, transform=None, *, transform_path=None,
                    synthetic_world=False, root=ROOT, collision_path=None,
                    identified_path=None, noise_path=None, head_path=None,
                    collision_models=None, forecast_max_horizon_s=None):
        """Resolve the old command without changing its defaults or any file.

        A dict selection/transform is accepted for existing Python callers;
        these two in-memory inputs are explicitly content-hashed inline. CLI
        migration uses a selection file and optional transform file instead.
        Legacy hold duration was hardcoded to .1, regardless of selection
        metadata; preserve that behavior in the resolved observer settings.
        """
        root = Path(root).resolve()
        resources = {}
        if isinstance(selection, dict):
            selected = deepcopy(selection)
            resources["selection"] = _inline_resource(selected)
        else:
            if not isinstance(selection, (str, os.PathLike)):
                raise ValueError("legacy selection must be a JSON object or a file path")
            resources["selection"] = _file_resource(selection, root)
            selected = _json(Path(selection).read_bytes(), "selection")
        paths = {
            "collision": collision_path or root / LEGACY_PATHS["collision"],
            "identified": identified_path or root / LEGACY_PATHS["identified"],
            "noise": noise_path or root / LEGACY_PATHS["noise"],
        }
        for name, path in paths.items():
            resources[name] = _file_resource(path, root)
        collision_path = Path(paths["collision"]).resolve()
        collision = _json(collision_path.read_bytes(), "collision")
        identified = _json(Path(paths["identified"]).read_bytes(), "identified")
        if collision_models is None:
            collision_models = {name: collision_path.parent / "models" / filename
                                for name, filename in COLLISION_FILENAMES.items()}
        if set(collision_models) != set(COLLISION_RESOURCES):
            raise ValueError("collision models must explicitly contain ridge1/ridge2/gp")
        for name, path in collision_models.items():
            resources[COLLISION_RESOURCES[name]] = _file_resource(path, root)
        resources["head"] = None
        trained_horizon = None
        if selected.get("head"):
            actual_head = Path(head_path) if head_path is not None else root / selected["head"]
            resources["head"] = _file_resource(actual_head, root)
            head = _json(actual_head.read_bytes(), "residual head")
            trained_horizon = head["metadata"]["maximum_trained_horizon_s"]
        elif head_path is not None:
            raise ValueError("head override supplied for a selection without a residual head")
        if transform_path is not None:
            if transform is not None:
                raise ValueError("supply an inline transform or a transform path")
            resources["transform"] = _file_resource(transform_path, root)
            transform = _json(Path(transform_path).read_bytes(), "transform")
        if transform is not None:
            frame_mode = "explicit_transform"
            resolved_transform = _transform(transform)
        elif synthetic_world:
            frame_mode = "synthetic_world"
            height = float(collision["upstream"]["height"])
            resolved_transform = _transform({
                "R_world_from_model": np.eye(3).tolist(),
                "translation_world_from_model_m": [0, 0, .033-height],
            })
        else:
            raise ValueError("supply calibrated --transform or explicit --synthetic-world")
        if "transform" not in resources:
            resources["transform"] = _inline_resource(resolved_transform)
        manifest = {
            "schema": SCHEMA,
            "resources": resources,
            "contract": deepcopy(CONTRACT),
            "settings": {
                "candidate": selected["candidate"],
                "observer": {"model_id": selected["model_id"], "experiment": selected["experiment"],
                             "policy": "hold100", "hold_s": .1,
                             "filter_config": _resolved_filter(identified["filter"]),
                             **_observer_recipe(selected["experiment"])},
                "forecast": {"max_horizon_s": selected["max_horizon_s"], "grid_step_s": .02,
                             "velocity_difference_step_s": .001, "second_contact_tolerance_m": 1e-5},
                "residual": {"enabled": bool(selected.get("head")),
                             "maximum_trained_horizon_s": trained_horizon,
                             "wait_s": .05, "ramp_s": .05, "maximum_correction_m": .15,
                             "correction_reference_horizon_s": .5, "horizon_fade_s": .2},
                "frame": {"mode": frame_mode, **resolved_transform},
            },
        }
        manifest["fingerprint"] = bundle_fingerprint(manifest)
        bundle = cls(manifest, root)
        return bundle if forecast_max_horizon_s is None else bundle.with_forecast_horizon(forecast_max_horizon_s)

    def with_forecast_horizon(self, forecast_max_horizon_s):
        """Derive an explicit experiment in memory; preserve every model resource.

        This changes only the supported future physics domain. It does not
        retrain a head, extend its <=700ms domain, change observations/transform
        or alter the first/second-ground-contact rules. The source bundle and
        files are untouched. Omitting this API retains the original manifest,
        fingerprint and 1.5s selection behavior.
        """
        manifest = self.manifest
        forecast = manifest["settings"]["forecast"]
        horizon = _experiment_horizon(forecast_max_horizon_s, forecast["grid_step_s"])
        selected_horizon = self._selection["max_horizon_s"]
        forecast.pop("horizon_experiment", None)
        forecast["max_horizon_s"] = selected_horizon
        base_fingerprint = bundle_fingerprint(manifest)
        forecast["max_horizon_s"] = horizon
        forecast["horizon_experiment"] = _horizon_experiment(horizon, selected_horizon, base_fingerprint)
        manifest["fingerprint"] = bundle_fingerprint(manifest)
        return type(self)(manifest, self.base_dir)

    def _json_resource(self, name):
        return _json(self._data[name], name)

    def _validate_settings(self, collision, identified, noise):
        selected = self._selection
        if selected.get("schema") != "CONTROLLER_PREDICTOR/1":
            raise ValueError("unsupported controller selection schema")
        if collision.get("schema") != "FROZEN_PRECONTACT/1":
            raise ValueError("unsupported collision checkpoint schema")
        if identified.get("schema") != "TENNIS_IDENTIFIED/1":
            raise ValueError("unsupported identified checkpoint schema")
        settings = self._manifest.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("bundle settings are required")
        observer, forecast, residual, frame = (settings[name] for name in
                                               ("observer", "forecast", "residual", "frame"))
        if (settings["candidate"] != selected["candidate"]
                or observer["model_id"] != selected["model_id"]
                or observer["experiment"] != selected["experiment"]):
            raise ValueError("bundle settings disagree with the frozen selection")
        if observer["model_id"] not in {"fixed", "ridge1", "ridge2", "gp"}:
            raise ValueError("unknown bundle collision model")
        if observer["policy"] != "hold100":
            raise ValueError("unsupported bundle observation policy")
        if observer["filter_config"] != _resolved_filter(identified["filter"]):
            raise ValueError("resolved filter settings disagree with the identified checkpoint/defaults")
        for name, value in _observer_recipe(observer["experiment"]).items():
            if observer[name] != value:
                raise ValueError(f"bundle observer recipe disagrees with this implementation: {name}")
        _positive(observer["hold_s"], "hold_s")
        selected_horizon = _positive(selected["max_horizon_s"], "selected max_horizon_s")
        if selected_horizon > 1.5:
            raise ValueError("unvalidated forecast horizon")
        _positive(forecast["max_horizon_s"], "max_horizon_s")
        for name in ("grid_step_s", "velocity_difference_step_s", "second_contact_tolerance_m"):
            _positive(forecast[name], name)
        if "horizon_experiment" not in forecast:
            if forecast["max_horizon_s"] != selected["max_horizon_s"]:
                raise ValueError("bundle settings disagree with the frozen selection")
        else:
            horizon = _experiment_horizon(forecast["max_horizon_s"], forecast["grid_step_s"])
            baseline = self.manifest
            baseline_forecast = baseline["settings"]["forecast"]
            baseline_forecast.pop("horizon_experiment")
            baseline_forecast["max_horizon_s"] = selected["max_horizon_s"]
            expected = _horizon_experiment(horizon, selected["max_horizon_s"], bundle_fingerprint(baseline))
            if forecast["horizon_experiment"] != expected:
                raise ValueError("invalid or stale physics horizon experiment declaration")
        for name in ("ramp_s", "maximum_correction_m", "correction_reference_horizon_s", "horizon_fade_s"):
            _positive(residual[name], name)
        if (not isinstance(residual["wait_s"], (int, float)) or isinstance(residual["wait_s"], bool)
                or not math.isfinite(residual["wait_s"]) or residual["wait_s"] < 0):
            raise ValueError("residual wait_s must be finite and nonnegative")
        # These two factors remain inside the unchanged existing head/domain APIs.
        if residual["correction_reference_horizon_s"] != .5 or residual["horizon_fade_s"] != .2:
            raise ValueError("unsupported residual cap/fade formula")
        enabled = self._manifest["resources"]["head"] is not None
        if not isinstance(residual["enabled"], bool) or residual["enabled"] != enabled or enabled != bool(selected.get("head")):
            raise ValueError("residual enablement disagrees with declared resources/selection")
        head_metadata = None
        if enabled:
            head = self._json_resource("head")
            head_metadata = head["metadata"]
            trained = _positive(head_metadata["maximum_trained_horizon_s"], "trained residual horizon")
            if trained > MAX_RESIDUAL_HORIZON_S:
                raise ValueError("residual training domain must not exceed 0.7 seconds")
            if residual["maximum_trained_horizon_s"] != trained:
                raise ValueError("residual training domain disagrees with the head")
        elif residual["maximum_trained_horizon_s"] is not None:
            raise ValueError("a disabled residual must not declare a training domain")
        resolved_transform = _transform(self._json_resource("transform"))
        if _transform(frame) != resolved_transform:
            raise ValueError("bundle frame disagrees with its transform resource")
        if frame["mode"] not in {"synthetic_world", "explicit_transform"}:
            raise ValueError("unknown transform provenance")
        height = float(collision["upstream"]["height"])
        if not math.isfinite(height):
            raise ValueError("collision contact height must be finite")
        if frame["mode"] == "synthetic_world":
            expected = {"R_world_from_model": np.eye(3).tolist(),
                        "translation_world_from_model_m": [0., 0., .033-height]}
            if resolved_transform != expected:
                raise ValueError("synthetic-world transform must retain the legacy height-only alignment")
        sigma = np.asarray(noise["sigma_xyz_m"], float)
        if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("training noise proxy must contain positive finite XYZ")
        if not isinstance(identified["filter"], dict) or not isinstance(collision["parameters"], dict):
            raise ValueError("physics parameters and filter config must be objects")
        self._effective = {
            "selection": deepcopy(selected),
            "physics_parameters": deepcopy(collision["parameters"]),
            "contact_height_m": height,
            "filter": deepcopy(observer["filter_config"]),
            "noise_sigma_xyz_m": sigma.tolist(),
            "residual_head_metadata": deepcopy(head_metadata),
            "transform": resolved_transform,
        }

    @property
    def selection(self):
        return deepcopy(self._selection)

    @property
    def settings(self):
        return deepcopy(self._manifest["settings"])

    @property
    def effective_config(self):
        return deepcopy(self._effective)

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def resource_path(self, name):
        if name not in self._paths:
            raise ValueError(f"resource is absent or inline, not a file: {name}")
        return self._paths[name]

    @property
    def collision_model_paths(self):
        return {model_id: self.resource_path(name) for model_id, name in COLLISION_RESOURCES.items()}

    @property
    def resolved_manifest(self):
        result = self.manifest
        for name, path in self._paths.items():
            result["resources"][name]["resolved_path"] = str(path)
        result["manifest_path"] = None if self.manifest_path is None else str(self.manifest_path)
        result["effective_config"] = self.effective_config
        return result

    def write(self, path):
        """Write one new manifest, rebasing references without copying resources."""
        path = Path(path).resolve()
        manifest = self.manifest
        for name, resource_path in self._paths.items():
            manifest["resources"][name]["path"] = os.path.relpath(resource_path, path.parent)
        # Rebasing changes locations only, so this identity remains unchanged.
        manifest["fingerprint"] = bundle_fingerprint(manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="write only a new manifest; never alter/copy models")
    create.add_argument("--selection", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--project-root", type=Path, default=ROOT)
    frame = create.add_mutually_exclusive_group(required=True)
    frame.add_argument("--transform", type=Path)
    frame.add_argument("--synthetic-world", action="store_true")
    create.add_argument("--collision", type=Path)
    create.add_argument("--identified", type=Path)
    create.add_argument("--noise", type=Path)
    create.add_argument("--head", type=Path)
    create.add_argument("--forecast-max-horizon-s", type=float,
                        help="显式物理预测域实验，0.8--2.0s且对齐20ms；不扩展残差训练域")
    horizon = commands.add_parser("forecast-horizon", help="derive a new experimental manifest; never change the source bundle/models")
    horizon.add_argument("--bundle", type=Path, required=True)
    horizon.add_argument("--forecast-max-horizon-s", type=float, required=True)
    horizon.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="verify all hashes and print resolved inputs/settings")
    inspect.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            bundle = ModelBundle.from_legacy(
                args.selection, transform_path=args.transform, synthetic_world=args.synthetic_world,
                root=args.project_root, collision_path=args.collision,
                identified_path=args.identified, noise_path=args.noise, head_path=args.head,
                forecast_max_horizon_s=args.forecast_max_horizon_s,
            )
            output = bundle.write(args.output)
            print(json.dumps({"manifest": str(output), "bundle_fingerprint": bundle.fingerprint,
                              "operation": "manifest_only_no_model_changes"}, ensure_ascii=False))
        elif args.command == "forecast-horizon":
            bundle = ModelBundle.load(args.bundle).with_forecast_horizon(args.forecast_max_horizon_s)
            output = bundle.write(args.output)
            print(json.dumps({"manifest": str(output), "bundle_fingerprint": bundle.fingerprint,
                              "operation": "experimental_manifest_only_no_model_changes",
                              "forecast_max_horizon_s": args.forecast_max_horizon_s}, ensure_ascii=False))
        else:
            print(json.dumps(ModelBundle.load(args.manifest).resolved_manifest,
                             ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"model bundle error: {error}\n")


if __name__ == "__main__":
    main()
