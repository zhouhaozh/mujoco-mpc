"""Small, deterministic NumPy MLP for causal ball-position residuals.

Each row represents one history window and one forecast horizon.  The caller
must construct causal features, keep complete trajectories/acquisition groups
in their fixed split, and pass targets ``observed_position - physics_forecast``.
This module never reads a test split.  Input and target normalization use only
training rows; validation position error alone chooses the saved epoch.

The network does not infer real spin, and it does not change the physics model.
Its predictions are corrections in metres, with no hidden output clipping.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

FORMAT_VERSION = 1
DEFAULT_SEED = 20261008


def _matrix(value: np.ndarray, name: str, columns: int | None = None,
            allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or (not allow_empty and len(result) == 0):
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if result.shape[1] == 0 or (columns is not None and result.shape[1] != columns):
        raise ValueError(f"{name} has the wrong number of columns")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _weights(sample_weights: np.ndarray | None, count: int) -> np.ndarray:
    if sample_weights is None:
        return np.ones(count, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if (weights.shape != (count,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("sample_weights must be finite, nonnegative, and have positive total")
    # Normalize before summing to avoid overflow on otherwise valid weights.
    weights = weights / np.max(weights)
    return weights / np.mean(weights)


def _normalization(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(values * weights[:, None], axis=0) / np.sum(weights)
    variance = np.sum((values - mean) ** 2 * weights[:, None], axis=0) / np.sum(weights)
    scale = np.sqrt(variance)
    # Constant features/targets use unit scale; their standardized value is zero.
    scale = np.where(scale > 1e-8, scale, 1.0)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("Training normalization overflowed")
    return mean, scale


def _forward(model: dict[str, Any], normalized: np.ndarray) -> list[np.ndarray]:
    activations = [normalized]
    weights, biases = model["weights"], model["biases"]
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        output = activations[-1] @ weight + bias
        if index < len(weights) - 1:
            output = np.tanh(output)
        activations.append(output)
    return activations


def predict_residual(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Return N x D corrections in target units; never silently clip them."""
    X = _matrix(X, "X", int(model["input_dim"]), allow_empty=True)
    normalized = (X - model["x_mean"]) / model["x_scale"]
    correction = _forward(model, normalized)[-1] * model["y_scale"] + model["y_mean"]
    if not np.all(np.isfinite(correction)):
        raise FloatingPointError("MLP returned a nonfinite correction")
    return correction


def _rmse(error: np.ndarray, weights: np.ndarray | None = None) -> float:
    """RMS output-vector norm, in target units (metres for XYZ residuals)."""
    squared_norm = np.sum(error * error, axis=1)
    if weights is None:
        return float(np.sqrt(np.mean(squared_norm)))
    return float(np.sqrt(np.sum(weights * squared_norm) / np.sum(weights)))


def save_residual_mlp(model: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """Save parameters as non-pickled NPZ and normalization/schema as JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, (weight, bias) in enumerate(zip(model["weights"], model["biases"])):
        arrays[f"weight_{index}"] = weight
        arrays[f"bias_{index}"] = bias
    parameters = out_dir / "residual_mlp.npz"
    normalizer = out_dir / "residual_mlp.json"
    np.savez_compressed(parameters, **arrays)
    metadata = {key: model[key] for key in
                ("format_version", "input_dim", "output_dim", "hidden", "activation", "seed", "target_kind")}
    metadata.update({key: np.asarray(model[key]).tolist()
                     for key in ("x_mean", "x_scale", "y_mean", "y_scale")})
    metadata.update(normalization_source="training_rows_only",
                    output_clipping=None)
    normalizer.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"parameters": str(parameters), "normalizer": str(normalizer)}


def load_residual_mlp(out_dir: str | Path) -> dict[str, Any]:
    """Load and shape-check a NumPy/JSON checkpoint without pickle execution."""
    out_dir = Path(out_dir)
    metadata = json.loads((out_dir / "residual_mlp.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION or metadata.get("activation") != "tanh":
        raise ValueError("Unsupported residual MLP checkpoint version or activation")
    hidden = tuple(metadata["hidden"])
    dimensions = (int(metadata["input_dim"]), *hidden, int(metadata["output_dim"]))
    if len(hidden) != 2 or any(not isinstance(n, int) or n <= 0 for n in dimensions):
        raise ValueError("Invalid MLP checkpoint dimensions")
    model: dict[str, Any] = {key: metadata[key] for key in
                            ("format_version", "input_dim", "output_dim", "activation", "seed", "target_kind")}
    model["hidden"] = hidden
    model["weights"], model["biases"] = [], []
    with np.load(out_dir / "residual_mlp.npz", allow_pickle=False) as arrays:
        required = {f"{kind}_{i}" for i in range(3) for kind in ("weight", "bias")}
        if set(arrays.files) != required:
            raise ValueError("Unexpected parameter arrays in residual MLP checkpoint")
        for index, (nin, nout) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            weight = np.asarray(arrays[f"weight_{index}"], dtype=np.float64)
            bias = np.asarray(arrays[f"bias_{index}"], dtype=np.float64)
            if (weight.shape != (nin, nout) or bias.shape != (nout,)
                    or not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias))):
                raise ValueError("Invalid MLP checkpoint parameter shape or value")
            model["weights"].append(weight.copy())
            model["biases"].append(bias.copy())
    for key in ("x_mean", "x_scale", "y_mean", "y_scale"):
        value = np.asarray(metadata[key], dtype=np.float64)
        expected = dimensions[0] if key.startswith("x_") else dimensions[-1]
        if value.shape != (expected,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid MLP checkpoint normalizer")
        if key.endswith("scale") and np.any(value <= 0):
            raise ValueError("MLP checkpoint scales must be positive")
        model[key] = value
    return model


def fit_residual_mlp(train_X: np.ndarray, train_Y: np.ndarray,
                     val_X: np.ndarray, val_Y: np.ndarray,
                     out_dir: str | Path,
                     sample_weights: np.ndarray | None = None,
                     seed: int = DEFAULT_SEED,
                     hidden: tuple[int, int] = (32, 32),
                     max_epochs: int = 300, *,
                     target_kind: str = "future_position_residual_m",
                     val_sample_weights: np.ndarray | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit tanh(32)-tanh(32)-linear residuals with Adam and validation early stop.

    ``sample_weights`` may assign equal total weight to each training trajectory.
    Validation selection uses RMS output-vector error.  ``val_sample_weights``
    can assign equal total weight to each validation trajectory independently
    of its duration or number of rows.  Without them, rows are equally weighted.
    Neither validation features nor validation weights enter normalization.
    Default targets are position residuals in metres.  Other heads must provide
    an explicit ``target_kind``; mixed-unit vector error is a model-selection
    score and must not be presented as a physical position error.
    The return value is ``(model, report)``.  No test-data argument exists.
    """
    train_X = _matrix(train_X, "train_X")
    train_Y = _matrix(train_Y, "train_Y")
    val_X = _matrix(val_X, "val_X", train_X.shape[1])
    val_Y = _matrix(val_Y, "val_Y", train_Y.shape[1])
    if len(train_X) != len(train_Y) or len(val_X) != len(val_Y):
        raise ValueError("Features and targets must have equal row counts")
    if (len(hidden) != 2 or any(not isinstance(n, int) or n <= 0 for n in hidden)
            or not isinstance(max_epochs, int) or max_epochs <= 0):
        raise ValueError("Two positive hidden sizes and a positive max_epochs are required")
    if not isinstance(target_kind, str) or not target_kind:
        raise ValueError("target_kind must describe the output semantics and units")
    weights = _weights(sample_weights, len(train_X))
    val_weights = _weights(val_sample_weights, len(val_X))
    rng = np.random.default_rng(seed)
    x_mean, x_scale = _normalization(train_X, weights)
    y_mean, y_scale = _normalization(train_Y, weights)
    x = (train_X - x_mean) / x_scale
    y = (train_Y - y_mean) / y_scale
    dimensions = (x.shape[1], *hidden, y.shape[1])
    model: dict[str, Any] = {
        "format_version": FORMAT_VERSION, "input_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]), "hidden": tuple(hidden), "activation": "tanh",
        "seed": int(seed), "x_mean": x_mean, "x_scale": x_scale,
        "y_mean": y_mean, "y_scale": y_scale, "weights": [], "biases": [],
        "target_kind": target_kind,
    }
    for nin, nout in zip(dimensions[:-1], dimensions[1:]):
        bound = np.sqrt(6.0 / (nin + nout))
        model["weights"].append(rng.uniform(-bound, bound, (nin, nout)))
        model["biases"].append(np.zeros(nout))
    parameters = [value for pair in zip(model["weights"], model["biases"]) for value in pair]
    first_moments = [np.zeros_like(value) for value in parameters]
    second_moments = [np.zeros_like(value) for value in parameters]
    learning_rate, regularization = 0.003, 1e-4
    batch_size, patience, min_epochs = 512, 35, min(20, max_epochs)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    best_value = _rmse(predict_residual(model, val_X) - val_Y, val_weights)
    best_parameters = [value.copy() for value in parameters]
    best_epoch, stale, step = 0, 0, 0
    position_targets = target_kind == "future_position_residual_m"

    def history_entry(epoch: int, train_error: float, val_error: float) -> dict[str, Any]:
        entry = {"epoch": epoch, "train_output_rmse": train_error, "val_output_rmse": val_error}
        if position_targets:
            entry.update(train_position_rmse_m=train_error, val_position_rmse_m=val_error)
        return entry

    history = [history_entry(0, _rmse(predict_residual(model, train_X) - train_Y, weights), best_value)]
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        permutation = rng.permutation(len(x))
        for offset in range(0, len(x), batch_size):
            indices = permutation[offset:offset + batch_size]
            activations = _forward(model, x[indices])
            delta = (2.0 / (len(indices) * y.shape[1])) * (activations[-1] - y[indices])
            delta *= weights[indices, None]
            gradients = [None] * len(parameters)
            for layer in range(2, -1, -1):
                gradients[2 * layer] = activations[layer].T @ delta + regularization * model["weights"][layer]
                gradients[2 * layer + 1] = np.sum(delta, axis=0)
                if layer:
                    delta = (delta @ model["weights"][layer].T) * (1 - activations[layer] ** 2)
            step += 1
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                first_moments[index] *= beta1
                first_moments[index] += (1 - beta1) * gradient
                second_moments[index] *= beta2
                second_moments[index] += (1 - beta2) * gradient ** 2
                corrected_m = first_moments[index] / (1 - beta1 ** step)
                corrected_v = second_moments[index] / (1 - beta2 ** step)
                parameter -= learning_rate * corrected_m / (np.sqrt(corrected_v) + epsilon)
        train_rmse = _rmse(predict_residual(model, train_X) - train_Y, weights)
        val_rmse = _rmse(predict_residual(model, val_X) - val_Y, val_weights)
        if not np.isfinite(train_rmse) or not np.isfinite(val_rmse):
            raise FloatingPointError("MLP optimization diverged")
        history.append(history_entry(epoch, train_rmse, val_rmse))
        if val_rmse < best_value:
            best_value, best_epoch = val_rmse, epoch
            best_parameters = [value.copy() for value in parameters]
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    for parameter, saved in zip(parameters, best_parameters):
        parameter[:] = saved
    baseline_train = _rmse(train_Y, weights)
    baseline_val = _rmse(val_Y, val_weights)
    report = {
        "schema": "tennis_online_residual_mlp_training_v1", "seed": int(seed),
        "architecture": {"hidden": list(hidden), "activation": "tanh", "output": "linear",
                         "input_dim": int(x.shape[1]), "output_dim": int(y.shape[1]),
                         "parameter_count": int(sum(value.size for value in parameters))},
        "optimization": {"method": "Adam", "learning_rate": learning_rate,
                         "l2_weight_decay": regularization, "batch_size": batch_size,
                         "patience_epochs": patience, "min_epochs": min_epochs,
                         "max_epochs": max_epochs, "sample_weights": sample_weights is not None},
        "normalization_source": "training_rows_only",
        "selection_source": "validation_rows_only",
        "selection_metric": (("weighted" if val_sample_weights is not None else "unweighted")
                             + ("_rms_euclidean_position_error_m" if position_targets
                                else "_rms_output_vector_error_in_target_units")),
        "validation_weighting": {
            "sample_weights_supplied": val_sample_weights is not None,
            "metric": "sqrt_weighted_mean_squared_output_vector_norm",
            "normalization": "weights_rescaled_to_mean_one",
            "trajectory_balance": ("caller_supplied_weights; use inverse_rows_per_trajectory_for_equal_totals"
                                   if val_sample_weights is not None else "equal_rows; not automatically trajectory_balanced"),
            "effective_row_count": float(np.sum(val_weights) ** 2 / np.sum(val_weights ** 2)),
        },
        "target_kind": target_kind,
        "train_rows": len(train_X), "validation_rows": len(val_X),
        "rows_are_independent_trajectories": False,
        "best_epoch": best_epoch, "actual_epochs": history[-1]["epoch"],
        "early_stopped": history[-1]["epoch"] < max_epochs,
        "training_elapsed_s": time.perf_counter() - start,
        "baseline_zero_correction_train_output_rmse": baseline_train,
        "baseline_zero_correction_val_output_rmse": baseline_val,
        "train_output_rmse": _rmse(predict_residual(model, train_X) - train_Y, weights),
        "val_output_rmse": best_value,
        "validation_improved_over_zero_correction": bool(best_value < baseline_val),
        "output_clipping": None, "history": history,
    }
    if position_targets:
        report.update(baseline_zero_correction_train_position_rmse_m=baseline_train,
                      baseline_zero_correction_val_position_rmse_m=baseline_val,
                      train_position_rmse_m=report["train_output_rmse"],
                      val_position_rmse_m=best_value)
    report["checkpoint"] = save_residual_mlp(model, out_dir)
    Path(out_dir, "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return model, report
