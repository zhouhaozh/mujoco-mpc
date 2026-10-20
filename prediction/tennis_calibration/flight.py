"""Gravity/quadratic-drag calibration from cleaned position observations.

The shared drag parameter sees training trajectories only.  Forecasts estimate
initial conditions from an early observation window, while full-segment fits
are explicitly reconstructions used to estimate impact velocities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

GRAVITY = np.array([0.0, 0.0, -9.81])
MAX_DRAG_PER_M = 0.5


def _advance(p: np.ndarray, v: np.ndarray, h: np.ndarray,
             k_drag: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized RK4, one possibly different step length per trajectory."""
    def acceleration(velocity: np.ndarray) -> np.ndarray:
        return GRAVITY - k_drag * np.linalg.norm(velocity, axis=-1, keepdims=True) * velocity

    a1 = acceleration(v)
    v2 = v + h * a1 / 2
    a2 = acceleration(v2)
    v3 = v + h * a2 / 2
    a3 = acceleration(v3)
    v4 = v + h * a3
    a4 = acceleration(v4)
    return (p + h * (v + 2 * v2 + 2 * v3 + v4) / 6,
            v + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6)


def simulate_batch(times: np.ndarray, initial_states: np.ndarray, k_drag: float,
                   max_step_s: float = 0.004) -> np.ndarray:
    """Return N x M x 6 states at true, nonuniform times relative to each start."""
    times = np.asarray(times, dtype=float)
    states = np.asarray(initial_states, dtype=float)
    if times.ndim != 2 or states.shape != (times.shape[0], 6):
        raise ValueError("Expected times N x M and initial_states N x 6")
    if max_step_s <= 0 or np.any(times < 0) or np.any(np.diff(times, axis=1) < 0):
        raise ValueError("Times must be nonnegative and monotonic; max_step_s must be positive")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(states)):
        raise ValueError("Nonfinite integration input")
    if k_drag < 0 or not np.isfinite(k_drag):
        raise ValueError("Drag must be nonnegative and finite")
    if k_drag == 0:
        t = times[:, :, None]
        p = states[:, None, :3] + t * states[:, None, 3:] + 0.5 * t * t * GRAVITY
        v = states[:, None, 3:] + t * GRAVITY
        return np.concatenate((p, v), axis=2)
    result = np.empty((times.shape[0], times.shape[1], 6))
    p, v = states[:, :3].copy(), states[:, 3:].copy()
    previous = np.zeros(times.shape[0])
    for j in range(times.shape[1]):
        interval = times[:, j] - previous
        count = max(1, int(np.ceil(float(np.max(interval)) / max_step_s)))
        h = (interval / count)[:, None]
        for _ in range(count):
            p, v = _advance(p, v, h, k_drag)
        result[:, j, :3], result[:, j, 3:] = p, v
        previous = times[:, j]
    return result


def simulate(times: np.ndarray, initial_state: np.ndarray, k_drag: float,
             max_step_s: float = 0.004) -> np.ndarray:
    return simulate_batch(np.asarray(times)[None, :], np.asarray(initial_state)[None, :],
                          k_drag, max_step_s)[0]


def simulate_batch_with_jacobian(times: np.ndarray, initial_states: np.ndarray,
                                 k_drag: float, max_step_s: float = 0.004) -> tuple[np.ndarray, np.ndarray]:
    """RK4 positions and exact discrete sensitivities to p0/v0/k_D.

    The final Jacobian axis is [p0x,p0y,p0z,v0x,v0y,v0z,k_D].
    Differentiating the same RK4 stages avoids finite-difference noise in the
    large shared-parameter fit.  It also avoids seven extra ODE evaluations.
    """
    times, initial_states = np.asarray(times), np.asarray(initial_states)
    if times.ndim != 2 or initial_states.shape != (len(times), 6):
        raise ValueError("Expected times N x M and initial_states N x 6")
    if np.any(times < 0) or np.any(np.diff(times, axis=1) < 0) or max_step_s <= 0:
        raise ValueError("Nonnegative monotonic times and positive max step required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(initial_states)) or not np.isfinite(k_drag):
        raise ValueError("Nonfinite sensitivity integration input")
    p, v = initial_states[:, :3].copy(), initial_states[:, 3:].copy()
    dp, dv = np.zeros((len(times), 3, 7)), np.zeros((len(times), 3, 7))
    dp[:, :, :3] = np.eye(3)
    dv[:, :, 3:6] = np.eye(3)
    positions = np.empty((len(times), times.shape[1], 3))
    jacobian = np.empty((len(times), times.shape[1], 3, 7))
    previous = np.zeros(len(times))

    def acceleration_and_derivative(velocity: np.ndarray, sensitivity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        speed = np.linalg.norm(velocity, axis=1)
        norm_derivative = np.sum(velocity[:, :, None] * sensitivity, axis=1) / np.maximum(speed[:, None], 1e-15)
        derivative = -k_drag * (speed[:, None, None] * sensitivity + velocity[:, :, None] * norm_derivative[:, None, :])
        derivative[:, :, 6] -= speed[:, None] * velocity
        return GRAVITY - k_drag * speed[:, None] * velocity, derivative

    for j in range(times.shape[1]):
        interval = times[:, j] - previous
        count = max(1, int(np.ceil(float(np.max(interval)) / max_step_s)))
        h = (interval / count)[:, None]
        dh = h[:, :, None]
        for _ in range(count):
            a1, da1 = acceleration_and_derivative(v, dv)
            v2, dv2 = v + h * a1 / 2, dv + dh * da1 / 2
            a2, da2 = acceleration_and_derivative(v2, dv2)
            v3, dv3 = v + h * a2 / 2, dv + dh * da2 / 2
            a3, da3 = acceleration_and_derivative(v3, dv3)
            v4, dv4 = v + h * a3, dv + dh * da3
            a4, da4 = acceleration_and_derivative(v4, dv4)
            p += h * (v + 2 * v2 + 2 * v3 + v4) / 6
            dp += dh * (dv + 2 * dv2 + 2 * dv3 + dv4) / 6
            v += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
            dv += dh * (da1 + 2 * da2 + 2 * da3 + da4) / 6
        positions[:, j] = p
        jacobian[:, j] = dp
        previous = times[:, j]
    return positions, jacobian


def state_at(relative_time_s: float, initial_state: np.ndarray,
             k_drag: float) -> np.ndarray:
    """High-accuracy endpoint integration; supports short backward extrapolation."""
    if relative_time_s == 0:
        return np.asarray(initial_state).copy()

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        v = state[3:]
        return np.r_[v, GRAVITY - k_drag * np.linalg.norm(v) * v]

    solution = solve_ivp(rhs, (0, relative_time_s), initial_state, rtol=1e-9,
                         atol=1e-11, max_step=0.004)
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise RuntimeError("Endpoint integration failed")
    return solution.y[:, -1]


def initial_guess(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Robust quadratic position fit; never take a raw sample difference."""
    t = np.asarray(times) - float(times[0])
    design = np.column_stack((np.ones(len(t)), t, 0.5 * t * t))
    weights = np.ones(len(t))
    coefficients = np.zeros((3, 3))
    for _ in range(8):
        coefficients = np.linalg.lstsq(design * weights[:, None],
                                       positions * weights[:, None], rcond=None)[0]
        distance = np.linalg.norm(design @ coefficients - positions, axis=1)
        scale = max(0.001, 1.4826 * float(np.median(np.abs(distance - np.median(distance)))))
        weights = np.sqrt(np.minimum(1, 1.5 * scale / np.maximum(distance, 1e-12)))
    return np.r_[coefficients[0], np.clip(coefficients[1], -149, 149)]


def fit_segment(times: np.ndarray, positions: np.ndarray, k_drag: float,
                representative_points: int = 64, loss_scale_m: float = 0.004) -> dict[str, Any]:
    times, positions = np.asarray(times), np.asarray(positions)
    if len(times) < 8 or times[-1] - times[0] < 0.025:
        raise ValueError("Too few observations or too short a flight segment")
    idx = np.unique(np.linspace(0, len(times) - 1, min(representative_points, len(times))).astype(int))
    relative = times[idx] - times[0]
    guess = initial_guess(times, positions)
    fit = least_squares(lambda state: (simulate(relative, state, k_drag)[:, :3] - positions[idx]).ravel(),
                        guess, bounds=(np.r_[[-1000] * 3, [-150] * 3],
                                       np.r_[[1000] * 3, [150] * 3]),
                        loss="soft_l1", f_scale=loss_scale_m, max_nfev=80,
                        xtol=1e-9, ftol=1e-9, gtol=1e-9)
    prediction = simulate(times - times[0], fit.x, k_drag)
    return {"initial_state": fit.x, "t0_s": float(times[0]), "prediction": prediction,
            "converged": bool(fit.success), "nfev": int(fit.nfev),
            "position_errors_m": np.linalg.norm(prediction[:, :3] - positions, axis=1)}


def fit_shared_drag(segments: list[tuple[np.ndarray, np.ndarray]],
                    representative_points: int = 48, loss_scale_m: float = 0.004) -> dict[str, Any]:
    """Joint training-only fit, sparse derivatives isolate nuisance initial states."""
    if len(segments) < 2:
        raise ValueError("At least two training flight segments are required")
    count = min(representative_points, min(len(t) for t, _ in segments))
    t_fit, p_fit, guesses = [], [], []
    for t, p in segments:
        idx = np.linspace(0, len(t) - 1, count).astype(int)
        t_fit.append(t[idx] - t[0])
        p_fit.append(p[idx])
        guesses.append(initial_guess(t, p))
    t_fit, p_fit = np.asarray(t_fit), np.asarray(p_fit)
    n = len(segments)
    # Initial conditions are nuisance fit variables, not physical joint limits.
    # Unbounded p0/v0 avoid huge Coleman-Li distance scaling from arbitrary
    # position +/-1000 m bounds. Only the nonnegative drag is bounded.
    lower = np.r_[0, np.full(n * 6, -np.inf)]
    upper = np.r_[MAX_DRAG_PER_M, np.full(n * 6, np.inf)]
    x0 = np.r_[0.02, np.asarray(guesses).ravel()]
    row_count = n * count * 3
    jacobian_rows = np.repeat(np.arange(row_count), 7)
    jacobian_cols = np.empty((row_count, 7), dtype=int)
    jacobian_cols[:, 0] = 0
    jacobian_cols[:, 1:] = 1 + 6 * np.repeat(np.arange(n), count * 3)[:, None] + np.arange(6)[None, :]
    cached_x: np.ndarray | None = None
    cached_residual: np.ndarray | None = None
    cached_jacobian: csr_matrix | None = None

    def evaluate(x: np.ndarray) -> None:
        nonlocal cached_x, cached_residual, cached_jacobian
        if cached_x is not None and np.array_equal(cached_x, x):
            return
        prediction, sensitivity = simulate_batch_with_jacobian(t_fit, x[1:].reshape(n, 6), x[0])
        values = np.concatenate((sensitivity[:, :, :, 6:7], sensitivity[:, :, :, :6]), axis=3)
        cached_jacobian = csr_matrix((values.ravel(), (jacobian_rows, jacobian_cols.ravel())),
                                     shape=(row_count, 1 + n * 6))
        cached_residual = (prediction - p_fit).ravel()
        cached_x = x.copy()

    def residual(x: np.ndarray) -> np.ndarray:
        evaluate(x)
        return cached_residual.copy()

    def jacobian(x: np.ndarray) -> csr_matrix:
        evaluate(x)
        # SciPy's robust-loss scaling mutates Jacobian data in place.
        return cached_jacobian.copy()

    fit = least_squares(residual, x0, bounds=(lower, upper),
                        jac=jacobian, loss="soft_l1", f_scale=loss_scale_m,
                        max_nfev=400, xtol=1e-8, ftol=1e-8, gtol=1e-7,
                        tr_solver="lsmr", tr_options={"atol": 1e-10, "btol": 1e-10, "maxiter": 2000},
                        x_scale="jac")
    return {"k_drag_per_m": float(fit.x[0]), "initial_states": fit.x[1:].reshape(n, 6),
            "converged": bool(fit.success), "message": str(fit.message),
            "nfev": int(fit.nfev), "cost": float(fit.cost),
            "optimality": float(fit.optimality), "representative_points_per_trace": count,
            "optimizer": {"method": "trf", "jacobian": "exact discrete RK4 sensitivities",
                          "linear_solver": "lsmr", "max_nfev": 400, "ftol": 1e-8,
                          "xtol": 1e-8, "gtol": 1e-7, "lsmr_atol_btol": 1e-10,
                          "nuisance_initial_conditions_bounds": "unbounded"},
            "at_parameter_bound": bool(fit.x[0] < 1e-7 or fit.x[0] > MAX_DRAG_PER_M - 1e-7)}


def forecast_from_initial_window(times: np.ndarray, positions: np.ndarray,
                                 k_drag: float, window_s: float = 0.08, *,
                                 initialization_observations: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    """Estimate p/v using only the leading window, then forecast later observations."""
    if window_s <= 0 or not np.isfinite(window_s):
        raise ValueError("Initial window must be positive and finite")
    initial_t, initial_p = initialization_observations if initialization_observations is not None else (times, positions)
    t0 = float(initial_t[0])
    use_initial = initial_t <= t0 + window_s + 1e-12
    evaluate = times > t0 + window_s + 1e-12
    if np.any(times < t0) or int(np.sum(use_initial)) < 8 or int(np.sum(evaluate)) < 8:
        raise ValueError("Initial window or forecast interval has fewer than eight observations")
    fit = fit_segment(initial_t[use_initial], initial_p[use_initial], k_drag)
    forecast = simulate(times - t0, fit["initial_state"], k_drag)
    return {"initial_state": fit["initial_state"], "t0_s": t0,
            "init_end_t_s": float(initial_t[use_initial][-1]), "init_samples": int(np.sum(use_initial)),
            "forecast_samples": int(np.sum(evaluate)), "evaluation_mask": evaluate,
            "prediction": forecast, "converged": fit["converged"],
            "position_errors_m": np.linalg.norm(forecast[evaluate, :3] - positions[evaluate], axis=1)}


def error_stats(errors: np.ndarray | list[float]) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"samples": 0, "rmse_m": None, "p50_m": None, "p95_m": None, "max_m": None}
    return {"samples": int(len(values)), "rmse_m": float(np.sqrt(np.mean(values ** 2))),
            "p50_m": float(np.percentile(values, 50)), "p95_m": float(np.percentile(values, 95)),
            "max_m": float(np.max(values))}


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               default=_jsonable, allow_nan=False) + "\n")


def _fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def _resolve_clean(clean_dir: Path, name: str) -> Path:
    options = [clean_dir / name, clean_dir / Path(name).name, clean_dir.parent / name]
    for path in options:
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def _load_clean(path: Path, *, only_kept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        if only_kept and str(row.get("keep", "1")).lower() not in ("1", "true", "yes"):
            continue
        values = [_float(row, key) for key in ("t_s", "x_m", "y_m", "z_m")]
        if all(value is not None for value in values) and (not result or values[0] > result[-1][0]):
            result.append(values)
    array = np.asarray(result, dtype=float)
    if array.ndim != 2 or len(array) < 8:
        raise ValueError("Cleaned trace has fewer than eight valid points")
    return array[:, 0], array[:, 1:]


def _split_map(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    obj = json.loads(path.read_text())
    mapping: dict[str, str] = {}
    for name in ("train", "validation", "test"):
        entries = obj.get(name, obj.get("val", []) if name == "validation" else [])
        if not isinstance(entries, list):
            raise ValueError("Split values must be complete trajectory ID lists")
        for entry in entries:
            trajectory_id = entry if isinstance(entry, str) else entry["trajectory_id"]
            if trajectory_id in mapping:
                raise ValueError(f"Duplicate trajectory ID in split: {trajectory_id}")
            mapping[trajectory_id] = name
    if not mapping or not any(value == "train" for value in mapping.values()):
        raise ValueError("Split requires a nonempty training group")
    return mapping, obj


def predicted_contact(initial_state: np.ndarray, k_drag: float, ground_height_m: float,
                      ball_radius_m: float, max_horizon_s: float = 5.0) -> dict[str, Any]:
    """Contact forecast is available only after independent ground calibration."""
    threshold = ground_height_m + ball_radius_m
    if initial_state[2] <= threshold:
        return {"available": False, "reason": "initial_ball_center_at_or_below_contact_height"}

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        v = state[3:]
        return np.r_[v, GRAVITY - k_drag * np.linalg.norm(v) * v]

    def event(_t: float, state: np.ndarray) -> float:
        return state[2] - threshold

    event.terminal = True
    event.direction = -1
    sol = solve_ivp(rhs, (0, max_horizon_s), initial_state, events=event,
                    rtol=1e-9, atol=1e-11, max_step=0.004)
    if not sol.success or not len(sol.t_events[0]):
        return {"available": False, "reason": "no_descending_contact_in_horizon"}
    return {"available": True, "relative_time_s": float(sol.t_events[0][0]),
            "position_m": sol.y_events[0][0, :3], "velocity_m_s": sol.y_events[0][0, 3:]}


def run_flight(clean_dir: str | Path, annotations: str | Path,
               split_json: str | Path, out_dir: str | Path, *,
               init_window_s: float = 0.08, ground_height_m: float | None = None,
               ball_radius_m: float = 0.0335) -> dict[str, Any]:
    if init_window_s <= 0 or not np.isfinite(init_window_s):
        raise ValueError("Initial window must be positive and finite")
    if ball_radius_m <= 0 or not np.isfinite(ball_radius_m):
        raise ValueError("Ball radius must be positive and finite")
    if ground_height_m is not None and not np.isfinite(ground_height_m):
        raise ValueError("Ground height must be finite")
    clean_dir, annotations = Path(clean_dir), Path(annotations)
    split_json, out_dir = Path(split_json), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments, split_description = _split_map(split_json)
    with annotations.open(newline="") as stream:
        annotated = list(csv.DictReader(stream))
    session_assignments: dict[str, str] = {}
    for row in annotated:
        session_id = row.get("session_id", "")
        group = assignments.get(row["trajectory_id"])
        if session_id and group:
            if session_id in session_assignments and session_assignments[session_id] != group:
                raise ValueError(f"Inferred acquisition session leaks across splits: {session_id}")
            session_assignments[session_id] = group
    traces, exclusions, fingerprints = [], [], []
    seen: set[str] = set()
    for row in annotated:
        trajectory_id = row["trajectory_id"]
        if trajectory_id in seen:
            raise ValueError(f"Repeated annotation ID: {trajectory_id}")
        seen.add(trajectory_id)
        try:
            if trajectory_id not in assignments:
                raise ValueError("trajectory_not_in_frozen_split")
            start, end = _float(row, "pre_start_t_s"), _float(row, "pre_end_t_s")
            if start is None or end is None:
                raise ValueError("no_annotated_preimpact_segment")
            path = _resolve_clean(clean_dir, row["cleaned_file"])
            times, positions = _load_clean(path)
            raw_times, raw_positions = _load_clean(path, only_kept=False)
            use = (times >= start) & (times <= end)
            if np.sum(use) < 24 or end - start < 0.14:
                raise ValueError("insufficient_preimpact_air_segment")
            pre_times, pre_positions = times[use], positions[use]
            if pre_times[-1] - pre_times[0] < 0.14:
                raise ValueError("valid_preimpact_duration_below_140ms")
            fingerprints.append(_fingerprint(path))
            traces.append({"id": trajectory_id, "split": assignments[trajectory_id],
                           "row": row, "times": pre_times, "positions": pre_positions,
                           "raw_times": raw_times, "raw_positions": raw_positions,
                           "all_times": times, "all_positions": positions})
        except (ValueError, FileNotFoundError) as exc:
            exclusions.append({"trajectory_id": trajectory_id, "reason": str(exc)})
    unknown_ids = set(assignments) - seen
    if unknown_ids:
        raise ValueError(f"Split contains IDs absent from annotations: {sorted(unknown_ids)}")
    training = [trace for trace in traces if trace["split"] == "train"]
    shared = fit_shared_drag([(trace["times"], trace["positions"]) for trace in training])
    k_drag = shared["k_drag_per_m"]
    fits, metric_rows = [], []
    pools: dict[str, dict[str, list[float]]] = {
        group: {"drag_forecast": [], "ballistic_forecast": [], "drag_reconstruction": [],
                "ballistic_reconstruction": [], "per_trace_drag_forecast_rmse": [],
                "per_trace_ballistic_forecast_rmse": []}
        for group in ("train", "validation", "test")}
    validation_sensitivity = {str(window): [] for window in (0.08, 0.15, 0.25)}
    for trace in traces:
        t, p, row = trace["times"], trace["positions"], trace["row"]
        group = trace["split"]
        item: dict[str, Any] = {"trajectory_id": trace["id"], "split": group,
                                "annotation_confidence": row.get("confidence"),
                                "annotation_must_review": row.get("must_review"),
                                "pre_start_t_s": float(t[0]), "pre_end_t_s": float(t[-1]),
                                "pre_samples": len(t), "k_drag_per_m": k_drag}
        summary_row: dict[str, Any] = {"trajectory_id": trace["id"], "split": group}
        reconstructions = {}
        for label, coefficient in (("drag", k_drag), ("ballistic", 0.0)):
            reconstruction = fit_segment(t, p, coefficient)
            reconstructions[label] = reconstruction
            stats = error_stats(reconstruction["position_errors_m"])
            pools[group][label + "_reconstruction"].extend(reconstruction["position_errors_m"].tolist())
            item[label + "_reconstruction"] = {
                "t0_s": reconstruction["t0_s"], "p0_m": reconstruction["initial_state"][:3],
                "v0_m_s": reconstruction["initial_state"][3:], "position_error": stats,
                "converged": reconstruction["converged"]}
            summary_row[label + "_reconstruction_rmse_m"] = stats["rmse_m"]
            try:
                forecast = forecast_from_initial_window(
                    t, p, coefficient, init_window_s,
                    initialization_observations=(trace["raw_times"], trace["raw_positions"]))
                stats = error_stats(forecast["position_errors_m"])
                pools[group][label + "_forecast"].extend(forecast["position_errors_m"].tolist())
                pools[group]["per_trace_" + label + "_forecast_rmse"].append(stats["rmse_m"])
                info = {"t0_s": forecast["t0_s"], "p0_m": forecast["initial_state"][:3],
                        "v0_m_s": forecast["initial_state"][3:], "init_end_t_s": forecast["init_end_t_s"],
                        "init_samples": forecast["init_samples"], "forecast_samples": forecast["forecast_samples"],
                        "position_error": stats, "converged": forecast["converged"]}
                bounce_t = _float(row, "first_bounce_t_s")
                if bounce_t is not None:
                    predicted = state_at(bounce_t - forecast["t0_s"], forecast["initial_state"], coefficient)
                    info["forecast_at_observed_bounce"] = {
                        "t_s": bounce_t, "position_m": predicted[:3], "velocity_m_s": predicted[3:],
                        "note": "Event time is measured annotation, not a model touchdown prediction"}
                if ground_height_m is None:
                    info["predicted_contact"] = {"available": False, "reason": "ground_and_ball_center_reference_unconfirmed"}
                else:
                    info["predicted_contact"] = predicted_contact(forecast["initial_state"], coefficient,
                                                                   ground_height_m, ball_radius_m)
                    if info["predicted_contact"]["available"]:
                        info["predicted_contact"]["t_s"] = forecast["t0_s"] + info["predicted_contact"]["relative_time_s"]
                item[label + "_forecast"] = info
                summary_row[label + "_forecast_rmse_m"] = stats["rmse_m"]
                if label == "drag":
                    trajectory_path = out_dir / "predictions" / (hashlib.sha256(trace["id"].encode()).hexdigest()[:20] + ".csv")
                    trajectory_path.parent.mkdir(exist_ok=True)
                    with trajectory_path.open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["t_s", "measured_x_m", "measured_y_m", "measured_z_m",
                                         "forecast_x_m", "forecast_y_m", "forecast_z_m",
                                         "forecast_vx_m_s", "forecast_vy_m_s", "forecast_vz_m_s", "evaluate"])
                        for j in range(len(t)):
                            writer.writerow([t[j], *p[j], *forecast["prediction"][j], int(forecast["evaluation_mask"][j])])
                    item["prediction_csv"] = str(trajectory_path.relative_to(out_dir))
            except ValueError as exc:
                item[label + "_forecast"] = {"available": False, "reason": str(exc)}
        bounce_t = _float(row, "first_bounce_t_s")
        if bounce_t is not None:
            reconstruction = reconstructions["drag"]
            delta = bounce_t - t[-1]
            if 0 <= delta <= 0.15:
                before = state_at(bounce_t - t[0], reconstruction["initial_state"], k_drag)
                item["impact_reconstruction"] = {
                    "first_bounce_t_s": bounce_t, "p_minus_m": before[:3], "v_minus_m_s": before[3:],
                    "pre_extrapolation_s": float(delta),
                    "note": "Full preimpact fit; reconstruction, not heldout forecast"}
                post_start, post_end = _float(row, "post_start_t_s"), _float(row, "post_end_t_s")
                if post_start is not None and post_end is not None:
                    use_post = (trace["all_times"] >= post_start) & (trace["all_times"] <= post_end)
                    post_t, post_p = trace["all_times"][use_post], trace["all_positions"][use_post]
                    try:
                        if len(post_t) < 16 or post_t[-1] - post_t[0] < 0.06:
                            raise ValueError("insufficient_postimpact_air_segment")
                        if post_t[0] - bounce_t > 0.15:
                            raise ValueError("postimpact_backward_extrapolation_exceeds_150ms")
                        post_fit = fit_segment(post_t, post_p, k_drag)
                        after = state_at(bounce_t - post_t[0], post_fit["initial_state"], k_drag)
                        item["impact_reconstruction"].update({
                            "p_plus_m": after[:3], "v_plus_m_s": after[3:],
                            "post_extrapolation_s": float(post_t[0] - bounce_t),
                            "post_position_error": error_stats(post_fit["position_errors_m"]),
                            "post_fit_converged": post_fit["converged"],
                            "post_t0_s": float(post_t[0]), "post_p0_m": post_fit["initial_state"][:3],
                            "post_v0_m_s": post_fit["initial_state"][3:]})
                    except ValueError as exc:
                        item["impact_reconstruction"]["post_unavailable_reason"] = str(exc)
            else:
                item["impact_reconstruction"] = {"available": False, "reason": "preimpact_extrapolation_outside_0_to_150ms"}
        if group == "validation":
            for window in (0.08, 0.15, 0.25):
                try:
                    check = forecast_from_initial_window(
                        t, p, k_drag, window,
                        initialization_observations=(trace["raw_times"], trace["raw_positions"]))
                    common_suffix = t > check["t0_s"] + 0.25 + 1e-12
                    common_errors = np.linalg.norm(check["prediction"][common_suffix, :3] - p[common_suffix], axis=1)
                    validation_sensitivity[str(window)].append({
                        "trajectory_id": trace["id"],
                        "own_forecast_horizon": error_stats(check["position_errors_m"]),
                        "common_suffix_after_250ms": error_stats(common_errors)})
                except ValueError:
                    pass
        fits.append(item)
        metric_rows.append(summary_row)
    shared_output = {key: value for key, value in shared.items() if key != "initial_states"}
    shared_output.update({"model": "dp/dt=v; dv/dt=(0,0,-9.81)-k_D*norm(v)*v",
                          "gravity_m_s2": GRAVITY.tolist(), "k_drag_unit": "1/m",
                          "training_trajectories": len(training), "fit_trajectory_ids": [trace["id"] for trace in training],
                          "parameter_limits_per_m": [0, MAX_DRAG_PER_M],
                          "loss": "soft_l1", "loss_scale_m": 0.004,
                          "reference_ball_radius_m": ball_radius_m, "ground_height_m": ground_height_m,
                          "assumptions": ["input t in seconds and positions converted from millimeters to meters",
                                          "measured z is vertically upward", "no observed spin or wind",
                                          "constant shared equivalent drag coefficient",
                                          "ball radius and ground/ball-center reference remain unconfirmed unless provided"]})
    metrics = {group: {label: error_stats(values) for label, values in grouped.items()}
               for group, grouped in pools.items()}
    output = {"shared_parameters": shared_output, "metrics": metrics,
              "trajectories_used": len(traces), "excluded": exclusions,
              "initial_window_s": init_window_s,
              "forecast_definition": "Predict retained observations after initial window; p/v fit uses finite chronological raw SI observations in that window, independent of future-derived cleaning keep flags",
              "reconstruction_definition": "Whole airborne segment fitted; impact velocities are inferred, not directly observed",
              "split_description": split_description,
              "session_assignments_checked": session_assignments,
              "validation_initial_window_sensitivity": validation_sensitivity,
              "limitations": ["Cleaning/annotations and coordinate units remain conditional on capture metadata",
                              "Trajectory points are autocorrelated; pooled samples are not independent trials",
                              "No fixed ground => no independent touchdown-time or landing-point error certification",
                              "Pre/post impact reconstruction residuals must be screened before restitution fitting",
                              "No aerodynamic model complexity selected from test set"],
              "source_fingerprints": [_fingerprint(Path(__file__)), _fingerprint(annotations),
                                      _fingerprint(split_json), *fingerprints]}
    if not shared["converged"]:
        output["calibration_status"] = "candidate_optimizer_did_not_converge"
    elif shared["at_parameter_bound"]:
        output["calibration_status"] = "candidate_drag_at_bound_requires_review"
    else:
        output["calibration_status"] = "conditional_candidate_pending_metadata_and_residual_review"
    _write_json(out_dir / "shared_parameters.json", shared_output)
    _write_json(out_dir / "trajectory_fits.json", {"trajectories": fits})
    _write_json(out_dir / "flight_metrics.json", output)
    fields = ["trajectory_id", "split", "drag_reconstruction_rmse_m", "ballistic_reconstruction_rmse_m",
              "drag_forecast_rmse_m", "ballistic_forecast_rmse_m"]
    with (out_dir / "per_trace_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init-window-s", type=float, default=0.08)
    parser.add_argument("--ground-height-m", type=float)
    parser.add_argument("--ball-radius-m", type=float, default=0.0335)
    args = parser.parse_args()
    result = run_flight(args.clean_dir, args.annotations, args.split_json, args.out_dir,
                        init_window_s=args.init_window_s, ground_height_m=args.ground_height_m,
                        ball_radius_m=args.ball_radius_m)
    print(json.dumps({"trajectories_used": result["trajectories_used"],
                      "k_drag_per_m": result["shared_parameters"]["k_drag_per_m"],
                      "converged": result["shared_parameters"]["converged"],
                      "heldout_test": result["metrics"]["test"]}, default=_jsonable))


if __name__ == "__main__":
    main()
