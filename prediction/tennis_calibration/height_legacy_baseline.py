"""Frozen legacy EKF6 / EKF6+MLP baselines on the height-policy clock.

The old checkpoint retains its original six-state filter, parameters, feature
normalization and position residual head. It is never attached to the newer
Magnus/GP filter. Only its continuous observation policy is evaluated here.
Offline first-contact references are evaluation labels, not filter inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .identified_run import dump, plain
from .online_data import (aggregate_predictions, causal_anchors, future_labels,
                          history_features, load_dataset)
from .online_features import FEATURE_SCHEMA, base_features, position_features
from .online_filter import forecast_batch
from .online_mlp import predict_residual
from .online_predictor import OnlinePredictor
from .online_run import write_csv


ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUN = ROOT / "artifacts/tennis_online/20261008_final"
DEFAULT_OUT = ROOT / "artifacts/tennis_height_policy/20261010_height_hold100"
HORIZONS = (0., .1, .2, .5)
CANDIDATES = ("legacy_ekf6", "legacy_ekf6_mlp")


def checkpoint_hashes(run_dir):
    """Include immutable inference configuration, weights and normalizers."""
    run_dir = Path(run_dir)
    paths = [run_dir / "config.json"]
    for name in ("mlp_position", "mlp_landing"):
        directory = run_dir / "models" / name
        if not directory.exists():
            directory = run_dir / name
        paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def update_filter_only(predictor, capture_t_s, xyz):
    """Exact OnlinePredictor.update history policy without future/landing work."""
    before = predictor.filter.last_t
    snapshot = predictor.filter.update(capture_t_s, xyz)
    if predictor.filter.last_t is not None and predictor.filter.last_t != before:
        predictor._history.append((float(capture_t_s), np.asarray(xyz, dtype=float).copy()))
        while (len(predictor._history) > 2
               and predictor._history[1][0] < predictor.filter.last_t - .12):
            predictor._history.popleft()
    return snapshot


def forecast_legacy(predictor, snapshot):
    """Original physics and 98-D MLP contract; h=0 is the same posterior."""
    if not snapshot["initialized"]:
        return {candidate: [None] * len(HORIZONS) for candidate in CANDIDATES}, {
            candidate: ["not_initialized"] * len(HORIZONS) for candidate in CANDIDATES}
    physics = forecast_batch(snapshot["state"][None], np.array([snapshot["bounced"]]),
                             HORIZONS, **predictor.physical,
                             config=predictor.config["filter"])[0]
    raw_times = np.array([item[0] for item in predictor._history])
    raw_xyz = np.array([item[1] for item in predictor._history])
    history = history_features(raw_times, raw_xyz, len(raw_times)-1,
                               window_s=FEATURE_SCHEMA["history_window_s"],
                               points=FEATURE_SCHEMA["history_points"])
    base = base_features(history, snapshot, 0.)
    corrected = physics.copy()
    reasons = ["current_posterior_no_mlp"]
    for index, horizon in enumerate(HORIZONS[1:], start=1):
        if predictor._mlp_delay_reason(0.):
            raise ValueError("legacy zero-delay inference is not validated")
        if not np.any(np.isclose(predictor.supported_horizons, horizon, rtol=0, atol=1e-9)):
            raise ValueError("legacy requested horizon is not validated")
        try:
            corrected[index] += predict_residual(
                predictor.position_model, position_features(base, horizon)[None])[0]
            reasons.append("validation_selected")
        except FloatingPointError:
            # Preserve the established live fallback; do not clip or retrain.
            reasons.append("nonfinite_mlp_fallback_to_physics")
    return {"legacy_ekf6": list(physics), "legacy_ekf6_mlp": list(corrected)}, {
        "legacy_ekf6": ["current_posterior" if h == 0 else "legacy_physics_forecast"
                        for h in HORIZONS], "legacy_ekf6_mlp": reasons}


def verify_reference_prediction(predictor, predictions):
    """A few anchors compare direct work against the original live predictor."""
    before = predictor.filter.snapshot()
    live = predictor.predict(horizons=list(HORIZONS[1:]))
    for index, row in enumerate(live["forecasts"], start=1):
        np.testing.assert_array_equal(row["physics_position_xyz_m"],
                                      predictions["legacy_ekf6"][index])
        np.testing.assert_array_equal(row["position_xyz_m"],
                                      predictions["legacy_ekf6_mlp"][index])
    np.testing.assert_array_equal(predictor.filter.state, before["state"])
    np.testing.assert_array_equal(predictor.filter.covariance, before["covariance"])
    if predictor.filter.last_t != before["t_s"]:
        raise AssertionError("legacy reference projection modified capture clock")
    return {"capture_t_s": float(before["t_s"]), "future_forecasts_compared": 6,
            "position_difference_m": 0., "posterior_unchanged": True}


def run_legacy_baselines(traces, out, *, run_dir=LEGACY_RUN, verify_reference=False):
    """Return common-clock rows and write evidence without altering checkpoints.

    ``traces`` must already be the requested frozen validation/test selection.
    h=0 measures posterior tracking, whereas positive horizons are forecasts.
    Every first-frame 20-ms slot includes initialization/unavailable output.
    """
    out, run_dir = Path(out), Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    before_hashes = checkpoint_hashes(run_dir)
    predictor = OnlinePredictor(run_dir)
    if predictor.config["augmented"] or predictor.config["selected_filter"] != "physics6":
        raise ValueError("expected the original frozen physics6 legacy bundle")
    if predictor.position_model is None or not predictor.config["position_mlp_enabled"]:
        raise ValueError("legacy position MLP must retain its frozen enabled selection")
    if abs(predictor.config["filter"]["init_window_s"]-.08) > 1e-12:
        raise ValueError("legacy initialization contract must remain 80 ms")
    rows, diagnostic, reference_checks = [], Counter(), []
    started = perf_counter()
    for trace_index, trace in enumerate(traces):
        predictor.reset()
        times, positions = np.asarray(trace["raw_times"]), np.asarray(trace["raw_xyz"])
        if (not len(times) or positions.shape != (len(times), 3)
                or not np.isfinite(times).all() or not np.isfinite(positions).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError("raw replay must be finite and strictly chronological")
        reference = trace.get("bounce_reference_s")
        if reference is None or not np.isfinite(reference):
            raise ValueError("legacy comparison requires a first-contact scoring reference")
        scope = trace.get("eval_scope_end_s")
        scope = float(times[-1]) if scope is None else float(scope)
        anchors = {a["current_index"] for a in causal_anchors(
            trace, initialization_s=0., stride_s=.02)}
        checked = set()
        for index, (stamp, xyz) in enumerate(zip(times, positions)):
            stamp = float(stamp)
            if stamp > scope + 1e-12:
                break
            snapshot = update_filter_only(predictor, stamp, xyz)
            diagnostic[snapshot["reason"]] += 1
            if index not in anchors:
                continue
            predictions, reasons = forecast_legacy(predictor, snapshot)
            if verify_reference and snapshot["initialized"]:
                verification_stage = ("post200" if stamp >= reference+.2 else
                                      "post" if stamp >= reference+.02 else "early")
                if verification_stage not in checked:
                    check = verify_reference_prediction(predictor, predictions)
                    reference_checks.append({"trajectory_id": trace["id"],
                                             "stage": verification_stage, **check})
                    checked.add(verification_stage)
            labels = future_labels(trace, stamp, HORIZONS)
            for candidate in CANDIDATES:
                for horizon_index, label in enumerate(labels):
                    rows.append(plain({
                        "trajectory_id": trace["id"], "split": trace["split"],
                        "candidate": candidate, "model": candidate,
                        "policy": "continuous", "capture_t_s": stamp,
                        "elapsed_capture_s": stamp-float(times[0]),
                        "relative_capture_s": stamp-float(reference),
                        "reference_contact_t_s": float(reference), **label,
                        "predicted_xyz": predictions[candidate][horizon_index],
                        "prediction_reason": (reasons[candidate][horizon_index]
                                              if predictions[candidate][horizon_index] is None else ""),
                        "inference_note": reasons[candidate][horizon_index],
                        "raw_rows_received": index+1,
                        "current_state_metric": label["horizon_s"] == 0.,
                        "observations_after_anchor_used": False}))
        print(f"LEGACY_BASELINE {trace_index+1}/{len(traces)} {trace['split']} "
              f"{trace['id']}", flush=True)
    after_hashes = checkpoint_hashes(run_dir)
    if after_hashes != before_hashes:
        raise RuntimeError("legacy checkpoint changed during read-only evaluation")
    summary = {
        "platform": platform.system(), "platform_detail": platform.platform(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": perf_counter()-started, "n_trajectories": len(traces),
        "n_rows": len(rows), "candidates": list(CANDIDATES),
        "split_trajectories": dict(Counter(t["split"] for t in traces)),
        "checkpoint_sha256_before": before_hashes,
        "checkpoint_sha256_after": after_hashes,
        "checkpoint_unchanged": True,
        "legacy_configuration": predictor.config,
        "protocol": {
            "observations": "all raw finite chronological points; no offline keep mask",
            "anchor_grid": "20ms from first received frame; first received sample at/after slot",
            "warmup": ">=8 received observations over >=80ms; unavailable anchors retained",
            "scope": "first-bounce evaluation scope; no second-bounce scoring",
            "horizons_s": list(HORIZONS), "delay_s": 0.,
            "current_state": "posterior tracking residual; MLP never applied at horizon zero",
            "future_state": "original OnlineFilter6 and optional original position MLP",
            "training_or_tuning": False, "height_hold_policy_evaluated": False,
            "landing_mlp_applied": False,
            "reference_contact_role": "offline labels/score clock/verification anchors only"},
        "filter_diagnostics": dict(diagnostic),
        "live_reference_checks": reference_checks,
        "summary": {split: {candidate: aggregate_predictions(
            [r for r in rows if r["split"] == split and r["candidate"] == candidate])
            for candidate in CANDIDATES} for split in sorted({t["split"] for t in traces})},
        "source_sha256": {name: hashlib.sha256((ROOT/"tennis_calibration"/name).read_bytes()).hexdigest()
                           for name in ("height_legacy_baseline.py", "online_filter.py",
                                        "online_predictor.py", "online_features.py", "online_mlp.py",
                                        "online_data.py")},
        "limitations": ["historical validation/test groups, not new blind data",
                        "original legacy parameters/features differ from the new Magnus/GP models",
                        "legacy MLP is future XYZ residual, not learned collision outgoing velocity",
                        "only continuous legacy policy evaluated; no legacy height-hold combinations"]}
    write_csv(out/"legacy_predictions.csv", rows)
    dump(out/"legacy_baseline_summary.json", plain(summary))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path,
                        default=ROOT/"artifacts/new_ubuntu/tennis_calibration")
    parser.add_argument("--run-dir", type=Path, default=LEGACY_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true",
                        help="one trajectory per split and three original-predictor checks each")
    args = parser.parse_args(argv)
    if (platform.system() != "Linux"
            or Path(sys.prefix).resolve() != (ROOT/".venv-tennis").resolve()):
        parser.error("formal legacy baseline replay must run in the Ubuntu project environment")
    traces = [t for t in load_dataset(args.calibration_dir)
              if t["split"] in ("validation", "test") and t["bounce_reference_s"] is not None]
    if args.smoke:
        traces = [next(t for t in traces if t["split"] == split)
                  for split in ("validation", "test")]
    run_legacy_baselines(traces, args.out, run_dir=args.run_dir,
                         verify_reference=args.smoke)


if __name__ == "__main__":
    main()
