"""Read-only Ubuntu numerical audit of frozen precontact experiment outputs."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from .collision_mapping import load_model, pre_features, post_lift_targets
from .collision_predictor import CollisionPredictor
from .online_data import load_dataset


def _read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _vector(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = json.loads(value)
    a = np.asarray(value, dtype=float)
    return a


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())} if len(values) else {"count": 0}


def audit(out, calibration):
    if platform.system() != "Linux":
        raise RuntimeError("Numeric audits must run on Ubuntu")
    rows = _read_csv(out/"post_predictions.csv")
    events = _read_csv(out/"event_predictions.csv")
    report = json.loads((out/"summary.json").read_text())
    selected = json.loads((out/"selected_collision.json").read_text())
    cache = json.loads((out/"prefix_cache.json").read_text())
    mapping = json.loads((out/"mapping_training.json").read_text())
    post_labels = json.loads((out.parent/"20261009_post_labels_with_covariance.json").read_text())
    labels = {r["trajectory_id"]: r for r in post_labels["records"]}
    nonfinite = Counter()
    position_groups = defaultdict(list)
    for row in rows:
        key = (row["split"], row["model"], float(row["gap_s"]), float(row["horizon_s"]))
        position_groups[key].append(row)
        for name in ("predicted_xyz", "target_xyz", "velocity_predicted"):
            a = _vector(row.get(name))
            if a is not None and (a.shape != (3,) or not np.isfinite(a).all()):
                nonfinite["position_csv_"+name] += 1
    summary_cells = {(r["split"], r["model"], r["gap_s"], r["horizon_s"]): r
                     for r in report["metrics"]["position"]["cells"]}
    recomputed = []
    comparison_deltas = []
    for key, cell in sorted(position_groups.items()):
        errors = []
        valid_labels = failures = 0
        for row in cell:
            target, prediction = _vector(row["target_xyz"]), _vector(row["predicted_xyz"])
            if row["valid"].lower() != "true" or target is None:
                continue
            valid_labels += 1
            if prediction is None:
                failures += 1
            elif np.isfinite(prediction).all() and np.isfinite(target).all():
                errors.append(float(np.linalg.norm(prediction-target)))
        error_array = np.asarray(errors)
        rmse = float(np.sqrt(np.mean(error_array**2))) if len(error_array) else None
        official = summary_cells[key]
        official_rmse = official["position_error"]["rmse_m"]
        if rmse is not None and official_rmse is not None:
            comparison_deltas.append(abs(rmse-official_rmse))
        recomputed.append({"split": key[0], "model": key[1], "gap_s": key[2],
                           "horizon_s": key[3], "n_requested": len(cell),
                           "n_labeled": valid_labels, "n_prediction_failures": failures,
                           "n_evaluated": len(errors), "rmse_3d_m": rmse,
                           "counts_match_summary": bool(len(errors) == official["n_evaluated"]
                               and failures == official["n_prediction_failures"]
                               and valid_labels == official["n_labeled"])})
    event_summary = []
    buckets = defaultdict(list)
    for row in events:
        buckets[(row["split"], row["model"])].append(row)
        for name in ("predicted_v_minus", "predicted_v_plus", "predicted_b_plus", "predicted_pb"):
            a = _vector(row.get(name))
            if a is not None and (a.shape != (3,) or not np.isfinite(a).all()):
                nonfinite["event_csv_"+name] += 1
    for key, cell in sorted(buckets.items()):
        failed = nonpositive = 0
        for row in cell:
            vp = _vector(row["predicted_v_plus"])
            if vp is None:
                failed += 1
            else:
                nonpositive += int(vp[2] <= 0)
        event_summary.append({"split": key[0], "model": key[1], "n_event_variants": len(cell),
                              "n_independent_events": len({r["trajectory_id"] for r in cell}),
                              "missing_predictions": failed, "nonpositive_rebound_vz": nonpositive})
    # Use fixed_carry event outputs to recover incident features.  Its b_plus
    # is exactly b_minus and contains no learned head or offline reference.
    train = [r for r in events if r["split"] == "train" and r["model"] == "fixed_carry"
             and _vector(r["predicted_v_minus"]) is not None]
    test = [r for r in events if r["split"] == "test" and r["model"] == "fixed_carry"
            and _vector(r["predicted_v_minus"]) is not None]
    vm_train = np.asarray([_vector(r["predicted_v_minus"]) for r in train])
    bm_train = np.asarray([_vector(r["predicted_b_plus"]) for r in train])
    X_train, _ = pre_features(vm_train, bm_train)
    X_test, _ = pre_features([_vector(r["predicted_v_minus"]) for r in test],
                             [_vector(r["predicted_b_plus"]) for r in test])
    _, _, sigma = post_lift_targets(vm_train, bm_train, [labels.get(r["trajectory_id"], {}) for r in train])
    reliable = sigma <= .08
    domains = []
    for model_path in sorted((out/"models").iterdir()):
        if not model_path.is_file():
            continue
        model = load_model(model_path)
        source = X_train[reliable] if model_path.name.startswith("lift_") else X_train
        standardized = (source-model["x_mean"])/model["x_scale"]
        candidates = (X_test-model["x_mean"])/model["x_scale"]
        lower, upper = standardized.min(axis=0), standardized.max(axis=0)
        outside_axes = (candidates < lower-1e-12) | (candidates > upper+1e-12)
        outside = outside_axes.any(axis=1)
        nearest = np.sqrt(np.min(np.sum((candidates[:, None, :]-standardized[None, :, :])**2, axis=2), axis=1))
        domains.append({"model": model_path.name, "kind": model["kind"],
                        "n_train_variants": len(source), "n_test_variants": len(test),
                        "n_test_variants_outside_coordinate_box": int(outside.sum()),
                        "n_test_collision_events_with_any_variant_outside": len({r["trajectory_id"] for r, yes in zip(test, outside) if yes}),
                        "outside_feature_axis_counts": outside_axes.sum(axis=0).tolist(),
                        "normalized_training_min": lower.tolist(), "normalized_training_max": upper.tolist(),
                        "test_nearest_training_normalized_distance": _stats(nearest),
                        "definition": "Diagnostic coordinatewise training bounding box, not a calibrated safety or support guarantee"})
    latency = []
    for mode, cases in cache.items():
        groups = defaultdict(list)
        for case in cases.values():
            if not isinstance(case, dict) or "fit_elapsed_ms" not in case:
                continue
            groups[case["split"]].append(case["fit_elapsed_ms"])
            groups["all"].append(case["fit_elapsed_ms"])
        for split, values in sorted(groups.items()):
            latency.append({"window": mode, "split": split, "prefix_total_latency_ms": _stats(values),
                            "note": "Measured during formal cache generation; mode composition differs, split counts reported"})
    shared = set(cache["full"]) & set(cache["recent"])
    paired_latency = {mode: _stats([cache[mode][key]["fit_elapsed_ms"] for key in shared])
                      for mode in ("full", "recent")}
    # Inference API consistency: raw prefix only, explicit shared-clock targets.
    chosen_name = selected["selected_model"]["name"]
    api_rows = [r for r in rows if r["split"] == "test" and r["model"] == chosen_name
                and abs(float(r["gap_s"])-.04) < 1e-12 and _vector(r["predicted_xyz"]) is not None]
    case_id = sorted({r["trajectory_id"] for r in api_rows})[0]
    api_rows = sorted([r for r in api_rows if r["trajectory_id"] == case_id], key=lambda r: float(r["target_t_s"]))
    trace = next(t for t in load_dataset(calibration) if t["id"] == case_id)
    cut = trace["bounce_reference_s"]-.04
    prefix = trace["raw_times"] <= cut
    target_times = np.asarray([float(r["target_t_s"]) for r in api_rows])
    start = perf_counter()
    api = CollisionPredictor(out/"selected_collision.json")
    construction_ms = (perf_counter()-start)*1000
    prediction = api.predict(trace["raw_times"][prefix], trace["raw_xyz"][prefix], cut,
                             target_times_s=target_times)
    csv_positions = np.asarray([_vector(r["predicted_xyz"]) for r in api_rows])
    delta = prediction["predicted_states"][:, :3]-csv_positions
    fit_cache = cache[selected["upstream"]["mode"]][case_id+"|0.040000"]["fit"]
    api_evidence = {"trajectory_id": case_id, "split": "test", "gap_s": .04,
                    "model": chosen_name, "valid": bool(prediction["valid"]),
                    "raw_prefix_count": int(prefix.sum()), "cut_t_s": cut,
                    "target_times_s_absolute": target_times.tolist(),
                    "csv_predicted_positions_m": csv_positions.tolist(),
                    "api_predicted_positions_m": prediction["predicted_states"][:, :3].tolist(),
                    "max_position_vector_difference_m": float(np.max(np.linalg.norm(delta, axis=1))),
                    "max_fitted_state_difference": float(np.max(np.abs(np.asarray(prediction["fit"]["state"])-fit_cache["state"]))),
                    "api_prediction_latency_ms": prediction["latency_ms"],
                    "model_construction_latency_ms": construction_ms,
                    "consistent_to_1e_minus_6_m": bool(np.max(np.abs(delta)) <= 1e-6),
                    "note": "Single-call timing, not a real-time-rate acceptance test; targets are common absolute reference clock"}
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [out/"post_predictions.csv", out/"event_predictions.csv", out/"summary.json", out/"selected_collision.json", out/"prefix_cache.json"]}
    artifact = {"schema": "TENNIS_COLLISION_INDEPENDENT_NUMERIC_AUDIT/1",
                "hostname": platform.node(), "source_outputs_sha256": hashes,
                "recomputed_test_rmse_cells": [r for r in recomputed if r["split"] == "test"],
                "max_rmse_difference_from_summary_m": max(comparison_deltas, default=0.),
                "all_summary_cell_counts_match": all(r["counts_match_summary"] for r in recomputed),
                "nonfinite_or_wrong_shape_counts": dict(nonfinite), "event_checks": event_summary,
                "test_feature_domain_checks": domains, "prefix_latency": latency,
                "paired_window_latency_ms": {"n_shared_prefix_variants": len(shared), **paired_latency},
                "independent_training_counts": {"velocity_collision_events": len({r["trajectory_id"] for r in train}),
                    "velocity_prefix_variants": len(train),
                    "lift_reliable_collision_events": len({r["trajectory_id"] for r, yes in zip(train, reliable) if yes}),
                    "lift_reliable_prefix_variants": int(reliable.sum()),
                    "mapping_report_velocity_events": mapping["n_independent_training_events"],
                    "mapping_report_lift_variants": mapping["n_lift_reliable_variants"]},
                "inference_api_evidence": api_evidence,
                "read_only_policy": "No fitted model, candidate choice, CSV, summary or tuning threshold was modified"}
    target = out/"independent_numeric_audit.json"
    target.write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"audit": str(target), "rmse_max_delta_m": artifact["max_rmse_difference_from_summary_m"],
                      "counts_match": artifact["all_summary_cell_counts_match"],
                      "nonfinite": artifact["nonfinite_or_wrong_shape_counts"],
                      "training_counts": artifact["independent_training_counts"],
                      "api_delta_m": api_evidence["max_position_vector_difference_m"],
                      "api_latency_ms": api_evidence["api_prediction_latency_ms"]}), flush=True)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    audit(args.out, args.calibration)


if __name__ == "__main__":
    main()
