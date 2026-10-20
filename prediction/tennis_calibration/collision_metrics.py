"""Metrics for frozen, strictly pre-contact collision forecasts.

Gap variants of one trajectory are repeated evaluations of one collision, not
independent training or statistical samples. Historical post-observation EKF
scores are deliberately not read or combined with this task.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def read_collision_rows(path):
    """Read emitted position/event CSVs without treating 'False' as truthy."""
    numbers = {"gap_s", "planned_gap_s", "actual_gap_s", "horizon_s", "capture_t_s",
        "target_t_s", "cut_t_s", "input_end_t_s", "tb_ref", "predicted_tb",
        "tb_pred", "t_b_pred", "predicted_t_b", "input_count"}
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in tuple(row.items()):
            if key in ("valid", "filter_bounced_at_cut", "normal_rebound_positive"):
                row[key] = value.lower() in ("true", "1")
            elif key in numbers:
                row[key] = float(value) if value else None
            elif value and value[0] in "[{":
                row[key] = json.loads(value)
            elif not value and key not in ("reason", "prediction_reason"):
                row[key] = None
    return rows


def _vector(value, sizes=(3,)):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    return result if result.shape in [(size,) for size in sizes] and np.isfinite(result).all() else None


def _scalar(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _get(row, *keys):
    return next((row[key] for key in keys if key in row and row[key] is not None), None)


def _stats(errors, unit):
    values = np.asarray(errors, float)
    values = values[np.isfinite(values)]
    output = {"count": int(len(values))}
    for field in ("rmse", "p50", "p95", "max"):
        output[f"{field}_{unit}"] = None
    if len(values):
        output.update({f"rmse_{unit}": float(np.sqrt(np.mean(values**2))),
            f"p50_{unit}": float(np.quantile(values, .5)), f"p95_{unit}": float(np.quantile(values, .95)),
            f"max_{unit}": float(np.max(values))})
    return output


def _macro(per_trajectory, unit):
    values = [float(np.sqrt(np.mean(np.square(errors)))) for errors in per_trajectory.values() if errors]
    return {"n_trajectories": len(values),
        f"mean_per_trajectory_rmse_{unit}": float(np.mean(values)) if values else None,
        "per_trajectory_rmse_distribution": _stats(values, unit)}


def _coverage(rows, labeled, success, reasons):
    return {"n_requested": len(rows), "n_labeled": labeled, "n_evaluated": success,
        "n_prediction_failures": labeled-success, "n_missing_labels": len(rows)-labeled,
        "n_unique_collision_events_requested": len({row["trajectory_id"] for row in rows}),
        "label_coverage": labeled/len(rows) if rows else None,
        "prediction_coverage_given_label": success/labeled if labeled else None,
        "total_success_coverage": success/len(rows) if rows else None,
        "missing_or_failure_reasons": dict(reasons)}


def summarize_position_rows(rows: Iterable[dict]):
    rows = list(rows)
    errors, axes, per_trace, reasons = [], [], defaultdict(list), Counter()
    labeled = 0
    for row in rows:
        target = _vector(row.get("target_xyz"))
        if target is None or not row.get("valid", True):
            reasons[row.get("reason") or "missing_label"] += 1
            continue
        labeled += 1
        prediction = _vector(row.get("predicted_xyz"))
        if prediction is None:
            reasons[row.get("prediction_reason") or "prediction_failure"] += 1
            continue
        delta = prediction-target
        error = float(np.linalg.norm(delta))
        errors.append(error)
        axes.append(delta)
        per_trace[row["trajectory_id"]].append(error)
    result = _coverage(rows, labeled, len(errors), reasons)
    result.update(position_error=_stats(errors, "m"), trajectory_macro=_macro(per_trace, "m"),
        axis_rmse_m=np.sqrt(np.mean(np.square(axes), axis=0)).tolist() if axes else None)
    return result


def _event_values(row):
    return {
        "vm_ref": _vector(_get(row, "v_minus_ref", "v_minus_ref_m_s")),
        "vp_ref": _vector(_get(row, "v_plus_ref", "v_plus_ref_m_s")),
        "pb_ref": _vector(_get(row, "pb_ref", "p_b_ref", "p_b_ref_m"), (2, 3)),
        "tb_ref": _scalar(_get(row, "tb_ref", "t_b_ref", "t_b_ref_s")),
        "vm_pred": _vector(_get(row, "predicted_v_minus", "v_minus_pred", "predicted_vminus", "vminus_pred", "predicted_v_minus_m_s")),
        "vp_pred": _vector(_get(row, "predicted_v_plus", "v_plus_pred", "predicted_vplus", "vplus_pred", "predicted_v_plus_m_s")),
        "pb_pred": _vector(_get(row, "predicted_pb", "pb_pred", "predicted_p_b", "p_b_pred", "predicted_p_b_m"), (2, 3)),
        "tb_pred": _scalar(_get(row, "predicted_tb", "tb_pred", "predicted_t_b", "t_b_pred", "predicted_t_b_s")),
    }


def summarize_event_rows(rows: Iterable[dict]):
    rows = list(rows)
    metric_names = {"v_minus_error": "m_s", "v_plus_error": "m_s", "impact_xy_error": "m", "impact_time_abs_error": "s"}
    errors = {name: [] for name in metric_names}
    per_trace = {name: defaultdict(list) for name in metric_names}
    reasons = Counter()
    labeled = success = 0
    for row in rows:
        event = _event_values(row)
        if not row.get("valid", True) or any(event[key] is None for key in ("vp_ref", "pb_ref", "tb_ref")):
            reasons[row.get("reason") or "missing_event_reference"] += 1
            continue
        labeled += 1
        if any(event[key] is None for key in ("vp_pred", "pb_pred", "tb_pred")):
            reasons[row.get("prediction_reason") or "event_prediction_failure"] += 1
            continue
        success += 1
        values = {"v_plus_error": np.linalg.norm(event["vp_pred"]-event["vp_ref"]),
            "impact_xy_error": np.linalg.norm(event["pb_pred"][:2]-event["pb_ref"][:2]),
            "impact_time_abs_error": abs(event["tb_pred"]-event["tb_ref"])}
        if event["vm_ref"] is not None and event["vm_pred"] is not None:
            values["v_minus_error"] = np.linalg.norm(event["vm_pred"]-event["vm_ref"])
        for name, value in values.items():
            errors[name].append(float(value))
            per_trace[name][row["trajectory_id"]].append(float(value))
    result = _coverage(rows, labeled, success, reasons)
    for name, unit in metric_names.items():
        result[name] = _stats(errors[name], unit)
        result[f"{name}_trajectory_macro"] = _macro(per_trace[name], unit)
    result["sampling_note"] = "Gap variants of a trajectory are the same collision; pooled rows are not independent events."
    return result


def _gap(row):
    value = _scalar(_get(row, "gap_s", "planned_gap_s"))
    if value is None or value < 0:
        raise ValueError("Nonnegative finite planned gap_s required")
    return value


def _group(rows, event=False):
    rows = list(rows)
    grouped = defaultdict(list)
    for row in rows:
        key = (row["split"], row["model"], _gap(row))
        if not event:
            key += (float(row["horizon_s"]),)
        grouped[key].append(row)
    function = summarize_event_rows if event else summarize_position_rows
    cells = []
    for key, members in sorted(grouped.items()):
        result = {"split": key[0], "model": key[1], "gap_s": key[2], **function(members)}
        if not event:
            result["horizon_s"] = key[3]
        actual = [value for row in members if (value := _scalar(row.get("actual_gap_s"))) is not None]
        result["actual_gap_s_range"] = [min(actual), max(actual)] if actual else None
        cells.append(result)
    return {"overall": function(rows), "cells": cells}


def _identity(row, event=False):
    return (row["trajectory_id"], _gap(row)) if event else (
        row["trajectory_id"], _gap(row), float(row["horizon_s"]), float(row["target_t_s"]))


def audit_comparison_grid(position_rows):
    """Verify methods requested identical clocks and labels within each split."""
    by_split = defaultdict(lambda: defaultdict(dict))
    for row in position_rows:
        key = _identity(row)
        target = _vector(row.get("target_xyz"))
        label = (bool(row.get("valid", True)), row.get("reason") or "",
                 None if target is None else tuple(float(value) for value in target))
        bucket = by_split[row["split"]][row["model"]]
        if key in bucket:
            raise ValueError(f"Duplicate collision prediction row: {row['split']}/{row['model']}/{key}")
        bucket[key] = label
    result = {}
    for split, models in sorted(by_split.items()):
        names = sorted(models)
        reference = models[names[0]]
        if any(models[name] != reference for name in names[1:]):
            raise ValueError(f"Different requested label clocks or validity across models in split {split}")
        result[split] = {"models": names, "identical_requested_target_grid": True,
            "n_target_keys_per_model": len(reference),
            "n_unique_collision_events": len({key[0] for key in reference})}
    return result


def _paired_positions(rows):
    models = defaultdict(set)
    grouped = defaultdict(list)
    for row in rows:
        models[row["split"]].add(row["model"])
        grouped[(row["split"], _identity(row))].append(row)
    output = []
    for (split, _), members in grouped.items():
        if {row["model"] for row in members} == models[split] and all(
                row.get("valid", True) and _vector(row.get("target_xyz")) is not None
                and _vector(row.get("predicted_xyz")) is not None for row in members):
            output.extend(members)
    return output


def aggregate_collision_metrics(position_rows, event_rows=()):
    position_rows, event_rows = list(position_rows), list(event_rows)
    grid = audit_comparison_grid(position_rows)
    return {"schema": "TENNIS_PRECONTACT_COLLISION_METRICS/1",
        "position": _group(position_rows), "event": _group(event_rows, event=True),
        "comparison": {"label_grid": grid, "common_success_position": _group(_paired_positions(position_rows)),
            "note": "Use common-success errors together with marginal coverage; reused historical holdout is regression, not new blind validation."},
        "sampling_note": "Trajectory is the collision unit. Gaps/horizons are correlated repeated evaluations, not independent samples."}
