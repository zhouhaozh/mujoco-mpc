"""Read-only, causal-local-fit measurement discrepancy diagnostic.

This is a *proxy*, not an estimate of the device's true measurement sigma. A
current raw observation is compared with a quadratic extrapolation fitted only
to earlier raw observations. Offline event labels restrict diagnostic regions;
they are never part of an online estimator. Raw values and scoring labels are
not changed, and no model or default filter is selected by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .online_data import causal_anchors, load_dataset


def _robust_extrapolation(times, xyz, current_t, window_s, iterations, mad_floor_m):
    """Fit three quadratic axes on real timestamps, excluding the current row."""
    u = (times - current_t) / window_s
    design = np.column_stack((np.ones(len(u)), u, u * u))
    prediction = np.empty(3)
    for axis in range(3):
        values = xyz[:, axis]
        weights = np.ones(len(values))
        for _ in range(iterations):
            root_w = np.sqrt(weights)
            coef, _, rank, _ = np.linalg.lstsq(
                design * root_w[:, None], values * root_w, rcond=None)
            if rank != 3:
                raise ValueError("rank_deficient_fit")
            residual = values - design @ coef
            centered = residual - np.median(residual)
            scale = max(mad_floor_m, 1.4826 * float(np.median(np.abs(centered))))
            weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(centered), 1e-15))
        # Apply the final robust weights too; this does not add a future point.
        root_w = np.sqrt(weights)
        coef, _, rank, _ = np.linalg.lstsq(
            design * root_w[:, None], values * root_w, rcond=None)
        if rank != 3 or not np.all(np.isfinite(coef)):
            raise ValueError("invalid_fit")
        prediction[axis] = coef[0]
    return prediction


def _trajectory_statistics(errors):
    values = np.asarray(errors, dtype=float)
    if not len(values):
        return None
    median = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - median), axis=0)
    norms = np.linalg.norm(values, axis=1)
    return {
        "n_scored_anchors": int(len(values)),
        "signed_axis_median_m": median.tolist(),
        "signed_axis_mad_sigma_proxy_m": mad.tolist(),
        "norm_p50_m": float(np.median(norms)),
        "norm_p95_m": float(np.quantile(norms, .95)),
        "norm_max_m": float(np.max(norms)),
    }


def _aggregate(trajectories):
    valid = [r["statistics"] for r in trajectories if r["statistics"] is not None]
    counts = Counter()
    for row in trajectories:
        counts.update(row["counts"])
    result = {
        "n_trajectories_total": len(trajectories),
        "n_trajectories_scored": len(valid),
        "counts": dict(sorted(counts.items())),
        "aggregation": "Compute each trajectory statistic first, then weight trajectories equally.",
        "trajectory_equal_mean_signed_axis_median_m": None,
        "trajectory_equal_mean_signed_axis_mad_sigma_proxy_m": None,
        "trajectory_equal_mean_norm_p95_m": None,
        "across_trajectory_norm_p95_p50_m": None,
        "across_trajectory_norm_p95_p95_m": None,
    }
    if valid:
        medians = np.asarray([s["signed_axis_median_m"] for s in valid])
        mads = np.asarray([s["signed_axis_mad_sigma_proxy_m"] for s in valid])
        p95s = np.asarray([s["norm_p95_m"] for s in valid])
        result.update(
            trajectory_equal_mean_signed_axis_median_m=np.mean(medians, axis=0).tolist(),
            trajectory_equal_mean_signed_axis_mad_sigma_proxy_m=np.mean(mads, axis=0).tolist(),
            trajectory_equal_mean_norm_p95_m=float(np.mean(p95s)),
            across_trajectory_norm_p95_p50_m=float(np.median(p95s)),
            across_trajectory_norm_p95_p95_m=float(np.quantile(p95s, .95)),
        )
    return result


def probe_measurement_noise(
    traces: Iterable[dict], *, windows_s=(.04, .08), stride_s=.02,
    min_samples=10, min_span_s=.025, event_clearance_s=.1,
    max_gap_s=.012, irls_iterations=5, mad_floor_m=.001,
):
    """Return JSON-safe per-split proxy diagnostics from ``load_dataset`` traces.

    The current point never enters its fit. A window ends at the latest earlier
    received point, while its interval extends to the actual current timestamp
    for gap/event checks. The entire interval must be >=100 ms from every known
    event. Traces without any finite event label are explicitly unscored.
    """
    windows = tuple(float(w) for w in windows_s)
    if (not windows or any(not np.isfinite(w) or w <= 0 for w in windows)
            or len(set(windows)) != len(windows)):
        raise ValueError("windows_s must contain distinct positive finite values")
    positive = (stride_s, min_span_s, event_clearance_s, max_gap_s, mad_floor_m)
    if any(not np.isfinite(x) or x <= 0 for x in positive):
        raise ValueError("time and MAD settings must be positive and finite")
    if min_samples < 3 or min_samples != int(min_samples):
        raise ValueError("min_samples must be an integer >=3")
    if irls_iterations < 3 or irls_iterations != int(irls_iterations):
        raise ValueError("irls_iterations must be an integer >=3")
    result = {
        "schema_version": 1,
        "diagnostic": "current_raw_minus_causal_robust_quadratic_extrapolation",
        "units": "m and s",
        "configuration": {
            "windows_s": list(windows), "anchor_stride_s": float(stride_s),
            "min_past_samples": int(min_samples), "min_past_span_s": float(min_span_s),
            "event_clearance_s": float(event_clearance_s), "max_gap_s": float(max_gap_s),
            "irls_iterations": int(irls_iterations), "huber_multiplier": 1.5,
            "irls_mad_scale_floor_m": float(mad_floor_m),
            "current_point_in_fit": False, "current_time_interpolated": False,
            "fit_degree": 2, "fit_timestamp_basis": "actual_received_capture_t_s",
            "event_check_scope": "entire_past_fit_interval_through_current_anchor",
            "unknown_event_trace_policy": "skip_no_reference_events",
            "raw_input_keep_flags_used": False,
        },
        "interpretation_limits": [
            "A discrepancy proxy includes current observation error, past observation error, extrapolation/model mismatch and time/coordinate error.",
            "It cannot identify true device noise, constant coordinate bias or a physical ground plane without independent reference measurements.",
            "MAD times 1.4826 is a robust scale convention, not evidence of Gaussian device noise.",
            "Offline reference events only exclude diagnostic regions; this is not a deployable event-aware online prefilter.",
            "The 40/80 ms proxy is a local measurement consistency probe, not a 40/80 ms future-prediction error score.",
            "Overlapping windows and anchors are correlated; row counts are not independent sample counts.",
            "Equal means of per-trajectory MADs/P95s are not the pooled MAD/P95 or an estimate of true measurement sigma.",
            "No raw values, evaluation references, selected weights or default filter settings are modified.",
        ],
        "by_split": {},
    }
    grouped = defaultdict(lambda: defaultdict(list))
    seen = set()
    for trace in traces:
        identity, split = str(trace["id"]), str(trace["split"])
        if identity in seen:
            raise ValueError(f"duplicate trajectory: {identity}")
        seen.add(identity)
        times = np.asarray(trace["raw_times"], dtype=float)
        xyz = np.asarray(trace["raw_xyz"], dtype=float)
        if (times.ndim != 1 or xyz.shape != (len(times), 3)
                or not np.all(np.isfinite(times)) or not np.all(np.isfinite(xyz))
                or not np.all(np.diff(times) > 0)):
            raise ValueError(f"invalid chronological raw arrays: {identity}")
        event_values = trace.get("bounce_event_times_s", [])
        if len(event_values) == 0 and trace.get("bounce_reference_s") is not None:
            event_values = [trace["bounce_reference_s"]]
        events = np.asarray(event_values, dtype=float)
        events = np.unique(events[np.isfinite(events)])
        anchors = causal_anchors(trace, initialization_s=0.0, stride_s=stride_s)
        for window in windows:
            counts = Counter(raw_rows=len(times), anchors_total=len(anchors), scored=0)
            for reason in ("no_reference_events", "insufficient_past_samples",
                           "insufficient_past_span", "gap_over_limit", "event_crossing",
                           "event_clearance", "fit_failure"):
                counts["skip_" + reason] = 0
            errors = []
            for anchor in anchors:
                index, current_t = anchor["current_index"], anchor["t_s"]
                if not len(events):
                    counts["skip_no_reference_events"] += 1
                    continue
                left = int(np.searchsorted(times, current_t-window-1e-12, side="left"))
                past_t, past_xyz = times[left:index], xyz[left:index]
                if len(past_t) < min_samples:
                    counts["skip_insufficient_past_samples"] += 1
                    continue
                if past_t[-1] - past_t[0] < min_span_s-1e-12:
                    counts["skip_insufficient_past_span"] += 1
                    continue
                interval_t = times[left:index+1]
                if np.max(np.diff(interval_t)) > max_gap_s+1e-12:
                    counts["skip_gap_over_limit"] += 1
                    continue
                if np.any((events >= past_t[0]-1e-12) & (events <= current_t+1e-12)):
                    counts["skip_event_crossing"] += 1
                    continue
                distance = np.maximum(np.maximum(past_t[0]-events, events-current_t), 0.0)
                if np.any(distance < event_clearance_s-1e-12):
                    counts["skip_event_clearance"] += 1
                    continue
                try:
                    predicted = _robust_extrapolation(
                        past_t, past_xyz, current_t, window, int(irls_iterations), mad_floor_m)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    counts["skip_fit_failure"] += 1
                    continue
                error = xyz[index] - predicted
                if not np.all(np.isfinite(error)):
                    counts["skip_fit_failure"] += 1
                    continue
                errors.append(error)
                counts["scored"] += 1
            if counts["anchors_total"] != counts["scored"] + sum(
                    n for key, n in counts.items() if key.startswith("skip_")):
                raise AssertionError("anchor disposition counts do not balance")
            grouped[split][f"{window * 1000:g}ms"].append({
                "trajectory_id": identity, "n_reference_events": int(len(events)),
                "counts": dict(sorted(counts.items())),
                "statistics": _trajectory_statistics(errors),
            })
    for split, window_rows in sorted(grouped.items()):
        result["by_split"][split] = {
            window: {"summary": _aggregate(rows), "trajectories": rows}
            for window, rows in window_rows.items()
        }
    return result


def run_noise_probe(calibration_dir: str | Path, **kwargs):
    """Load the frozen dataset read-only, then return the diagnostic object."""
    return probe_measurement_noise(load_dataset(Path(calibration_dir)), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_noise_probe(args.calibration_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
