"""Frozen selected online EKF evaluated on strictly pre-contact prefixes.

This is a fair frozen-input baseline, not the older streaming score that could
ingest post-contact observations. References choose benchmark cuts/targets;
only chronological raw tXYZ and previously frozen parameters enter the EKF.
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

from .collision_metrics import aggregate_collision_metrics
from .collision_run import GAPS, HORIZONS, fingerprint
from .identified_dynamics import impact_map
from .identified_filter import IdentifiedFilter, forecast_states_batch, landing_batch
from .identified_run import dump, plain
from .online_data import load_dataset, future_labels
from .online_run import write_csv


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "frozen_selected_EKF_preonly"


def snapshot_prefixes(times, xyz, cutoffs, selected):
    """Replay once chronologically, freeze independent snapshots at each cut.

    Inputs contain no contact labels, cleaning flags or post-flight state. The
    cutoff values are requested capture boundaries; no known event time enters
    a filter feature or its first-contact logic.
    """
    times, xyz, cutoffs = np.asarray(times, float), np.asarray(xyz, float), np.asarray(cutoffs, float)
    if times.ndim != 1 or xyz.shape != (len(times), 3) or cutoffs.ndim != 1:
        raise ValueError("Expected t[N], XYZ[N,3], cutoffs[K]")
    if not np.isfinite(times).all() or not np.isfinite(xyz).all() or not np.isfinite(cutoffs).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Finite strictly chronological raw observations required")
    tracker = IdentifiedFilter(selected["parameters"], selected["contact_height"],
                               selected["filter"], selected.get("augmented", True))
    records, timings, cursor = [None]*len(cutoffs), [], 0
    reasons = Counter()
    replay_started = perf_counter()
    for index in np.argsort(cutoffs, kind="stable"):
        cutoff = float(cutoffs[index])
        while cursor < len(times) and times[cursor] <= cutoff:
            initialized_before = tracker.state is not None
            started = perf_counter()
            snapshot = tracker.update(float(times[cursor]), xyz[cursor])
            timings.append({"t_s": float(times[cursor]), "ms": (perf_counter()-started)*1000,
                "initialization": not initialized_before and snapshot["initialized"],
                "initialization_buffer": not initialized_before and not snapshot["initialized"],
                "reason": snapshot["reason"]})
            reasons[snapshot["reason"]] += 1
            cursor += 1
        # plain produces a deep JSON-safe copy so later causal updates cannot
        # mutate an earlier prefix's state, covariance or diagnostics.
        records[index] = {"cut_t_s": cutoff, "input_count": cursor,
            "input_end_t_s": None if cursor == 0 else float(times[cursor-1]),
            "input_sha256": fingerprint(times[:cursor], xyz[:cursor]),
            "snapshot": plain(tracker.snapshot()), "replay_reason_counts": dict(reasons),
            "prefix_replay_elapsed_ms": (perf_counter()-replay_started)*1000}
    return records, timings


def predict_frozen(snapshot, targets, selected):
    """Predict against absolute target times without any observation updates."""
    result = {"states": None, "prediction_reason": "not_initialized",
        "predicted_v_minus": None, "predicted_v_plus": None,
        "predicted_pb": None, "predicted_tb": None, "event_reason": "not_initialized"}
    if not snapshot.get("initialized"):
        return result
    state = np.asarray(snapshot["state"], float)
    capture = float(snapshot["t_s"])
    horizons = np.asarray(targets, float)-capture
    if np.any(horizons < -1e-12):
        raise ValueError("Target must not precede frozen observation capture")
    horizons = np.maximum(horizons, 0.)
    bounced = bool(snapshot["bounced"])
    started = perf_counter()
    prediction = forecast_states_batch(state[None], np.array([bounced]), horizons,
        selected["parameters"], selected["contact_height"], selected["filter"])[0]
    result.update(states=prediction, prediction_reason="", forecast_elapsed_ms=(perf_counter()-started)*1000)
    if bounced:
        # The live filter has already switched mode from its own dynamics.
        # Continue its post-contact state; do not find/manufacture a second
        # landing or pretend the current outgoing speed is the event speed.
        result["event_reason"] = "filter_already_bounced_first_event_state_not_exposed"
        return result
    started = perf_counter()
    contact = landing_batch(state[None], np.array([False]), selected["parameters"],
        selected["contact_height"], selected["filter"])
    result["contact_elapsed_ms"] = (perf_counter()-started)*1000
    if not contact["valid"][0]:
        result["event_reason"] = str(contact["reason"][0])
        return result
    b = state[6:9] if len(state) == 9 else np.zeros(3)
    incident = np.r_[contact["position"][0], contact["velocity"][0], b]
    outgoing = impact_map(incident, selected["parameters"])
    result.update(predicted_v_minus=contact["velocity"][0].tolist(),
        predicted_v_plus=outgoing[3:6].tolist(), predicted_pb=contact["position"][0].tolist(),
        predicted_tb=capture+float(contact["time_to_contact"][0]), event_reason="")
    return result


def _latency(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0, "p50_ms": None, "p99_ms": None, "max_ms": None}
    return {"n": len(values), "p50_ms": float(np.quantile(values, .5)),
        "p99_ms": float(np.quantile(values, .99)), "max_ms": float(np.max(values)),
        "over_1ms": int((values>1).sum()), "over_2ms": int((values>2).sum())}


def run_baseline(calibration, selected_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    selected_bytes = Path(selected_path).read_bytes()
    selected = json.loads(selected_bytes)
    frozen = hashlib.sha256(selected_bytes).hexdigest()
    (out/"selected_model_frozen.json").write_bytes(selected_bytes)
    dump(out/"execution.json", {"schema": "FROZEN_ONLINE_PRECONTACT_BASELINE/1", "hostname": platform.node(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "python_prefix": sys.prefix,
        "selected_sha256_frozen_before_replay": frozen,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_selection": "none: parameters were previously frozen, no current validation or test tuning"})
    traces = load_dataset(calibration)
    impacts = {row["trajectory_id"]: row for row in json.loads((Path(calibration)/"bounce/summary.json").read_text())["impact_estimates"]}
    eligible = [trace for trace in traces if trace["bounce_reference_s"] is not None
        and (trace["split"] != "train" or impacts.get(trace["id"], {}).get("used_for_fit"))]
    positions, events, snapshots, update_times, prediction_times, event_times = [], [], [], [], [], []
    for number, trace in enumerate(eligible, 1):
        reference = float(trace["bounce_reference_s"])
        # These benchmark cut definitions are external to the predictor.
        prefixes, timings = snapshot_prefixes(trace["raw_times"], trace["raw_xyz"],
            [reference-gap for gap in GAPS], selected)
        update_times.extend({**row, "trajectory_id": trace["id"], "split": trace["split"]} for row in timings)
        for gap, prefix in zip(GAPS, prefixes):
            snapshot = prefix["snapshot"]
            capture = prefix["input_end_t_s"]
            base = {"trajectory_id": trace["id"], "groupid": trace["groupid"], "split": trace["split"],
                "model": MODEL_NAME, "gap_s": gap, "planned_gap_s": gap,
                "actual_gap_s": None if capture is None else reference-capture,
                "capture_t_s": capture, "cut_t_s": prefix["cut_t_s"],
                "input_count": prefix["input_count"], "input_end_t_s": capture,
                "filter_bounced_at_cut": bool(snapshot["bounced"])}
            frozen_prediction = predict_frozen(snapshot, [reference+h for h in HORIZONS], selected)
            prediction_times.append(frozen_prediction.get("forecast_elapsed_ms"))
            event_times.append(frozen_prediction.get("contact_elapsed_ms"))
            for index, label in enumerate(future_labels(trace, reference, HORIZONS)):
                predicted = frozen_prediction["states"]
                positions.append(dict(base, **label,
                    predicted_xyz=None if predicted is None else predicted[index,:3].tolist(),
                    velocity_predicted=None if predicted is None else predicted[index,3:6].tolist(),
                    prediction_reason=frozen_prediction["prediction_reason"]))
            event_reference = impacts.get(trace["id"], {})
            events.append(dict(base, tb_ref=reference, pb_ref=event_reference.get("p_b_m"),
                v_minus_ref=event_reference.get("v_minus_m_s"), v_plus_ref=event_reference.get("v_plus_m_s"),
                predicted_v_minus=frozen_prediction["predicted_v_minus"],
                predicted_v_plus=frozen_prediction["predicted_v_plus"], predicted_pb=frozen_prediction["predicted_pb"],
                predicted_tb=frozen_prediction["predicted_tb"], prediction_reason=frozen_prediction["event_reason"]))
            snapshots.append({**base, **prefix})
        if number % 10 == 0:
            print(f"ONLINE_PRECONTACT_BASELINE {number}/{len(eligible)}", flush=True)
    positions, events = plain(positions), plain(events)
    write_csv(out/"post_predictions.csv", positions)
    write_csv(out/"event_predictions.csv", events)
    write_csv(out/"update_latency.csv", plain(update_times))
    dump(out/"frozen_prefix_snapshots.json", snapshots)
    latency = {"all_raw_updates": _latency([row["ms"] for row in update_times]),
        "initialization": _latency([row["ms"] for row in update_times if row["initialization"]]),
        "steady_updates": _latency([row["ms"] for row in update_times if not row["initialization"] and not row["initialization_buffer"]]),
        "five_target_forecast": _latency([value for value in prediction_times if value is not None]),
        "first_contact_query": _latency([value for value in event_times if value is not None]),
        "measurement_condition": "Ubuntu wall time under concurrent collision fitting and native MuJoCo viewer load; not an isolated real-time benchmark"}
    dump(out/"latency.json", latency)
    summary = {"schema": "FROZEN_ONLINE_PRECONTACT_BASELINE/1", "completed": True,
        "hostname": platform.node(), "model": MODEL_NAME, "upstream_selected": selected["selected_name"],
        "selected_sha256_frozen_before_replay": frozen, "wall_elapsed_s": perf_counter()-started,
        "n_trajectories": len(eligible), "split_counts": dict(Counter(trace["split"] for trace in eligible)),
        "already_bounced_prefixes": sum(row["filter_bounced_at_cut"] for row in snapshots),
        "gaps_s": GAPS, "horizons_s_after_reference_contact": HORIZONS,
        "metrics": aggregate_collision_metrics(positions, events), "latency": latency,
        "input_boundary": "Only raw tXYZ at/before each cutoff; prefix snapshots independent copies; no post observations or cleaning masks; no reference contact feature",
        "already_bounced_rule": "Continue frozen state with bounced=True; first-event state unavailable, no second-impact search",
        "test_role": "Previously exposed historical groups reused as regression, not new blind validation",
        "comparison_boundary": "Compare only with same absolute reference-contact targets in this frozen-prefix task; do not use old post-observation streaming errors"}
    dump(out/"summary.json", summary)
    print("ONLINE_PRECONTACT_BASELINE_REPORT="+str(out/"summary.json"), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT/"artifacts/tennis_calibration/20261008_calibration")
    parser.add_argument("--selected", type=Path, default=ROOT/"artifacts/tennis_identified/20261009_training_prior/selected_model.json")
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/tennis_collision/20261009_frozen_pre/online_baseline")
    args = parser.parse_args()
    if platform.system() != "Linux" or Path(sys.prefix).resolve() != (ROOT/".venv-tennis").resolve():
        raise RuntimeError("Formal frozen online baseline must execute in Ubuntu project .venv-tennis")
    run_baseline(args.calibration, args.selected, args.out)


if __name__ == "__main__":
    main()
