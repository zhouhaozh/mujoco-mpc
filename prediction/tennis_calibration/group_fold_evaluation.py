"""Replay fixed-recipe group-fold checkpoints and aggregate out-of-fold errors.

Training and replay are separate commands. Each replay reads the unchanged
original scoring labels, saves compressed CSV, and can be resumed fold by fold.
Outer-fold rankings are descriptive sensitivity results, never a blind estimate
of a hold duration selected using those same outer folds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .gp_analysis_plots import _style
from .group_fold_training import FOLD_GROUPS, fold_plan
from .height_policy_run import phase, row_error
from .hold_sweep import (HOLDS_MS, HORIZONS, aggregate, candidate, digest, render,
                         _worker_init, run_case)
from .identified_run import dump, plain
from .hybrid_hold_evaluation import describe
from .online_data import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SCORING_SOURCE_FILES = ("group_fold_evaluation.py", "hold_sweep.py", "height_policy.py",
    "height_policy_run.py", "height_forecast.py", "gp_online.py", "identified_filter.py",
    "identified_dynamics.py", "collision_mapping.py", "hybrid_hold_evaluation.py", "online_data.py",
    "player_bounce.py", "collision_model_bank.py", "collision_forecast.py", "flight.py", "online_filter.py")


def read_json_csv(path):
    """Every CSV cell is a JSON scalar/array, preserving null/bool/float types."""
    with gzip.open(path, "rt", newline="") as stream:
        return [{k: json.loads(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def source_hashes():
    return {name: digest(ROOT/"tennis_calibration"/name) for name in SCORING_SOURCE_FILES}


def calibration_hashes(calibration):
    files = [calibration/"split.json", calibration/"clean/annotations.json",
             calibration/"clean/summary.json", calibration/"bounce/summary.json",
             *sorted((calibration/"clean").rglob("*.csv"))]
    return {str(p.relative_to(calibration)): digest(p) for p in files}


def checked_model(model_root, calibration, fold):
    folder = model_root/("fold"+str(fold))
    plan = json.loads((folder/"fold_plan.json").read_text())
    expected = fold_plan(json.loads((calibration/"split.json").read_text()), fold)
    if plan != expected:
        raise ValueError("fold model plan differs from the frozen group assignment")
    trained = json.loads((folder/"training_summary.json").read_text())
    if not trained.get("completed") or trained["fold"] != fold or trained["plan"] != plan:
        raise ValueError("fold model has no matching completed training summary")
    for name, sha in trained["trained_model_hashes"].items():
        if digest(folder/name) != sha:
            raise ValueError("trained checkpoint changed: "+name)
    audit = json.loads((folder/"training_input_audit.json").read_text())
    for name in ("raw_data", "flight_data", "impact_law", "gp_fit"):
        ids = set(audit[name]["unique_training_ids"])
        if ids-set(plan["train_ids"]) or ids & set(plan["evaluation_ids"]):
            raise ValueError("training/evaluation leakage in "+name)
    if audit["evaluation_raw_files_read"]:
        raise ValueError("training loader read evaluation raw files")
    checkpoint, selected_path = folder/"selected_collision.json", folder/"selected_model.json"
    selected = json.loads(selected_path.read_text())
    collision = json.loads(checkpoint.read_text())
    if selected["fold"] != fold or collision["fold"] != fold:
        raise ValueError("checkpoint fold mismatch")
    if selected["filter"]["initial_magnus_sigma"] != collision["prefix_config"]["b_prior_sigma_per_s"]:
        raise ValueError("fold prefix/filter priors differ")
    protected = {str(p.relative_to(folder)): digest(p) for p in
                 (checkpoint, selected_path, folder/"models/velocity_0",
                  folder/"models/velocity_1", folder/"models/velocity_3",
                  folder/"fold_plan.json", folder/"training_summary.json", folder/"training_input_audit.json")}
    return folder, plan, trained, checkpoint, selected, protected


def labeled_coverage(rows, events):
    counts = {e["candidate"]: Counter() for e in events}
    for row in rows:
        if row["elapsed_capture_s"] < .1-1e-12 or phase(row["relative_capture_s"]) is None:
            continue
        c = counts[row["candidate"]]
        c["requested"] += 1
        if not row["valid"]:
            c["missing_label"] += 1
        else:
            c["labeled"] += 1
            c["successful" if row_error(row) is not None else "failed_labeled"] += 1
    return [{"candidate": name, **dict(c),
             "prediction_coverage_given_label": c["successful"]/c["labeled"] if c["labeled"] else None,
             "tracking_failures": sum(bool(e["tracking_failure"]) for e in events if e["candidate"] == name)}
            for name, c in sorted(counts.items())]


def summarize(rows, events):
    result = aggregate(rows, events)
    result.pop("frozen_model_group_resampling", None)
    coverage = labeled_coverage(rows, events)
    by_arm = {r["candidate"]: r for r in coverage}
    key_coverage = {r["candidate"]: r for r in result["successful_key_coverage"]}
    for r in result["ranking"]:
        r["eligible_for_descriptive_ranking"] = (
            by_arm[r["candidate"]].get("failed_labeled", 0) == 0 and
            by_arm[r["candidate"]]["tracking_failures"] == 0 and
            key_coverage[r["candidate"]]["equal_to_reference"])
    result["absolute_labeled_coverage"] = coverage
    first200 = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (row["elapsed_capture_s"] >= .1-1e-12 and row["horizon_s"] == .5
                and 0 <= row["relative_capture_s"] < .2):
            error = row_error(row)
            if error is not None:
                first200[row["candidate"]][row["trajectory_id"]].append(error)
    result["first200ms_h500_metrics"] = [
        {"candidate": name, "horizon_s": .5, "capture_window": "[0,200ms)", **describe(values)}
        for name, values in sorted(first200.items())]
    rankable = [r for r in result["ranking"] if r["eligible_for_descriptive_ranking"]]
    result["descriptive_best_candidate"] = min(rankable, key=lambda r:
        (r["balanced_future_rmse_m"], r["candidate"]))["candidate"] if rankable else None
    return result


def execute_fold(args):
    out = args.out/("fold"+str(args.fold))
    if (out/"summary.json").exists():
        raise RuntimeError("fold evaluation is already completed; preserve it")
    if (out/"predictions.csv.gz").exists():
        raise RuntimeError("partial predictions exist; choose a fresh output root before replay")
    out.mkdir(parents=True, exist_ok=True)
    model_dir, plan, trained, checkpoint, selected, model_sha = checked_model(
        args.models_root, args.calibration, args.fold)
    original = calibration_hashes(args.calibration)
    scoring = source_hashes()
    data = load_dataset(args.calibration)
    expected = set(plan["evaluation_ids"])
    traces = []
    excluded = []
    for trace in data:
        if trace["id"] not in expected:
            continue
        if trace["bounce_reference_s"] is None:
            excluded.append({"id": trace["id"], "reason": "no_original_first_contact_reference"})
            continue
        trace = dict(trace, original_split=trace["split"], split="oof", fold=args.fold)
        traces.append(trace)
    if {t["id"] for t in traces} | {r["id"] for r in excluded} != expected:
        raise ValueError("evaluation identities do not match calibration")
    if not traces:
        raise ValueError("no evaluable trajectories in this fold")
    started = perf_counter()
    execution = {"schema": "GP_GROUP_FOLD_EVALUATION/1", "fold": args.fold,
        "created_utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
        "python_prefix": sys.prefix, "plan": plan, "excluded": excluded,
        "n_evaluation_trajectories": len(traces), "holds_ms": list(HOLDS_MS),
        "scoring_source_hashes": scoring, "calibration_source_hashes": original,
        "model_hashes": model_sha, "model_directory": str(model_dir),
        "csv_encoding": "each CSV cell is JSON encoded; gzip compressed; read with json.loads",
        "original_scoring_references_unchanged": True,
        "model_training_summary": trained}
    dump(out/"execution.json", execution)
    rows, events = [], []
    trace_map = {t["id"]: t for t in traces}
    writer = None
    with gzip.open(out/"predictions.csv.gz", "wt", newline="") as stream:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                                 initargs=(checkpoint, selected)) as pool:
            futures = [pool.submit(run_case, t, h) for t in traces for h in HOLDS_MS]
            for index, future in enumerate(as_completed(futures), 1):
                batch, event = future.result()
                original_split = trace_map[event["trajectory_id"]]["original_split"]
                for r in batch:
                    r.update(fold=args.fold, original_split=original_split)
                event.update(fold=args.fold, original_split=original_split)
                rows.extend(batch); events.append(event)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(batch[0])); writer.writeheader()
                for row in batch:
                    writer.writerow({k: json.dumps(plain(v), ensure_ascii=False, allow_nan=False)
                                     for k, v in row.items()})
                if index % 10 == 0 or index == len(futures):
                    print("FOLD_REPLAY "+json.dumps({"fold": args.fold, "completed": index,
                          "total": len(futures), "elapsed_s": perf_counter()-started}), flush=True)
    by_id = defaultdict(list)
    for e in events: by_id[e["trajectory_id"]].append(e)
    for identity, members in by_id.items():
        if {e["hold_ms"] for e in members} != set(HOLDS_MS):
            raise ValueError("missing or duplicate hold arms for "+identity)
        if len(members) != len(HOLDS_MS): raise ValueError("duplicate case")
        if not any(e["tracking_failure"] for e in members):
            for field in ("height_contact_t_s", "height_discovery_t_s", "initialization_t_s"):
                if len({e[field] for e in members}) != 1:
                    raise ValueError("hold arms changed pre-event state/clock")
    if calibration_hashes(args.calibration) != original or source_hashes() != scoring:
        raise RuntimeError("calibration or scoring sources changed during fold replay")
    if model_sha != {name: digest(model_dir/name) for name in model_sha}:
        raise RuntimeError("trained model changed during fold replay")
    result = summarize(rows, events)
    result.update(**execution, completed=True, elapsed_s=perf_counter()-started,
                  events=sorted(events, key=lambda e: (e["trajectory_id"], e["hold_ms"])),
                  predictions_sha256=digest(out/"predictions.csv.gz"),
                  semantics="original offline references score only; fold data parameters retrained; six matched height-event holds",
                  ranking_role="descriptive outer-fold comparison; selecting its best arm is not blind validation")
    result["files"] = render(result, out, "oof", HOLDS_MS)
    dump(out/"summary.json", result)
    print("FOLD_EVALUATED "+json.dumps({"fold": args.fold, "seconds": result["elapsed_s"],
          "ranking": result["ranking"], "out": str(out)}), flush=True)


def fold_comparison_plot(fold_summaries, overall, out):
    _style()
    matrix, labels = [], []
    for item in fold_summaries:
        index = {r["candidate"]: r for r in item["ranking"]}
        matrix.append([index.get(candidate(h), {}).get("balanced_future_rmse_m", np.nan)*100 for h in HOLDS_MS])
        labels.append("第 %d 折（N=%d）" % (item["fold"]+1, item["n_evaluation_trajectories"]))
    index = {r["candidate"]: r for r in overall["ranking"]}
    matrix.append([index.get(candidate(h), {}).get("balanced_future_rmse_m", np.nan)*100 for h in HOLDS_MS])
    labels.append("合并留出预测（OOF）")
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(bottom=.2, left=.22, top=.84, right=.96)
    upper = float(np.nanmax(values)) if np.isfinite(values).any() else 1.
    ax.imshow(values, cmap="Blues", vmin=0, vmax=max(upper*1.7, .01), aspect="auto")
    ax.set_xticks(range(len(HOLDS_MS)), [str(h)+" ms" for h in HOLDS_MS])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("估计高度触地后停止融合的时长")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, "—" if not np.isfinite(values[i, j]) else f"{values[i,j]:.2f}",
                    ha="center", va="center", fontsize=12, color="#11283c")
    fig.suptitle("不同采集组留出验证：阶段平衡综合 RMSE（cm）", fontsize=16)
    fig.text(.06, .065, "每折重新拟合物理参数、GP 与先验；原评分参考固定。数值越低越好。", fontsize=10)
    fig.text(.06, .03, "各折最低值仅描述敏感性；不能把同批外层验证选出的最优值作为独立盲测结果。", fontsize=10)
    files = []
    for suffix in ("png", "pdf"):
        path = out/("fold_hold_comparison."+suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight"); files.append(str(path))
    plt.close(fig)
    return files


def aggregate_folds(args):
    fold_summaries, rows, events, seen = [], [], [], set()
    data = load_dataset(args.calibration)
    eligible = {t["id"] for t in data if t["bounce_reference_s"] is not None}
    expected_calibration = calibration_hashes(args.calibration)
    expected_scoring = None
    for fold in range(len(FOLD_GROUPS)):
        folder = args.out/("fold"+str(fold))
        if not (folder/"summary.json").exists():
            if args.allow_partial: continue
            raise RuntimeError("missing completed evaluation fold "+str(fold))
        item = json.loads((folder/"summary.json").read_text())
        if not item.get("completed") or item["fold"] != fold:
            raise ValueError("invalid fold evaluation summary")
        if item["calibration_source_hashes"] != expected_calibration:
            raise ValueError("fold scoring references differ")
        if expected_scoring is None: expected_scoring = item["scoring_source_hashes"]
        if item["scoring_source_hashes"] != expected_scoring:
            raise ValueError("folds used different scoring implementations")
        if item["holds_ms"] != list(HOLDS_MS): raise ValueError("hold grids differ")
        if digest(folder/"predictions.csv.gz") != item["predictions_sha256"]:
            raise ValueError("fold predictions changed after scoring")
        _, plan, _, _, _, model_sha = checked_model(args.models_root, args.calibration, fold)
        if item["plan"] != plan or item["model_hashes"] != model_sha:
            raise ValueError("fold evaluation was made with different training artifacts")
        expected_ids = set(plan["evaluation_ids"]) & eligible
        actual_ids = {e["trajectory_id"] for e in item["events"]}
        if actual_ids != expected_ids or actual_ids & seen:
            raise ValueError("outer evaluation identities duplicate or differ")
        seen.update(actual_ids)
        batch = read_json_csv(folder/"predictions.csv.gz")
        if {r["trajectory_id"] for r in batch} != actual_ids or {r["fold"] for r in batch} != {fold}:
            raise ValueError("prediction rows do not match fold summary")
        if any(r["split"] != "oof" for r in batch): raise ValueError("OOF split missing")
        rows.extend(batch); events.extend(item["events"]); fold_summaries.append(item)
    if not fold_summaries: raise ValueError("no completed fold evaluations")
    complete = len(fold_summaries) == len(FOLD_GROUPS)
    if complete and seen != eligible: raise ValueError("complete OOF predictions do not cover every eligible trace")
    result = summarize(rows, events)
    fold_table = []
    variability = []
    for item in fold_summaries:
        index = {r["candidate"]: r for r in item["ranking"]}
        base = index.get(candidate(100), {}).get("balanced_future_rmse_m")
        for hold in HOLDS_MS:
            cell = index.get(candidate(hold))
            if not cell: continue
            fold_table.append({"fold": item["fold"], "evaluation_groups": item["plan"]["evaluation_groups"],
                "N": item["n_evaluation_trajectories"], "hold_ms": hold,
                "balanced_rmse_cm": cell["balanced_future_rmse_m"]*100,
                "delta_vs_hold100_cm": (cell["balanced_future_rmse_m"]-base)*100 if base is not None else None,
                "eligible_for_descriptive_ranking": cell["eligible_for_descriptive_ranking"]})
    for hold in HOLDS_MS:
        values = np.asarray([r["balanced_rmse_cm"] for r in fold_table if r["hold_ms"] == hold])
        if not len(values): continue
        variability.append({"hold_ms": hold, "fold_count": len(values),
            "mean_across_folds_cm": float(values.mean()), "min_fold_cm": float(values.min()),
            "max_fold_cm": float(values.max()),
            "sample_std_across_folds_cm": float(values.std(ddof=1)) if len(values)>1 else None,
            "definition": "equal-fold descriptive dispersion; not a confidence interval or pooled OOF RMSE"})
    result.update(schema="GP_GROUP_FOLD_OOF/1", completed=complete,
        created_utc=datetime.now(timezone.utc).isoformat(), hostname=platform.node(), python_prefix=sys.prefix,
        holds_ms=list(HOLDS_MS), horizons_s=HORIZONS, folds_present=[r["fold"] for r in fold_summaries],
        N=len(seen), expected_eligible_N=len(eligible), calibration_source_hashes=expected_calibration,
        scoring_source_hashes=expected_scoring, fold_ranking_table=fold_table,
        fold_variability=variability,
        per_fold=[{key: item[key] for key in ("fold", "plan", "n_evaluation_trajectories", "excluded",
            "ranking", "phase_metrics", "group_ranking", "group_phase_metrics", "resume_first50ms_metrics",
            "absolute_labeled_coverage", "successful_key_coverage", "first200ms_h500_metrics",
            "descriptive_best_candidate", "model_hashes")}
            for item in fold_summaries],
        score_definition="within trajectory phase/horizon balanced MSE, then trajectory equal-weight RMSE; all OOF rows scored together",
        p95_definition="step inverse distance ECDF; every trajectory has total weight 1/N; original acquisition group ids preserved",
        evaluation_role="historical group-fold validation of a previously selected fixed recipe, not nested hyperparameter selection",
        limitations=["inferred acquisition groups, metadata unconfirmed",
            "family and fixed hyperparameters were chosen in previous experiments on these historical data",
            "outer-fold hold rankings are exploratory; choosing their minimum and reporting it as blind accuracy is invalid",
            "same-device provisional scoring references retain measurement error",
            "20ms capture grid; recovery peaks are not all-frame maxima",
            "no deployment/default model changes"])
    args.out.mkdir(parents=True, exist_ok=True)
    result["files"] = render(result, args.out, "oof", HOLDS_MS)+fold_comparison_plot(fold_summaries, result, args.out)
    dump(args.out/"fold_comparison_table.json", {"rows": fold_table, "dispersion": variability})
    dump(args.out/"summary.json", result)
    print("OOF_AGGREGATED "+json.dumps({"folds": result["folds_present"], "N": len(seen),
          "completed": complete, "ranking": result["ranking"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT/"artifacts/new_ubuntu/tennis_calibration")
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold", type=int, choices=range(5))
    mode.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if platform.system() != "Linux" or Path(sys.prefix).resolve() != (ROOT/".venv-tennis").resolve():
        raise RuntimeError("Formal group-fold replay/aggregation must run in Ubuntu project .venv-tennis")
    if args.workers < 1: raise ValueError("workers must be positive")
    if args.aggregate_only: aggregate_folds(args)
    else: execute_fold(args)


if __name__ == "__main__":
    main()
