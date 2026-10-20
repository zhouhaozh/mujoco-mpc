"""Train the frozen GP recipe again with entire inferred groups held out.

This is historical, fixed-recipe cross-validation, not a new blind experiment.
No evaluation trajectory is loaded by the training-data loader. Each fold
refits drag, impact labels/law/height, Magnus prior, and GP normalization/weights.
The original cleaned reference labels remain untouched for downstream scoring.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

import numpy as np

from .bounce import impact_estimate, fit_law
from .collision_mapping import (fit_model, save_model, trajectory_weights,
                                velocity_targets)
from .collision_prefix import estimate_prefix, predict_contact
from .flight import _load_clean, _resolve_clean, fit_shared_drag
from .identified_dynamics import ASSUMPTIONS, fit_shared_drag_magnus
from .identified_run import dump, plain

ROOT = Path(__file__).resolve().parents[1]
FOLD_GROUPS = ((6,), (5, 8, 9), (4, 7), (3, 0), (1, 2, 10))
GAPS = (0., .005, .01, .02, .04, .08)
RECIPE = {
    "fold_groups": FOLD_GROUPS,
    "grouping": "fixed count-balanced inferred acquisition groups; no model errors used",
    "gp": {"kind": "gp", "length_scale": 1., "noise": .6},
    "prefix_window_s": .4, "b_policy": "carry", "height_policy": "legacy_training_median",
    "gap_variants_s": GAPS, "magnus_initial_factor": 1., "magnus_drive_factor": .3,
    "aero_starts": (.01, .02, .05),
    "role": "historical fixed-recipe group CV; hyperparameter family was selected historically",
}


def fold_plan(split, fold):
    if fold not in range(len(FOLD_GROUPS)):
        raise ValueError("fold must be in 0..4")
    groups = split["trajectory_to_group"]
    ids = [i for part in ("train", "validation", "test") for i in split[part]]
    if len(ids) != len(set(ids)) or set(ids) != set(groups):
        raise ValueError("original split must partition all grouped identities exactly once")
    if set(groups.values()) != set(g for f in FOLD_GROUPS for g in f):
        raise ValueError("frozen fold plan does not match the acquisition groups")
    held = set(FOLD_GROUPS[fold])
    train = sorted(i for i in ids if groups[i] not in held)
    evaluation = sorted(i for i in ids if groups[i] in held)
    return {"fold": fold, "train_ids": train, "evaluation_ids": evaluation,
            "train_groups": sorted(set(groups.values())-held),
            "evaluation_groups": sorted(held), "grouping_is_inferred": True}


def assert_training_ids(ids, plan):
    identities = list(ids)
    forbidden = set(identities) & set(plan["evaluation_ids"])
    unknown = set(identities)-set(plan["train_ids"])
    if forbidden or unknown:
        raise ValueError("non-training identity passed to fitting: "+repr(sorted(forbidden | unknown)))
    return {"unique_training_ids": sorted(set(identities)),
            "evaluation_overlap": [], "n_rows": len(identities)}


def training_bundle(calibration, plan):
    """Read per-trajectory data only after training membership is established."""
    calibration = Path(calibration)
    annotations = json.loads((calibration/"clean/annotations.json").read_text())
    by_id = {r["trajectory_id"]: r for r in annotations}
    if len(by_id) != len(annotations) or set(by_id) != set(plan["train_ids"]+plan["evaluation_ids"]):
        raise ValueError("annotations and fold identities differ")
    bundle = []
    for identity in plan["train_ids"]:
        row = deepcopy(by_id[identity])
        path = _resolve_clean(calibration/"clean", row["cleaned_file"])
        t, p = _load_clean(path)
        raw_t, raw_p = _load_clean(path, only_kept=False)
        bundle.append({"id": identity, "annotation": row, "times": t, "xyz": p,
                       "raw_times": raw_t, "raw_xyz": raw_p,
                       "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert_training_ids([r["id"] for r in bundle], plan)
    return bundle


def pre_segments(bundle):
    segments, identities, excluded = [], [], []
    for item in bundle:
        row = item["annotation"]
        start, end = row.get("pre_start_t_s"), row.get("pre_end_t_s")
        if start is None or end is None:
            excluded.append({"id": item["id"], "reason": "no_pre_segment"})
            continue
        keep = (item["times"] >= start) & (item["times"] <= end)
        if keep.sum() < 24 or np.ptp(item["times"][keep]) < .14:
            excluded.append({"id": item["id"], "reason": "insufficient_pre_segment"})
            continue
        segments.append((item["times"][keep], item["xyz"][keep]))
        identities.append(item["id"])
    return segments, identities, excluded


def _impact_task(payload):
    item, k = payload
    row, t, p = item["annotation"], item["times"], item["xyz"]
    try:
        if row.get("first_bounce_t_s") is None:
            raise ValueError("no_bounce_annotation")
        base = impact_estimate(t, p, row, k)
        sensitivity = []
        for window in (.08, .18):
            try:
                sensitivity.append(impact_estimate(t, p, row, k, window))
            except ValueError:
                pass
        base.update(trajectory_id=item["id"], split="train",
                    annotation_confidence=float(row.get("confidence", 0)),
                    window_e_span=float(np.ptp([base["e_n_observed"]]+[s["e_n_observed"] for s in sensitivity])),
                    window_alpha_span=float(np.ptp([base["alpha_t_observed"]]+[s["alpha_t_observed"] for s in sensitivity])),
                    sensitivity_windows_successful=len(sensitivity), sensitivity_windows=sensitivity)
        # These are the exact acceptance checks in bounce.run_bounce, before
        # the subsequent training-derived contact-height consistency check.
        reasons = []
        if not base["converged"]: reasons.append("local_optimizer_not_converged")
        if not sensitivity: reasons.append("window_sensitivity_not_available")
        if base["position_error"]["rmse_m"] > .08: reasons.append("position_reconstruction_error_above_80mm")
        if base["time_at_bound"]: reasons.append("impact_time_at_bracket_bound")
        if base["window_e_span"] > .18: reasons.append("normal_velocity_window_unstable")
        if base["window_alpha_span"] > .12: reasons.append("tangential_velocity_window_unstable")
        if not .1 < base["e_n_observed"] < 1.3: reasons.append("observed_normal_ratio_unusual")
        if base["annotation_confidence"] < .55: reasons.append("annotation_low_confidence")
        base["quality_reasons"] = reasons
        return {"record": plain(base)}
    except (ValueError, KeyError) as exc:
        return {"excluded": {"id": item["id"], "reason": str(exc)}}


def _map(function, payloads, workers):
    if workers == 1:
        return list(map(function, payloads))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, payloads, chunksize=1))


def impact_calibration(bundle, k, workers):
    results = _map(_impact_task, [(r, k) for r in bundle], workers)
    records = [r["record"] for r in results if "record" in r]
    excluded = [r["excluded"] for r in results if "excluded" in r]
    training = [r for r in records if not r["quality_reasons"]]
    if len(training) < 5:
        raise RuntimeError("fewer_than_five_quality_training_impacts")
    heights = np.array([r["p_b_m"][2] for r in training])
    height = float(np.median(heights))
    spread = max(.010, 1.4826*float(np.median(abs(heights-height))))
    for r in records:
        if abs(r["p_b_m"][2]-height) > max(.04, 4*spread):
            r["quality_reasons"].append("contact_height_inconsistent_with_training")
        r["used_for_fit"] = not r["quality_reasons"]
    training = [r for r in records if r["used_for_fit"]]
    law = fit_law(training)
    height = float(np.median([r["p_b_m"][2] for r in training]))
    return {"parameters": {**law, "effective_contact_center_height_m": height},
            "impact_estimates": records, "excluded": excluded,
            "label_role": "training-only outgoing-velocity reconstruction with fold training drag"}


def prior_configs(aero_states, template):
    b = np.asarray(aero_states)[:, 6:9]
    sigma = max(.02, float(np.sqrt(np.mean(np.sum(b*b, axis=1))/3)))
    cfg = deepcopy(template)
    cfg.update(initial_magnus_sigma=sigma, magnus_drive_sigma=max(.003, .3*sigma))
    prefix = {"window_s": .4, "b_prior_sigma_per_s": sigma}
    return cfg, prefix


def _prefix_task(payload):
    item, impact, gap, parameters, height, cfg = payload
    cut = float(impact["t_b_s"])-gap
    take = item["raw_times"] <= cut
    t, p = item["raw_times"][take], item["raw_xyz"][take]
    fit = estimate_prefix(t, p, cut, parameters, height, cfg)
    contact = predict_contact(fit, parameters, height)
    return {"id": item["id"], "gap_s": gap, "cut_s": cut,
            "input_count": len(t), "input_end_s": float(t[-1]) if len(t) else None,
            "fit": fit, "contact": contact, "v_plus_label": impact["v_plus_m_s"]}


def train_fold(calibration, out, fold, workers, filter_template):
    started = perf_counter()
    split = json.loads((calibration/"split.json").read_text())
    plan = fold_plan(split, fold)
    out.mkdir(parents=True, exist_ok=False)
    stage = "load_training"
    timers = {}
    def mark(name):
        nonlocal stage
        stage = name
        print("FOLD_STAGE "+json.dumps({"fold": fold, "stage": name,
              "elapsed_s": perf_counter()-started}), flush=True)
    execution = {"platform": platform.platform(), "hostname": platform.node(),
                 "python_prefix": sys.prefix, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "recipe": RECIPE, "plan": plan,
                 "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT/"tennis_calibration").glob("*.py")}}
    dump(out/"execution.json", execution)
    dump(out/"fold_plan.json", plan)
    try:
        bundle = training_bundle(calibration, plan)
        segments, ids, excluded = pre_segments(bundle)
        input_audit = {"raw_data": assert_training_ids([r["id"] for r in bundle], plan),
                       "flight_data": assert_training_ids(ids, plan),
                       "training_source_sha256": {r["id"]: r["source_sha256"] for r in bundle},
                       "evaluation_raw_files_read": [],
                       "evaluation_reference": "unchanged original calibration labels, scorer only"}
        dump(out/"training_input_audit.json", input_audit)
        mark("six_state_drag")
        tick = perf_counter(); drag = fit_shared_drag(segments)
        timers[stage] = perf_counter()-tick
        dump(out/"six_state_drag.json", {**drag, "trajectory_ids": ids, "excluded": excluded})
        if not drag["converged"]: raise RuntimeError("six-state shared drag did not converge")
        mark("training_impact_labels_and_law")
        tick = perf_counter(); bounce = impact_calibration(bundle, drag["k_drag_per_m"], workers)
        timers[stage] = perf_counter()-tick
        dump(out/"training_bounce.json", bounce)
        if not bounce["parameters"]["converged"]: raise RuntimeError("bounce law did not converge")
        impact_ids = [r["trajectory_id"] for r in bounce["impact_estimates"] if r["used_for_fit"]]
        input_audit["impact_law"] = assert_training_ids(impact_ids, plan)
        mark("nine_state_drag_magnus")
        tick = perf_counter()
        aero = fit_shared_drag_magnus(segments, starts=RECIPE["aero_starts"], diagnose=False)
        timers[stage] = perf_counter()-tick
        dump(out/"aero_fit.json", {**aero, "trajectory_ids": ids, "excluded": excluded,
                                 "diagnostic_svd_omitted": True})
        if not aero["converged"]: raise RuntimeError("nine-state shared drag/Magnus fit did not converge")
        cfg, prefix = prior_configs(aero["initial_states"], filter_template)
        law = bounce["parameters"]
        height = law["effective_contact_center_height_m"]
        parameters = {**ASSUMPTIONS, "k_drag": aero["k_drag_per_m"], "e_n": law["e_n"],
                      "alpha_t": law["alpha_t"], "impact_mode": "effective"}
        dump(out/"training_prior.json", {"trajectory_ids": ids,
             "sigma_per_s": cfg["initial_magnus_sigma"],
             "drive_sigma_per_s_sqrt_s": cfg["magnus_drive_sigma"],
             "prefix_prior_sigma_per_s": prefix["b_prior_sigma_per_s"],
             "formula": "max(.02,sqrt(mean(norm(b)^2)/3)); initial factor1; drive max(.003,.3*sigma)",
             "prefix_default_overridden": True})
        mark("training_prefix_variants")
        impact = {r["trajectory_id"]: r for r in bounce["impact_estimates"] if r["used_for_fit"]}
        payloads = [(item, impact[item["id"]], gap, parameters, height, prefix)
                    for item in bundle if item["id"] in impact for gap in GAPS]
        assert_training_ids([p[0]["id"] for p in payloads], plan)
        tick = perf_counter(); cases = _map(_prefix_task, payloads, workers)
        timers[stage] = perf_counter()-tick
        dump(out/"training_prefix_cases.json", cases)
        active = [c for c in cases if c["contact"]["valid"]]
        if len({c["id"] for c in active}) < 20:
            raise RuntimeError("fewer than 20 independent successful training events")
        identities = [c["id"] for c in active]
        input_audit["gp_fit"] = assert_training_ids(identities, plan)
        vm = np.asarray([c["contact"]["v_minus"] for c in active])
        bm = np.asarray([c["contact"]["b_minus"] for c in active])
        targets = np.asarray([c["v_plus_label"] for c in active])
        x, y = velocity_targets(vm, bm, targets, parameters["e_n"], parameters["alpha_t"])
        weights = trajectory_weights(identities)
        mark("gp_training")
        tick = perf_counter(); reports = {}
        # BounceModelBank loads these three frozen keys. The two Ridge fits
        # provide checkpoint compatibility; they are not selected or scored.
        for key, model_cfg in (("velocity_0", {"kind": "ridge", "degree": 1, "ridge": .1}),
                               ("velocity_1", {"kind": "ridge", "degree": 2, "ridge": 1.}),
                               ("velocity_3", RECIPE["gp"])):
            model, reports[key] = fit_model(x, y, weights, model_cfg)
            save_model(model, out/"models"/key)
        timers[stage] = perf_counter()-tick
        selected_collision = {"schema": "FROZEN_PRECONTACT/1",
             "selected_model": {"name": "velocity_3_carry", "velocity_model": "velocity_3",
                                "b_policy": "carry", "lift_model": None},
             "upstream": {"mode": "recent", "height_name": "legacy", "height": height},
             "parameters": parameters, "prefix_config": prefix,
             "selection": "historical fixed recipe; no fold validation model selection",
             "training_audit_file": "training_input_audit.json", "fold": fold,
             "true_spin_output": False}
        selected_filter = {"schema": "TENNIS_IDENTIFIED/1",
             "selected_name": "fixed_recipe_group_fold_"+str(fold), "parameters": parameters,
             "contact_height": height, "filter": cfg, "augmented": True,
             "selection_source": "historical fixed recipe; data parameters refitted on fold train only",
             "fold": fold}
        dump(out/"selected_collision.json", selected_collision)
        dump(out/"selected_model.json", selected_filter)
        dump(out/"training_input_audit.json", input_audit)
        dump(out/"mapping_training.json", {"reports": reports, "n_training_variants": len(active),
             "n_requested_variants": len(cases), "n_independent_training_events": len(set(identities)),
             "per_event_weight_sum": {i: float(weights[np.array(identities)==i].sum()) for i in sorted(set(identities))},
             "failed_prefixes": [{"id": c["id"], "gap_s": c["gap_s"], "reason": c["contact"]["reason"]}
                                   for c in cases if not c["contact"]["valid"]]})
        result = {"completed": True, "fold": fold, "plan": plan, "timing_s": timers,
                  "total_s": perf_counter()-started, "parameters": parameters,
                  "contact_height": height, "initial_magnus_sigma": cfg["initial_magnus_sigma"],
                  "prefix_prior_sigma": prefix["b_prior_sigma_per_s"],
                  "n_training_events": len(set(identities)), "n_training_variants": len(active),
                  "trained_model_hashes": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (out/"selected_collision.json", out/"selected_model.json", out/"models/velocity_3")},
                  "evaluation_reference_labels_mutated": False,
                  "limitations": ["inferred session metadata", "historically chosen fixed family and hyperparameters",
                                  "not nested model selection and not a new blind data set"]}
        dump(out/"training_summary.json", result)
        print("FOLD_TRAINED "+json.dumps({"fold": fold, "total_s": result["total_s"],
               "out": str(out), "events": len(set(identities))}), flush=True)
        return result
    except Exception as exc:
        dump(out/"failure.json", {"stage": stage, "error": type(exc).__name__+": "+str(exc),
             "traceback": traceback.format_exc(), "timing_s": timers, "total_s": perf_counter()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT/"artifacts/new_ubuntu/tennis_calibration")
    parser.add_argument("--filter-template", type=Path, default=ROOT/"artifacts/tennis_identified/20261009_training_prior/selected_model.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if platform.system() != "Linux" or Path(sys.prefix).resolve() != (ROOT/".venv-tennis").resolve():
        raise RuntimeError("Formal group-fold training must use Ubuntu project .venv-tennis")
    if args.workers < 1: raise ValueError("workers must be positive")
    template = json.loads(args.filter_template.read_text())["filter"]
    train_fold(args.calibration, args.out, args.fold, args.workers, template)


if __name__ == "__main__":
    main()
