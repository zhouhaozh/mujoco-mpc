"""Train-only small residual heads over frozen observer replay caches.

The input caches are produced by robust_optimization. This second stage never
reruns or modifies an observer, height detector, measurement stream or label.
All numerical training, scoring and plotting is intended for Ubuntu.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from time import perf_counter

import numpy as np

from .identified_run import plain
from .residual_head import FEATURE_NAMES, ResidualHead, fit_residual_head


CACHE_FIELDS = ("X", "pred", "ref", "H", "gate", "ids", "capture_t_s",
    "elapsed_capture_s", "relative_capture_s", "target_t_s", "anchor", "dense",
    "valid", "initialized", "prediction_ok", "split", "groupid", "candidate")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dump(path, payload):
    Path(path).write_text(json.dumps(plain(payload), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def read_rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        return [{key: json.loads(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def write_rows(path, rows):
    if not rows:
        raise ValueError("no residual evaluation rows")
    fields = sorted({key for row in rows for key in row})
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(plain(row.get(key)), ensure_ascii=False,
                                             allow_nan=False) for key in fields})


def read_training_ids(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("train_ids")
    if not isinstance(payload, list) or not payload or len(set(payload)) != len(payload):
        raise ValueError("training-ID JSON must be a nonempty unique list or object.train_ids")
    return {str(value) for value in payload}


def read_cache(path, parent):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(CACHE_FIELDS)-set(archive.files)
        if missing:
            raise ValueError("cache is missing fields: "+", ".join(sorted(missing)))
        cache = {name: archive[name].copy() for name in CACHE_FIELDS}
    n = len(cache["ids"])
    for name, values in cache.items():
        expected = (n, len(FEATURE_NAMES)) if name == "X" else ((n, 3) if name in ("pred", "ref") else (n,))
        if values.shape != expected:
            raise ValueError("invalid cache shape: "+name)
    if not n or set(map(str, cache["candidate"])) != {parent}:
        raise ValueError("cache does not contain exactly the requested observer")
    if (not np.isfinite(cache["gate"]).all() or np.any(cache["gate"] < 0)
            or np.any(cache["gate"] > 1)):
        raise ValueError("invalid cached causal gate")
    if np.any(~np.isfinite(cache["H"])) or np.any(cache["H"] <= 0):
        raise ValueError("residual replay expects positive fixed horizons")
    # No target/reference array is used to manufacture or repair input features.
    return cache


def training_arrays(cache, allowed_train_ids):
    """Select only explicit training IDs, common anchors and a fully open gate."""
    ids = cache["ids"].astype(str)
    membership = np.isin(ids, sorted(allowed_train_ids))
    eligible = (membership & cache["anchor"].astype(bool)
                & np.isclose(cache["gate"], 1., atol=1e-12, rtol=0.)
                & cache["valid"].astype(bool) & cache["prediction_ok"].astype(bool)
                & cache["initialized"].astype(bool)
                & (cache["elapsed_capture_s"] >= .1-1e-12)
                & np.isfinite(cache["X"]).all(axis=1)
                & np.isfinite(cache["pred"]).all(axis=1)
                & np.isfinite(cache["ref"]).all(axis=1))
    if not np.any(eligible):
        raise ValueError("no eligible fully-open-gate training anchors")
    used = set(ids[eligible])
    assert used <= set(allowed_train_ids)
    result = {"allowed_training_ids": sorted(allowed_train_ids),
        "observed_training_ids": sorted(used),
        "allowed_ids_without_eligible_rows": sorted(set(allowed_train_ids)-used),
        "cache_rows": len(ids), "cache_rows_in_training_ids": int(membership.sum()),
        "eligible_training_rows": int(eligible.sum()),
        "gate_requirement": "gate == 1, after actual resume + 100ms with default gate",
        "anchor_requirement": "common 20ms anchor, no dense-frame oversampling",
        "training_rows_per_trajectory": dict(sorted(Counter(ids[eligible]).items())),
        "training_weighting": "each eligible trajectory has equal total weight",
        "targets": "legal reference XYZ minus original parent forecast XYZ",
        "normalization": "fit on eligible explicit training IDs only"}
    return (cache["X"][eligible], cache["ref"][eligible]-cache["pred"][eligible],
            ids[eligible].tolist(), result)


def cache_lookup(cache):
    result = {}
    for index, (identity, capture, horizon) in enumerate(zip(cache["ids"], cache["capture_t_s"], cache["H"])):
        key = str(identity), float(capture), float(horizon)
        if key in result:
            raise ValueError("duplicate capture/horizon key in residual cache")
        result[key] = index
    return result


def apply_head_to_rows(head, cache, rows, parent, candidate, *, maximum_correction_m=.15):
    """Reuse the original labels/clocks and apply only causal output correction."""
    lookup = cache_lookup(cache)
    selected = [row for row in rows if row["candidate"] == parent]
    if not selected:
        raise ValueError("no evaluation rows for residual parent: "+parent)
    indices = []
    for row in selected:
        key = str(row["trajectory_id"]), float(row["capture_t_s"]), float(row["horizon_s"])
        if key not in lookup:
            raise ValueError("CSV/cached prediction key mismatch")
        i = lookup[key]
        if str(cache["split"][i]) != row["split"] or bool(cache["valid"][i]) != bool(row["valid"]):
            raise ValueError("CSV/cache split or label-validity mismatch")
        if not np.isclose(cache["target_t_s"][i], row["target_t_s"], atol=1e-12, rtol=0.):
            raise ValueError("CSV/cache target-time mismatch")
        if row["predicted_xyz"] is not None:
            np.testing.assert_allclose(cache["pred"][i], row["predicted_xyz"], atol=1e-12, rtol=0.)
        if row["target_xyz"] is not None:
            np.testing.assert_allclose(cache["ref"][i], row["target_xyz"], atol=1e-12, rtol=0.)
        indices.append(i)
    indices = np.asarray(indices, dtype=int)
    gates = cache["gate"][indices]
    has_prediction = np.asarray([row["predicted_xyz"] is not None and not row.get("prediction_reason")
                                 for row in selected])
    active = (gates > 0) & has_prediction
    if np.any(active & ~np.isfinite(cache["X"][indices]).all(axis=1)):
        raise ValueError("active residual output has no finite causal features")
    displacements = np.zeros((len(selected), 3))
    if np.any(active):
        displacements[active] = (gates[active, None]
            *head.predict_bounded(cache["X"][indices[active]], cache["H"][indices[active]],
                                  maximum_correction_m=maximum_correction_m))
    results = []
    for row, index, delta in zip(selected, indices, displacements):
        output = dict(row, candidate=candidate, residual_parent=parent,
                      residual_kind=head.kind, residual_gate=float(cache["gate"][index]),
                      residual_displacement_xyz=delta.tolist())
        if row["predicted_xyz"] is not None:
            output["predicted_xyz"] = (np.asarray(row["predicted_xyz"])+delta).tolist()
        results.append(output)
    norms = np.linalg.norm(displacements, axis=1)
    return results, {"parent": parent, "candidate": candidate,
        "evaluation_rows": len(results), "nonzero_gate_prediction_rows": int(active.sum()),
        "zero_gate_rows_unchanged": bool(np.all(displacements[gates == 0] == 0)),
        "maximum_applied_correction_m": float(norms.max()),
        "same_parent_event_clock": True, "same_parent_label_targets": True,
        "cap_m_at_h500": maximum_correction_m,
        "cap_formula": "cap * min(horizon_s / 0.5, 1) * causal_gate"}


def run(args):
    # Import only after argument validation; this function is shared with the
    # observer experiment and intentionally keeps dense peaks out of ranking.
    from .robust_optimization import summarize

    if (args.out/"summary.json").exists() or (args.out/"execution.json").exists():
        raise RuntimeError("residual output already exists; choose a fresh directory")
    parents = list(dict.fromkeys(args.parents))
    if not parents or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in parents):
        raise ValueError("unsupported parent observer identifier")
    allowed = read_training_ids(args.train_ids)
    source_rows = read_rows(args.eval_run/"predictions.csv.gz")
    source_rows = [row for row in source_rows if row["candidate"] in parents
                   and row["split"] in args.eval_splits]
    source_events = json.loads((args.eval_run/"events.json").read_text())
    source_events = [event for event in source_events if event["candidate"] in parents
                     and event["split"] in args.eval_splits]
    evaluation_ids = {str(row["trajectory_id"]) for row in source_rows}
    if not evaluation_ids or evaluation_ids & allowed:
        raise ValueError("missing evaluation rows or training/evaluation trajectory overlap")
    if set(event["candidate"] for event in source_events) != set(parents):
        raise ValueError("missing evaluation observer event stream")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"models").mkdir()
    started = perf_counter()
    execution = {"schema": "SMALL_CAUSAL_RESIDUAL_EVALUATION/1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
        "python_prefix": sys.prefix, "parent_observers": parents,
        "train_run": str(args.train_run), "eval_run": str(args.eval_run),
        "evaluation_splits": args.eval_splits, "evaluation_ids": sorted(evaluation_ids),
        "training_ids": sorted(allowed), "train_evaluation_disjoint": True,
        "feature_names": list(FEATURE_NAMES), "residual_kinds": args.kinds,
        "gate_policy": "reuse parent cached causal gate; full training gate only",
        "source_hashes": {"training_ids": digest(args.train_ids),
            "evaluation_predictions": digest(args.eval_run/"predictions.csv.gz"),
            "evaluation_events": digest(args.eval_run/"events.json"),
            "residual_head.py": digest(Path(__file__).with_name("residual_head.py")),
            "residual_evaluation.py": digest(Path(__file__))},
        "candidate_settings": {"ridge_alpha": args.ridge_alpha, "hidden": args.hidden,
            "epochs": args.epochs, "seed": args.seed, "mlp_l2": args.mlp_l2,
            "maximum_correction_m": args.maximum_correction_m},
        "historical_exposed_evaluation_not_new_blind_test": True}
    for parent in parents:
        for label, directory in (("train", args.train_run), ("evaluation", args.eval_run)):
            execution["source_hashes"][label+"_cache_"+parent] = digest(directory/("cache_"+parent+".npz"))
    dump(args.out/"execution.json", execution)
    rows = [dict(row, residual_parent=row["candidate"], residual_kind="none",
                 residual_displacement_xyz=[0., 0., 0.]) for row in source_rows]
    events = [dict(event) for event in source_events]
    training_audits, application_audits = [], []
    for parent in parents:
        training = read_cache(args.train_run/("cache_"+parent+".npz"), parent)
        evaluation = read_cache(args.eval_run/("cache_"+parent+".npz"), parent)
        x, y, identities, audit = training_arrays(training, allowed)
        audit["parent"] = parent
        for kind in args.kinds:
            name = parent+"__residual_"+kind
            head = fit_residual_head(kind, x, y, identities, allowed_train_ids=allowed,
                ridge_alpha=args.ridge_alpha, hidden=args.hidden, epochs=args.epochs,
                seed=args.seed, mlp_l2=args.mlp_l2)
            model_path = args.out/"models"/(name+".json")
            head.save(model_path)
            # Verify saved trained normalization and weights, without touching
            # validation/test statistics for model fitting or optimizer control.
            restored = ResidualHead.load(model_path)
            np.testing.assert_array_equal(head.predict(x[:8]), restored.predict(x[:8]))
            new_rows, applied = apply_head_to_rows(restored, evaluation, source_rows, parent, name,
                                                    maximum_correction_m=args.maximum_correction_m)
            rows.extend(new_rows)
            events.extend(dict(event, candidate=name, residual_parent=parent, residual_kind=kind)
                          for event in source_events if event["candidate"] == parent)
            training_audits.append(dict(audit, kind=kind, candidate=name,
                saved_model_sha256=digest(model_path), fitted_metadata=head.metadata,
                normalization_sha256=hashlib.sha256(head.mean.tobytes()+head.scale.tobytes()
                                                   +head.target_scale.tobytes()).hexdigest()))
            application_audits.append(applied)
            print("RESIDUAL_FIT "+json.dumps({"candidate": name, "training_rows": len(x),
                  "training_trajectories": len(set(identities)), "elapsed_s": perf_counter()-started}), flush=True)
    result = summarize(rows, events)
    # The outer-fold/evaluation minimum is descriptive only. Original validation
    # may select a candidate; no historical test value enters this choice.
    validation = [r for r in result["ranking"] if r["split"] == "validation"]
    validation = [r for r in validation if r.get("eligible_for_ranking",
                        r.get("eligible_for_descriptive_ranking", True))]
    result["residual_evaluation"] = {"execution": execution, "training_audits": training_audits,
        "application_audits": application_audits,
        "selected_on_original_validation": min(validation, key=lambda r:
            (r["balanced_future_rmse_m"], r["candidate"]))["candidate"] if validation else None,
        "selection_scope": "original validation only if present; outer fold minima remain descriptive",
        "elapsed_s": perf_counter()-started}
    write_rows(args.out/"predictions.csv.gz", rows)
    dump(args.out/"events.json", events)
    dump(args.out/"training_audit.json", training_audits)
    dump(args.out/"application_audit.json", application_audits)
    dump(args.out/"summary.json", result)
    print("RESIDUAL_COMPLETE "+json.dumps({"out": str(args.out), "rows": len(rows),
          "ranking": result["ranking"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--parents", nargs="+", required=True)
    parser.add_argument("--eval-splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", choices=("ridge", "mlp"), default=["ridge", "mlp"])
    parser.add_argument("--ridge-alpha", type=float, default=.03)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1931)
    parser.add_argument("--mlp-l2", type=float, default=.01)
    parser.add_argument("--maximum-correction-m", type=float, default=.15)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
