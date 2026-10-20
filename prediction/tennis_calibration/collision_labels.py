"""Offline-only post-flight labels for the frozen-prefix collision task."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import platform

import numpy as np

from .collision_mapping import fit_post_labels
from .online_data import load_dataset


def build_labels(calibration_dir, k_drag, out_path, *, b_prior_sigma=.15):
    if platform.system() != "Linux":
        raise RuntimeError("Formal trajectory label fitting must run on Ubuntu")
    records = []
    for index, trace in enumerate(load_dataset(calibration_dir)):
        row = {"trajectory_id": trace["trajectory_id"], "split": trace["split"],
               "groupid": trace["groupid"], "offline_label_only": True}
        try:
            tb = trace["bounce_reference_s"]
            if tb is None:
                raise ValueError("no_first_bounce_reference")
            times, xyz = trace["eval_times"], trace["eval_xyz"]
            scope_end = trace.get("eval_scope_end_s")
            if scope_end is not None:
                select = times <= float(scope_end)
                times, xyz = times[select], xyz[select]
            row.update(fit_post_labels(times, xyz, tb, k_drag,
                                       b_prior_sigma=b_prior_sigma))
        except (ValueError, FloatingPointError) as error:
            row.update(available=False, reason=str(error))
        records.append(row)
        if (index+1) % 10 == 0:
            print(json.dumps({"post_labels_completed": index+1}, ensure_ascii=False), flush=True)
    summary = {"rows": len(records), "available": sum(r["available"] for r in records),
               "velocity_quality": sum(r["available"] and not r["quality_reasons"] for r in records),
               "b_label_usable": sum(r["available"] and r["b_label_usable"] for r in records),
               "failure_reasons": dict(Counter(r["reason"] for r in records if not r["available"])),
               "b_prior_sigma_per_s": b_prior_sigma,
               "k_drag_per_m": float(k_drag),
               "data_role": "Offline output labels only; reference time, cleaned post data and fitted states never enter pre-only prediction features",
               "true_spin_measured": False}
    report = {"summary": summary, "records": records}
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k-drag", type=float, required=True)
    parser.add_argument("--b-prior-sigma", type=float, default=.15)
    args = parser.parse_args()
    build_labels(args.calibration_dir, args.k_drag, args.out,
                 b_prior_sigma=args.b_prior_sigma)


if __name__ == "__main__":
    main()
