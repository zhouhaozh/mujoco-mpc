"""Audit and segment measured tennis trajectories without changing the archive.

Event detection is provisional: z is assumed vertical, but no ground level or
ball radius is imposed. Discovery smoothing is never exported as observations.
Outlier fits are split at each discovered bounce so a real velocity jump remains.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def _mad(x: np.ndarray, floor: float = 0.0) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not len(x):
        return floor
    return max(floor, 1.4826 * float(np.median(np.abs(x - np.median(x)))))


def robust_poly(t: np.ndarray, y: np.ndarray, at: float, degree: int = 2):
    """Huber IRLS on real timestamps; ascending coefficients centered at at."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    degree = min(degree, len(t) - 1)
    if degree < 0:
        return None
    scale = max(float(np.ptp(t)), 1e-5)
    u = (t - at) / scale
    a = np.vander(u, degree + 1, increasing=True)
    w = np.ones(len(t))
    coef = np.zeros(degree + 1)
    for _ in range(7):
        coef = np.linalg.lstsq(a * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
        residual = y - a @ coef
        sigma = _mad(residual, 1e-6)
        w = np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual), 1e-12))
    coef /= scale ** np.arange(degree + 1)
    return coef, _mad(residual, 1e-6)


def _fit_window(t, y, center, lo, hi):
    m = (t >= center + lo) & (t <= center + hi)
    if np.count_nonzero(m) < 8 or np.ptp(t[m]) < .025:
        return None
    return robust_poly(t[m], y[m], center)


def find_bounces(t: np.ndarray, z: np.ndarray) -> list[dict]:
    """Find sustained descending-to-ascending transitions, not raw minima."""
    if len(t) < 30 or np.ptp(t) < .18:
        return []
    # 50 ms robust discovery fits prevent a single low measurement becoming an
    # event. These crossing-window estimates only discover events, never clean
    # the observations or estimate the exported incident/outgoing velocities.
    discovery = np.empty(len(t))
    for i, ti in enumerate(t):
        m = np.abs(t - ti) <= .025
        fit = robust_poly(t[m], z[m], ti)
        discovery[i] = fit[0][0] if fit is not None else z[i]
    raw = []
    for i in range(1, len(t) - 1):
        if not (discovery[i] <= discovery[i-1] and discovery[i] < discovery[i+1]):
            continue
        ti = float(t[i])
        pre = _fit_window(t, z, ti, -.13, -.02)
        post = _fit_window(t, z, ti, .02, .13)
        if pre is None or post is None:
            continue
        left, ls = pre
        right, rs = post
        if len(left) < 2 or len(right) < 2:
            continue
        if left[1] > -.4 or right[1] < .4:
            continue
        delta = right - left
        roots = np.polynomial.polynomial.polyroots(delta)
        roots = [float(r.real) for r in roots if abs(r.imag) < 1e-7 and abs(r.real) < .065]
        if not roots:
            continue
        offset = min(roots, key=abs)
        event_t = ti + offset
        vz_pre = float(np.polynomial.polynomial.polyval(offset, np.arange(1, len(left)) * left[1:]))
        vz_post = float(np.polynomial.polynomial.polyval(offset, np.arange(1, len(right)) * right[1:]))
        if vz_pre >= -.4 or vz_post <= .4:
            continue
        jump = vz_post - vz_pre
        dt = float(np.median(np.diff(t)))
        interval_half = min(.05, max(1.5 * dt, 1.96 * np.hypot(ls, rs) / jump))
        noise = float(np.hypot(ls, rs))
        confidence = float(np.clip(1 - noise / max(.10, .08 * jump) - abs(offset) / .20, .0, .99))
        h = float(np.polynomial.polynomial.polyval(offset, left))
        raw.append({"time_s": event_t, "interval_start_s": event_t - interval_half,
                    "interval_end_s": event_t + interval_half, "confidence": confidence,
                    "contact_z_m": h, "vz_pre_local_mps": vz_pre,
                    "vz_post_local_mps": vz_post, "local_noise_m": noise})
    # Nearby estimates refer to the same event. Preserve the best independent
    # left/right fit, then retain chronological ordering for first bounce.
    merged: list[dict] = []
    for event in sorted(raw, key=lambda x: x["time_s"]):
        if merged and event["time_s"] - merged[-1]["time_s"] < .12:
            if event["confidence"] > merged[-1]["confidence"]:
                merged[-1] = event
        else:
            merged.append(event)
    return merged


def clean_rows(rows: np.ndarray) -> tuple[list[dict], dict]:
    """Pure trajectory cleaning, SI input; all source rows stay in the output."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError("rows must have shape N x 4: time and three positions")
    n = len(rows)
    reasons = [[] for _ in range(n)]
    keep = np.all(np.isfinite(rows), axis=1)
    for i in np.flatnonzero(~keep):
        reasons[i].append("nonfinite_or_malformed")
    previous = -np.inf
    for i in np.flatnonzero(keep):
        if rows[i, 0] <= previous:
            keep[i] = False
            reasons[i].append("nonmonotonic_or_duplicate_time")
        else:
            previous = rows[i, 0]
    good_idx = np.flatnonzero(keep)
    if len(good_idx) < 8:
        out = [{"source_line": i+1, "t_s": rows[i, 0], "x_m": rows[i, 1],
                "y_m": rows[i, 2], "z_m": rows[i, 3], "keep": bool(keep[i]),
                "reason": ";".join(reasons[i]), "segment": "unresolved"} for i in range(n)]
        return out, {"status": "incomplete", "incomplete": True, "must_review": True,
                     "confidence": 0., "n_raw": n, "n_keep": int(keep.sum()), "n_outliers": 0,
                     "first_bounce_t_s": None, "events": []}
    t = rows[good_idx, 0]
    p = rows[good_idx, 1:]
    dt_med = float(np.median(np.diff(t)))
    gap_limit = max(.012, 4.0 * dt_med)
    for j in np.flatnonzero(np.diff(t) > gap_limit):
        reasons[good_idx[j]].append("gap_after")
        reasons[good_idx[j+1]].append("gap_before")
    for j in np.flatnonzero(np.all(np.diff(p, axis=0) == 0, axis=1)):
        reasons[good_idx[j+1]].append("duplicate_position")
    # A very short boundary interval and sub-0.1 mm coordinate precision are
    # flags, not proof of interpolation. Exclude micro-dt rows from estimation.
    for j in (0, len(t)-2):
        if t[j+1] - t[j] < .35 * dt_med:
            k = good_idx[0] if j == 0 else good_idx[-1]
            keep[k] = False
            reasons[k].append("boundary_micro_dt")
    for k in (good_idx[0], good_idx[-1]):
        if np.any(np.abs(rows[k, 1:] / .0001 - np.round(rows[k, 1:] / .0001)) > .001):
            reasons[k].append("boundary_subgrid_precision_suspected_interpolation")
    event_idx = np.flatnonzero(keep)
    events = find_bounces(rows[event_idx, 0], rows[event_idx, 3])
    credible = [e for e in events if e["confidence"] >= .55]
    first = credible[0] if credible else (events[0] if events else None)
    # Separate every discovered event to avoid fitting across a real bounce.
    cuts = [e["time_s"] for e in events]
    seg_ids = np.searchsorted(cuts, rows[:, 0], side="right")
    flagged = []
    for seg in np.unique(seg_ids[keep]):
        inds = np.flatnonzero(keep & (seg_ids == seg))
        if len(inds) < 10:
            continue
        tt, pp = rows[inds, 0], rows[inds, 1:]
        residual = np.zeros_like(pp)
        for j, ti in enumerate(tt):
            m = (np.abs(tt - ti) <= .035)
            m[j] = False  # Leave-one-out avoids one point masking itself.
            if m.sum() < 6 or np.ptp(tt[m]) < .012:
                residual[j] = np.nan
                continue
            for axis in range(3):
                fit = robust_poly(tt[m], pp[m, axis], ti)
                residual[j, axis] = pp[j, axis] - fit[0][0]
        scale = np.array([_mad(residual[:, axis], .003) for axis in range(3)])
        scores = np.sqrt(np.nansum((residual / scale) ** 2, axis=1))
        distances = np.linalg.norm(np.nan_to_num(residual), axis=1)
        bad = (scores > 7.0) & (distances > .03)
        for k in inds[bad]:
            flagged.append(k)
    for k in flagged:
        keep[k] = False
        reasons[k].append("position_outlier_local_robust")
    # Refinement uses only retained observations; no smoothed values are used.
    refine_idx = np.flatnonzero(keep)
    refined = find_bounces(rows[refine_idx, 0], rows[refine_idx, 3])
    if first is not None:
        near = [e for e in refined if abs(e["time_s"] - first["time_s"]) < .1]
        if near:
            first = max(near, key=lambda x: x["confidence"])
    first_t = first["time_s"] if first else None
    segments = []
    for ti in rows[:, 0]:
        if first is None or not np.isfinite(ti):
            segments.append("unresolved")
        elif ti < first["interval_start_s"] - .012:
            segments.append("pre")
        elif ti > first["interval_end_s"] + .012:
            segments.append("post")
        else:
            segments.append("collision_excluded")
    out = [{"source_line": i+1, "t_s": rows[i, 0], "x_m": rows[i, 1],
            "y_m": rows[i, 2], "z_m": rows[i, 3], "keep": bool(keep[i]),
            "reason": ";".join(reasons[i]), "segment": segments[i]} for i in range(n)]
    retained_t = rows[keep, 0]
    next_events = [e for e in refined if first_t is not None and e["time_s"] > first_t + .12]
    post_end = (next_events[0]["interval_start_s"] - .012 if next_events else
                float(retained_t[-1]) if len(retained_t) else None)
    meta = {"n_raw": n, "n_keep": int(keep.sum()), "n_outliers": len(flagged),
            "dt_median_s": dt_med, "max_gap_s": float(np.max(np.diff(t))),
            "gap_count": int(np.sum(np.diff(t) > gap_limit)), "events": refined,
            "incomplete": first is None, "status": "candidate" if first else "incomplete",
            "confidence": first["confidence"] if first else 0., "must_review": True,
            "first_bounce_t_s": first_t,
            "pre_start_t_s": float(retained_t[0]) if len(retained_t) else None,
            "post_end_t_s": post_end}
    if first:
        meta.update({"bounce_interval_start_s": first["interval_start_s"],
                     "bounce_interval_end_s": first["interval_end_s"],
                     "contact_z_m": first["contact_z_m"],
                     "pre_end_t_s": first["interval_start_s"] - .012,
                     "post_start_t_s": first["interval_end_s"] + .012,
                     "pre_coverage_s": first_t - retained_t[0] if len(retained_t) else 0.,
                     "post_coverage_s": post_end - first_t if post_end is not None else 0.})
    return out, meta


def infer_sessions(names: list[str]) -> dict[str, str]:
    """Filename/GID-derived grouping; intentionally marked inferred, not truth."""
    parsed = []
    for name in names:
        m = re.search(r"trajectory_(\d+)_gid(\d+)_(\d+)", name)
        parsed.append((int(m[2]) if m else -1, int(m[1]) if m else -1, name))
    previous_gid = previous_number = None
    session = 0
    out = {}
    for gid, number, name in sorted(parsed):
        if previous_gid is None or gid < 0 or gid - previous_gid > 50000 or number <= previous_number:
            session += 1
        out[name] = f"inferred_session_{session:02d}"
        previous_gid, previous_number = gid, number
    return out


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_archive(zip_path, out_dir, position_scale=.001, seed=20261008) -> dict:
    """Read numeric members without extraction; output reproducible audit files."""
    zip_path, out_dir = Path(zip_path), Path(out_dir)
    if position_scale <= 0 or not np.isfinite(position_scale):
        raise ValueError("position_scale must be positive and finite")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "cleaned").mkdir()
    annotations, member_hashes, totals = [], {}, {}
    with ZipFile(zip_path) as archive:
        members = sorted((i for i in archive.infolist() if not i.is_dir() and
                         not i.filename.startswith("__MACOSX/") and i.filename.lower().endswith((".txt", ".csv"))),
                         key=lambda i: i.filename)
        sessions = infer_sessions([m.filename for m in members])
        if not members:
            raise ValueError("archive has no .txt or .csv numeric trajectory members")
        seen = set()
        for member in members:
            if member.filename in seen:
                raise ValueError(f"duplicate zip member name: {member.filename}")
            seen.add(member.filename)
            if member.file_size > 20_000_000:
                raise ValueError(f"trajectory member exceeds 20 MB limit: {member.filename}")
            data = archive.read(member)
            member_hashes[member.filename] = hashlib.sha256(data).hexdigest()
            text = data.decode("utf-8-sig")
            rows, originals = [], []
            for line in text.splitlines():
                originals.append(line)
                tokens = line.replace(",", " ").split()
                try:
                    row = [float(x) for x in tokens] if len(tokens) == 4 else [np.nan] * 4
                except ValueError:
                    row = [np.nan] * 4
                rows.append(row)
            array = np.asarray(rows, dtype=float).reshape((-1, 4))
            array[:, 1:] *= position_scale
            cleaned, meta = clean_rows(array)
            for row, raw in zip(cleaned, originals):
                row["raw_line"] = raw
                for reason in row["reason"].split(";"):
                    if reason:
                        totals[reason] = totals.get(reason, 0) + 1
            basename = Path(member.filename).name
            key = hashlib.sha256(member.filename.encode()).hexdigest()[:12]
            relative = f"cleaned/{basename}.{key}.csv"
            _write_csv(out_dir / relative, cleaned,
                       ["source_line", "t_s", "x_m", "y_m", "z_m", "keep", "reason", "segment", "raw_line"])
            meta.update({"trajectory_id": member.filename, "cleaned_file": relative,
                         "session_id": sessions[member.filename], "session_inferred": True,
                         "raw_sha256": member_hashes[member.filename]})
            annotations.append(meta)
    fields = ["trajectory_id", "cleaned_file", "session_id", "session_inferred", "raw_sha256",
              "n_raw", "n_keep", "n_outliers", "dt_median_s", "max_gap_s", "gap_count",
              "first_bounce_t_s", "bounce_interval_start_s", "bounce_interval_end_s", "contact_z_m",
              "pre_start_t_s", "pre_end_t_s", "post_start_t_s", "post_end_t_s",
              "pre_coverage_s", "post_coverage_s", "confidence", "must_review", "status", "incomplete"]
    _write_csv(out_dir / "annotations.csv", annotations, fields)
    (out_dir / "annotations.json").write_text(json.dumps(annotations, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"archive_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
              "source_archive": str(zip_path.resolve()), "position_scale": position_scale,
              "unit_assumption": "raw t in seconds; raw xyz multiplied by position_scale; z vertical",
              "ground_assumption": "unknown; contact_z is trajectory intersection, not ground height",
              "seed": seed, "n_trajectories": len(annotations), "n_raw": sum(a["n_raw"] for a in annotations),
              "n_keep": sum(a["n_keep"] for a in annotations),
              "n_bounce_candidates": sum(a["first_bounce_t_s"] is not None for a in annotations),
              "n_incomplete": sum(a["incomplete"] for a in annotations),
              "n_inferred_sessions": len(set(sessions.values())), "reason_counts": totals,
              "rules": {"boundary_micro_dt_fraction": .35, "gap_limit": "max(12ms,4*median_dt)",
                        "outlier": "segment-split leave-one-out 70ms robust quadratic; norm>7 sigma and>30mm",
                        "event": "sustained vz_pre<-0.4 and vz_post>0.4; independent intersection fits",
                        "event_interval": "heuristic local-residual/slope-jump uncertainty, not a calibrated statistical confidence interval",
                        "contact_exclusion_padding_s": .012, "must_review": "all provisional automatic labels",
                        "session_grouping": "inferred GID gap>50000 or trajectory number reset"},
              "member_sha256": member_hashes}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", "--zip", dest="archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--position-scale", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    print(json.dumps(clean_archive(args.archive, args.output_dir, args.position_scale, args.seed), indent=2))


if __name__ == "__main__":
    main()
