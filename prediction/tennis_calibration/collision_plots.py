"""Readable static plots for the frozen pre-contact collision task on Ubuntu."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .collision_metrics import aggregate_collision_metrics


def _number(value):
    return np.nan if value is None else float(value)


def _display(name):
    labels = {"fixed_impact": "Fixed impact map", "constant_impact": "Constant impact",
        "baseline": "Physical baseline", "prefix_baseline": "Prefix physics",
        "ekf_baseline": "Pre-contact EKF", "repaired6": "Repaired EKF6",
        "magnus_effective_prior_s1_q2": "Identified Magnus EKF",
        "frozen_selected_EKF_preonly": "Previous selected EKF (frozen)",
        "ridge": "Ridge residual", "gp": "GP residual", "mlp": "MLP residual"}
    return labels.get(name, str(name).replace("_", " "))


def _selected(summary):
    value = (summary or {}).get("selected_model")
    if isinstance(value, dict):
        return value.get("name", value.get("selected_name", value.get("model")))
    return value


def _save(fig, directory, name):
    path = directory/name
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _lookup(cells):
    return {(row["split"], row["model"], row["gap_s"], row.get("horizon_s")): row for row in cells}


def _macro(cell, metric="position_error"):
    if metric == "position_error":
        return _number(cell.get("trajectory_macro", {}).get("mean_per_trajectory_rmse_m"))
    return _number(cell.get(f"{metric}_trajectory_macro", {}).get("mean_per_trajectory_rmse_m_s"))


def make_collision_plots(out, position_rows, event_rows, summary=None):
    """Render only supplied frozen-prefix evaluations; never select a model."""
    position_rows, event_rows = list(position_rows), list(event_rows)
    output = Path(out)/"plots"
    output.mkdir(parents=True, exist_ok=True)
    metrics = aggregate_collision_metrics(position_rows, event_rows)
    marginal = _lookup(metrics["position"]["cells"])
    paired = _lookup(metrics["comparison"]["common_success_position"]["cells"])
    events = _lookup(metrics["event"]["cells"])
    models = sorted({row["model"] for row in position_rows})
    gaps = sorted({float(row.get("gap_s", row.get("planned_gap_s"))) for row in position_rows})
    horizons = sorted({float(row["horizon_s"]) for row in position_rows})
    splits = [split for split in ("validation", "test") if any(row["split"] == split for row in position_rows)]
    colors = {model: plt.get_cmap("tab10")(index % 10) for index, model in enumerate(models)}
    selected = _selected(summary)
    paths = []
    if not gaps or not horizons or not splits:
        return paths
    common_note = "Frozen pre-contact input; shared successful targets; exposed holdout = regression"
    for split in splits:
        ncols, nrows = min(3, len(gaps)), int(np.ceil(len(gaps)/min(3, len(gaps))))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.6*nrows), squeeze=False, constrained_layout=True)
        for ax, gap in zip(axes.ravel(), gaps):
            for model in models:
                values = [_macro(paired.get((split, model, gap, horizon), {}))*100 for horizon in horizons]
                ax.plot(np.asarray(horizons)*1000, values, "o-", color=colors[model], label=_display(model), linewidth=1.5)
            ax.set(xlabel="After reference contact (ms)", ylabel="Trajectory macro 3D RMSE (cm)",
                   title=f"Missing pre-contact tail: {gap*1000:g} ms")
            ax.set_xscale("log")
            ax.set_xticks(np.asarray(horizons)*1000, [f"{h*1000:g}" for h in horizons])
            ax.grid(alpha=.22)
            ax.legend(fontsize=7)
        for ax in axes.ravel()[len(gaps):]:
            ax.set_visible(False)
        fig.suptitle(f"{split.capitalize()}: {common_note}", fontsize=11)
        paths.append(_save(fig, output, f"collision_horizon_gap_{split}.png"))

    if selected in models:
        matrices = [np.array([[_macro(marginal.get((split, selected, gap, h), {}))*100 for h in horizons]
                     for gap in gaps]) for split in splits]
        finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
        high = float(np.max(finite)) if len(finite) else 1.
        fig, axes = plt.subplots(1, len(splits), figsize=(7*len(splits), 4.6), squeeze=False, constrained_layout=True)
        for ax, split, matrix in zip(axes.ravel(), splits, matrices):
            image = ax.imshow(np.ma.masked_invalid(matrix), origin="upper", aspect="auto", vmin=0, vmax=high, cmap="viridis")
            ax.set_xticks(np.arange(len(horizons)), [f"{h*1000:g}" for h in horizons])
            ax.set_yticks(np.arange(len(gaps)), [f"{g*1000:g}" for g in gaps])
            ax.set(xlabel="After reference contact (ms)", ylabel="Planned missing tail (ms)", title=split.capitalize())
            for i in range(len(gaps)):
                for j in range(len(horizons)):
                    text = "NA" if not np.isfinite(matrix[i, j]) else f"{matrix[i,j]:.1f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=9,
                            color="white" if not np.isfinite(matrix[i,j]) or matrix[i,j]<.5*high else "black")
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Marginal trajectory macro 3D RMSE (cm)")
        fig.suptitle(f"Frozen selected model: {_display(selected)}; see coverage for failures; reused holdout")
        paths.append(_save(fig, output, "collision_selected_gap_horizon_heatmap.png"))

    longest = horizons[-1]
    fig, axes = plt.subplots(len(splits), 2, figsize=(13, 4.2*len(splits)), squeeze=False, constrained_layout=True)
    for i, split in enumerate(splits):
        for model in models:
            coverage, tail = [], []
            for gap in gaps:
                cell = marginal.get((split, model, gap, longest), {})
                coverage.append(_number(cell.get("prediction_coverage_given_label"))*100)
                tail.append(_number(paired.get((split, model, gap, longest), {}).get("position_error", {}).get("p95_m"))*100)
            axes[i, 0].plot(np.asarray(gaps)*1000, coverage, "o-", color=colors[model], label=_display(model))
            axes[i, 1].plot(np.asarray(gaps)*1000, tail, "o-", color=colors[model], label=_display(model))
        label_coverage = [_number(marginal.get((split, models[0], gap, longest), {}).get("label_coverage"))*100 for gap in gaps]
        axes[i, 0].plot(np.asarray(gaps)*1000, label_coverage, "k--", label="Target label coverage")
        axes[i, 0].set(ylabel="Coverage (%)", ylim=(-2, 102), title=f"{split.capitalize()}: {longest*1000:g} ms")
        axes[i, 1].set(ylabel="Common-success 3D P95 error (cm)", title=f"{split.capitalize()}: error tail")
        for ax in axes[i]:
            ax.set_xlabel("Planned missing pre-contact tail (ms)")
            ax.grid(alpha=.22)
            ax.legend(fontsize=7)
    fig.suptitle("Frozen input; failures remain in coverage; gap variants are correlated; exposed holdout regression")
    paths.append(_save(fig, output, "collision_coverage_and_error_tail.png"))

    if event_rows:
        fig, axes = plt.subplots(1, len(splits), figsize=(7*len(splits), 4.5), squeeze=False, constrained_layout=True)
        for ax, split in zip(axes.ravel(), splits):
            for model in models:
                values = [_macro(events.get((split, model, gap, None), {}), "v_plus_error") for gap in gaps]
                ax.plot(np.asarray(gaps)*1000, values, "o-", color=colors[model], label=_display(model))
            ax.set(xlabel="Planned missing pre-contact tail (ms)", ylabel="Trajectory macro outgoing velocity RMSE (m/s)",
                   title=split.capitalize())
            ax.legend(fontsize=7)
            ax.grid(alpha=.22)
        fig.suptitle("Outgoing speed vector errors; marginal successful events; each trajectory is one collision; reused holdout")
        paths.append(_save(fig, output, "collision_outgoing_velocity_gap.png"))
    return paths
