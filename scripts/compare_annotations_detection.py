#!/usr/bin/env python3
"""Create an interactive annotation-versus-detection comparison report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LABEL_COLORS = {
    "Normal": "rgba(34,197,94,0.12)",
    "Anomalous": "rgba(239,68,68,0.16)",
    "Verify": "rgba(245,158,11,0.18)",
    "Unlabeled": "rgba(148,163,184,0.14)",
}


def boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def local_image_uri(value) -> str:
    """Convert an annotation image path to a browser-readable local URI."""
    if value is None or pd.isna(value) or not str(value).strip():
        return ""
    try:
        return Path(str(value)).expanduser().resolve().as_uri()
    except ValueError:
        return ""


def load_and_align(annotation_csv: Path, scores_csv: Path) -> pd.DataFrame:
    annotations = pd.read_csv(annotation_csv, low_memory=False)
    scores = pd.read_csv(scores_csv, low_memory=False)
    required_annotations = {"safety_area", "frame_id", "label"}
    required_scores = {
        "safety_area", "anomaly_score", "threshold", "normalized_score",
        "is_anomalous",
    }
    if missing := required_annotations.difference(annotations.columns):
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
    if missing := required_scores.difference(scores.columns):
        raise ValueError(f"Score CSV is missing columns: {sorted(missing)}")

    annotations["alignment_index"] = annotations.groupby("safety_area").cumcount()
    scores["alignment_index"] = scores.groupby("safety_area").cumcount()
    counts_a = annotations.groupby("safety_area").size().to_dict()
    counts_s = scores.groupby("safety_area").size().to_dict()
    if counts_a != counts_s:
        raise ValueError(
            "Cannot safely align by per-area order; row counts differ: "
            f"annotations={counts_a}, scores={counts_s}"
        )
    merged = annotations.merge(
        scores, on=["safety_area", "alignment_index"], how="inner",
        validate="one_to_one", suffixes=("_annotation", "_detection"),
    )
    merged["label"] = merged["label"].fillna("Unlabeled").str.strip().str.title()
    merged["detected_anomalous"] = boolean_series(merged["is_anomalous"])
    for column in ("processed_image_path", "raw_image_path"):
        merged[f"{column}_uri"] = (
            merged[column].map(local_image_uri) if column in merged else ""
        )
    return merged.sort_values(["safety_area", "frame_id"]).reset_index(drop=True)


def binary_metrics(group: pd.DataFrame) -> dict:
    evaluated = group[group["label"].isin(("Normal", "Anomalous"))].copy()
    truth = evaluated["label"].eq("Anomalous")
    prediction = evaluated["detected_anomalous"]
    tp = int((truth & prediction).sum())
    tn = int((~truth & ~prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    has_annotated_anomalies = bool(truth.any())

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    ranking = ranking_curve_data(group)
    return {
        "total_frames": (
            int(group["frame_id"].nunique())
            if "frame_id" in group.columns else int(len(group))
        ),
        "total_area_rows": int(len(group)),
        "total_normal": int((group["label"] == "Normal").sum()),
        "total_anomalous": int((group["label"] == "Anomalous").sum()),
        "total_verify": int((group["label"] == "Verify").sum()),
        "total_unlabeled": int((group["label"] == "Unlabeled").sum()),
        "evaluated": len(evaluated), "excluded_verify": int((group["label"] == "Verify").sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": divide(tp, tp + fp), "recall": recall,
        "specificity": specificity,
        "f1": divide(2 * tp, 2 * tp + fp + fn) if has_annotated_anomalies else None,
        "accuracy": divide(tp + tn, len(evaluated)),
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None else None
        ),
        "false_positive_rate": divide(fp, fp + tn),
        "false_negative_rate": divide(fn, fn + tp),
        "auroc": ranking["auroc"],
        "auprc": ranking["auprc"],
        "aupro": None,
        "aupro_status": (
            "unavailable: requires pixel-level ground-truth masks and anomaly maps"
        ),
    }


def ranking_curve_data(group: pd.DataFrame) -> dict:
    """Compute frame-level ROC and precision-recall curves without sklearn."""
    empty = {
        "fpr": np.array([]), "tpr": np.array([]),
        "recall": np.array([]), "precision": np.array([]),
        "auroc": None, "auprc": None,
    }
    if "normalized_score" not in group.columns:
        return empty
    evaluated = group[group["label"].isin(("Normal", "Anomalous"))]
    truth = evaluated["label"].eq("Anomalous").to_numpy(dtype=np.int8)
    scores = evaluated["normalized_score"].to_numpy(dtype=float)
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    if not positives or not negatives or not len(scores):
        return empty
    order = np.argsort(-scores, kind="mergesort")
    truth = truth[order]
    scores = scores[order]
    cumulative_tp = np.cumsum(truth)
    cumulative_fp = np.cumsum(1 - truth)
    distinct_ends = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]
    tp = cumulative_tp[distinct_ends].astype(float)
    fp = cumulative_fp[distinct_ends].astype(float)
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    recall = np.r_[0.0, tp / positives]
    precision = np.r_[1.0, tp / np.maximum(tp + fp, 1.0)]
    return {
        "fpr": fpr, "tpr": tpr,
        "recall": recall, "precision": precision,
        "auroc": float(np.trapz(tpr, fpr)),
        "auprc": float(np.sum(np.diff(recall) * precision[1:])),
    }


def label_runs(group: pd.DataFrame):
    rows = group.sort_values("frame_id")
    start = previous = int(rows.iloc[0]["frame_id"])
    label = rows.iloc[0]["label"]
    for _, row in rows.iloc[1:].iterrows():
        frame = int(row["frame_id"])
        if row["label"] != label or frame != previous + 1:
            yield start, previous, label
            start, label = frame, row["label"]
        previous = frame
    yield start, previous, label


def scenario_runs(group: pd.DataFrame):
    """Yield contiguous global-frame ranges for source scenarios."""
    rows = group.drop_duplicates("frame_id").sort_values("frame_id")
    scenario_column = (
        "source_scenario_id" if "source_scenario_id" in rows.columns
        else "scenario_id" if "scenario_id" in rows.columns else None
    )
    if not scenario_column or rows.empty:
        return
    start = previous = int(rows.iloc[0]["frame_id"])
    scenario = str(rows.iloc[0][scenario_column])
    description = str(rows.iloc[0].get("scenario_description", "") or "")
    for _, row in rows.iloc[1:].iterrows():
        frame = int(row["frame_id"])
        current = str(row[scenario_column])
        if current != scenario or frame != previous + 1:
            yield start, previous, scenario, description
            start = frame
            scenario = current
            description = str(row.get("scenario_description", "") or "")
        previous = frame
    yield start, previous, scenario, description


def plot_custom_data(group: pd.DataFrame) -> np.ndarray:
    """Build hover/preview data, including unified-video source provenance."""
    size = len(group)
    scenario = (
        group["source_scenario_id"] if "source_scenario_id" in group
        else group["scenario_id"] if "scenario_id" in group
        else pd.Series([""] * size, index=group.index)
    )
    description = group.get(
        "scenario_description", pd.Series([""] * size, index=group.index)
    ).fillna("")
    source_frame = group.get(
        "source_frame_id", group["frame_id"]
    )
    return np.column_stack((
        group["label"], group["anomaly_score"], group["threshold"],
        group.get("filename", pd.Series([""] * size, index=group.index)),
        group["processed_image_path_uri"], group["raw_image_path_uri"],
        group.get("note", pd.Series([""] * size, index=group.index)).fillna(""),
        group["detected_anomalous"], group["score_strategy"],
        scenario.fillna("").astype(str), description.astype(str), source_frame,
    ))


def make_figure(data: pd.DataFrame) -> go.Figure:
    areas = list(dict.fromkeys(data["safety_area"]))
    first_strategy = data["score_strategy"].iloc[0]
    timeline = data[
        (data["safety_area"] == areas[0])
        & (data["score_strategy"] == first_strategy)
    ]
    source_runs = list(scenario_runs(timeline))
    figure = make_subplots(
        rows=len(areas), cols=1, shared_xaxes=True, vertical_spacing=0.045,
        subplot_titles=areas,
    )
    for row_number, area in enumerate(areas, start=1):
        group = data[data["safety_area"] == area].sort_values("frame_id")
        strategy_colors = {"max": "#2563eb", "percentile": "#9333ea"}
        for strategy, strategy_group in group.groupby("score_strategy", sort=False):
            custom = plot_custom_data(strategy_group)
            figure.add_trace(go.Scatter(
                x=strategy_group["frame_id"], y=strategy_group["normalized_score"],
                mode="lines", line={"color": strategy_colors.get(strategy), "width": 1.6},
                name=f"{strategy}: score / threshold", legendgroup=f"score-{strategy}",
                showlegend=row_number == 1, customdata=custom,
                hovertemplate=(
                    "Strategy %{customdata[8]}<br>Scenario %{customdata[9]}<br>"
                    "%{customdata[10]}<br>Global frame %{x}<br>"
                    "Source frame %{customdata[11]}<br>"
                    "Normalized score %{y:.3f}×<br>Annotation %{customdata[0]}<br>"
                    "Raw score %{customdata[1]:.5f}<br>Threshold %{customdata[2]:.5f}"
                    "<br>%{customdata[3]}<br>%{customdata[6]}<extra></extra>"
                ),
            ), row=row_number, col=1)
        axis_name = "y" if row_number == 1 else f"y{row_number}"
        figure.add_shape(
            type="line", x0=float(group.frame_id.min()), x1=float(group.frame_id.max()),
            y0=1, y1=1, line={"color": "#111827", "dash": "dash", "width": 1},
            xref="x" if row_number == 1 else f"x{row_number}", yref=axis_name,
        )
        y_max = max(1.1, float(group["normalized_score"].max()) * 1.08)
        for start, _, _, _ in source_runs[1:]:
            figure.add_shape(
                type="line", x0=start - 0.5, x1=start - 0.5, y0=0, y1=y_max,
                line={"color": "#475569", "dash": "dot", "width": 1.2},
                xref="x" if row_number == 1 else f"x{row_number}", yref=axis_name,
            )
        annotation_group = group.drop_duplicates("frame_id").sort_values("frame_id")
        for start, end, label in label_runs(annotation_group):
            figure.add_shape(
                type="rect", x0=start - 0.5, x1=end + 0.5, y0=0, y1=y_max,
                fillcolor=LABEL_COLORS.get(label, LABEL_COLORS["Unlabeled"]),
                line={"width": 0}, layer="below",
                xref="x" if row_number == 1 else f"x{row_number}", yref=axis_name,
            )
        for strategy, strategy_group in group.groupby("score_strategy", sort=False):
            false_positive = strategy_group[(strategy_group["label"] == "Normal") & strategy_group["detected_anomalous"]]
            false_negative = strategy_group[(strategy_group["label"] == "Anomalous") & ~strategy_group["detected_anomalous"]]
            for subset, name, color, symbol in (
                (false_positive, "FP", "#2563eb", "x"),
                (false_negative, "FN", "#dc2626", "circle-open"),
            ):
                marker_custom = plot_custom_data(subset)
                figure.add_trace(go.Scatter(
                    x=subset["frame_id"], y=subset["normalized_score"], mode="markers",
                    marker={"color": color, "symbol": symbol, "size": 7},
                    name=f"{strategy} {name}", legendgroup=f"{strategy}-{name}",
                    showlegend=row_number == 1, customdata=marker_custom,
                    hovertemplate=(
                        f"{strategy} {name}<br>Scenario %{{customdata[9]}}<br>"
                        "%{customdata[10]}<br>Global frame %{x}<br>"
                        "Source frame %{customdata[11]}<br>"
                        "Score %{y:.3f}×<extra></extra>"
                    ),
                ), row=row_number, col=1)
        figure.update_yaxes(title_text="score / threshold", range=[0, y_max], row=row_number, col=1)
    if source_runs:
        figure.update_xaxes(
            title_text="Source scenario along unified global-frame timeline",
            tickmode="array",
            tickvals=[(start + end) / 2 for start, end, _, _ in source_runs],
            ticktext=[scenario for _, _, scenario, _ in source_runs],
            tickangle=-45,
            row=len(areas), col=1,
        )
    else:
        figure.update_xaxes(title_text="Frame ID", row=len(areas), col=1)
    figure.update_layout(
        height=max(850, 290 * len(areas)), template="plotly_white", hovermode="x unified",
        autosize=True,
        legend={"orientation": "h", "y": 1.03},
        margin={"l": 58, "r": 18, "t": 100, "b": 95},
        title="Annotation versus model detection",
    )
    return figure


def format_metric(value):
    return "—" if value is None else f"{value:.3f}"


def load_threshold_metadata(data: pd.DataFrame, threshold_dir: Path) -> list[dict]:
    """Load the calibration JSON corresponding to every plotted strategy/area."""
    records = []
    for (strategy, area), group in data.groupby(
        ["score_strategy", "safety_area"], sort=False
    ):
        first = group.iloc[0]
        csv_metadata = {}
        for key in (
            "score_func", "reconstruction_mode", "offset", "sigma", "quantile",
            "threshold_strategy",
        ):
            if key in group.columns and pd.notna(first.get(key)):
                csv_metadata[key] = first[key]
        area_dir = threshold_dir / area
        candidates = []
        if "threshold_file" in group.columns and pd.notna(first.get("threshold_file")):
            threshold_file = Path(str(first["threshold_file"]))
            candidates.append(
                threshold_file if threshold_file.is_absolute()
                else area_dir / threshold_file.name
            )
        if all(key in csv_metadata for key in ("offset", "sigma", "quantile")):
            candidates.append(area_dir / (
                f"threshold_{area}_{strategy}_off{csv_metadata['offset']}"
                f"_sig{csv_metadata['sigma']}_q{csv_metadata['quantile']}.json"
            ))
            # Calibration encodes strategy parameters in some filenames, e.g.
            # percentile99.0, while the JSON/score CSV records "percentile".
            candidates.extend(sorted(area_dir.glob(
                f"threshold_{area}_{strategy}*_off{csv_metadata['offset']}"
                f"_sig{csv_metadata['sigma']}_q{csv_metadata['quantile']}.json"
            )))
        candidates.extend((
            area_dir / f"threshold_{area}_{strategy}.json",
            area_dir / f"threshold_{area}.json",
        ))
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        record = {
            "strategy": strategy, "area": area, "path": path,
            "available": path.is_file(),
            "csv_threshold": float(group["threshold"].iloc[0]),
        }
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            record.update(payload)
            record["threshold_matches_csv"] = bool(np.isclose(
                float(payload["threshold"]), record["csv_threshold"]
            ))
        # The score CSV records what inference actually used and is authoritative.
        record.update(csv_metadata)
        records.append(record)
    return records


def confusion_matrix_card(title: str, values: dict) -> str:
    """Render an actual-by-predicted 2x2 confusion matrix."""
    return f"""<article class="cm-card"><h3>{html.escape(title)}</h3>
<table class="cm"><thead><tr><th>Actual \\ Predicted</th><th>Normal</th><th>Anomalous</th></tr></thead>
<tbody><tr><th>Normal</th><td class="tn"><b>{values['tn']}</b><small>TN</small></td><td class="fp"><b>{values['fp']}</b><small>FP</small></td></tr>
<tr><th>Anomalous</th><td class="fn"><b>{values['fn']}</b><small>FN</small></td><td class="tp"><b>{values['tp']}</b><small>TP</small></td></tr></tbody></table>
<p>{values['evaluated']} evaluated · {values['excluded_verify']} Verify excluded</p></article>"""


def load_model_metadata(threshold_dir: Path, area: str) -> dict:
    """Load the compact training provenance saved beside model checkpoints."""
    models_dir = threshold_dir.parent / "train" / "models"
    candidates = sorted(models_dir.glob(f"model_{area}_*_config.json"))
    if not candidates:
        return {"available": False, "path": str(models_dir)}
    path = candidates[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "path": str(path.resolve()),
        "model_type": payload.get("model_type"),
        "suffix": payload.get("suffix"),
        "saved_at": payload.get("saved_at"),
        "epochs_trained": payload.get("epochs_trained", payload.get("epoch")),
        "dataset": payload.get("dataset", {}),
        "training": payload.get("training", {}),
        "model_parameters": payload.get("params", {}),
        "augmentation": payload.get("augmentation", {}),
        "notes": payload.get("notes"),
    }


def json_value(value):
    """Convert NumPy/Pandas/Path values to strict JSON-compatible values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def default_evaluation_path(score_csv: Path, threshold_dir: Path) -> Path:
    stem = re.sub(r"^(?:video|rosbag|frames|cropped)_", "", score_csv.stem)
    return threshold_dir.parent / "evaluation" / f"evaluation_{stem}.json"


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")


def plot_confusion_matrix(path: Path, title: str, metrics: dict) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]],
                       [metrics["fn"], metrics["tp"]]], dtype=int)
    figure, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            color = "white" if value > matrix.max() * 0.55 else "#111827"
            code = (("TN", "FP"), ("FN", "TP"))[row][column]
            ax.text(column, row, f"{value:,}\n{code}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=color)
    ax.set_xticks((0, 1), ("Predicted Normal", "Predicted Anomalous"))
    ax.set_yticks((0, 1), ("Actual Normal", "Actual Anomalous"))
    ax.set_title(title)
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_plots(data: pd.DataFrame, output_dir: Path) -> dict:
    """Save per-area and cumulative diagnostic plots for each strategy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    metric_names = ("precision", "recall", "f1", "accuracy", "balanced_accuracy")
    for strategy, strategy_group in data.groupby("score_strategy", sort=False):
        strategy_name = safe_filename(strategy)
        groups = [(area, group) for area, group in strategy_group.groupby(
            "safety_area", sort=False
        )]
        groups.append(("cumulative", strategy_group))
        strategy_artifacts = {"confusion_matrices": {}, "aupro": {
            "available": False,
            "reason": "requires pixel-level ground-truth masks and anomaly maps",
        }}

        for name, group in groups:
            path = output_dir / f"{strategy_name}_{safe_filename(name)}_confusion_matrix.png"
            plot_confusion_matrix(
                path, f"{strategy} — {name} confusion matrix", binary_metrics(group)
            )
            strategy_artifacts["confusion_matrices"][name] = str(path.resolve())

        figure, (roc_ax, pr_ax) = plt.subplots(
            1, 2, figsize=(14, 5.5), constrained_layout=True
        )
        curve_metrics = {}
        for name, group in groups:
            curve = ranking_curve_data(group)
            curve_metrics[name] = {
                "auroc": curve["auroc"], "auprc": curve["auprc"]
            }
            if curve["auroc"] is None:
                continue
            width = 2.5 if name == "cumulative" else 1.5
            roc_ax.plot(curve["fpr"], curve["tpr"], linewidth=width,
                        label=f"{name} (AUROC={curve['auroc']:.3f})")
            pr_ax.plot(curve["recall"], curve["precision"], linewidth=width,
                       label=f"{name} (AUPRC={curve['auprc']:.3f})")
        roc_ax.plot((0, 1), (0, 1), "--", color="#64748b", linewidth=1)
        roc_ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
                   title=f"ROC curves — {strategy}", xlim=(0, 1), ylim=(0, 1))
        pr_ax.set(xlabel="Recall", ylabel="Precision",
                  title=f"Precision–recall curves — {strategy}", xlim=(0, 1), ylim=(0, 1))
        for ax in (roc_ax, pr_ax):
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        curves_path = output_dir / f"{strategy_name}_roc_precision_recall_curves.png"
        figure.savefig(curves_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        strategy_artifacts["roc_precision_recall"] = str(curves_path.resolve())
        strategy_artifacts["curve_metrics"] = curve_metrics

        columns = 2
        rows = int(np.ceil(len(groups) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(14, 3.6 * rows),
                                    constrained_layout=True, squeeze=False)
        for ax, (name, group) in zip(axes.flat, groups):
            evaluated = group[group["label"].isin(("Normal", "Anomalous"))]
            for label, color in (("Normal", "#16a34a"), ("Anomalous", "#dc2626")):
                values = evaluated.loc[evaluated["label"] == label, "normalized_score"]
                if len(values):
                    ax.hist(values, bins=70, density=True, alpha=0.48,
                            label=f"{label} (n={len(values):,})", color=color)
            ax.axvline(1.0, color="#111827", linestyle="--", label="Decision threshold")
            ax.set(title=name, xlabel="Normalized anomaly score", ylabel="Density")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        for ax in axes.flat[len(groups):]:
            ax.set_visible(False)
        distributions_path = output_dir / f"{strategy_name}_score_distributions.png"
        figure.suptitle(f"Normal versus anomalous score distributions — {strategy}")
        figure.savefig(distributions_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        strategy_artifacts["score_distributions"] = str(distributions_path.resolve())

        labels = [name for name, _ in groups]
        metrics_by_group = [binary_metrics(group) for _, group in groups]
        x = np.arange(len(labels))
        width = 0.15
        figure, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        for index, metric in enumerate(metric_names):
            values = [item[metric] if item[metric] is not None else np.nan
                      for item in metrics_by_group]
            ax.bar(x + (index - 2) * width, values, width, label=metric.replace("_", " "))
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Metric value")
        ax.set_title(f"Evaluation metric summary — {strategy}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncols=3)
        metrics_path = output_dir / f"{strategy_name}_metrics_summary.png"
        figure.savefig(metrics_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        strategy_artifacts["metrics_summary"] = str(metrics_path.resolve())
        artifacts[strategy] = strategy_artifacts
    return artifacts


def write_evaluation_json(
    output: Path,
    data: pd.DataFrame,
    annotation_csv: Path,
    scores_csvs: list[Path],
    report_html: Path,
    threshold_dir: Path,
    plot_artifacts: dict | None = None,
) -> None:
    threshold_records = load_threshold_metadata(data, threshold_dir)
    threshold_by_key = {
        (record["strategy"], record["area"]): record
        for record in threshold_records
    }
    evaluations = {}
    for strategy, strategy_group in data.groupby("score_strategy", sort=False):
        areas = {}
        for area, area_group in strategy_group.groupby("safety_area", sort=False):
            threshold = dict(threshold_by_key[(strategy, area)])
            threshold["path"] = str(Path(threshold["path"]).resolve())
            areas[area] = {
                "metrics": binary_metrics(area_group),
                "model_training": load_model_metadata(threshold_dir, area),
                "threshold_calibration": threshold,
                "inference": {
                    key: json_value(area_group.iloc[0][key])
                    for key in (
                        "threshold_strategy", "score_func", "reconstruction_mode",
                        "offset", "sigma", "quantile", "threshold",
                    ) if key in area_group.columns
                },
            }
        evaluations[strategy] = {
            "cumulative_metrics": binary_metrics(strategy_group),
            "safety_areas": areas,
        }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "annotations": str(annotation_csv),
            "scores": [str(path) for path in scores_csvs],
            "html_report": str(report_html),
            "threshold_directory": str(threshold_dir),
        },
        "plot_artifacts": plot_artifacts or {},
        "evaluation": evaluations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_report(output: Path, data: pd.DataFrame, annotation_csv: Path,
                 scores_csvs: list[Path], threshold_dir: Path) -> None:
    metrics = {
        f"{strategy}:{area}": binary_metrics(group)
        for (strategy, area), group in data.groupby(
            ["score_strategy", "safety_area"], sort=False
        )
    }
    plot = make_figure(data).to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id="comparison-plot",
        config={"responsive": True, "scrollZoom": True},
    )
    metric_rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in (
            html.escape(key.split(":", 1)[0]), html.escape(key.split(":", 1)[1]),
            values["evaluated"], values["excluded_verify"],
            values["tp"], values["tn"], values["fp"], values["fn"],
            format_metric(values["precision"]), format_metric(values["recall"]),
            format_metric(values["specificity"]), format_metric(values["f1"]),
            format_metric(values["accuracy"]), format_metric(values["balanced_accuracy"]),
        )) + "</tr>" for key, values in metrics.items()
    )
    confusion_cards = "".join(
        confusion_matrix_card(f"{key.split(':', 1)[0]} — {key.split(':', 1)[1]}", values)
        for key, values in metrics.items()
    )
    cumulative_metrics = {
        strategy: binary_metrics(group)
        for strategy, group in data.groupby("score_strategy", sort=False)
    }
    cumulative_cards = "".join(
        confusion_matrix_card(f"{strategy} — all safety areas", values)
        for strategy, values in cumulative_metrics.items()
    )
    threshold_records = load_threshold_metadata(data, threshold_dir)
    threshold_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            record["strategy"], record["area"],
            record.get("score_func", "metadata unavailable"),
            record.get("reconstruction_mode", "legacy-unspecified"),
            record.get("offset", "—"), record.get("sigma", "—"),
            record.get("quantile", "—"), record.get("threshold", record["csv_threshold"]),
            record.get("threshold_strategy", "—"), record.get("n_images", "—"),
            record.get("epochs_trained", "—"), record.get("computed_at", "—"),
            "yes" if record.get("threshold_matches_csv") else "NO" if record["available"] else "unknown",
            record["path"],
        )) + "</tr>" for record in threshold_records
    )
    disagreement = data[
        data["label"].isin(("Normal", "Anomalous"))
        & (data["label"].eq("Anomalous") != data["detected_anomalous"])
    ].copy()
    disagreement["error"] = np.where(
        disagreement["detected_anomalous"], "False positive", "False negative"
    )
    disagreement_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row.score_strategy, row.safety_area, row.frame_id, row.label, row.error,
            f"{row.normalized_score:.3f}", getattr(row, "filename", ""),
            getattr(row, "note", ""),
        )) + "</tr>" for row in disagreement.itertuples()
    )
    highest = data.sort_values("normalized_score", ascending=False).head(100)
    high_score_rows = "".join(
        "<tr>" + "".join((
            f"<td>{html.escape(str(row.safety_area))}</td>",
            f"<td>{html.escape(str(row.score_strategy))}</td>",
            f"<td>{int(row.frame_id)}</td>",
            f"<td>{html.escape(str(row.label))}</td>",
            f"<td>{row.normalized_score:.3f}×</td>",
            f"<td>{row.anomaly_score:.5f}</td>",
            f'<td><a href="{html.escape(row.processed_image_path_uri)}" target="_blank">processed</a></td>' if row.processed_image_path_uri else "<td>—</td>",
            f'<td><a href="{html.escape(row.raw_image_path_uri)}" target="_blank">raw</a></td>' if row.raw_image_path_uri else "<td>—</td>",
        )) + "</tr>" for row in highest.itertuples()
    )
    description = data.get("scenario_description", pd.Series([""])).dropna()
    scenario_description = description.iloc[0] if len(description) else ""
    score_sources = "<br>".join(
        f"<b>Scores:</b> <code>{html.escape(str(path))}</code>" for path in scores_csvs
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Annotation versus detection</title><style>
body{{font-family:system-ui;background:#f3f6f9;color:#172033;margin:0}}main{{width:calc(100vw - 16px);max-width:none;margin:0 auto;padding:8px;box-sizing:border-box}}
section{{background:white;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 1px 4px #0001}}
.plot-section{{padding:6px 2px;overflow:hidden}}#comparison-plot,.plotly-graph-div{{width:100%!important}}
.legend span{{display:inline-block;padding:6px 12px;border-radius:12px;margin-right:8px}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:7px 9px;border-bottom:1px solid #dbe2ea;text-align:left}}th{{position:sticky;top:0;background:#eaf0f6}}
.scroll{{max-height:520px;overflow:auto}}code{{word-break:break-all}}
.cm-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.cm-card{{border:1px solid #dbe2ea;border-radius:10px;padding:12px}}.cm-card h3{{margin:0 0 10px}}
.cm{{table-layout:fixed}}.cm td{{text-align:center;font-size:24px;border:5px solid white;border-radius:10px}}
.cm td small{{display:block;font-size:12px;font-weight:600}}.cm .tn{{background:#dcfce7}}.cm .tp{{background:#ffedd5}}.cm .fp{{background:#dbeafe}}.cm .fn{{background:#fee2e2}}
details>summary{{cursor:pointer;font-size:1.35rem;font-weight:700;padding:4px 0}}details>summary:hover{{color:#2563eb}}details[open]>summary{{margin-bottom:14px}}
#frame-preview{{position:fixed;right:18px;top:18px;width:min(620px,44vw);z-index:20;background:#111827;color:white;padding:12px;border:8px solid #64748b;border-radius:14px;box-shadow:0 8px 30px #0007;display:none}}
#frame-preview .images{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}#frame-preview img{{width:100%;max-height:360px;object-fit:contain;background:#05070a}}
#frame-preview .hint{{color:#cbd5e1;font-size:12px}}#preview-status{{display:inline-block;font-weight:800;font-size:16px;padding:5px 10px;margin:8px 0;border-radius:8px;color:white}}
@media(max-width:900px){{#frame-preview{{width:calc(100vw - 60px);top:auto;bottom:15px}}}}
</style></head><body><main>
<h1>Annotation versus model detection</h1><p>{html.escape(str(scenario_description))}</p>
<section><b>Alignment:</b> per-safety-area row order, verified equal counts. Binary metrics exclude <i>Verify</i> frames.<br>
<b>Annotation:</b> <code>{html.escape(str(annotation_csv))}</code><br>{score_sources}</section>
<section class="legend"><span style="background:#dcfce7">TN: safe, correct</span><span style="background:#ffedd5">TP: anomaly, correct</span><span style="background:#dbeafe">FP: normal, flagged</span><span style="background:#fee2e2">FN: anomaly, missed</span><span style="background:#e2e8f0">Verify: excluded</span><span>Dashed line = detection threshold (1.0×)</span><p>Hover a score to preview its frames. Click to lock the preview; click another point to replace it; use Close to unlock.</p></section>
<section><details><summary>Threshold calibration and anomaly-score configuration</summary>
<p><b>TAAS offset</b> controls spatial tolerance, <b>sigma</b> controls Gaussian smoothing of the residual map, and <b>quantile</b> selects the residual-map tail used as the frame score. These are anomaly-score parameters. The <b>threshold strategy</b> is a separate operation that converts validation-frame scores into the final decision boundary.</p>
<div class="scroll"><table><thead><tr><th>Strategy</th><th>Area</th><th>Score function</th><th>Reconstruction mode</th><th>TAAS offset</th><th>TAAS sigma</th><th>TAAS quantile</th><th>Threshold</th><th>Calibration strategy</th><th>Calibration images</th><th>Model epochs</th><th>Computed at</th><th>Matches score CSV</th><th>Metadata file</th></tr></thead><tbody>{threshold_rows}</tbody></table></div></details></section>
<section><h2>Cumulative confusion matrix</h2><p>All evaluated safety-area rows combined for each threshold strategy. Verify and Unlabeled rows are excluded.</p><div class="cm-grid">{cumulative_cards}</div></section>
<section><h2>Confusion matrices by safety area</h2><p>Rows are ground truth; columns are model predictions.</p><div class="cm-grid">{confusion_cards}</div></section>
<section><h2>Metrics by threshold strategy and safety area</h2><table><thead><tr><th>Strategy</th><th>Area</th><th>Evaluated</th><th>Verify excluded</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>Accuracy</th><th>Balanced accuracy</th></tr></thead><tbody>{metric_rows}</tbody></table></section>
<section class="plot-section">{plot}</section>
<section><h2>Highest 100 normalized scores</h2><div class="scroll"><table><thead><tr><th>Area</th><th>Strategy</th><th>Frame</th><th>Annotation</th><th>Normalized score</th><th>Raw score</th><th>Processed image</th><th>Raw image</th></tr></thead><tbody>{high_score_rows}</tbody></table></div></section>
<section><h2>Disagreements ({len(disagreement)})</h2><div class="scroll"><table><thead><tr><th>Strategy</th><th>Area</th><th>Frame</th><th>Annotation</th><th>Error</th><th>Normalized score</th><th>Filename</th><th>Note</th></tr></thead><tbody>{disagreement_rows}</tbody></table></div></section>
<aside id="frame-preview"><div><b id="preview-title"></b> <button id="preview-close" style="float:right">Close</button></div><div id="preview-status"></div><div id="preview-meta"></div><div class="images"><div><small>Processed safety area</small><img id="preview-processed"></div><div><small>Raw frame</small><img id="preview-raw"></div></div><div class="hint">Images are loaded from dataset paths and are not embedded in this report.</div></aside>
<script type="application/json" id="comparison-metrics">{html.escape(json.dumps(metrics))}</script>
<script>
const plot=document.getElementById('comparison-plot'), preview=document.getElementById('frame-preview');
const title=document.getElementById('preview-title'), meta=document.getElementById('preview-meta');
const statusBadge=document.getElementById('preview-status');
const processed=document.getElementById('preview-processed'), raw=document.getElementById('preview-raw');
let locked=false;
function classification(label,detected){{
  if(label==='Verify'||label==='Unlabeled')return {{code:'VERIFY',text:'Needs verification — excluded from metrics',color:'#64748b'}};
  if(label==='Anomalous'&&detected)return {{code:'TP',text:'Anomaly correctly detected',color:'#f97316'}};
  if(label==='Anomalous'&&!detected)return {{code:'FN',text:'Anomaly missed by model',color:'#dc2626'}};
  if(label==='Normal'&&detected)return {{code:'FP',text:'Normal frame incorrectly flagged',color:'#2563eb'}};
  return {{code:'TN',text:'Safe frame correctly classified',color:'#16a34a'}};
}}
function showFrame(point){{
  const d=point.customdata;if(!d)return;
  const result=classification(d[0],d[7]===true||String(d[7]).toLowerCase()==='true');
  title.textContent=`Scenario ${{d[9]}} — Global frame ${{point.x}} — ${{d[0]}}`;
  statusBadge.textContent=`${{result.code}} — ${{result.text}}`;statusBadge.style.background=result.color;preview.style.borderColor=result.color;
  meta.textContent=`${{d[10]}} | Source frame ${{d[11]}} | ${{d[8]}} | Score ${{Number(point.y).toFixed(3)}}× | raw ${{Number(d[1]).toFixed(5)}} | threshold ${{Number(d[2]).toFixed(5)}}`;
  processed.src=d[4]||'';raw.src=d[5]||'';processed.style.display=d[4]?'block':'none';raw.style.display=d[5]?'block':'none';preview.style.display='block';
}}
plot.on('plotly_hover',e=>{{if(!locked&&e.points.length)showFrame(e.points[0]);}});
plot.on('plotly_unhover',()=>{{if(!locked)preview.style.display='none';}});
plot.on('plotly_click',e=>{{if(e.points.length){{locked=true;showFrame(e.points[0]);}}}});
document.getElementById('preview-close').onclick=()=>{{locked=false;preview.style.display='none';}};
</script>
</main></body></html>""", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path, nargs="+")
    parser.add_argument(
        "--threshold-dir", type=Path,
        help="Threshold metadata root; default: sibling thresholds directory under results version.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluation-json", type=Path,
        help=(
            "Evaluation JSON path. Default: "
            "results/<dataset-version>/evaluation/evaluation_<score-stem>.json"
        ),
    )
    parser.add_argument(
        "--plots-dir", type=Path,
        help=(
            "Evaluation plot directory. Default: "
            "results/<dataset-version>/evaluation/plots/<evaluation-name>."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = args.annotations.expanduser().resolve()
    scores = [path.expanduser().resolve() for path in args.scores]
    threshold_dir = (
        args.threshold_dir.expanduser().resolve() if args.threshold_dir else
        scores[0].parent.parent / "thresholds"
    )
    if args.output:
        output = args.output.expanduser().resolve()
    elif len(scores) == 1:
        output = scores[0].with_name(f"{scores[0].stem}_annotation_comparison.html")
    else:
        output = scores[0].with_name("rosbag_threshold_strategies_annotation_comparison.html")
    datasets = []
    for score_path in scores:
        data = load_and_align(annotations, score_path)
        if "threshold_strategy" in data.columns:
            strategies = data["threshold_strategy"].dropna().astype(str).unique()
        else:
            strategies = []
        if len(strategies) == 1:
            strategy = strategies[0]
        else:
            match = re.search(
                r"_(max|percentile|mean_std)(?:_|_scores$)", score_path.stem
            )
            strategy = match.group(1) if match else score_path.stem
        data["score_strategy"] = strategy
        datasets.append(data)
    data = pd.concat(datasets, ignore_index=True)
    write_report(output, data, annotations, scores, threshold_dir)
    evaluation_output = (
        args.evaluation_json.expanduser().resolve()
        if args.evaluation_json else default_evaluation_path(scores[0], threshold_dir)
    )
    plots_output = (
        args.plots_dir.expanduser().resolve()
        if args.plots_dir else
        evaluation_output.parent / "plots" / evaluation_output.stem
    )
    plot_artifacts = save_evaluation_plots(data, plots_output)
    write_evaluation_json(
        evaluation_output, data, annotations, scores, output, threshold_dir,
        plot_artifacts,
    )
    print(f"Aligned rows: {len(data)}")
    for (strategy, area), group in data.groupby(["score_strategy", "safety_area"], sort=False):
        values = binary_metrics(group)
        print(
            f"{strategy}/{area}: precision={format_metric(values['precision'])} "
            f"recall={format_metric(values['recall'])} F1={format_metric(values['f1'])} "
            f"FP={values['fp']} FN={values['fn']} verify={values['excluded_verify']}"
        )
    print(f"Report: {output}")
    print(f"Evaluation JSON: {evaluation_output}")
    print(f"Evaluation plots: {plots_output}")


if __name__ == "__main__":
    main()
