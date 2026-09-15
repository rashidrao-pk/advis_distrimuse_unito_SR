#!/usr/bin/env python3
"""Create an interactive annotation-versus-detection comparison report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    annotations = pd.read_csv(annotation_csv)
    scores = pd.read_csv(scores_csv)
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

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    return {
        "evaluated": len(evaluated), "excluded_verify": int((group["label"] == "Verify").sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp), "f1": divide(2 * tp, 2 * tp + fp + fn),
        "accuracy": divide(tp + tn, len(evaluated)),
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


def make_figure(data: pd.DataFrame) -> go.Figure:
    areas = list(dict.fromkeys(data["safety_area"]))
    figure = make_subplots(
        rows=len(areas), cols=1, shared_xaxes=True, vertical_spacing=0.045,
        subplot_titles=areas,
    )
    for row_number, area in enumerate(areas, start=1):
        group = data[data["safety_area"] == area].sort_values("frame_id")
        custom = np.column_stack((
            group["label"], group["anomaly_score"], group["threshold"],
            group.get("filename", pd.Series([""] * len(group))),
            group["processed_image_path_uri"], group["raw_image_path_uri"],
            group.get("note", pd.Series([""] * len(group))).fillna(""),
        ))
        figure.add_trace(go.Scatter(
            x=group["frame_id"], y=group["normalized_score"], mode="lines",
            line={"color": "#2563eb", "width": 1.5}, name="score / threshold",
            legendgroup="score", showlegend=row_number == 1, customdata=custom,
            hovertemplate=(
                "Frame %{x}<br>Normalized score %{y:.3f}×<br>"
                "Annotation %{customdata[0]}<br>Raw score %{customdata[1]:.5f}<br>"
                "Threshold %{customdata[2]:.5f}<br>%{customdata[3]}"
                "<br>%{customdata[6]}<extra></extra>"
            ),
        ), row=row_number, col=1)
        axis_name = "y" if row_number == 1 else f"y{row_number}"
        figure.add_shape(
            type="line", x0=float(group.frame_id.min()), x1=float(group.frame_id.max()),
            y0=1, y1=1, line={"color": "#111827", "dash": "dash", "width": 1},
            xref="x" if row_number == 1 else f"x{row_number}", yref=axis_name,
        )
        y_max = max(1.1, float(group["normalized_score"].max()) * 1.08)
        for start, end, label in label_runs(group):
            figure.add_shape(
                type="rect", x0=start - 0.5, x1=end + 0.5, y0=0, y1=y_max,
                fillcolor=LABEL_COLORS.get(label, LABEL_COLORS["Unlabeled"]),
                line={"width": 0}, layer="below",
                xref="x" if row_number == 1 else f"x{row_number}", yref=axis_name,
            )
        false_positive = group[(group["label"] == "Normal") & group["detected_anomalous"]]
        false_negative = group[(group["label"] == "Anomalous") & ~group["detected_anomalous"]]
        for subset, name, color, symbol in (
            (false_positive, "False positive", "#f97316", "x"),
            (false_negative, "False negative", "#dc2626", "circle-open"),
        ):
            figure.add_trace(go.Scatter(
                x=subset["frame_id"], y=subset["normalized_score"], mode="markers",
                marker={"color": color, "symbol": symbol, "size": 7}, name=name,
                legendgroup=name, showlegend=row_number == 1,
                hovertemplate=f"{name}<br>Frame %{{x}}<br>Score %{{y:.3f}}×<extra></extra>",
            ), row=row_number, col=1)
        figure.update_yaxes(title_text="score / threshold", range=[0, y_max], row=row_number, col=1)
    figure.update_xaxes(title_text="Frame ID", row=len(areas), col=1)
    figure.update_layout(
        height=max(850, 290 * len(areas)), template="plotly_white", hovermode="x unified",
        legend={"orientation": "h", "y": 1.03}, margin={"t": 100},
        title="Annotation versus model detection",
    )
    return figure


def format_metric(value):
    return "—" if value is None else f"{value:.3f}"


def write_report(output: Path, data: pd.DataFrame, annotation_csv: Path,
                 scores_csv: Path) -> None:
    metrics = {area: binary_metrics(group) for area, group in data.groupby("safety_area", sort=False)}
    plot = make_figure(data).to_html(include_plotlyjs=True, full_html=False, div_id="comparison-plot")
    metric_rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in (
            html.escape(area), values["evaluated"], values["excluded_verify"],
            values["tp"], values["tn"], values["fp"], values["fn"],
            format_metric(values["precision"]), format_metric(values["recall"]),
            format_metric(values["specificity"]), format_metric(values["f1"]),
            format_metric(values["accuracy"]),
        )) + "</tr>" for area, values in metrics.items()
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
            row.safety_area, row.frame_id, row.label, row.error,
            f"{row.normalized_score:.3f}", getattr(row, "filename", ""),
            getattr(row, "note", ""),
        )) + "</tr>" for row in disagreement.itertuples()
    )
    highest = data.sort_values("normalized_score", ascending=False).head(100)
    high_score_rows = "".join(
        "<tr>" + "".join((
            f"<td>{html.escape(str(row.safety_area))}</td>",
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Annotation versus detection</title><style>
body{{font-family:system-ui;background:#f3f6f9;color:#172033;margin:0}}main{{max-width:1800px;margin:auto;padding:22px}}
section{{background:white;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 1px 4px #0001}}
.legend span{{display:inline-block;padding:6px 12px;border-radius:12px;margin-right:8px}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:7px 9px;border-bottom:1px solid #dbe2ea;text-align:left}}th{{position:sticky;top:0;background:#eaf0f6}}
.scroll{{max-height:520px;overflow:auto}}code{{word-break:break-all}}
#frame-preview{{position:fixed;right:18px;top:18px;width:min(620px,44vw);z-index:20;background:#111827;color:white;padding:12px;border-radius:12px;box-shadow:0 8px 30px #0007;display:none}}
#frame-preview .images{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}#frame-preview img{{width:100%;max-height:360px;object-fit:contain;background:#05070a}}
#frame-preview .hint{{color:#cbd5e1;font-size:12px}}@media(max-width:900px){{#frame-preview{{width:calc(100vw - 60px);top:auto;bottom:15px}}}}
</style></head><body><main>
<h1>Annotation versus model detection</h1><p>{html.escape(str(scenario_description))}</p>
<section><b>Alignment:</b> per-safety-area row order, verified equal counts. Binary metrics exclude <i>Verify</i> frames.<br>
<b>Annotation:</b> <code>{html.escape(str(annotation_csv))}</code><br><b>Scores:</b> <code>{html.escape(str(scores_csv))}</code></section>
<section class="legend"><span style="background:#dcfce7">Normal annotation</span><span style="background:#fee2e2">Anomalous annotation</span><span style="background:#fef3c7">Verify</span><span>Dashed line = detection threshold (1.0×)</span><p>Hover a score to preview its frames. Click to lock the preview; click another point to replace it; use Close to unlock.</p></section>
<section>{plot}</section>
<section><h2>Metrics by safety area</h2><table><thead><tr><th>Area</th><th>Evaluated</th><th>Verify excluded</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>Accuracy</th></tr></thead><tbody>{metric_rows}</tbody></table></section>
<section><h2>Highest 100 scores</h2><div class="scroll"><table><thead><tr><th>Area</th><th>Frame</th><th>Annotation</th><th>Normalized score</th><th>Raw score</th><th>Processed image</th><th>Raw image</th></tr></thead><tbody>{high_score_rows}</tbody></table></div></section>
<section><h2>Disagreements ({len(disagreement)})</h2><div class="scroll"><table><thead><tr><th>Area</th><th>Frame</th><th>Annotation</th><th>Error</th><th>Normalized score</th><th>Filename</th><th>Note</th></tr></thead><tbody>{disagreement_rows}</tbody></table></div></section>
<aside id="frame-preview"><div><b id="preview-title"></b> <button id="preview-close" style="float:right">Close</button></div><div id="preview-meta"></div><div class="images"><div><small>Processed safety area</small><img id="preview-processed"></div><div><small>Raw frame</small><img id="preview-raw"></div></div><div class="hint">Images are loaded from dataset paths and are not embedded in this report.</div></aside>
<script type="application/json" id="comparison-metrics">{html.escape(json.dumps(metrics))}</script>
<script>
const plot=document.getElementById('comparison-plot'), preview=document.getElementById('frame-preview');
const title=document.getElementById('preview-title'), meta=document.getElementById('preview-meta');
const processed=document.getElementById('preview-processed'), raw=document.getElementById('preview-raw');
let locked=false;
function showFrame(point){{
  const d=point.customdata;if(!d)return;
  title.textContent=`Frame ${{point.x}} — ${{d[0]}}`;
  meta.textContent=`Score ${{Number(point.y).toFixed(3)}}× | raw ${{Number(d[1]).toFixed(5)}} | threshold ${{Number(d[2]).toFixed(5)}}`;
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
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = args.annotations.expanduser().resolve()
    scores = args.scores.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              scores.with_name(f"{scores.stem}_annotation_comparison.html"))
    data = load_and_align(annotations, scores)
    write_report(output, data, annotations, scores)
    print(f"Aligned rows: {len(data)}")
    for area, group in data.groupby("safety_area", sort=False):
        values = binary_metrics(group)
        print(
            f"{area}: precision={format_metric(values['precision'])} "
            f"recall={format_metric(values['recall'])} F1={format_metric(values['f1'])} "
            f"FP={values['fp']} FN={values['fn']} verify={values['excluded_verify']}"
        )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
