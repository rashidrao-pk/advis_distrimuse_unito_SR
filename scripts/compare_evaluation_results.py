#!/usr/bin/env python3
"""Compare TAAS evaluation JSON files and write an interactive HTML dashboard."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRICS = (
    "balanced_accuracy", "f1", "auroc", "auprc", "precision", "recall",
    "specificity", "accuracy", "false_positive_rate", "false_negative_rate",
)
HEATMAP_METRICS = METRICS[:8]
RANK_METRICS = (*METRICS, "fp", "fn")
LOWER_IS_BETTER = {"false_positive_rate", "false_negative_rate", "fp", "fn"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evaluation_dir", type=Path,
        help="Directory containing evaluation_*.json files.",
    )
    parser.add_argument(
        "--rank-by", choices=RANK_METRICS, default="balanced_accuracy",
        help="Primary cumulative ranking metric (default: balanced_accuracy).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output HTML (default: <evaluation_dir>/evaluation_comparison.html).",
    )
    return parser.parse_args()


def variant_from_path(path: Path) -> str:
    return re.sub(r"^evaluation_|\.json$", "", path.name).removesuffix("_scores")


def load_evaluations(evaluation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_rows = []
    area_rows = []
    for path in sorted(evaluation_dir.glob("evaluation_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant_base = variant_from_path(path)
        for strategy, evaluation in (payload.get("evaluation") or {}).items():
            areas = evaluation.get("safety_areas") or {}
            first = next(iter(areas.values()), {})
            inference = first.get("inference") or {}
            calibration = first.get("threshold_calibration") or {}
            variant = variant_base if len(payload.get("evaluation") or {}) == 1 else f"{variant_base}_{strategy}"
            common = {
                "variant": variant,
                "strategy": strategy,
                "offset": inference.get("offset", calibration.get("offset")),
                "sigma": inference.get("sigma", calibration.get("sigma")),
                "quantile": inference.get("quantile", calibration.get("quantile")),
                "threshold_percentile": calibration.get("threshold_percentile"),
                "score_func": inference.get("score_func", calibration.get("score_func")),
                "reconstruction_mode": inference.get(
                    "reconstruction_mode", calibration.get("reconstruction_mode")
                ),
                "source_json": str(path.resolve()),
                "source_html": (payload.get("sources") or {}).get("html_report", ""),
            }
            cumulative_rows.append({
                **common, **(evaluation.get("cumulative_metrics") or {})
            })
            for area, details in areas.items():
                area_rows.append({
                    **common,
                    "safety_area": area,
                    **(details.get("metrics") or {}),
                    "threshold": (details.get("inference") or {}).get("threshold"),
                    "training_images": (
                        (details.get("model_training") or {}).get("dataset") or {}
                    ).get("n_train_images"),
                    "validation_images": (
                        (details.get("model_training") or {}).get("dataset") or {}
                    ).get("n_val_images"),
                    "calibration_images": (
                        details.get("threshold_calibration") or {}
                    ).get("n_images"),
                })
    if not cumulative_rows:
        raise FileNotFoundError(f"No usable evaluation_*.json files in {evaluation_dir}")
    return pd.DataFrame(cumulative_rows), pd.DataFrame(area_rows)


def ranked(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    result = frame.copy()
    result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return result.sort_values(
        metric, ascending=metric in LOWER_IS_BETTER, na_position="last"
    ).reset_index(drop=True)


def metric_text(value):
    return "—" if value is None or pd.isna(value) else f"{float(value):.4f}"


def best_card(frame: pd.DataFrame, metric: str) -> str:
    valid = frame.dropna(subset=[metric])
    if valid.empty:
        return ""
    row_index = valid[metric].idxmin() if metric in LOWER_IS_BETTER else valid[metric].idxmax()
    row = valid.loc[row_index]
    qualifier = "Lowest" if metric in LOWER_IS_BETTER else "Best"
    return (
        f'<article class="best"><small>{qualifier} cumulative {html.escape(metric.replace("_", " "))}</small>'
        f'<strong>{float(row[metric]):.4f}</strong><span>{html.escape(str(row.variant))}</span></article>'
    )


def make_tradeoff_plot(frame: pd.DataFrame) -> go.Figure:
    custom = np.column_stack((
        frame["variant"], frame["offset"], frame["sigma"], frame["quantile"],
        frame["f1"], frame["balanced_accuracy"], frame["auroc"], frame["auprc"],
    ))
    figure = go.Figure(go.Scatter(
        x=frame["recall"], y=frame["precision"], mode="markers+text",
        text=frame["variant"].str.extract(r"(off\d+_sig[\d.]+_q[\d.]+)", expand=False),
        textposition="top center",
        marker={
            "size": 13, "color": frame["balanced_accuracy"], "colorscale": "Viridis",
            "showscale": True, "colorbar": {"title": "Balanced accuracy"},
        },
        customdata=custom,
        hovertemplate=(
            "%{customdata[0]}<br>Offset %{customdata[1]} · sigma %{customdata[2]} · "
            "quantile %{customdata[3]}<br>Precision %{y:.4f}<br>Recall %{x:.4f}<br>"
            "F1 %{customdata[4]:.4f}<br>Balanced accuracy %{customdata[5]:.4f}<br>"
            "AUROC %{customdata[6]:.4f}<br>AUPRC %{customdata[7]:.4f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title="Cumulative precision–recall trade-off", template="plotly_white",
        xaxis_title="Recall", yaxis_title="Precision", height=620,
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
    )
    return figure


def make_metric_heatmap(frame: pd.DataFrame) -> go.Figure:
    values = frame[list(HEATMAP_METRICS)].astype(float).to_numpy()
    figure = go.Figure(go.Heatmap(
        z=values, x=[name.replace("_", " ") for name in HEATMAP_METRICS],
        y=frame["variant"], colorscale="Viridis", zmin=0, zmax=1,
        text=np.where(np.isnan(values), "—", np.vectorize(lambda x: f"{x:.3f}")(values)),
        texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
    ))
    figure.update_layout(
        title="Cumulative metric heatmap", template="plotly_white",
        height=max(520, 34 * len(frame) + 170), margin={"l": 270, "r": 30, "t": 70},
    )
    return figure


def make_area_heatmaps(area_frame: pd.DataFrame) -> go.Figure:
    variants = list(dict.fromkeys(area_frame["variant"]))
    areas = list(dict.fromkeys(area_frame["safety_area"]))
    figure = make_subplots(rows=1, cols=2, subplot_titles=("F1 by safety area", "Balanced accuracy by safety area"))
    for column, metric in enumerate(("f1", "balanced_accuracy"), start=1):
        pivot = area_frame.pivot(index="variant", columns="safety_area", values=metric).reindex(index=variants, columns=areas)
        values = pivot.to_numpy(dtype=float)
        figure.add_trace(go.Heatmap(
            z=values, x=areas, y=variants, colorscale="Viridis", zmin=0, zmax=1,
            text=np.where(np.isnan(values), "—", np.vectorize(lambda x: f"{x:.3f}")(values)),
            texttemplate="%{text}", showscale=column == 2,
            hovertemplate="%{y}<br>%{x}<br>Score %{z:.4f}<extra></extra>",
        ), row=1, col=column)
    figure.update_layout(
        title="Safety-area robustness across TAAS variants", template="plotly_white",
        height=max(570, 34 * len(variants) + 180), margin={"l": 270, "r": 30, "t": 80},
    )
    return figure


def make_parameter_plot(frame: pd.DataFrame, rank_by: str) -> go.Figure:
    figure = go.Figure()
    for (offset, sigma), group in frame.groupby(["offset", "sigma"], sort=True):
        group = group.sort_values("quantile")
        figure.add_trace(go.Scatter(
            x=group["quantile"], y=group[rank_by], mode="lines+markers",
            name=f"offset={offset}, sigma={sigma}", customdata=group[["variant"]],
            hovertemplate="%{customdata[0]}<br>quantile %{x}<br>score %{y:.4f}<extra></extra>",
        ))
    figure.update_layout(
        title=f"Effect of TAAS parameters on cumulative {rank_by.replace('_', ' ')}",
        template="plotly_white", xaxis_title="TAAS quantile", yaxis_title=rank_by.replace("_", " "),
        height=560, legend={"orientation": "h", "y": 1.08},
    )
    return figure


def results_table(frame: pd.DataFrame, rank_by: str) -> str:
    columns = (
        "variant", "offset", "sigma", "quantile", "threshold_percentile",
        "tp", "tn", "fp", "fn", *METRICS,
    )
    rows = []
    for index, row in frame.iterrows():
        cells = [f"<td>{index + 1}</td>"]
        for column in columns:
            value = row.get(column)
            if column in METRICS:
                rendered = metric_text(value)
            else:
                rendered = "—" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        cells.append(f'<td><a href="{Path(row.source_json).as_uri()}">JSON</a></td>')
        cells.append(
            f'<td><a href="{Path(row.source_html).as_uri()}">HTML</a></td>'
            if row.source_html else "<td>—</td>"
        )
        rows.append(f'<tr class="{"winner" if index == 0 else ""}">' + "".join(cells) + "</tr>")
    headers = ("Rank", *(name.replace("_", " ") for name in columns), "Evaluation JSON", "Detailed report")
    numeric_columns = {
        "Rank", "offset", "sigma", "quantile", "threshold percentile",
        "tp", "tn", "fp", "fn", *(metric.replace("_", " ") for metric in METRICS),
    }
    header_cells = []
    for index, value in enumerate(headers):
        key = value.replace(" ", "_")
        is_numeric = value in numeric_columns
        first_direction = (
            "asc" if key in LOWER_IS_BETTER or value == "Rank"
            else "desc" if is_numeric else "asc"
        )
        header_cells.append(
            f'<th data-column="{index}" data-type="{"number" if is_numeric else "text"}" '
            f'data-first="{first_direction}">{html.escape(value)}</th>'
        )
    return (
        "<table id='results-table'><thead><tr>"
        + "".join(header_cells)
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def write_dashboard(output: Path, cumulative: pd.DataFrame, areas: pd.DataFrame, rank_by: str) -> None:
    ordered = ranked(cumulative, rank_by)
    cards = "".join(best_card(cumulative, metric) for metric in (
        rank_by, "f1", "auroc", "auprc", "recall", "precision",
        "false_positive_rate", "false_negative_rate",
    ))
    plots = [
        make_tradeoff_plot(ordered), make_metric_heatmap(ordered),
        make_area_heatmaps(areas), make_parameter_plot(ordered, rank_by),
    ]
    plot_sections = "".join(
        "<section>" + figure.to_html(
            full_html=False, include_plotlyjs=index == 0,
            config={"responsive": True, "scrollZoom": True},
        ) + "</section>" for index, figure in enumerate(plots)
    )
    best = ordered.iloc[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>TAAS evaluation comparison</title><style>
body{{font-family:system-ui;background:#f3f6f9;color:#172033;margin:0}}main{{width:calc(100vw - 20px);margin:auto}}
section{{background:white;border-radius:12px;padding:18px;margin:12px 0;box-shadow:0 1px 4px #0001}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}.best{{padding:14px;border:1px solid #dbe2ea;border-radius:10px}}
.best small,.best span{{display:block}}.best strong{{display:block;font-size:28px;color:#166534;margin:5px 0}}
.scroll{{overflow:auto;max-height:720px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:7px 9px;border-bottom:1px solid #dbe2ea;text-align:right}}th{{position:sticky;top:0;background:#eaf0f6;cursor:pointer;user-select:none}}th:hover{{background:#dbe7f2}}th.sorted::after{{content:attr(data-arrow);margin-left:6px}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr.winner{{background:#dcfce7}}
.note{{border-left:5px solid #f59e0b;padding-left:12px}}code{{word-break:break-all}}
</style></head><body><main><h1>TAAS evaluation comparison</h1>
<section><p>Compared <b>{len(ordered)}</b> evaluated variants from <code>{html.escape(str(output.parent))}</code>.</p>
<p>Initially ranked by cumulative <b>{html.escape(rank_by.replace('_', ' '))}</b>. The current leader is <b>{html.escape(str(best.variant))}</b> at <b>{float(best[rank_by]):.4f}</b>.</p>
<p><b>Why this is first:</b> balanced accuracy is the default because it averages recall (detecting anomalous frames) and specificity (accepting normal frames), giving both classes equal importance even when normal frames greatly outnumber anomalies. The first row has the highest value for that default objective; it is not necessarily the variant with the fewest false alarms or fewest missed anomalies.</p>
<p class="note"><b>Choose the behavior you need:</b> click any table heading to sort. For fewer false alarms, sort <b>false positive rate</b> or <b>FP</b> ascending. For fewer missed anomalies, sort <b>false negative rate</b> or <b>FN</b> ascending, or <b>recall</b> descending. Use <b>F1</b> for a precision–recall compromise. Lower is better for FP, FN, false-positive rate and false-negative rate; higher is better for the other metrics. AUPRO is unavailable because these evaluations contain frame-level labels and scalar scores, not pixel-level anomaly maps.</p></section>
<section class="cards">{cards}</section>
<section><h2>Ranked cumulative results</h2><div class="scroll">{results_table(ordered, rank_by)}</div></section>
{plot_sections}
<script>
(() => {{
  const table = document.getElementById('results-table');
  const body = table.tBodies[0];
  const headers = [...table.tHead.rows[0].cells];
  headers.forEach(header => header.addEventListener('click', () => {{
    const column = Number(header.dataset.column);
    const direction = header.dataset.direction || header.dataset.first;
    const multiplier = direction === 'asc' ? 1 : -1;
    const rows = [...body.rows];
    rows.sort((a, b) => {{
      const av = a.cells[column].textContent.trim();
      const bv = b.cells[column].textContent.trim();
      if (av === '—') return 1;
      if (bv === '—') return -1;
      if (header.dataset.type === 'number') return multiplier * (Number(av) - Number(bv));
      return multiplier * av.localeCompare(bv, undefined, {{numeric: true, sensitivity: 'base'}});
    }});
    rows.forEach((row, index) => {{
      row.cells[0].textContent = index + 1;
      row.classList.toggle('winner', index === 0);
      body.appendChild(row);
    }});
    headers.forEach(item => {{ item.classList.remove('sorted'); delete item.dataset.arrow; }});
    header.classList.add('sorted');
    header.dataset.arrow = direction === 'asc' ? '▲' : '▼';
    header.dataset.direction = direction === 'asc' ? 'desc' : 'asc';
  }}));
}})();
</script></main></body></html>""", encoding="utf-8")


def main():
    args = parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")
    cumulative, areas = load_evaluations(evaluation_dir)
    output = (
        args.output.expanduser().resolve()
        if args.output else evaluation_dir / "evaluation_comparison.html"
    )
    write_dashboard(output, cumulative, areas, args.rank_by)
    leader = ranked(cumulative, args.rank_by).iloc[0]
    print(f"Compared variants: {len(cumulative)}")
    print(f"Best by {args.rank_by}: {leader.variant} ({leader[args.rank_by]:.4f})")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
