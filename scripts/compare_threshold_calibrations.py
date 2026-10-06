#!/usr/bin/env python3
"""Compare supervised threshold-calibration combinations in an HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go


RANK_METRICS = (
    "balanced_accuracy", "f1", "auc", "recall", "precision", "specificity",
    "accuracy", "false_positive_rate", "false_negative_rate", "fp", "fn",
    "separation_score",
)
LOWER_IS_BETTER = {"false_positive_rate", "false_negative_rate", "fp", "fn"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "calibration_dir", type=Path,
        help="Scenario calibration directory, or its json/ directory.",
    )
    parser.add_argument(
        "--rank-by", choices=RANK_METRICS, default="balanced_accuracy",
        help="Default ranking metric (default: balanced_accuracy).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output HTML (default: <scenario directory>/calibration_comparison.html).",
    )
    return parser.parse_args(argv)


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def load_combinations(calibration_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Load comparison JSON artifacts and derive confusion-matrix metrics."""
    calibration_dir = calibration_dir.expanduser().resolve()
    json_dir = calibration_dir if calibration_dir.name == "json" else calibration_dir / "json"
    scenario_dir = json_dir.parent
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Calibration JSON directory not found: {json_dir}")

    rows = []
    for path in sorted(json_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("artifact_type") != "test_threshold_combination":
            continue
        metrics = payload.get("metrics") or {}
        parameters = payload.get("score_parameters") or {}
        tn = int(metrics.get("true_negative") or 0)
        fp = int(metrics.get("false_positive") or 0)
        fn = int(metrics.get("false_negative") or 0)
        tp = int(metrics.get("true_positive") or 0)
        recall = metrics.get("recall")
        specificity = _divide(tn, tn + fp)
        balanced_accuracy = (
            (float(recall) + specificity) / 2
            if recall is not None and specificity is not None else None
        )
        plot_file = payload.get("plot_file")
        plot_path = (json_dir / plot_file).resolve() if plot_file else None
        rows.append({
            "method": payload.get("method", path.stem),
            "safety_area": payload.get("safety_area", "unknown"),
            "offset": parameters.get("offset"),
            "sigma": parameters.get("sigma"),
            "quantile": parameters.get("quantile"),
            "taas_variant": parameters.get("taas_variant"),
            "threshold": payload.get("threshold"),
            "selection_metric": payload.get("selection_metric"),
            "saved_rank": payload.get("rank"),
            "saved_winner": bool(payload.get("is_best")),
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": balanced_accuracy,
            "precision": metrics.get("precision"),
            "recall": recall,
            "specificity": specificity,
            "f1": metrics.get("f1"),
            "auc": metrics.get("auc"),
            # Older artifacts used the misleading name binormal_auc. Display
            # it under its actual meaning without modifying source artifacts.
            "separation_score": metrics.get(
                "separation_score", metrics.get("binormal_auc")
            ),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "false_positive_rate": _divide(fp, fp + tn),
            "false_negative_rate": _divide(fn, fn + tp),
            "source_json": path,
            "source_plot": plot_path if plot_path and plot_path.is_file() else None,
        })
    if not rows:
        raise FileNotFoundError(f"No test_threshold_combination JSON files in {json_dir}")
    return pd.DataFrame(rows), scenario_dir


def rank_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    result = frame.copy()
    result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return result.sort_values(
        metric, ascending=metric in LOWER_IS_BETTER, na_position="last"
    ).reset_index(drop=True)


def winner(frame: pd.DataFrame, metric: str) -> pd.Series | None:
    ranked = rank_frame(frame, metric).dropna(subset=[metric])
    return None if ranked.empty else ranked.iloc[0]


def _metric(value, digits=4):
    return "—" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def plot_anchor(method: str) -> str:
    """Return a stable in-page anchor for one calibration method."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(method)).strip("-")
    return f"calibration-plot-{slug or 'unknown'}"


def _winner_card(frame: pd.DataFrame, metric: str, title: str) -> str:
    row = winner(frame, metric)
    if row is None:
        return ""
    direction = "Lowest" if metric in LOWER_IS_BETTER else "Highest"
    return (
        '<article class="card">'
        f'<small>{html.escape(title)}</small><strong>{_metric(row[metric])}</strong>'
        f'<span>{html.escape(str(row.method))}</span>'
        f'<em>{direction} {html.escape(metric.replace("_", " "))}</em></article>'
    )


def tradeoff_figure(frame: pd.DataFrame) -> go.Figure:
    custom = np.column_stack((
        frame["method"], frame["balanced_accuracy"], frame["f1"], frame["auc"],
        frame["fp"], frame["fn"], frame["threshold"], frame["saved_winner"],
    ))
    figure = go.Figure(go.Scatter(
        x=frame["recall"], y=frame["precision"], mode="markers",
        marker={
            "size": np.where(frame["saved_winner"], 19, 12),
            "color": frame["balanced_accuracy"], "colorscale": "Viridis",
            "showscale": True, "colorbar": {"title": "Balanced<br>accuracy"},
            "line": {
                "width": np.where(frame["saved_winner"], 4, 1),
                "color": np.where(frame["saved_winner"], "#ef4444", "#ffffff"),
            },
        },
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Recall %{x:.4f}<br>Precision %{y:.4f}"
            "<br>Balanced accuracy %{customdata[1]:.4f}<br>F1 %{customdata[2]:.4f}"
            "<br>AUROC %{customdata[3]:.4f}<br>FP %{customdata[4]} · FN %{customdata[5]}"
            "<br>Threshold %{customdata[6]:.6f}<br>Saved winner: %{customdata[7]}"
            "<extra></extra>"
        ),
    ))
    figure.update_layout(
        title="Precision–recall trade-off", template="plotly_white", height=590,
        xaxis_title="Recall — higher means fewer missed anomalies",
        yaxis_title="Precision — higher means fewer false alarms",
        margin={"l": 65, "r": 35, "t": 70, "b": 65},
    )
    return figure


def error_figure(frame: pd.DataFrame) -> go.Figure:
    custom = np.column_stack((
        frame["method"], frame["f1"], frame["balanced_accuracy"],
        frame["precision"], frame["recall"],
    ))
    figure = go.Figure(go.Scatter(
        x=frame["fp"], y=frame["fn"], mode="markers",
        marker={
            "size": 13, "color": frame["f1"], "colorscale": "Plasma",
            "showscale": True, "colorbar": {"title": "F1"},
        },
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>False positives %{x}<br>False negatives %{y}"
            "<br>F1 %{customdata[1]:.4f}<br>Balanced accuracy %{customdata[2]:.4f}"
            "<br>Precision %{customdata[3]:.4f}<br>Recall %{customdata[4]:.4f}"
            "<extra></extra>"
        ),
    ))
    figure.update_layout(
        title="False-alarm versus missed-anomaly trade-off", template="plotly_white",
        height=590, xaxis_title="False positives — lower is better",
        yaxis_title="False negatives — lower is better",
        margin={"l": 65, "r": 35, "t": 70, "b": 65},
    )
    return figure


def parameter_figure(frame: pd.DataFrame, metric: str) -> go.Figure:
    figure = go.Figure()
    for (offset, sigma), group in frame.groupby(["offset", "sigma"], sort=True):
        group = group.sort_values("quantile")
        figure.add_trace(go.Scatter(
            x=group["quantile"], y=group[metric], mode="lines+markers",
            name=f"offset={offset}, sigma={sigma}", customdata=group[["method"]],
            hovertemplate="%{customdata[0]}<br>quantile %{x}<br>score %{y:.4f}<extra></extra>",
        ))
    figure.update_layout(
        title=f"TAAS parameters versus {metric.replace('_', ' ')}",
        template="plotly_white", height=560, xaxis_title="Quantile",
        yaxis_title=metric.replace("_", " "),
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 65, "r": 35, "t": 95, "b": 65},
    )
    return figure


def results_table(frame: pd.DataFrame, rank_by: str) -> str:
    ranked = rank_frame(frame, rank_by)
    columns = (
        "method", "offset", "sigma", "quantile", "threshold", "tp", "tn", "fp", "fn",
        "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
        "auc", "separation_score", "saved_rank",
    )
    numeric = set(columns) - {"method"}
    headers = ["Rank", *[column.replace("_", " ") for column in columns], "Artifacts"]
    head = []
    for index, label in enumerate(headers):
        key = label.replace(" ", "_")
        is_number = label == "Rank" or key in numeric
        first = "asc" if key in LOWER_IS_BETTER or label == "Rank" else "desc"
        head.append(
            f'<th data-column="{index}" data-type="{"number" if is_number else "text"}" '
            f'data-first="{first}">{html.escape(label)}</th>'
        )
    body = []
    for index, row in ranked.iterrows():
        rank = index + 1
        if row.source_plot:
            rank_cell = (
                f'<td><a class="rank-link" href="#{plot_anchor(row.method)}" '
                f'title="Open calibration plot for {html.escape(str(row.method))}">'
                f'{rank}</a></td>'
            )
        else:
            rank_cell = f'<td title="Calibration plot unavailable">{rank}</td>'
        cells = [rank_cell]
        for column in columns:
            value = row[column]
            rendered = str(value) if column == "method" else _metric(value, 6 if column == "threshold" else 4)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        links = [f'<a href="{row.source_json.as_uri()}">JSON</a>']
        if row.source_plot:
            links.append(f'<a href="{row.source_plot.as_uri()}">Plot</a>')
        cells.append(f'<td>{" · ".join(links)}</td>')
        css = "saved-winner" if row.saved_winner else ("default-winner" if index == 0 else "")
        body.append(f'<tr class="{css}">' + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table id="results"><thead><tr>'
        + "".join(head) + "</tr></thead><tbody>" + "".join(body)
        + "</tbody></table></div>"
    )


def calibration_plot_gallery(frame: pd.DataFrame, rank_by: str) -> str:
    """Render all existing calibration PNGs in default ranking order."""
    figures = []
    for index, row in rank_frame(frame, rank_by).iterrows():
        rank = index + 1
        anchor = plot_anchor(row.method)
        badges = (
            f'Balanced accuracy <b>{_metric(row.balanced_accuracy)}</b> · '
            f'F1 <b>{_metric(row.f1)}</b> · AUROC <b>{_metric(row.auc)}</b> · '
            f'FP <b>{int(row.fp)}</b> · FN <b>{int(row.fn)}</b>'
        )
        if row.source_plot:
            visual = (
                f'<a href="{row.source_plot.as_uri()}" target="_blank" '
                f'title="Open full-size calibration plot">'
                f'<img loading="lazy" src="{row.source_plot.as_uri()}" '
                f'alt="Calibration plot for {html.escape(str(row.method))}"></a>'
            )
        else:
            visual = '<div class="plot-missing">Calibration plot unavailable</div>'
        winner_badge = (
            '<span class="saved-badge">Saved winner</span>'
            if row.saved_winner else ""
        )
        figures.append(
            f'<figure class="calibration-figure" id="{anchor}">'
            f'<figcaption><div><span class="rank-badge">Rank {rank}</span>'
            f'{winner_badge}<strong>{html.escape(str(row.method))}</strong></div>'
            f'<small>{badges}</small></figcaption>{visual}'
            f'<a class="back-link" href="#results-section">↑ Back to ranking table</a>'
            '</figure>'
        )
    return "".join(figures)


def write_report(output: Path, frame: pd.DataFrame, rank_by: str) -> None:
    ranked = rank_frame(frame, rank_by)
    default_winner = ranked.iloc[0]
    saved = frame[frame["saved_winner"]]
    saved_winner = saved.iloc[0] if not saved.empty else None
    mismatch = saved_winner is not None and saved_winner.method != default_winner.method
    selection_metric = next(
        (str(value) for value in frame["selection_metric"].dropna().unique()), "unknown"
    )
    warning = ""
    if mismatch:
        warning = (
            '<div class="warning"><strong>The saved winner differs from the report winner.</strong> '
            f'The calibration saved <code>{html.escape(str(saved_winner.method))}</code> using '
            f'<code>{html.escape(selection_metric)}</code>, while ranking by '
            f'<code>{html.escape(rank_by)}</code> selects '
            f'<code>{html.escape(str(default_winner.method))}</code>. No threshold files were changed.</div>'
        )

    cards = "".join((
        _winner_card(frame, rank_by, f"Recommended by {rank_by.replace('_', ' ')}"),
        _winner_card(frame, "f1", "Best F1 balance"),
        _winner_card(frame, "auc", "Best threshold-independent AUROC"),
        _winner_card(frame, "fn", "Fewest missed anomalies"),
        _winner_card(frame, "fp", "Fewest false alarms"),
    ))
    tradeoff = tradeoff_figure(frame).to_html(
        full_html=False, include_plotlyjs=True, config={"responsive": True}
    )
    errors = error_figure(frame).to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True}
    )
    parameters = parameter_figure(frame, rank_by).to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True}
    )
    plot_gallery = calibration_plot_gallery(frame, rank_by)
    area = ", ".join(sorted(frame["safety_area"].astype(str).unique()))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TAAS calibration comparison — {html.escape(area)}</title>
<style>
:root{{--bg:#f4f7fb;--card:#fff;--ink:#172033;--muted:#64748b;--line:#dbe3ee;--accent:#2563eb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif}}
main{{max-width:1900px;margin:auto;padding:28px}} h1{{margin:0 0 6px}} .subtitle{{color:var(--muted);margin:0 0 20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:18px 0}}
.card,.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 20px #1e293b0d}}
.card{{padding:16px;display:flex;flex-direction:column;gap:5px}} .card small,.card em{{color:var(--muted);font-style:normal}}
.card strong{{font-size:1.65rem}} .card span{{font-weight:700;overflow-wrap:anywhere}} .panel{{padding:16px;margin:15px 0}}
.warning{{padding:14px 16px;border:1px solid #f59e0b;background:#fffbeb;border-radius:12px;margin:16px 0}}
details{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:13px 16px;margin:15px 0}} summary{{font-weight:700;cursor:pointer}}
.plots{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}} .plots .wide{{grid-column:1/-1}}
.table-wrap{{overflow:auto;max-height:760px;border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:separate;border-spacing:0;width:100%;font-size:.86rem}} th,td{{padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}} th{{position:sticky;top:0;background:#172033;color:#fff;cursor:pointer;z-index:2}}
tr.default-winner{{background:#eff6ff}} tr.saved-winner{{background:#fff7ed}} code{{background:#e8eef7;padding:2px 5px;border-radius:4px}}
.rank-link{{display:inline-flex;align-items:center;justify-content:center;min-width:30px;height:28px;border-radius:8px;background:#dbeafe;color:#1d4ed8;font-weight:800;text-decoration:none}}
.rank-link:hover{{background:#2563eb;color:#fff}} .gallery-heading{{margin:32px 0 8px}}
.plot-gallery{{display:grid;grid-template-columns:1fr;gap:20px}}
.calibration-figure{{scroll-margin-top:18px;margin:0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 5px 20px #1e293b0d}}
.calibration-figure:target{{border:4px solid var(--accent);box-shadow:0 0 0 5px #2563eb22}}
.calibration-figure figcaption{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:12px}}
.calibration-figure figcaption div{{display:flex;align-items:center;gap:9px;flex-wrap:wrap}} .calibration-figure figcaption strong{{font-size:1.12rem}}
.calibration-figure figcaption small{{color:var(--muted);text-align:right}} .calibration-figure img{{display:block;width:100%;height:auto;border-radius:9px;border:1px solid var(--line)}}
.rank-badge,.saved-badge{{display:inline-block;padding:5px 9px;border-radius:999px;font-size:.78rem;font-weight:800}} .rank-badge{{background:#dbeafe;color:#1d4ed8}} .saved-badge{{background:#ffedd5;color:#c2410c}}
.back-link{{display:inline-block;margin-top:10px;color:var(--accent);font-weight:650;text-decoration:none}} .plot-missing{{padding:50px;text-align:center;background:#f1f5f9;color:var(--muted);border-radius:9px}}
@media(max-width:1000px){{.plots{{grid-template-columns:1fr}} .plots .wide{{grid-column:auto}} main{{padding:15px}}}}
</style></head><body><main>
<h1>TAAS calibration comparison — {html.escape(area)}</h1>
<p class="subtitle">{len(frame)} combinations · default ranking: {html.escape(rank_by.replace('_', ' '))} · saved selection metric: {html.escape(selection_metric)}</p>
{warning}<section class="cards">{cards}</section>
<details><summary>How to interpret this comparison</summary>
<p><b>Balanced accuracy</b> equally weights anomaly recall and normal-frame specificity, so class imbalance does not dominate. <b>F1</b> balances precision and recall but does not use true negatives. <b>AUROC</b> compares score ordering independently of the selected threshold. For safety, inspect false negatives and recall; for alert burden, inspect false positives and precision.</p>
<p><b>Separation score</b> is the legacy value previously called <code>binormal_auc</code>. It is an unbounded effect-size-like distance between correctly classified normal and anomalous scores, not an AUC. It should not be interpreted on a 0–1 scale.</p>
<p>The red-outlined point in the first chart is the winner saved by calibration. This report does not modify that saved choice.</p></details>
<section class="plots"><div class="panel">{tradeoff}</div><div class="panel">{errors}</div><div class="panel wide">{parameters}</div></section>
<section class="panel" id="results-section"><h2>Sortable results</h2><p>Click a column heading to sort for a different operating objective. Click a value in the <b>Rank</b> column to jump to that method's calibration plot.</p>{results_table(frame, rank_by)}</section>
<h2 class="gallery-heading">Calibration plots in default rank order</h2>
<p class="subtitle">Plots are linked from the existing calibration artifacts and are not embedded in this HTML. Click an image to open it at full size.</p>
<section class="plot-gallery">{plot_gallery}</section>
</main><script>
document.querySelectorAll('#results th').forEach(th=>th.addEventListener('click',()=>{{
 const table=th.closest('table'), body=table.tBodies[0], rows=[...body.rows], col=Number(th.dataset.column);
 const previous=table.dataset.sortColumn, initial=th.dataset.first||'asc';
 const direction=previous===String(col)&&table.dataset.sortDirection===initial?(initial==='asc'?'desc':'asc'):initial;
 rows.sort((a,b)=>{{let av=a.cells[col].textContent.trim(),bv=b.cells[col].textContent.trim();
  if(th.dataset.type==='number'){{av=parseFloat(av);bv=parseFloat(bv);av=Number.isNaN(av)?Infinity:av;bv=Number.isNaN(bv)?Infinity:bv;return direction==='asc'?av-bv:bv-av}}
  return direction==='asc'?av.localeCompare(bv):bv.localeCompare(av)}});
 rows.forEach(row=>body.appendChild(row)); table.dataset.sortColumn=String(col);table.dataset.sortDirection=direction;
}}));
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    frame, scenario_dir = load_combinations(args.calibration_dir)
    output = (args.output or scenario_dir / "calibration_comparison.html").expanduser().resolve()
    write_report(output, frame, args.rank_by)
    best = winner(frame, args.rank_by)
    print(f"Compared combinations: {len(frame)}")
    print(f"Best by {args.rank_by}: {best.method} ({float(best[args.rank_by]):.6f})")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
