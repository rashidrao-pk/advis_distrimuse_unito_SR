#!/usr/bin/env python3
"""Assess training usefulness of extracted safety-area sequences by frame motion."""

import argparse
import csv
from html import escape
from pathlib import Path
import sys

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

UTILS_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(UTILS_DIR))
from utils import motion_score  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_area_directories(scenario_path, camera):
    processed = scenario_path / camera / "processed"
    if not processed.is_dir():
        raise FileNotFoundError(f"Processed safety-area directory not found: {processed}")
    return sorted(path for path in processed.iterdir() if path.is_dir())


def classify_area(frame_count, active_fraction, p95_score, args):
    if frame_count < args.min_frames:
        return (
            "DO_NOT_USE",
            f"Only {frame_count} readable frames; minimum is {args.min_frames}.",
        )
    if active_fraction >= args.min_active_fraction:
        return (
            "USE",
            "Contains enough temporally changing frames for sequence training.",
        )
    if p95_score >= args.limited_p95:
        return (
            "LIMITED",
            "Mostly static with occasional change; downsample static frames and retain active intervals.",
        )
    return (
        "DO_NOT_USE_AS_SEQUENCE",
        "Negligible frame-to-frame change; keep only a few representatives if a static baseline is needed.",
    )


def analyze_area(area_dir, args):
    image_paths = sorted(
        path for path in area_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    readable = []
    unreadable = []
    for path in image_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            unreadable.append(str(path))
        else:
            readable.append((path, frame))

    scores = []
    score_rows = []
    if readable:
        previous_path, previous = readable[0]
        roi_mask = np.full(previous.shape[:2], 255, dtype=np.uint8)
        for index, (current_path, current) in enumerate(readable[1:], start=1):
            if current.shape[:2] != previous.shape[:2]:
                current = cv2.resize(
                    current,
                    (previous.shape[1], previous.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            score = float(
                motion_score(
                    previous,
                    current,
                    roi_mask,
                    pixel_threshold=args.pixel_threshold,
                )
            )
            scores.append(score)
            score_rows.append({
                "area": area_dir.name,
                "pair_index": index,
                "previous_frame": previous_path.name,
                "current_frame": current_path.name,
                "motion_score": score,
                "active": score > args.motion_threshold,
            })
            previous_path, previous = current_path, current

    values = np.asarray(scores, dtype=np.float64)
    active_fraction = float(np.mean(values > args.motion_threshold)) if len(values) else 0.0
    p95_score = float(np.percentile(values, 95)) if len(values) else 0.0
    decision, recommendation = classify_area(
        len(readable), active_fraction, p95_score, args
    )
    summary = {
        "area": area_dir.name,
        "files_found": len(image_paths),
        "readable_frames": len(readable),
        "unreadable_frames": len(unreadable),
        "frame_pairs": len(values),
        "mean_motion": float(values.mean()) if len(values) else 0.0,
        "median_motion": float(np.median(values)) if len(values) else 0.0,
        "p95_motion": p95_score,
        "max_motion": float(values.max()) if len(values) else 0.0,
        "active_pairs": int(np.count_nonzero(values > args.motion_threshold)),
        "active_fraction": active_fraction,
        "zero_motion_fraction": float(np.mean(values == 0.0)) if len(values) else 0.0,
        "decision": decision,
        "recommendation": recommendation,
    }
    return summary, score_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_motion_scores(path, score_rows, summaries, motion_threshold, rolling_window=15):
    """Create a dependency-free multi-panel motion plot using OpenCV."""
    grouped = {item["area"]: [] for item in summaries}
    for row in score_rows:
        grouped[row["area"]].append(float(row["motion_score"]))

    width = 1600
    panel_height = 270
    top_margin = 70
    bottom_margin = 35
    canvas = np.full(
        (top_margin + panel_height * len(grouped) + bottom_margin, width, 3),
        250,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas, "Safety-area frame-difference motion scores", (45, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2, cv2.LINE_AA,
    )

    left, right = 105, width - 40
    summary_map = {item["area"]: item for item in summaries}
    for panel_index, (area, scores) in enumerate(grouped.items()):
        y_top = top_margin + panel_index * panel_height + 25
        y_bottom = y_top + panel_height - 65
        plot_width = right - left
        plot_height = y_bottom - y_top
        values = np.asarray(scores, dtype=np.float64)
        y_max = max(
            motion_threshold * 1.25,
            float(values.max()) * 1.08 if len(values) else motion_threshold,
        )

        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = int(y_bottom - fraction * plot_height)
            cv2.line(canvas, (left, y), (right, y), (220, 220, 220), 1)
            cv2.putText(
                canvas, f"{fraction * y_max:.3f}", (25, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA,
            )
        cv2.line(canvas, (left, y_top), (left, y_bottom), (50, 50, 50), 1)
        cv2.line(canvas, (left, y_bottom), (right, y_bottom), (50, 50, 50), 1)

        threshold_y = int(
            y_bottom - min(1.0, motion_threshold / y_max) * plot_height
        )
        cv2.line(canvas, (left, threshold_y), (right, threshold_y), (40, 40, 210), 2)

        if len(values):
            x_values = np.linspace(left, right, len(values)).astype(np.int32)
            raw_y = (
                y_bottom - np.clip(values / y_max, 0, 1) * plot_height
            ).astype(np.int32)
            raw_points = np.column_stack((x_values, raw_y)).reshape((-1, 1, 2))
            cv2.polylines(canvas, [raw_points], False, (190, 105, 35), 1, cv2.LINE_AA)

            window = min(rolling_window, len(values))
            kernel = np.ones(window, dtype=np.float64) / window
            rolling = np.convolve(values, kernel, mode="same")
            rolling_y = (
                y_bottom - np.clip(rolling / y_max, 0, 1) * plot_height
            ).astype(np.int32)
            rolling_points = np.column_stack((x_values, rolling_y)).reshape((-1, 1, 2))
            cv2.polylines(canvas, [rolling_points], False, (0, 130, 230), 2, cv2.LINE_AA)

        summary = summary_map[area]
        title = (
            f"{area} | {summary['decision']} | active={summary['active_fraction']:.1%} "
            f"| p95={summary['p95_motion']:.4f}"
        )
        cv2.putText(
            canvas, title, (left, y_top - 9), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (25, 25, 25), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "frame pair", (right - 100, y_bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA,
        )

    legend_y = canvas.shape[0] - 12
    cv2.putText(
        canvas,
        f"blue: raw score   orange: rolling mean ({rolling_window})   red: active threshold ({motion_threshold})",
        (left, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 45), 1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to save motion plot: {path}")


def detect_peak_events(score_rows, args):
    grouped = {}
    for row in score_rows:
        grouped.setdefault(row["area"], []).append(row)

    events = []
    for area, rows in grouped.items():
        values = np.asarray([float(row["motion_score"]) for row in rows])
        active_indices = np.flatnonzero(values > args.motion_threshold)
        intervals = []
        if len(active_indices):
            start = previous = int(active_indices[0])
            for index in active_indices[1:]:
                index = int(index)
                if index - previous > args.peak_distance:
                    intervals.append((start, previous))
                    start = index
                previous = index
            intervals.append((start, previous))

        candidates = []
        for start, end in intervals:
            peak_index = start + int(np.argmax(values[start:end + 1]))
            if values[peak_index] >= args.peak_threshold:
                candidates.append((peak_index, start, end))
        selected = sorted(
            sorted(candidates, key=lambda item: values[item[0]], reverse=True)[
                :args.max_peaks_per_area
            ],
            key=lambda item: item[0],
        )

        for peak_index, start, end in selected:
            duration = end - start + 1
            strength = values[peak_index] / max(args.motion_threshold, 1e-12)
            if duration <= 2:
                pattern = "brief transient change"
            elif duration <= 15:
                pattern = "short movement episode"
            else:
                pattern = "sustained movement episode"
            if strength >= 5:
                pattern += " with a strong visual change"
            events.append({
                "area": area,
                "peak_pair": int(rows[peak_index]["pair_index"]),
                "peak_score": float(values[peak_index]),
                "start_pair": int(rows[start]["pair_index"]),
                "end_pair": int(rows[end]["pair_index"]),
                "active_duration_pairs": duration,
                "previous_frame": rows[start]["previous_frame"],
                "peak_frame": rows[peak_index]["current_frame"],
                "after_frame": rows[end]["current_frame"],
                "interpretation": pattern,
            })
    return events


def attach_dataset_image_paths(scenario_path, camera, events):
    """Reference original dataset images; never copy them into the report."""
    for event in events:
        source_dir = scenario_path / camera / "processed" / event["area"]
        event["previous_image_path"] = str(
            (source_dir / event["previous_frame"]).resolve()
        )
        event["peak_image_path"] = str(
            (source_dir / event["peak_frame"]).resolve()
        )
        event["after_image_path"] = str(
            (source_dir / event["after_frame"]).resolve()
        )


def write_interactive_plot(
    path, score_rows, summaries, events, args, scenario_path, camera
):
    areas = [item["area"] for item in summaries]
    grouped = {area: [row for row in score_rows if row["area"] == area] for area in areas}
    figure = make_subplots(
        rows=len(areas), cols=1, shared_xaxes=False,
        subplot_titles=[f"{item['area']} — {item['decision']}" for item in summaries],
        vertical_spacing=min(0.08, 0.25 / max(1, len(areas))),
    )
    for row_number, area in enumerate(areas, start=1):
        rows = grouped[area]
        x = [int(row["pair_index"]) for row in rows]
        y = np.asarray([float(row["motion_score"]) for row in rows])
        source_dir = scenario_path / camera / "processed" / area
        custom = [[
            "frame pair",
            row["current_frame"],
            (source_dir / row["previous_frame"]).resolve().as_uri(),
            (source_dir / row["current_frame"]).resolve().as_uri(),
        ] for row in rows]
        window = min(args.rolling_window, len(y)) if len(y) else 1
        rolling = np.convolve(y, np.ones(window) / window, mode="same") if len(y) else y
        figure.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=f"{area} raw", line={"width": 1},
            customdata=custom,
            hovertemplate=(
                "Pair %{x}<br>Score %{y:.6f}<br>Current: %{customdata[1]}"
                "<br>Hover preview uses original dataset images<extra></extra>"
            ),
        ), row=row_number, col=1)
        figure.add_trace(go.Scatter(
            x=x, y=rolling, mode="lines", name=f"{area} rolling mean",
            line={"width": 2, "color": "orange"}, hovertemplate="Pair %{x}<br>Rolling %{y:.6f}<extra></extra>",
        ), row=row_number, col=1)
        figure.add_hline(
            y=args.motion_threshold, line_dash="dash", line_color="red",
            annotation_text="active threshold", row=row_number, col=1,
        )
        area_events = [event for event in events if event["area"] == area]
        if area_events:
            figure.add_trace(go.Scatter(
                x=[event["peak_pair"] for event in area_events],
                y=[event["peak_score"] for event in area_events],
                mode="markers", name=f"{area} peaks",
                marker={"size": 11, "color": "crimson", "symbol": "diamond"},
                customdata=[[
                    event["interpretation"], event["peak_frame"],
                    Path(event["previous_image_path"]).as_uri(),
                    Path(event["peak_image_path"]).as_uri(),
                ] for event in area_events],
                hovertemplate=(
                    "Peak pair %{x}<br>Score %{y:.6f}<br>%{customdata[0]}"
                    "<br>Peak frame: %{customdata[1]}<extra></extra>"
                ),
            ), row=row_number, col=1)
        figure.update_xaxes(title_text="Frame pair", row=row_number, col=1)
        figure.update_yaxes(title_text="Motion", row=row_number, col=1)
    figure.update_layout(
        title="Interactive safety-area motion analysis",
        height=max(500, 360 * len(areas)), hovermode="x unified",
        template="plotly_white", legend={"orientation": "h"},
    )
    figure.write_html(
        str(path), include_plotlyjs=True, full_html=True, div_id="motion-plot"
    )
    html = path.read_text(encoding="utf-8")
    preview = r'''
<style>
#motion-image-preview {position:absolute;z-index:10000;background:rgba(20,20,20,.94);color:white;padding:12px;border-radius:8px;box-shadow:0 3px 16px #555;display:none;max-width:590px;font-family:sans-serif;pointer-events:none}
#motion-image-preview .images {display:flex;gap:10px}
#motion-image-preview img {width:270px;height:270px;object-fit:contain;background:#000;border:1px solid #777}
#motion-image-preview .label {font-size:13px;margin:5px 0}
</style>
<div id="motion-image-preview"><div id="motion-preview-title" class="label"></div><div class="images"><div><div class="label">Previous / before</div><img id="motion-preview-before"></div><div><div class="label">Current / peak</div><img id="motion-preview-current"></div></div></div>
<script>
(function () {
  const plot = document.getElementById('motion-plot');
  const preview = document.getElementById('motion-image-preview');
  const title = document.getElementById('motion-preview-title');
  const before = document.getElementById('motion-preview-before');
  const current = document.getElementById('motion-preview-current');
  let mouseX = 20;
  let mouseY = 20;
  function placePreview() {
    const previewWidth = 590;
    const previewHeight = 330;
    const pageWidth = document.documentElement.scrollWidth;
    const pageHeight = document.documentElement.scrollHeight;
    let left = mouseX + 22;
    let top = mouseY + 18;
    if (left + previewWidth > pageWidth - 12) left = mouseX - previewWidth - 22;
    if (top + previewHeight > pageHeight - 12) top = mouseY - previewHeight - 18;
    preview.style.left = Math.max(12, left) + 'px';
    preview.style.top = Math.max(12, top) + 'px';
  }
  document.addEventListener('mousemove', function (event) {
    mouseX = event.pageX;
    mouseY = event.pageY;
    if (preview.style.display === 'block') placePreview();
  });
  plot.on('plotly_hover', function (event) {
    const point = event.points.find(p => p.customdata && p.customdata.length >= 4);
    if (!point) return;
    title.textContent = point.customdata[0] + ' — ' + point.customdata[1];
    before.src = point.customdata[2];
    current.src = point.customdata[3];
    placePreview();
    preview.style.display = 'block';
  });
  plot.on('plotly_unhover', function () { preview.style.display = 'none'; });
})();
</script>
'''
    path.write_text(html.replace("</body>", preview + "\n</body>"), encoding="utf-8")


def write_markdown(
    path, scenario_path, camera, summaries, args, plot_filename,
    interactive_filename, events,
):
    lines = [
        f"# Safety-area motion report: {scenario_path.name}",
        "",
        f"- Scenario path: `{scenario_path}`",
        f"- Camera: `{camera}`",
        f"- Motion implementation: `scripts/src/utils.py::motion_score`",
        f"- Pixel-change threshold: `{args.pixel_threshold}`",
        f"- Active-pair threshold: motion score `> {args.motion_threshold}`",
        f"- USE threshold: active fraction `>= {args.min_active_fraction:.1%}`",
        f"- LIMITED threshold: 95th-percentile motion `>= {args.limited_p95}`",
        "",
        "## Motion plot",
        "",
        f"![Motion scores]({plot_filename})",
        "",
        f"[Open interactive Plotly chart]({interactive_filename})",
        "",
        "## Results",
        "",
        "| Area | Frames | Mean motion | P95 motion | Active pairs | Zero motion | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['area']} | {item['readable_frames']} | "
            f"{item['mean_motion']:.6f} | {item['p95_motion']:.6f} | "
            f"{item['active_fraction']:.1%} | {item['zero_motion_fraction']:.1%} | "
            f"**{item['decision']}** |"
        )
    lines.extend(["", "## Recommendations", ""])
    for item in summaries:
        lines.append(
            f"- **{item['area']} — {item['decision']}:** {item['recommendation']}"
        )
    lines.extend(["", "## Detected peak events", ""])
    if events:
        lines.extend([
            "| Area | Peak pair | Score | Active interval | Interpretation | Evidence |",
            "|---|---:|---:|---:|---|---|",
        ])
        for event in events:
            lines.append(
                f"| {event['area']} | {event['peak_pair']} | {event['peak_score']:.6f} | "
                f"{event['start_pair']}–{event['end_pair']} | {event['interpretation']} | "
                f"[peak frame]({event['peak_image_path']}) |"
            )
    else:
        lines.append("No peaks met the configured detection threshold.")
    lines.extend([
        "",
        "## Interpretation note",
        "",
        "This is a temporal-information screening heuristic, not a semantic-label guarantee. "
        "A static area can still contribute a small number of normal baseline images, but "
        "keeping hundreds of near-identical frames would over-represent that appearance.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate frame-difference training-usefulness reports."
    )
    parser.add_argument(
        "scenario_path", type=Path, nargs="?",
        help="One extracted scenario. Omit to analyze every discovered scenario.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/cf_dataset_mac.yaml"))
    parser.add_argument(
        "--scenarios-root", type=Path,
        help="Batch input root. Default: <config data.dataset_base>/extracted_frames.",
    )
    parser.add_argument("--camera", default="back_view")
    parser.add_argument("--areas", nargs="+", help="Analyze only selected areas.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Single output directory, or batch output root when no scenario is given.",
    )
    parser.add_argument("--pixel-threshold", type=int, default=20)
    parser.add_argument("--motion-threshold", type=float, default=0.01)
    parser.add_argument("--min-active-fraction", type=float, default=0.05)
    parser.add_argument("--limited-p95", type=float, default=0.005)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument(
        "--peak-threshold", type=float, default=0.02,
        help="Minimum motion score for an automatically detected peak.",
    )
    parser.add_argument(
        "--peak-distance", type=int, default=15,
        help="Merge active motion intervals separated by at most this many frame pairs.",
    )
    parser.add_argument("--max-peaks-per-area", type=int, default=10)
    parser.add_argument("--rolling-window", type=int, default=15)
    return parser.parse_args()


def analyze_scenario(args, scenario_path, output_dir):
    scenario_path = scenario_path.expanduser().resolve()
    scenario_id = scenario_path.name

    area_dirs = find_area_directories(scenario_path, args.camera)
    if args.areas:
        selected = set(args.areas)
        area_dirs = [path for path in area_dirs if path.name in selected]
        missing = selected - {path.name for path in area_dirs}
        if missing:
            raise ValueError(f"Safety-area folder(s) not found: {', '.join(sorted(missing))}")
    if not area_dirs:
        raise ValueError("No safety-area folders selected")

    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_scores = []
    for area_dir in area_dirs:
        summary, scores = analyze_area(area_dir, args)
        summaries.append(summary)
        all_scores.extend(scores)
        print(
            f"{summary['area']}: {summary['decision']} | "
            f"active={summary['active_fraction']:.1%} | "
            f"p95={summary['p95_motion']:.6f}"
        )

    write_csv(output_dir / f"{scenario_id}_summary.csv", summaries, list(summaries[0]))
    score_fields = [
        "area", "pair_index", "previous_frame", "current_frame",
        "motion_score", "active",
    ]
    print('='*100)
    print(f"Analyzing scenario: {scenario_id}")

    write_csv(output_dir / f"{scenario_id}_frame_scores.csv", all_scores, score_fields)
    plot_path = output_dir / f"{scenario_id}_motion_scores.png"
    plot_motion_scores(
        plot_path,
        all_scores,
        summaries,
        args.motion_threshold,
        rolling_window=args.rolling_window,
    )
    events = detect_peak_events(all_scores, args)
    attach_dataset_image_paths(scenario_path, args.camera, events)
    peak_fields = [
        "area", "peak_pair", "peak_score", "start_pair", "end_pair",
        "active_duration_pairs", "previous_frame", "peak_frame", "after_frame",
        "interpretation", "previous_image_path", "peak_image_path",
        "after_image_path",
    ]
    write_csv(output_dir / f"{scenario_id}_peak_events.csv", events, peak_fields)
    interactive_path = output_dir / f"{scenario_id}_motion_interactive.html"
    write_interactive_plot(
        interactive_path, all_scores, summaries, events, args,
        scenario_path, args.camera,
    )
    write_markdown(
        output_dir / f"{scenario_id}_report.md",
        scenario_path,
        args.camera,
        summaries,
        args,
        plot_path.name,
        interactive_path.name,
        events,
    )
    print(f"Report written to: {output_dir.resolve()}")
    return {
        "scenario_id": scenario_id,
        "output_dir": output_dir.resolve(),
        "interactive_path": interactive_path.resolve(),
        "report_path": (output_dir / f"{scenario_id}_report.md").resolve(),
        "summaries": summaries,
        "event_count": len(events),
    }


def scenario_sort_key(path):
    try:
        return tuple(int(part) for part in path.name.split("_"))
    except ValueError:
        return (sys.maxsize, path.name)


def write_combined_html(path, results, camera):
    sections = []
    navigation = []
    for result in results:
        scenario_id = result["scenario_id"]
        anchor = f"scenario-{scenario_id.replace('_', '-')}"
        navigation.append(f'<a href="#{anchor}">{escape(scenario_id)}</a>')
        rows = "".join(
            "<tr>"
            f"<td>{escape(item['area'])}</td>"
            f"<td>{item['readable_frames']}</td>"
            f"<td>{item['active_fraction']:.1%}</td>"
            f"<td>{item['p95_motion']:.6f}</td>"
            f"<td>{escape(item['decision'])}</td>"
            "</tr>"
            for item in result["summaries"]
        )
        interactive_relative = result["interactive_path"].relative_to(path.parent)
        report_relative = result["report_path"].relative_to(path.parent)
        iframe_height = max(560, 370 * len(result["summaries"]))
        sections.append(f"""
<section id="{anchor}">
  <h2>Scenario {escape(scenario_id)}</h2>
  <p>Detected peak events: {result['event_count']} · <a href="{report_relative}">Markdown report</a></p>
  <table><thead><tr><th>Area</th><th>Frames</th><th>Active</th><th>P95 motion</th><th>Decision</th></tr></thead><tbody>{rows}</tbody></table>
  <iframe src="{interactive_relative}" loading="lazy" style="height:{iframe_height}px"></iframe>
</section>""")
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>All scenario motion reports</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1d2733}} nav{{position:sticky;top:0;background:#17212b;padding:12px;z-index:2}} nav a{{color:white;margin:0 10px;text-decoration:none}} main{{max-width:1700px;margin:auto;padding:22px}} section{{background:white;margin:20px 0;padding:18px;border-radius:10px;box-shadow:0 2px 9px #ccd}} table{{border-collapse:collapse;margin-bottom:14px}} th,td{{border:1px solid #ccd;padding:7px 11px;text-align:left}} iframe{{width:100%;border:1px solid #ccd;border-radius:6px}}
</style></head><body><nav>{' '.join(navigation)}</nav><main>
<h1>Safety-area motion analysis — all scenarios</h1><p>Camera: {escape(camera)} · Scenarios: {len(results)}</p>
{''.join(sections)}
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    default_output_root = Path("reports") / "safety_area_motion"
    if args.scenario_path is not None:
        scenario_path = args.scenario_path.expanduser().resolve()
        output_dir = args.output_dir or (
            default_output_root / f"{scenario_path.name}_{args.camera}"
        )
        analyze_scenario(args, scenario_path, output_dir)
        return

    if args.scenarios_root:
        scenarios_root = args.scenarios_root.expanduser().resolve()
    else:
        with args.config.expanduser().open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        dataset_base = config.get("data", {}).get("dataset_base")
        if not dataset_base:
            raise ValueError("Config does not define data.dataset_base")
        scenarios_root = (Path(dataset_base).expanduser() / "extracted_frames").resolve()
    scenario_paths = sorted(
        (
            path for path in scenarios_root.iterdir()
            if path.is_dir() and (path / args.camera / "processed").is_dir()
        ),
        key=scenario_sort_key,
    )
    if not scenario_paths:
        raise ValueError(
            f"No scenarios with {args.camera}/processed found under {scenarios_root}"
        )

    output_root = (args.output_dir or default_output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Batch mode: analyzing {len(scenario_paths)} scenarios from {scenarios_root}")
    results = []
    for index, scenario_path in enumerate(scenario_paths, start=1):
        print(f"\n[{index}/{len(scenario_paths)}] Scenario {scenario_path.name}")
        results.append(analyze_scenario(
            args,
            scenario_path,
            output_root / f"{scenario_path.name}_{args.camera}",
        ))
    combined_path = output_root / f"all_scenarios_{args.camera}.html"
    write_combined_html(combined_path, results, args.camera)
    print(f"Combined HTML written to: {combined_path}")


if __name__ == "__main__":
    main()
