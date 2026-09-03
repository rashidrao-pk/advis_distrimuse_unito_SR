#!/usr/bin/env python3
"""Plot validation anomaly-score timelines for every safety area."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_AREAS = ("PLeft", "PRight", "RoboArm", "ConvBelt")
REQUIRED_COLUMNS = {"file_name", "anomaly_score"}


def parse_args():
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot validation anomaly-score timelines by safety area."
    )
    parser.add_argument("--dataset_version", default="V6")
    parser.add_argument(
        "--threshold_root",
        type=Path,
        default=None,
        help="Threshold results root (default: results/<dataset_version>/thresholds).",
    )
    parser.add_argument(
        "--safety_areas",
        nargs="+",
        default=list(DEFAULT_AREAS),
        help="Safety areas to plot.",
    )
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=25,
        help="Centered rolling-mean window; use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Combined PNG path (default: <threshold_root>/val_scores_timeline.png).",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also save one timeline PNG in each safety-area directory.",
    )
    args = parser.parse_args()
    args.threshold_root = (
        args.threshold_root
        or repository_root / "results" / args.dataset_version / "thresholds"
    ).expanduser().resolve()
    args.output = (
        args.output or args.threshold_root / "val_scores_timeline.png"
    ).expanduser().resolve()
    if args.rolling_window < 1:
        parser.error("--rolling_window must be at least 1")
    return args


def score_filename(area, args):
    return (
        f"val_scores_{area}_off{args.offset}_"
        f"sig{args.sigma}_q{args.quantile}.csv"
    )


def load_area_data(area, args):
    area_dir = args.threshold_root / area
    csv_path = area_dir / score_filename(area, args)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Validation score CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    frame["anomaly_score"] = pd.to_numeric(frame["anomaly_score"], errors="raise")

    threshold = None
    threshold_path = area_dir / f"threshold_{area}.json"
    if threshold_path.is_file():
        with threshold_path.open("r", encoding="utf-8") as stream:
            threshold = json.load(stream).get("threshold")
        if threshold is not None:
            threshold = float(threshold)
    return frame, threshold, csv_path


def plot_area(ax, area, frame, threshold, rolling_window):
    x_values = range(len(frame))
    scores = frame["anomaly_score"]
    ax.plot(x_values, scores, color="#4c78a8", alpha=0.28, linewidth=0.6,
            label="Validation score")
    if rolling_window > 1:
        rolling = scores.rolling(rolling_window, center=True, min_periods=1).mean()
        ax.plot(x_values, rolling, color="#174a7e", linewidth=1.2,
                label=f"Rolling mean ({rolling_window})")
    if threshold is not None:
        ax.axhline(threshold, color="#e45756", linestyle="--", linewidth=1.4,
                   label=f"Threshold = {threshold:.4f}")
    ax.set_title(f"{area} ({len(frame):,} validation samples)")
    ax.set_xlabel("Validation sample index")
    ax.set_ylabel("Anomaly score")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")


def main():
    args = parse_args()
    loaded = []
    for area in args.safety_areas:
        frame, threshold, csv_path = load_area_data(area, args)
        loaded.append((area, frame, threshold))
        print(f"[load] {area}: {len(frame):,} rows <- {csv_path}")

    figure, axes = plt.subplots(
        len(loaded), 1, figsize=(16, 2 * len(loaded)), squeeze=False,
        constrained_layout=True,
    )
    for ax, (area, frame, threshold) in zip(axes[:, 0], loaded):
        plot_area(ax, area, frame, threshold, args.rolling_window)
    figure.suptitle(f"Validation anomaly-score timelines — {args.dataset_version}", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[save] Combined timeline -> {args.output}")

    if args.individual:
        for area, frame, threshold in loaded:
            figure, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
            plot_area(ax, area, frame, threshold, args.rolling_window)
            output = args.threshold_root / area / f"val_scores_timeline_{area}.png"
            figure.savefig(output, dpi=180, bbox_inches="tight")
            plt.close(figure)
            print(f"[save] {area} timeline -> {output}")


if __name__ == "__main__":
    main()
