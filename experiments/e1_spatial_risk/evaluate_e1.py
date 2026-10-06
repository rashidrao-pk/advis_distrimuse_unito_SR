#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SpatialRiskWindowDataset
from model import SpatialRiskConvLSTM


def parse_args():
    p = argparse.ArgumentParser(
        "Evaluate E1-B2 spatial-risk forecasting with horizon-specific pre-onset windows"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test", nargs="+", required=True, help="Test scenario .npz files")
    p.add_argument("--history", type=int, default=15)
    p.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 15])
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument(
        "--risk-threshold",
        type=float,
        default=0.5,
        help="Pixel-level threshold used for IoU metrics",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=1.0,
        help="Normalized ADVIS scalar threshold for area anomaly state",
    )
    p.add_argument(
        "--min-anomaly-frames",
        type=int,
        default=3,
        help="Stable onset requires this many consecutive anomalous scalar-score frames",
    )
    p.add_argument(
        "--trend-window",
        type=int,
        default=5,
        help="Recent history frames used by the linear-trend baseline",
    )
    p.add_argument("--fps", type=float, default=5.0)

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/V6/e1_spatial_risk/eval_e1B2"),
    )
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def choose_device(cpu=False):
    if cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def persistence_forecast(x, n_horizons):
    # x: [B,T,1,H,W]
    return x[:, -1:].repeat(1, n_horizons, 1, 1, 1)


def linear_trend_forecast(x, horizons, trend_window=5):
    """
    Per-pixel least-squares extrapolation over the last N history maps.
    x: [B,T,1,H,W]
    returns: [B,K,1,H,W]
    """
    _, T, _, _, _ = x.shape
    n = min(int(trend_window), T)
    recent = x[:, -n:]

    t = torch.arange(n, device=x.device, dtype=x.dtype)
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum().clamp_min(1e-8)

    y_mean = recent.mean(dim=1)
    slope = (
        ((t - t_mean).view(1, n, 1, 1, 1)
         * (recent - y_mean.unsqueeze(1))).sum(dim=1)
        / denom
    )

    last = x[:, -1]
    preds = []
    for h in horizons:
        preds.append((last + slope * float(h)).clamp(0.0, 1.0))
    return torch.stack(preds, dim=1)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def batch_mae(pred, target):
    return (pred - target).abs().flatten(2).mean(dim=2)  # [B,K]


def batch_iou(pred, target, threshold=0.5):
    """
    IoU per sample/horizon.
    Empty-union samples are NaN rather than being counted as IoU=1.
    """
    p = (pred >= threshold).flatten(2)
    t = (target >= threshold).flatten(2)

    inter = (p & t).sum(dim=2).float()
    union = (p | t).sum(dim=2).float()

    out = torch.full_like(inter, float("nan"))
    valid = union > 0
    out[valid] = inter[valid] / union[valid]
    return out


def batch_positive_target_iou(pred, target, threshold=0.5):
    """
    IoU only when the target map itself contains threshold-positive pixels.
    """
    p = (pred >= threshold).flatten(2)
    t = (target >= threshold).flatten(2)

    inter = (p & t).sum(dim=2).float()
    union = (p | t).sum(dim=2).float()
    target_positive = t.sum(dim=2) > 0

    out = torch.full_like(inter, float("nan"))
    valid = target_positive & (union > 0)
    out[valid] = inter[valid] / union[valid]
    return out


def safe_nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def safe_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


# ---------------------------------------------------------------------
# Event / onset handling
# ---------------------------------------------------------------------

def stable_onset_frames(norm_scores, threshold=1.0, min_frames=3):
    """
    Return start frames of stable anomaly episodes.

    A stable anomaly onset at frame j means:
        score[j:j+min_frames] >= threshold

    Only the first frame of each stable contiguous episode is returned.
    """
    x = np.asarray(norm_scores, dtype=np.float32)
    abnormal = x >= threshold

    if len(abnormal) == 0:
        return []

    if min_frames <= 1:
        stable = abnormal.copy()
    else:
        stable = np.zeros_like(abnormal, dtype=bool)
        for j in range(0, len(abnormal) - min_frames + 1):
            if abnormal[j:j + min_frames].all():
                stable[j] = True

    # Only starts of stable episodes
    starts = []
    prev_stable = False
    for j, is_stable in enumerate(stable):
        if is_stable and not prev_stable:
            starts.append(j)
        prev_stable = bool(is_stable)

        # Once an episode stops being anomalous, allow a later new start.
        if j < len(abnormal) - 1 and not abnormal[j]:
            prev_stable = False

    return starts


class ScalarEventCache:
    """
    Loads normalized area scalar scores and precomputes stable anomaly onsets.
    """

    def __init__(self, files, threshold=1.0, min_frames=3):
        self.norm_by_file = {}
        self.area_to_idx = {}
        self.onsets = {}

        for f in files:
            p = str(Path(f))
            d = np.load(p, allow_pickle=True)

            areas = [str(a) for a in d["areas"]]
            scores = d["scores"].astype(np.float32)
            thresholds = d["thresholds"].astype(np.float32)
            norm = scores / np.maximum(thresholds, 1e-12)

            self.norm_by_file[p] = norm
            self.area_to_idx[p] = {a: i for i, a in enumerate(areas)}

            for area in areas:
                ai = self.area_to_idx[p][area]
                seq = norm[:, ai]
                self.onsets[(p, area)] = stable_onset_frames(
                    seq,
                    threshold=threshold,
                    min_frames=min_frames,
                )

    def sequence(self, file_path, area):
        p = str(Path(file_path))
        ai = self.area_to_idx[p][str(area)]
        return self.norm_by_file[p][:, ai]

    def onset_list(self, file_path, area):
        return self.onsets[(str(Path(file_path)), str(area))]


def horizon_category(
    norm_scores,
    onset_frames,
    start,
    history,
    horizon,
    score_threshold=1.0,
):
    """
    Strict horizon-specific category.

    Let stop = first future frame after the history window.

    already_anomalous:
        final history frame is already >= threshold

    pre_onset:
        history is currently normal AND a stable anomaly onset starts
        inside [stop, stop+horizon-1]

    transition_other:
        history is normal; no stable onset begins inside horizon, but at least
        one transient threshold crossing occurs inside the horizon

    normal_to_normal:
        history is normal and there is no threshold crossing inside horizon
    """
    stop = int(start) + int(history)
    current_idx = stop - 1

    if current_idx >= len(norm_scores):
        return "invalid", None

    if norm_scores[current_idx] >= score_threshold:
        return "already_anomalous", None

    future_end = min(stop + int(horizon) - 1, len(norm_scores) - 1)

    # Efficiently find the first stable onset >= stop.
    pos = bisect.bisect_left(onset_frames, stop)
    if pos < len(onset_frames):
        onset = onset_frames[pos]
        if onset <= future_end:
            lead_frames = onset - current_idx
            return "pre_onset", int(lead_frames)

    # No stable onset in the requested forecast horizon.
    future = norm_scores[stop:future_end + 1]
    if len(future) and np.any(future >= score_threshold):
        return "transition_other", None

    return "normal_to_normal", None


def scenario_from_file(path):
    stem = Path(path).stem
    return stem.replace("scenario_", "").replace("_maps", "")


def map_scenario_to_file(scenario, file_paths):
    scenario = str(scenario)
    for fp in file_paths:
        sc = scenario_from_file(fp)
        if scenario == sc or scenario in fp or sc in scenario:
            return fp
    raise RuntimeError(
        f"Could not map batch scenario '{scenario}' to a supplied test NPZ file"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.cpu)
    print(f"[device] {device}")

    print("[data] building test dataset...")
    test_ds = SpatialRiskWindowDataset(
        args.test,
        args.history,
        tuple(args.horizons),
        args.window_stride,
    )
    print(f"[data] test windows: {len(test_ds):,}")

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    try:
        ckpt = torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch.
        ckpt = torch.load(args.checkpoint, map_location=device)

    config = ckpt.get("config", {})
    hidden = int(config.get("hidden", 32))

    model = SpatialRiskConvLSTM(
        hidden,
        len(args.horizons),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[model] checkpoint={args.checkpoint}")
    print(f"[model] hidden={hidden} horizons={args.horizons}")

    file_paths = [str(Path(f)) for f in args.test]

    print("[events] precomputing normalized scores and stable anomaly onsets...")
    event_cache = ScalarEventCache(
        args.test,
        threshold=args.score_threshold,
        min_frames=args.min_anomaly_frames,
    )

    for fp in file_paths:
        sc = scenario_from_file(fp)
        d = np.load(fp, allow_pickle=True)
        for area in [str(a) for a in d["areas"]]:
            print(
                f"[events] scenario={sc:>5s} area={area:<8s} "
                f"stable_onsets={len(event_cache.onset_list(fp, area))}"
            )

    # Key:
    # (group_type, group_value, horizon_category, model_name, horizon_frames)
    metrics = defaultdict(lambda: {
        "mae": [],
        "iou": [],
        "positive_iou": [],
        "lead_frames": [],
        "count": 0,
    })

    # key: (scenario, area, horizon, category)
    category_counts = defaultdict(int)

    model_names = ("convlstm", "persistence", "linear_trend")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            preds = {
                "convlstm": model(x),
                "persistence": persistence_forecast(
                    x, len(args.horizons)
                ),
                "linear_trend": linear_trend_forecast(
                    x,
                    args.horizons,
                    args.trend_window,
                ),
            }

            computed = {}
            for name, pred in preds.items():
                computed[name] = {
                    "mae": batch_mae(pred, y).cpu().numpy(),
                    "iou": batch_iou(
                        pred, y, args.risk_threshold
                    ).cpu().numpy(),
                    "positive_iou": batch_positive_target_iou(
                        pred, y, args.risk_threshold
                    ).cpu().numpy(),
                }

            scenarios = batch["scenario"]
            areas = batch["area"]
            starts = batch["start"]

            if isinstance(scenarios, str):
                scenarios = [scenarios] * x.size(0)
            if isinstance(areas, str):
                areas = [areas] * x.size(0)
            if torch.is_tensor(starts):
                starts = starts.cpu().numpy().tolist()

            for b in range(x.size(0)):
                scenario = str(scenarios[b])
                area = str(areas[b])
                start = int(starts[b])

                fp = map_scenario_to_file(scenario, file_paths)
                norm_scores = event_cache.sequence(fp, area)
                onset_frames = event_cache.onset_list(fp, area)

                for hi, horizon in enumerate(args.horizons):
                    category, lead_frames = horizon_category(
                        norm_scores=norm_scores,
                        onset_frames=onset_frames,
                        start=start,
                        history=args.history,
                        horizon=horizon,
                        score_threshold=args.score_threshold,
                    )

                    if category == "invalid":
                        continue

                    category_counts[
                        (scenario, area, int(horizon), category)
                    ] += 1

                    group_pairs = [
                        ("overall", "all"),
                        ("scenario", scenario),
                        ("area", area),
                        ("scenario_area", f"{scenario}:{area}"),
                    ]

                    for model_name in model_names:
                        for group_type, group_value in group_pairs:
                            for cat in ("all", category):
                                key = (
                                    group_type,
                                    group_value,
                                    cat,
                                    model_name,
                                    int(horizon),
                                )
                                bucket = metrics[key]
                                bucket["mae"].append(
                                    float(computed[model_name]["mae"][b, hi])
                                )
                                bucket["iou"].append(
                                    float(computed[model_name]["iou"][b, hi])
                                )
                                bucket["positive_iou"].append(
                                    float(
                                        computed[model_name]["positive_iou"][b, hi]
                                    )
                                )
                                if lead_frames is not None:
                                    bucket["lead_frames"].append(
                                        int(lead_frames)
                                    )
                                bucket["count"] += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"[eval] {batch_idx + 1}/{len(loader)} batches")

    # -----------------------------------------------------------------
    # Save detailed metrics
    # -----------------------------------------------------------------

    rows = []
    for key, vals in metrics.items():
        group_type, group_value, category, model_name, horizon = key

        lead_mean_frames = safe_mean(vals["lead_frames"])
        lead_mean_sec = (
            lead_mean_frames / args.fps
            if not np.isnan(lead_mean_frames)
            else float("nan")
        )

        rows.append({
            "group_type": group_type,
            "group_value": group_value,
            "category": category,
            "model": model_name,
            "horizon_frames": horizon,
            "horizon_seconds": horizon / args.fps,
            "n_windows": vals["count"],
            "mean_lead_frames": lead_mean_frames,
            "mean_lead_seconds": lead_mean_sec,
            "mae": safe_nanmean(vals["mae"]),
            "iou": safe_nanmean(vals["iou"]),
            "positive_target_iou": safe_nanmean(vals["positive_iou"]),
        })

    rows.sort(
        key=lambda r: (
            r["group_type"],
            r["group_value"],
            r["category"],
            r["horizon_frames"],
            r["model"],
        )
    )

    metrics_csv = args.output_dir / "e1B2_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    count_rows = []
    for (scenario, area, horizon, category), n in sorted(
        category_counts.items()
    ):
        count_rows.append({
            "scenario": scenario,
            "area": area,
            "horizon_frames": horizon,
            "horizon_seconds": horizon / args.fps,
            "category": category,
            "n_windows": int(n),
        })

    counts_csv = args.output_dir / "e1B2_window_categories.csv"
    with counts_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "area",
                "horizon_frames",
                "horizon_seconds",
                "category",
                "n_windows",
            ],
        )
        writer.writeheader()
        writer.writerows(count_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "test_files": [str(x) for x in args.test],
        "history": args.history,
        "horizons": args.horizons,
        "fps": args.fps,
        "risk_threshold": args.risk_threshold,
        "score_threshold": args.score_threshold,
        "min_anomaly_frames": args.min_anomaly_frames,
        "trend_window": args.trend_window,
        "definition": {
            "pre_onset": (
                "Final history frame is normal and a stable anomaly onset "
                "begins within the evaluated forecast horizon."
            ),
            "already_anomalous": (
                "Final history frame is already above the normalized scalar "
                "ADVIS threshold."
            ),
            "normal_to_normal": (
                "History ends normal and no scalar threshold crossing occurs "
                "inside the evaluated forecast horizon."
            ),
            "transition_other": (
                "History ends normal and a transient threshold crossing occurs "
                "inside the horizon without satisfying stable-onset duration."
            ),
        },
        "rows": rows,
        "window_categories": count_rows,
    }

    summary_json = args.output_dir / "e1B2_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console headline tables
    # -----------------------------------------------------------------

    def lookup(category, model_name, horizon):
        for r in rows:
            if (
                r["group_type"] == "overall"
                and r["group_value"] == "all"
                and r["category"] == category
                and r["model"] == model_name
                and r["horizon_frames"] == horizon
            ):
                return r
        return None

    print("\n=== E1-B2 OVERALL ===")
    print(
        "horizon,"
        "model_MAE,persistence_MAE,trend_MAE,"
        "model_IoU,persistence_IoU,trend_IoU"
    )
    for h in args.horizons:
        m = lookup("all", "convlstm", h)
        p = lookup("all", "persistence", h)
        t = lookup("all", "linear_trend", h)
        print(
            f"{h},"
            f"{m['mae']:.6f},{p['mae']:.6f},{t['mae']:.6f},"
            f"{m['iou']:.6f},{p['iou']:.6f},{t['iou']:.6f}"
        )

    print("\n=== HORIZON-SPECIFIC PRE-ONSET ===")
    print(
        "horizon,horizon_sec,n_windows,mean_lead_sec,"
        "model_MAE,persistence_MAE,trend_MAE,"
        "model_posIoU,persistence_posIoU,trend_posIoU"
    )
    for h in args.horizons:
        m = lookup("pre_onset", "convlstm", h)
        p = lookup("pre_onset", "persistence", h)
        t = lookup("pre_onset", "linear_trend", h)

        if m is None:
            print(
                f"{h},{h / args.fps:.2f},0,NA,"
                "NA,NA,NA,NA,NA,NA"
            )
            continue

        print(
            f"{h},{h / args.fps:.2f},{m['n_windows']},"
            f"{m['mean_lead_seconds']:.3f},"
            f"{m['mae']:.6f},{p['mae']:.6f},{t['mae']:.6f},"
            f"{m['positive_target_iou']:.6f},"
            f"{p['positive_target_iou']:.6f},"
            f"{t['positive_target_iou']:.6f}"
        )

    print("\n=== HORIZON-SPECIFIC WINDOW COUNTS ===")
    totals = defaultdict(int)
    for r in count_rows:
        totals[(r["horizon_frames"], r["category"])] += r["n_windows"]

    for h in args.horizons:
        print(f"\n+h={h} frames ({h / args.fps:.2f}s)")
        for category in (
            "normal_to_normal",
            "pre_onset",
            "already_anomalous",
            "transition_other",
        ):
            print(
                f"  {category:20s} "
                f"{totals.get((h, category), 0):8d}"
            )

    print(f"\n[save] {metrics_csv}")
    print(f"[save] {counts_csv}")
    print(f"[save] {summary_json}")


if __name__ == "__main__":
    main()
