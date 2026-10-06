#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser("Evaluate E1-B spatial-risk forecasting")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test", nargs="+", required=True, help="Test scenario .npz files")
    p.add_argument("--history", type=int, default=15)
    p.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 15])
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--risk-threshold", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=1.0)
    p.add_argument("--min-anomaly-frames", type=int, default=3)
    p.add_argument("--trend-window", type=int, default=5)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/V6/e1_spatial_risk/eval_e1B"),
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


def persistence_forecast(x, n_horizons):
    return x[:, -1:].repeat(1, n_horizons, 1, 1, 1)


def linear_trend_forecast(x, horizons, trend_window=5):
    B, T, C, H, W = x.shape
    n = min(int(trend_window), T)
    recent = x[:, -n:]

    t = torch.arange(n, device=x.device, dtype=x.dtype)
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum().clamp_min(1e-8)

    y_mean = recent.mean(dim=1, keepdim=False)
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


def batch_mae(pred, target):
    return (pred - target).abs().flatten(2).mean(dim=2)


def batch_iou(pred, target, threshold=0.5):
    p = (pred >= threshold).flatten(2)
    t = (target >= threshold).flatten(2)

    intersection = (p & t).sum(dim=2).float()
    union = (p | t).sum(dim=2).float()

    out = torch.full_like(intersection, float("nan"))
    valid = union > 0
    out[valid] = intersection[valid] / union[valid]
    return out


def batch_positive_target_iou(pred, target, threshold=0.5):
    p = (pred >= threshold).flatten(2)
    t = (target >= threshold).flatten(2)

    intersection = (p & t).sum(dim=2).float()
    union = (p | t).sum(dim=2).float()
    target_positive = t.sum(dim=2) > 0

    out = torch.full_like(intersection, float("nan"))
    valid = target_positive & (union > 0)
    out[valid] = intersection[valid] / union[valid]
    return out


def safe_nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def stable_future_anomaly(norm_scores, start_idx, min_frames=3, threshold=1.0):
    x = np.asarray(norm_scores, dtype=np.float32)
    mask = x >= threshold

    if min_frames <= 1:
        return bool(mask[start_idx:].any())

    for i in range(start_idx, len(mask) - min_frames + 1):
        if mask[i:i + min_frames].all():
            return True
    return False


class ScalarScoreCache:
    def __init__(self, files):
        self.by_file = {}
        self.area_to_idx = {}

        for f in files:
            p = str(Path(f))
            d = np.load(p, allow_pickle=True)
            areas = [str(a) for a in d["areas"]]
            scores = d["scores"].astype(np.float32)
            thresholds = d["thresholds"].astype(np.float32)
            norm = scores / np.maximum(thresholds, 1e-12)

            self.by_file[p] = norm
            self.area_to_idx[p] = {a: i for i, a in enumerate(areas)}

    def sequence(self, file_path, area):
        p = str(Path(file_path))
        idx = self.area_to_idx[p][str(area)]
        return self.by_file[p][:, idx]


def classify_window(
    norm_scores,
    start,
    history,
    horizons,
    score_threshold=1.0,
    min_anomaly_frames=3,
):
    stop = int(start) + int(history)
    current_idx = stop - 1

    current_anom = norm_scores[current_idx] >= score_threshold

    future_indices = [stop + int(h) - 1 for h in horizons]
    future_indices = [i for i in future_indices if i < len(norm_scores)]
    future_values = norm_scores[future_indices] if future_indices else np.asarray([])

    if current_anom:
        return "already_anomalous"

    future_stable = stable_future_anomaly(
        norm_scores,
        start_idx=stop,
        min_frames=min_anomaly_frames,
        threshold=score_threshold,
    )

    if future_stable:
        return "pre_onset"

    if len(future_values) and np.all(future_values < score_threshold):
        return "normal_to_normal"

    return "transition_other"


def scenario_from_file(path):
    stem = Path(path).stem
    return stem.replace("scenario_", "").replace("_maps", "")


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

    # ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False,
    )
    
    config = ckpt.get("config", {})
    hidden = int(config.get("hidden", 32))

    model = SpatialRiskConvLSTM(hidden, len(args.horizons)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[model] checkpoint={args.checkpoint}")
    print(f"[model] hidden={hidden} horizons={args.horizons}")

    scalar_cache = ScalarScoreCache(args.test)
    file_paths = [str(Path(f)) for f in args.test]

    metrics = defaultdict(lambda: {
        "mae": [],
        "iou": [],
        "positive_iou": [],
        "count": 0,
    })
    category_counts = defaultdict(int)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            preds = {
                "convlstm": model(x),
                "persistence": persistence_forecast(x, len(args.horizons)),
                "linear_trend": linear_trend_forecast(
                    x, args.horizons, args.trend_window
                ),
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

            computed = {}
            for name, pred in preds.items():
                computed[name] = {
                    "mae": batch_mae(pred, y).cpu().numpy(),
                    "iou": batch_iou(pred, y, args.risk_threshold).cpu().numpy(),
                    "positive_iou": batch_positive_target_iou(
                        pred, y, args.risk_threshold
                    ).cpu().numpy(),
                }

            sample_meta = []
            for b in range(x.size(0)):
                scenario = str(scenarios[b])
                area = str(areas[b])
                start = int(starts[b])

                candidate_file = None
                for fp in file_paths:
                    sc = scenario_from_file(fp)
                    if scenario == sc or scenario in fp or sc in scenario:
                        candidate_file = fp
                        break
                if candidate_file is None:
                    raise RuntimeError(
                        f"Could not map scenario '{scenario}' to a test NPZ file"
                    )

                norm_scores = scalar_cache.sequence(candidate_file, area)
                category = classify_window(
                    norm_scores,
                    start,
                    args.history,
                    args.horizons,
                    args.score_threshold,
                    args.min_anomaly_frames,
                )
                category_counts[(scenario, area, category)] += 1
                sample_meta.append((scenario, area, category))

            for b, (scenario, area, category) in enumerate(sample_meta):
                group_pairs = [
                    ("overall", "all"),
                    ("scenario", scenario),
                    ("area", area),
                    ("scenario_area", f"{scenario}:{area}"),
                ]

                for model_name in preds:
                    for hi, horizon in enumerate(args.horizons):
                        for group_type, group_value in group_pairs:
                            for cat in ("all", category):
                                key = (
                                    group_type,
                                    group_value,
                                    cat,
                                    model_name,
                                    int(horizon),
                                )
                                metrics[key]["mae"].append(
                                    float(computed[model_name]["mae"][b, hi])
                                )
                                metrics[key]["iou"].append(
                                    float(computed[model_name]["iou"][b, hi])
                                )
                                metrics[key]["positive_iou"].append(
                                    float(computed[model_name]["positive_iou"][b, hi])
                                )
                                metrics[key]["count"] += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"[eval] {batch_idx + 1}/{len(loader)} batches")

    rows = []
    for key, vals in metrics.items():
        group_type, group_value, category, model_name, horizon = key
        rows.append({
            "group_type": group_type,
            "group_value": group_value,
            "category": category,
            "model": model_name,
            "horizon_frames": horizon,
            "horizon_seconds": horizon / args.fps,
            "n_windows": vals["count"],
            "mae": safe_nanmean(vals["mae"]),
            "iou": safe_nanmean(vals["iou"]),
            "positive_target_iou": safe_nanmean(vals["positive_iou"]),
        })

    rows.sort(key=lambda r: (
        r["group_type"],
        r["group_value"],
        r["category"],
        r["horizon_frames"],
        r["model"],
    ))

    metrics_csv = args.output_dir / "e1B_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    count_rows = [
        {
            "scenario": scenario,
            "area": area,
            "category": category,
            "n_windows": n,
        }
        for (scenario, area, category), n in sorted(category_counts.items())
    ]
    count_csv = args.output_dir / "e1B_window_categories.csv"
    with count_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "area", "category", "n_windows"],
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
        "rows": rows,
        "window_categories": count_rows,
    }
    summary_json = args.output_dir / "e1B_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

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

    print("\n=== OVERALL ===")
    print("horizon,model_MAE,persistence_MAE,trend_MAE,model_IoU,persistence_IoU,trend_IoU")
    for h in args.horizons:
        m = lookup("all", "convlstm", h)
        p = lookup("all", "persistence", h)
        t = lookup("all", "linear_trend", h)
        print(
            f"{h},"
            f"{m['mae']:.6f},{p['mae']:.6f},{t['mae']:.6f},"
            f"{m['iou']:.6f},{p['iou']:.6f},{t['iou']:.6f}"
        )

    print("\n=== PRE-ONSET ONLY ===")
    print("horizon,n_windows,model_MAE,persistence_MAE,trend_MAE,model_posIoU,persistence_posIoU,trend_posIoU")
    for h in args.horizons:
        m = lookup("pre_onset", "convlstm", h)
        p = lookup("pre_onset", "persistence", h)
        t = lookup("pre_onset", "linear_trend", h)
        if m is None:
            print(f"{h},0,NA,NA,NA,NA,NA,NA")
            continue
        print(
            f"{h},{m['n_windows']},"
            f"{m['mae']:.6f},{p['mae']:.6f},{t['mae']:.6f},"
            f"{m['positive_target_iou']:.6f},"
            f"{p['positive_target_iou']:.6f},"
            f"{t['positive_target_iou']:.6f}"
        )

    print("\n=== WINDOW CATEGORY COUNTS ===")
    totals = defaultdict(int)
    for r in count_rows:
        totals[r["category"]] += r["n_windows"]
    for category, n in sorted(totals.items()):
        print(f"{category:20s} {n:8d}")

    print(f"\n[save] {metrics_csv}")
    print(f"[save] {count_csv}")
    print(f"[save] {summary_json}")


if __name__ == "__main__":
    main()
