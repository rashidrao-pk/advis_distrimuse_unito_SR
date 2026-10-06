#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SpatialRiskWindowDataset
from model import SpatialRiskConvLSTM


def parse_args():
    p = argparse.ArgumentParser("Evaluate E1 against persistence")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--history", type=int, default=15)
    p.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 15])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--risk-threshold", type=float, default=0.5)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def device_for(cpu):
    if cpu: return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def batch_metrics(pred, target, threshold):
    eps = 1e-8
    mae = (pred - target).abs().mean(dim=(-1, -2, -3))
    pb, tb = pred >= threshold, target >= threshold
    inter = (pb & tb).sum(dim=(-1, -2, -3)).float()
    union = (pb | tb).sum(dim=(-1, -2, -3)).float()
    iou = torch.where(union > 0, inter / (union + eps), torch.ones_like(union))
    return mae, iou


def main():
    args = parse_args()
    device = device_for(args.cpu)
    ds = SpatialRiskWindowDataset(args.test, args.history, tuple(args.horizons))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model = SpatialRiskConvLSTM(args.hidden, len(args.horizons)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sums = defaultdict(lambda: np.zeros(len(args.horizons), dtype=np.float64))
    n = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            pred = model(x)
            persistence = x[:, -1:].repeat(1, len(args.horizons), 1, 1, 1)
            for name, out in (("model", pred), ("persistence", persistence)):
                mae, iou = batch_metrics(out, y, args.risk_threshold)
                sums[f"{name}_mae"] += mae.sum(dim=0).cpu().numpy()
                sums[f"{name}_iou"] += iou.sum(dim=0).cpu().numpy()
            n += x.size(0)

    print(f"test windows: {n}")
    print("horizon_frames,model_MAE,persistence_MAE,model_IoU,persistence_IoU")
    for j, h in enumerate(args.horizons):
        print(
            f"{h},{sums['model_mae'][j]/n:.6f},{sums['persistence_mae'][j]/n:.6f},"
            f"{sums['model_iou'][j]/n:.6f},{sums['persistence_iou'][j]/n:.6f}"
        )


if __name__ == "__main__":
    main()
