#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SpatialRiskWindowDataset
from model import SpatialRiskConvLSTM


def parse_args():
    p = argparse.ArgumentParser("Train E1 spatial-risk ConvLSTM")
    p.add_argument("--train", nargs="+", required=True, help="Training scenario .npz files")
    p.add_argument("--val", nargs="+", required=True, help="Validation scenario .npz files")
    p.add_argument("--history", type=int, default=15)
    p.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 15])
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("results/V6/e1_spatial_risk/convlstm_e1.pt"))
    p.add_argument("--seed", type=int, default=42)
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


def weighted_loss(pred, target):
    # Dense background dominates these maps. Give high-risk pixels more weight
    # while still preserving smooth-regression behavior.
    weight = 1.0 + 4.0 * target
    return (weight * (pred - target).abs()).mean()


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = weighted_loss(pred, y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.detach()) * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = choose_device(args.cpu)
    train_ds = SpatialRiskWindowDataset(args.train, args.history, tuple(args.horizons), args.window_stride)
    val_ds = SpatialRiskWindowDataset(args.val, args.history, tuple(args.horizons), args.window_stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = SpatialRiskConvLSTM(args.hidden, len(args.horizons)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device} | train windows={len(train_ds):,} | val windows={len(val_ds):,}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": history,
                "config": vars(args),
                "best_val_loss": best,
            }, args.output)
            print(f"[save] best -> {args.output}")

    args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
