#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError as exc:
    raise ImportError(
        "tqdm is required for verbose progress. Install it with: pip install tqdm"
    ) from exc

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
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/V6/e1_spatial_risk/convlstm_e1.pt"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help=(
            "Verbosity: 0=minimal, 1=epoch summaries, 2=batch progress, "
            "3=high debug including file/dataloader diagnostics"
        ),
    )
    return p.parse_args()


def log(message: str, args, level: int = 1):
    if args.verbose >= level:
        print(message, flush=True)


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


def print_cuda_info(device, args):
    if device.type != "cuda":
        return

    log(f"[device] CUDA available      : {torch.cuda.is_available()}", args, 1)
    log(f"[device] CUDA version        : {torch.version.cuda}", args, 1)
    log(f"[device] GPU count           : {torch.cuda.device_count()}", args, 1)
    log(f"[device] GPU                 : {torch.cuda.get_device_name(0)}", args, 1)

    if args.verbose >= 3:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        log(f"[device] GPU memory          : {total_gb:.2f} GB", args, 3)
        log(f"[device] compute capability  : {props.major}.{props.minor}", args, 3)


def build_dataset(files, history, horizons, stride, name, args):
    log(f"\n[data] Building {name} dataset...", args, 1)

    if args.verbose >= 3:
        for idx, path_str in enumerate(files, 1):
            p = Path(path_str)
            exists = p.is_file()
            size_mb = p.stat().st_size / (1024 ** 2) if exists else float("nan")
            log(
                f"[data:{name}] {idx:02d}/{len(files):02d} "
                f"exists={exists} size={size_mb:.1f} MB path={p}",
                args,
                3,
            )

    start = time.perf_counter()
    ds = SpatialRiskWindowDataset(files, history, tuple(horizons), stride)
    elapsed = time.perf_counter() - start

    log(
        f"[data] {name.capitalize()} dataset ready: "
        f"{len(ds):,} windows in {elapsed:.2f}s",
        args,
        1,
    )
    return ds


def build_loader(dataset, batch_size, shuffle, num_workers, pin_memory, name, args):
    log(
        f"[loader] Creating {name} DataLoader | "
        f"batch={batch_size} workers={num_workers} shuffle={shuffle} "
        f"pin_memory={pin_memory}",
        args,
        1,
    )

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # persistent_workers avoids worker restart overhead every epoch on HPC.
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    start = time.perf_counter()
    loader = DataLoader(**kwargs)
    elapsed = time.perf_counter() - start

    log(
        f"[loader] {name.capitalize()} DataLoader ready: "
        f"{len(loader):,} batches in {elapsed:.3f}s",
        args,
        1,
    )
    return loader


def run_epoch(
    model,
    loader,
    device,
    args,
    epoch,
    optimizer=None,
    show_first_batch=False,
):
    train = optimizer is not None
    model.train(train)

    mode = "train" if train else "val"
    total = 0.0
    count = 0

    if args.verbose >= 2:
        iterator = tqdm(
            loader,
            desc=f"{mode.upper():5s} {epoch:03d}/{args.epochs:03d}",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        )
    else:
        iterator = loader

    first_batch_seen = False
    batch_start = time.perf_counter()

    for batch_idx, batch in enumerate(iterator, start=1):
        if args.verbose >= 3 and batch_idx == 1:
            wait = time.perf_counter() - batch_start
            log(
                f"[{mode}] first batch received after {wait:.2f}s",
                args,
                3,
            )

        x = batch["x"].to(device, non_blocking=device.type == "cuda")
        y = batch["y"].to(device, non_blocking=device.type == "cuda")

        if show_first_batch and not first_batch_seen:
            log("\n[first batch]", args, 1)
            log(f"  x shape  : {tuple(x.shape)}", args, 1)
            log(f"  y shape  : {tuple(y.shape)}", args, 1)
            log(f"  x dtype  : {x.dtype}", args, 1)
            log(f"  y dtype  : {y.dtype}", args, 1)
            log(f"  x device : {x.device}", args, 1)
            log(f"  y device : {y.device}", args, 1)
            log(f"  x range  : [{x.min().item():.4f}, {x.max().item():.4f}]", args, 1)
            log(f"  y range  : [{y.min().item():.4f}, {y.max().item():.4f}]", args, 1)
            first_batch_seen = True

        if train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = weighted_loss(pred, y)

        if train:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            grad_norm = None

        batch_size = x.size(0)
        total += float(loss.detach()) * batch_size
        count += batch_size
        running_loss = total / max(count, 1)

        if args.verbose >= 2:
            postfix = {
                "loss": f"{loss.item():.5f}",
                "avg": f"{running_loss:.5f}",
            }
            if train and grad_norm is not None:
                postfix["grad"] = f"{float(grad_norm):.3f}"
            if device.type == "cuda":
                mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                postfix["gpu_mem"] = f"{mem_gb:.2f}G"
            iterator.set_postfix(postfix)

    return total / max(count, 1)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "=" * 80, flush=True)
    print("E1 Spatial Risk Forecasting — ConvLSTM Training", flush=True)
    print("=" * 80, flush=True)
    log(f"[config] train files        : {len(args.train)}", args, 1)
    log(f"[config] validation files   : {len(args.val)}", args, 1)
    log(f"[config] history            : {args.history}", args, 1)
    log(f"[config] horizons           : {args.horizons}", args, 1)
    log(f"[config] window stride      : {args.window_stride}", args, 1)
    log(f"[config] hidden channels    : {args.hidden}", args, 1)
    log(f"[config] epochs             : {args.epochs}", args, 1)
    log(f"[config] batch size         : {args.batch_size}", args, 1)
    log(f"[config] learning rate      : {args.lr}", args, 1)
    log(f"[config] num workers        : {args.num_workers}", args, 1)
    log(f"[config] seed               : {args.seed}", args, 1)
    log(f"[config] output             : {args.output}", args, 1)
    log(f"[config] verbosity          : {args.verbose}", args, 1)

    device = choose_device(args.cpu)
    log(f"\n[device] Selected device    : {device}", args, 1)
    print_cuda_info(device, args)

    train_ds = build_dataset(
        args.train,
        args.history,
        args.horizons,
        args.window_stride,
        "train",
        args,
    )
    val_ds = build_dataset(
        args.val,
        args.history,
        args.horizons,
        args.window_stride,
        "validation",
        args,
    )

    pin_memory = device.type == "cuda"
    train_loader = build_loader(
        train_ds,
        args.batch_size,
        True,
        args.num_workers,
        pin_memory,
        "train",
        args,
    )
    val_loader = build_loader(
        val_ds,
        args.batch_size,
        False,
        args.num_workers,
        pin_memory,
        "validation",
        args,
    )

    log("\n[model] Building ConvLSTM...", args, 1)
    model = SpatialRiskConvLSTM(args.hidden, len(args.horizons)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] total parameters    : {total_params:,}", args, 1)
    log(f"[model] trainable parameters: {trainable_params:,}", args, 1)

    best = float("inf")
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    log("\n[train] Starting training...", args, 1)
    log(
        f"[train] train windows={len(train_ds):,} | "
        f"val windows={len(val_ds):,} | "
        f"train batches={len(train_loader):,} | "
        f"val batches={len(val_loader):,}",
        args,
        1,
    )

    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        log(f"\n[epoch {epoch:03d}/{args.epochs:03d}] starting...", args, 1)

        train_loss = run_epoch(
            model,
            train_loader,
            device,
            args,
            epoch,
            optimizer=optimizer,
            show_first_batch=(epoch == 1),
        )

        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                device,
                args,
                epoch,
                optimizer=None,
                show_first_batch=False,
            )

        epoch_time = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_seconds": epoch_time,
            }
        )

        print(
            f"[epoch {epoch:03d}/{args.epochs:03d}] "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f} "
            f"time={epoch_time:.1f}s",
            flush=True,
        )

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "config": vars(args),
                    "best_val_loss": best,
                },
                args.output,
            )
            print(
                f"[save] new best val={best:.6f} -> {args.output}",
                flush=True,
            )

        args.output.with_suffix(".history.json").write_text(
            json.dumps(history, indent=2, default=str),
            encoding="utf-8",
        )

    total_time = time.perf_counter() - training_start

    print("\n" + "=" * 80, flush=True)
    print("Training complete", flush=True)
    print(f"[result] best validation loss : {best:.6f}", flush=True)
    print(f"[result] total training time  : {total_time / 60.0:.2f} min", flush=True)
    print(f"[result] best checkpoint      : {args.output}", flush=True)
    print(
        f"[result] history              : {args.output.with_suffix('.history.json')}",
        flush=True,
    )
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
