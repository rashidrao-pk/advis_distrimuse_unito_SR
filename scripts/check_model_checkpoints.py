#!/usr/bin/env python3
"""Read-only validation and summary of VAE-GAN model checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml


REQUIRED_STATE_KEYS = (
    "encoder_state_dict",
    "decoder_state_dict",
    "discriminator_state_dict",
)
OPTIMIZER_KEY_PAIRS = (
    ("optimizer_encdec_state_dict", "optimizer_d_state_dict"),
    ("optimizer_enc_state_dict", "optimizer_dec_state_dict"),
)


def history_length(history: Any) -> int:
    """Return the number of completed epochs represented by loss history."""
    if history is None:
        return 0
    try:
        return len(history)
    except TypeError:
        return 0


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Load one trusted local checkpoint and return a testable audit record."""
    path = Path(path).expanduser().resolve()
    report: dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "errors": [],
        "warnings": [],
    }
    if not path.is_file():
        report["errors"].append("checkpoint file does not exist")
        return report

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # A corrupt checkpoint must produce a report, not a crash.
        report["errors"].append(f"could not load checkpoint: {exc}")
        return report
    if not isinstance(checkpoint, dict):
        report["errors"].append("checkpoint root must be a dictionary")
        return report

    missing_states = [key for key in REQUIRED_STATE_KEYS if key not in checkpoint]
    if missing_states:
        report["errors"].append("missing model state: " + ", ".join(missing_states))

    optimizer_pair = next(
        (pair for pair in OPTIMIZER_KEY_PAIRS if all(key in checkpoint for key in pair)),
        None,
    )
    if optimizer_pair is None:
        report["warnings"].append("optimizer states are missing or incomplete")

    history = checkpoint.get("loss_history", [])
    completed_epochs = history_length(history)
    config = checkpoint.get("config") or {}
    dataset = config.get("dataset") or {}
    training = config.get("training") or {}
    params = config.get("params") or {}
    metadata_epochs = config.get("epochs_trained")
    saved_epoch_target = checkpoint.get("epoch")

    if metadata_epochs is not None and int(metadata_epochs) != completed_epochs:
        report["warnings"].append(
            f"epochs_trained metadata ({metadata_epochs}) differs from history "
            f"length ({completed_epochs})"
        )
    if saved_epoch_target is not None and int(saved_epoch_target) < completed_epochs:
        report["warnings"].append(
            f"saved epoch target ({saved_epoch_target}) is below completed epochs "
            f"({completed_epochs})"
        )

    n_train = dataset.get("n_train_images")
    n_val = dataset.get("n_val_images")
    report.update({
        "valid": not report["errors"],
        "suffix": config.get("suffix") or path.stem.removeprefix("model_"),
        "completed_epochs": completed_epochs,
        "saved_epoch_target": saved_epoch_target,
        "train_images": n_train,
        "validation_images": n_val,
        "training_image_exposures": (
            int(n_train) * completed_epochs if n_train is not None else None
        ),
        "dataset_name": dataset.get("dataset_name"),
        "training_directory": dataset.get("train_dir"),
        "batch_size": training.get("batch_size"),
        "device": training.get("device"),
        "latent_dim": params.get("latent_dim"),
        "image_size": params.get("image_size"),
        "saved_at": config.get("saved_at"),
        "optimizer_format": "new" if optimizer_pair == OPTIMIZER_KEY_PAIRS[0]
                            else "legacy" if optimizer_pair else None,
    })
    if not config:
        report["warnings"].append("checkpoint has no metadata config")
    if n_train is None:
        report["warnings"].append("number of training images is unavailable")
    return report


def checkpoint_directory(config_path: Path, override: Path | None) -> Path:
    if override is not None:
        value = override
    else:
        with Path(config_path).expanduser().open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        value = Path((config.get("models") or {}).get("checkpoints", ""))
        if not str(value):
            raise ValueError("config must define models.checkpoints")
    if not value.is_absolute():
        value = Path(__file__).resolve().parent.parent / value
    return value.expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cf_dataset_epito.yaml"))
    parser.add_argument("--checkpoints", type=Path, help="Override models.checkpoints.")
    parser.add_argument("--safety_area", nargs="+", help="Only inspect these safety areas.")
    parser.add_argument("--latent_dims", type=int, default=64)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = checkpoint_directory(args.config, args.checkpoints)
    if args.safety_area:
        paths = [root / f"model_{area}_{args.latent_dims}.pt" for area in args.safety_area]
    else:
        paths = sorted(root.glob("model_*.pt"))
    reports = [inspect_checkpoint(path) for path in paths]

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        print(f"Checkpoint directory: {root}")
        if not reports:
            print("No model checkpoints found.")
        for item in reports:
            status = "PASS" if item["valid"] else "FAIL"
            print(f"\n[{status}] {item['path']}")
            print(f"  completed epochs:       {item.get('completed_epochs', 'unknown')}")
            print(f"  saved epoch target:     {item.get('saved_epoch_target', 'unknown')}")
            print(f"  train / val images:     {item.get('train_images')} / {item.get('validation_images')}")
            print(f"  training exposures:     {item.get('training_image_exposures')}")
            print(f"  batch size / latent:    {item.get('batch_size')} / {item.get('latent_dim')}")
            print(f"  dataset:                 {item.get('dataset_name')}")
            for warning in item["warnings"]:
                print(f"  WARNING: {warning}")
            for error in item["errors"]:
                print(f"  ERROR: {error}")
    return 1 if not reports or any(not item["valid"] for item in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
