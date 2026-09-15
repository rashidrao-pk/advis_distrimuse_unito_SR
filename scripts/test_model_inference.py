#!/usr/bin/env python3
"""Smoke-test trained models by reconstructing normal or inference images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch
from torchvision import transforms
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils_model import Decoder, Encoder  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_AREAS = ("PLeft", "PRight", "RoboArm", "ConvBelt")


def resolve_path(value: str | Path, base: Path = REPOSITORY_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_images(root: Path, max_images: int) -> list[Path]:
    if root.is_file():
        candidates = [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    elif root.is_dir():
        candidates = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        candidates = []
    if len(candidates) <= max_images:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, max_images, dtype=int)
    return [candidates[index] for index in indices]


def area_input_root(base: Path, area: str, number_of_areas: int) -> Path:
    """Accept either a dataset root containing AREA or one area's direct path."""
    candidate = base / area
    if candidate.exists():
        return candidate
    if number_of_areas == 1:
        return base
    return candidate


def load_reconstruction_models(checkpoint_path: Path, latent_dim: int,
                               device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = Encoder(z_size=latent_dim).to(device)
    decoder = Decoder(z_size=latent_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, checkpoint


def reconstruct_batch(encoder, decoder, inputs: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
    """Use posterior mean for repeatable reconstruction diagnostics."""
    with torch.inference_mode():
        mean, _ = encoder(inputs.to(device))
        return decoder(mean)


def image_metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    residual = original - reconstruction
    mse = float(np.mean(np.square(residual)))
    return {
        "l1": float(np.mean(np.abs(residual))),
        "mse": mse,
        "psnr_db": float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse)),
        "dssim": float(1.0 - structural_similarity(
            original, reconstruction, data_range=1.0, channel_axis=-1,
        )),
    }


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return np.clip(
        ((tensor.detach().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0),
        0.0, 1.0,
    )


def save_preview(path: Path, rows: list[dict], images: list[np.ndarray],
                 reconstructions: list[np.ndarray]) -> None:
    tile_size, label_height = 256, 30
    count = min(len(rows), 8)
    canvas = Image.new("RGB", (tile_size * 2, (tile_size + label_height) * count), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        y = index * (tile_size + label_height)
        for column, array in enumerate((images[index], reconstructions[index])):
            tile = Image.fromarray((array * 255).astype(np.uint8)).resize((tile_size, tile_size))
            canvas.paste(tile, (column * tile_size, y + label_height))
        draw.text((5, y + 7), f"Input: {Path(rows[index]['image']).name}", fill="black")
        draw.text((tile_size + 5, y + 7),
                  f"Recon  MSE={rows[index]['mse']:.5f}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_area(area: str, input_root: Path, checkpoint_root: Path, latent_dim: int,
             max_images: int, batch_size: int, device: torch.device,
             output_root: Path) -> dict:
    checkpoint_path = checkpoint_root / f"model_{area}_{latent_dim}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    paths = find_images(input_root, max_images)
    if not paths:
        raise FileNotFoundError(f"No supported images found for {area}: {input_root}")

    encoder, decoder, checkpoint = load_reconstruction_models(
        checkpoint_path, latent_dim, device
    )
    image_size = ((checkpoint.get("config") or {}).get("params") or {}).get(
        "image_size", [128, 128]
    )
    height, width = (int(image_size[0]), int(image_size[1]))
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    records, originals, reconstructed_images = [], [], []
    output_shape = None
    finite = True
    raw_output_min, raw_output_max = float("inf"), float("-inf")
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch = torch.stack([
            transform(Image.open(path).convert("RGB")) for path in batch_paths
        ])
        reconstruction = reconstruct_batch(encoder, decoder, batch, device)
        output_shape = list(reconstruction.shape[1:])
        finite = finite and bool(torch.isfinite(reconstruction).all().item())
        raw_output_min = min(raw_output_min, float(reconstruction.min().item()))
        raw_output_max = max(raw_output_max, float(reconstruction.max().item()))
        if reconstruction.shape != batch.shape:
            raise RuntimeError(
                f"Shape mismatch for {area}: input={tuple(batch.shape)}, "
                f"output={tuple(reconstruction.shape)}"
            )
        for path, original_t, reconstruction_t in zip(batch_paths, batch, reconstruction):
            original = tensor_to_image(original_t)
            reconstructed = tensor_to_image(reconstruction_t)
            record = {"area": area, "image": str(path)}
            record.update(image_metrics(original, reconstructed))
            records.append(record)
            originals.append(original)
            reconstructed_images.append(reconstructed)

    if not finite:
        raise RuntimeError(f"Non-finite reconstruction generated by {area}")
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{area}_reconstruction_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    preview_path = output_root / f"{area}_reconstruction_preview.jpg"
    save_preview(preview_path, records, originals, reconstructed_images)

    summary = {
        "area": area,
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "input_root": str(input_root),
        "images_tested": len(records),
        "output_shape": output_shape,
        "finite_output": finite,
        "raw_output_range": [raw_output_min, raw_output_max],
        "mean_l1": float(np.mean([row["l1"] for row in records])),
        "mean_mse": float(np.mean([row["mse"] for row in records])),
        "mean_psnr_db": float(np.mean([row["psnr_db"] for row in records])),
        "mean_dssim": float(np.mean([row["dssim"] for row in records])),
        "metrics_csv": str(csv_path),
        "preview": str(preview_path),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cf_dataset_epito.yaml"))
    parser.add_argument("--data_source", choices=("training", "testing"), default="training")
    parser.add_argument("--input", type=Path, help="Override the selected config data path.")
    parser.add_argument("--checkpoints", type=Path, help="Override models.checkpoints.")
    parser.add_argument("--safety_area", nargs="+", default=list(DEFAULT_AREAS))
    parser.add_argument("--latent_dims", type=int)
    parser.add_argument("--max_images", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    if args.max_images < 1 or args.batch_size < 1:
        parser.error("--max_images and --batch_size must be positive")
    return args


def main() -> int:
    args = parse_args()
    with args.config.expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    data_config, model_config = config.get("data") or {}, config.get("models") or {}
    input_base = resolve_path(args.input or data_config.get(args.data_source, ""))
    checkpoint_root = resolve_path(
        args.checkpoints or model_config.get("checkpoints", "results/V6/train/models")
    )
    latent_dim = args.latent_dims or int(model_config.get("latent_dims", 64))
    output_root = resolve_path(
        args.output_dir or f"results/V6/inference_smoke/{args.data_source}"
    )
    device = choose_device(args.device)
    print(f"Device: {device}; input: {input_base}; checkpoints: {checkpoint_root}")

    summaries, failures = [], []
    for area in args.safety_area:
        try:
            summary = run_area(
                area, area_input_root(input_base, area, len(args.safety_area)),
                checkpoint_root, latent_dim, args.max_images, args.batch_size,
                device, output_root,
            )
            summaries.append(summary)
            print(
                f"[PASS] {area}: n={summary['images_tested']} "
                f"MSE={summary['mean_mse']:.6f} "
                f"PSNR={summary['mean_psnr_db']:.2f} dB "
                f"dSSIM={summary['mean_dssim']:.6f}"
            )
        except Exception as exc:
            failures.append({"area": area, "status": "FAIL", "error": str(exc)})
            print(f"[FAIL] {area}: {exc}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "inference_smoke_summary.json"
    summary_path.write_text(json.dumps(summaries + failures, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
