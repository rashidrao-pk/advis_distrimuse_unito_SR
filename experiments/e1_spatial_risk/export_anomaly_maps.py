#!/usr/bin/env python3
"""Export raw ADVIS spatial anomaly maps for E1 forecasting.

Designed to live under experiments/e1_spatial_risk/ in the ADVIS repository and
reuse scripts/infer_offline.py without changing the current inference pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import infer_offline as offline  # noqa: E402

ALL_AREAS = ("PLeft", "PRight", "RoboArm", "ConvBelt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export ADVIS raw anomaly maps for E1")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/cf_dataset_epito.yaml")
    p.add_argument("--scenario", required=True)
    p.add_argument("--dataset-version", default="V6")
    p.add_argument("--topic", default="/camera/back_view/image_raw")
    p.add_argument("--safety-areas", nargs="+", default=list(ALL_AREAS))
    p.add_argument("--checkpoints", type=Path)
    p.add_argument("--threshold-dir", type=Path)
    p.add_argument("--threshold-calibration-mode", choices=("auto", "test", "val"), default="val")
    p.add_argument("--threshold-strategy", choices=("max", "percentile", "mean_std", "f1c"), default="percentile")
    p.add_argument("--threshold-percentile", type=float, default=99.0)
    p.add_argument("--threshold-amplification", nargs="+", type=float, default=[1.0])
    p.add_argument("--offset", type=int, default=3)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--taas-backend", choices=("auto", "cython", "numpy"), default="auto")
    p.add_argument("--taas-variant", choices=("canonical", "minimization"), default="canonical")
    p.add_argument("--latent-dims", type=int, default=64)
    p.add_argument("--model-variant", choices=("old", "new"), default="old")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--skip-first", type=int, default=0)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--effective-fps", type=float, default=5.0)
    p.add_argument("--store-float32", action="store_true")
    return p.parse_args()


def select_device(cpu: bool) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path) -> dict:
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model_args(args: argparse.Namespace, config: dict) -> SimpleNamespace:
    model_cfg = config.get("models") or {}
    checkpoints = args.checkpoints or model_cfg.get("checkpoints") or f"results/{args.dataset_version}/train/models"
    checkpoints = offline.resolve_path(checkpoints, REPO_ROOT)
    threshold_dir = args.threshold_dir or (REPO_ROOT / "results" / args.dataset_version / "thresholds")
    threshold_dir = Path(threshold_dir).expanduser().resolve()

    factors = offline.resolve_threshold_amplifications(args.threshold_amplification, args.safety_areas)
    resolved = OrderedDict()
    for area in args.safety_areas:
        resolved[area] = offline.load_threshold(
            threshold_dir,
            area,
            args.threshold_strategy,
            args.threshold_percentile,
            args.offset,
            args.sigma,
            args.quantile,
            args.taas_variant,
            args.threshold_calibration_mode,
        )

    return SimpleNamespace(
        safety_areas=list(args.safety_areas),
        latent_dims=int(args.latent_dims),
        checkpoints=checkpoints,
        model_variant=args.model_variant,
        resolved_threshold_configs=resolved,
        threshold_amplification_by_area=factors,
        threshold_dir=threshold_dir,
        threshold_strategy=args.threshold_strategy,
        threshold_percentile=args.threshold_percentile,
        offset=args.offset,
        sigma=args.sigma,
        quantile=args.quantile,
        taas_variant=args.taas_variant,
        threshold_calibration_mode=args.threshold_calibration_mode,
    )


def raw_infer_crop(image, model, device, taas_backend: str):
    input_tensor = offline.to_tensor(image, offline.normalize_model_input, device)
    with torch.no_grad():
        mean, _ = model["encoder"](input_tensor)
        reconstruction = model["decoder"](mean)
    original = offline.tensor_to_hwc(input_tensor.squeeze(0))
    reconstructed = offline.tensor_to_hwc(reconstruction.squeeze(0))
    cfg = model["threshold"]
    distance = offline.distance_offset(
        original,
        reconstructed,
        int(cfg["offset"]),
        taas_backend,
        cfg["taas_variant"],
    )
    if float(cfg["sigma"]) > 0:
        distance = offline.gaussian_filter(distance, sigma=float(cfg["sigma"]))
    score = float(np.quantile(distance, float(cfg["quantile"])))
    threshold = float(cfg["threshold"])
    # Numerical spatial risk representation used by E1. The ADVIS threshold
    # corresponds to risk ~= 0.5 because the dashboard uses distance/(2*tau).
    risk = np.clip(distance / max(2.0 * threshold, 1e-12), 0.0, 1.0)
    return distance.astype(np.float32), risk.astype(np.float32), score, threshold


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data") or {}
    video_path, scenario_id = offline.resolve_video_scenario(args.scenario, data_cfg, args.topic)
    device = select_device(args.cpu)
    backend = offline.resolve_taas_backend(args.taas_backend)
    model_args = build_model_args(args, config)
    models = offline.load_models(model_args, device)
    masks_dir = Path(data_cfg["masks"]).expanduser().resolve()
    masks = offline.parse_masks([], args.safety_areas, masks_dir)

    iter_args = SimpleNamespace(
        input=video_path,
        skip_first=args.skip_first,
        frame_stride=args.frame_stride,
    )

    map_rows = []
    risk_rows = []
    score_rows = []
    threshold_rows = []
    sample_ids = []
    mask_geometries = None

    for i, (sample_id, frame) in enumerate(offline.iter_video(iter_args)):
        if mask_geometries is None:
            mask_geometries = OrderedDict(
                (area, offline.prepare_mask_geometry(masks[area], frame.shape))
                for area in args.safety_areas
            )
        frame_maps, frame_risks, frame_scores, frame_thresholds = [], [], [], []
        for area in args.safety_areas:
            crop = offline.crop_area(frame, masks[area], mask_geometries[area])
            distance, risk, score, threshold = raw_infer_crop(crop, models[area], device, backend)
            frame_maps.append(distance)
            frame_risks.append(risk)
            frame_scores.append(score)
            frame_thresholds.append(threshold)
        map_rows.append(np.stack(frame_maps, axis=0))
        risk_rows.append(np.stack(frame_risks, axis=0))
        score_rows.append(frame_scores)
        threshold_rows.append(frame_thresholds)
        sample_ids.append(str(sample_id))
        if args.max_frames is not None and (i + 1) >= args.max_frames:
            break

    if not map_rows:
        raise RuntimeError("No frames were exported")

    dtype = np.float32 if args.store_float32 else np.float16
    distance_maps = np.stack(map_rows).astype(dtype)
    risk_maps = np.stack(risk_rows).astype(dtype)
    scores = np.asarray(score_rows, dtype=np.float32)
    thresholds = np.asarray(threshold_rows, dtype=np.float32)

    output = args.output or (
        REPO_ROOT / "results" / args.dataset_version / "e1_spatial_risk" / "exports" /
        f"scenario_{scenario_id}_maps.npz"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "scenario": scenario_id,
        "source_video": str(video_path),
        "areas": list(args.safety_areas),
        "shape": list(risk_maps.shape),
        "effective_fps": float(args.effective_fps),
        "frame_stride": int(args.frame_stride),
        "skip_first": int(args.skip_first),
        "taas_backend": backend,
        "taas_variant": args.taas_variant,
        "threshold_strategy": args.threshold_strategy,
        "threshold_calibration_mode": args.threshold_calibration_mode,
        "offset": args.offset,
        "sigma": args.sigma,
        "quantile": args.quantile,
        "risk_definition": "clip(distance_map / (2 * effective_threshold), 0, 1)",
    }
    np.savez_compressed(
        output,
        distance_maps=distance_maps,
        risk_maps=risk_maps,
        scores=scores,
        thresholds=thresholds,
        sample_ids=np.asarray(sample_ids),
        areas=np.asarray(args.safety_areas),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"[save] {output}")
    print(f"[shape] risk_maps={risk_maps.shape} distance_maps={distance_maps.shape}")
    print(f"[device] {device} | [TAAS backend] {backend}")


if __name__ == "__main__":
    main()
