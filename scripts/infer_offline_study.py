#!/usr/bin/env python3
"""Run video inference and emit two row-wise videos for the XAI user study.

This entry point deliberately reuses the inference, TAAS, threshold, mask, and
dashboard implementation from ``infer_offline.py``.  It does not modify the
standard offline inference outputs.  The 2x2 ADVIS dashboard is split into:

* TL_TR: Input View + Unexpected Situations View
* BL_BR: AI View + Details
"""

from __future__ import annotations

import csv
import time
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import infer_offline as offline


STUDY_ROW_LABELS = {
    "TL_TR": "Input View + Unexpected Situations View",
    "BL_BR": "AI View + Details",
}


def split_study_dashboard(dashboard: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 2x2 dashboard into equal top-row and bottom-row frames."""
    if dashboard.ndim != 3 or dashboard.shape[2] != 3:
        raise ValueError(
            "Expected a BGR dashboard with shape (height, width, 3); "
            f"received {dashboard.shape}"
        )
    if dashboard.shape[0] < 2:
        raise ValueError("Dashboard must contain at least two pixel rows")
    split_y = dashboard.shape[0] // 2
    return (
        np.ascontiguousarray(dashboard[:split_y]),
        np.ascontiguousarray(dashboard[split_y:]),
    )


def study_output_paths(args) -> tuple[Path, Path, Path]:
    """Return top-video, bottom-video, and score-CSV paths."""
    repository_root = Path(__file__).resolve().parent.parent
    study_dir = (
        repository_root / "results" / args.dataset_version
        / "offline_inference" / "study_video"
    )
    detection_stem = args.output_video.stem
    score_name = args.output_csv.name
    return (
        study_dir / f"{detection_stem}_TL_TR.mp4",
        study_dir / f"{detection_stem}_BL_BR.mp4",
        study_dir / score_name,
    )


class StudyVideoOutput:
    """Fixed-size MP4 writer for one dashboard row."""

    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.frame_size = frame_size
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create study video: {path}")

    def write(self, frame: np.ndarray) -> None:
        width, height = self.frame_size
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def select_device(args) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_scores(path: Path, rows: list[dict]) -> None:
    fieldnames = (
        "sample_id", "safety_area", "anomaly_score", "threshold",
        "calibrated_threshold", "threshold_amplification",
        "normalized_score", "is_anomalous",
        "instantaneous_anomaly_score", "instantaneous_normalized_score",
        "rolling_policy", "rolling_window", "rolling_count",
        "threshold_strategy", "score_func", "calibration_score_func",
        "inference_score_func", "inference_score_backend", "reconstruction_mode",
        "offset", "sigma", "quantile", "taas_variant",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = offline.load_settings(offline.parse_args())
    if args.input_type != "video":
        raise ValueError(
            "infer_offline_study.py currently requires --input_type video; "
            "use infer_offline.py for frames, cropped data, or rosbags"
        )
    if not args.save_video:
        raise ValueError(
            "The study entry point creates videos; remove --scores-only"
        )

    args.taas_backend = offline.resolve_taas_backend(args.taas_backend)
    device = select_device(args)
    top_path, bottom_path, score_path = study_output_paths(args)
    print(f"[device] {device}")
    print(f"[input] video: {args.input}")
    print(f"[TAAS backend] {args.taas_backend}")
    print(f"[study] TL_TR ({STUDY_ROW_LABELS['TL_TR']}): {top_path}")
    print(f"[study] BL_BR ({STUDY_ROW_LABELS['BL_BR']}): {bottom_path}")
    print(f"[study] scores: {score_path}")

    models = offline.load_models(args, device)
    profiler = offline.TimingProfiler(args.profile_timing)
    masks = offline.parse_masks(
        args.mask, args.safety_areas, args.config_masks_dir
    )
    mask_geometries = None
    score_windows = OrderedDict(
        (area, deque(maxlen=args.rolling_window)) for area in args.safety_areas
    )
    rows = []
    processed_frames = 0
    processing_started = time.perf_counter()

    # make_advis_dashboard is 1600x1000, so each study row is 1600x500.
    frame_size = (1600, 500)
    top_video = StudyVideoOutput(top_path, args.output_fps, frame_size)
    bottom_video = StudyVideoOutput(bottom_path, args.output_fps, frame_size)
    try:
        progress = tqdm(
            offline.iter_video(args), total=offline.input_progress_total(args),
            desc="XAI study inference [video]", unit="frame", dynamic_ncols=True,
        )
        previous_finished = time.perf_counter()
        for sample_id, frame in progress:
            profiler.record(
                "input_decode", time.perf_counter() - previous_finished
            )
            if mask_geometries is None:
                with profiler.measure("mask_prepare_once"):
                    mask_geometries = OrderedDict(
                        (
                            area,
                            offline.prepare_mask_geometry(
                                masks[area], frame.shape
                            ),
                        )
                        for area in args.safety_areas
                    )

            frame_results = OrderedDict()
            for area in args.safety_areas:
                with profiler.measure("mask_crop"):
                    crop = offline.crop_area(
                        frame, masks[area], mask_geometries[area]
                    )
                result = offline.infer_crop(
                    crop, area, models[area], offline.normalize_model_input,
                    device, profiler, args.taas_backend,
                )
                with profiler.measure("rolling_policy"):
                    offline.apply_rolling_policy(
                        result, score_windows[area], args.rolling,
                        args.rolling_window,
                    )
                rows.append({
                    "sample_id": sample_id,
                    **offline.public_result(result),
                })
                frame_results[area] = result

            next_count = processed_frames + 1
            processing_fps = next_count / max(
                time.perf_counter() - processing_started, 1e-9
            )
            with profiler.measure("study_dashboard_render"):
                dashboard = offline.make_advis_dashboard(
                    frame, masks, frame_results, sample_id,
                    add_score_name=args.add_score_name,
                    add_fps_details=args.add_fps_details,
                    processing_fps=processing_fps,
                    output_fps=args.output_fps,
                    mask_geometries=mask_geometries,
                )
                top_row, bottom_row = split_study_dashboard(dashboard)
            with profiler.measure("study_TL_TR_video_write"):
                top_video.write(top_row)
            with profiler.measure("study_BL_BR_video_write"):
                bottom_video.write(bottom_row)

            processed_frames = next_count
            profiler.source_frames += 1
            progress.set_postfix_str(
                ", ".join(
                    f"{area}={result['normalized_score']:.2f}x"
                    for area, result in frame_results.items()
                )
            )
            previous_finished = time.perf_counter()
            if args.max_frames and processed_frames >= args.max_frames:
                break
    finally:
        top_video.close()
        bottom_video.close()

    write_scores(score_path, rows)
    print(f"[save] {len(rows):,} area results -> {score_path}")
    print(f"[save] TL_TR study video -> {top_path}")
    print(f"[save] BL_BR study video -> {bottom_path}")
    profiler.report()


if __name__ == "__main__":
    main()
