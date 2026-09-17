#!/usr/bin/env python3
"""Generate and combine scenario videos whose numeric IDs follow a cutoff ID."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np
import yaml


SCENARIO_PATTERN = re.compile(r"^(\d+)_(\d+)$")
LABEL_COLORS = {
    "Normal": (40, 180, 40),       # BGR green
    "Anomalous": (35, 35, 230),   # BGR red
    "Verify": (0, 165, 255),       # BGR orange
    "Unlabeled": (150, 150, 150),
}


def scenario_key(value: str) -> tuple[int, int]:
    match = SCENARIO_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid scenario ID {value!r}; expected for example 8_0")
    return int(match.group(1)), int(match.group(2))


def discover_scenarios(base_path: Path, after: str) -> list[str]:
    cutoff = scenario_key(after)
    scenarios = []
    for path in base_path.iterdir():
        if not path.is_dir() or not SCENARIO_PATTERN.fullmatch(path.name):
            continue
        if scenario_key(path.name) > cutoff:
            scenarios.append(path.name)
    return sorted(scenarios, key=scenario_key)


def scenario_video_path(base_path: Path, scenario: str, camera: str) -> Path:
    return (
        base_path / scenario / camera / "video"
        / f"s-{scenario}_c-{camera}.mp4"
    )


def annotated_video_path(base_path: Path, scenario: str, camera: str) -> Path:
    return (
        base_path / scenario / camera / "video"
        / f"s-{scenario}_c-{camera}_annotated_masks.mp4"
    )


def annotation_path(annotation_dir: Path, scenario: str, camera: str) -> Path:
    return annotation_dir / f"scenario_{scenario}_{camera}_annotations.csv"


def load_annotations(path: Path) -> tuple[dict[tuple[int, str], str], str]:
    labels = {}
    description = ""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"frame_id", "safety_area", "label"}
        if missing := required.difference(reader.fieldnames or []):
            raise ValueError(f"Annotation CSV {path} is missing: {sorted(missing)}")
        for row in reader:
            frame_id = int(row["frame_id"])
            area = row["safety_area"].strip()
            label = row["label"].strip().title() or "Unlabeled"
            key = (frame_id, area)
            if key in labels:
                raise ValueError(f"Duplicate annotation for frame/area {key} in {path}")
            labels[key] = label
            description = description or row.get("scenario_description", "").strip()
    return labels, description


def load_masks(config_path: Path, areas: list[str] | None) -> dict[str, np.ndarray]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    configured = (config.get("data") or {}).get("mask_types") or {}
    selected = areas or list(configured)
    unknown = [area for area in selected if area not in configured]
    if unknown:
        raise ValueError(
            f"Unknown masks {unknown}; configured areas are {list(configured)}"
        )
    masks = {}
    for area in selected:
        path = Path(configured[area]).expanduser()
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask for {area}: {path}")
        masks[area] = mask
    return masks


def draw_annotated_masks(frame: np.ndarray, masks: dict[str, np.ndarray],
                         labels: dict[tuple[int, str], str], frame_id: int,
                         scenario: str, description: str,
                         opacity: float) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    overlay = output.copy()
    outlines = []
    statuses = []
    for area, source_mask in masks.items():
        mask = source_mask
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        binary = (mask > 127).astype(np.uint8) * 255
        label = labels.get((frame_id, area), "Unlabeled")
        color = LABEL_COLORS.get(label, LABEL_COLORS["Unlabeled"])
        overlay[binary > 0] = color
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines.append((contours, color))
        moments = cv2.moments(binary)
        if moments["m00"]:
            center = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
            statuses.append((area, label, color, center))
    output = cv2.addWeighted(overlay, opacity, output, 1.0 - opacity, 0)
    for contours, color in outlines:
        cv2.drawContours(output, contours, -1, color, 5)
    for area, label, color, (x, y) in statuses:
        text = f"{area}: {label}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        x = max(4, min(x - text_width // 2, width - text_width - 4))
        y = max(text_height + 8, min(y, height - 5))
        cv2.rectangle(output, (x - 4, y - text_height - 6),
                      (x + text_width + 4, y + 5), (20, 20, 20), -1)
        cv2.putText(output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)
    header = f"Scenario {scenario} | frame {frame_id}"
    cv2.rectangle(output, (0, 0), (width, 66), (15, 15, 15), -1)
    cv2.putText(output, header, (14, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if description:
        cv2.putText(output, description[:110], (14, 53), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return output


def create_annotated_video(source: Path, destination: Path, annotations: Path | None,
                           masks: dict[str, np.ndarray], scenario: str,
                           save_every_n: int, opacity: float, force: bool,
                           progress: bool) -> bool:
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"[reuse annotated] {scenario}: {destination}")
        return True
    labels, description = load_annotations(annotations) if annotations else ({}, "")
    fps, width, height, expected_frames = inspect_video(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create annotated video: {destination}")
    capture = cv2.VideoCapture(str(source))
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotation_frame_id = written * save_every_n
            annotated = draw_annotated_masks(
                frame, masks, labels, annotation_frame_id, scenario,
                description, opacity,
            )
            writer.write(annotated)
            written += 1
            if progress and written % 500 == 0:
                print(f"  [annotate] {scenario}: {written}/{expected_frames}", end="\r")
    finally:
        capture.release()
        writer.release()
    if progress:
        print(f"  [annotate] {scenario}: {written}/{expected_frames}")
    return written > 0


def generate_video(args, scenario: str, destination: Path) -> bool:
    if destination.is_file() and destination.stat().st_size > 0 and not args.force:
        print(f"[reuse] {scenario}: {destination}")
        return True
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "process_rosbags_to_dataset.py"),
        "--config", str(args.config),
        "--scenario", scenario,
        "--camera", args.camera,
        "--process-to", "video",
        "--output-dir", str(args.base_path),
        "--video-fps", str(args.video_fps),
        "--save-every-n", str(args.save_every_n),
    ]
    if args.progress:
        command.append("--progress")
    if args.max_frames is not None:
        command.extend(("--max-frames", str(args.max_frames)))
    print(f"[generate] {scenario}")
    if args.dry_run:
        print("  " + " ".join(command))
        return False
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"[warning] processor failed for {scenario} with exit {result.returncode}")
        return False
    if not destination.is_file() or destination.stat().st_size == 0:
        print(f"[warning] expected video was not created: {destination}")
        return False
    return True


def inspect_video(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return fps, width, height, frames


def combine_videos(videos: list[tuple[str, Path]], output: Path,
                   output_fps: float | None, progress: bool) -> int:
    first_fps, width, height, _ = inspect_video(videos[0][1])
    fps = output_fps or first_fps
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create unified video: {output}")

    total_written = 0
    try:
        for index, (scenario, path) in enumerate(videos, start=1):
            source_fps, source_width, source_height, expected_frames = inspect_video(path)
            print(
                f"[combine {index}/{len(videos)}] {scenario}: "
                f"{expected_frames} frames, {source_width}x{source_height}, {source_fps:.3f} fps"
            )
            if abs(source_fps - fps) > 0.01:
                print(
                    f"  [warning] source FPS differs from output FPS "
                    f"({source_fps:.3f} -> {fps:.3f}); playback speed will change"
                )
            capture = cv2.VideoCapture(str(path))
            scenario_frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                scenario_frames += 1
                total_written += 1
                if progress and scenario_frames % 500 == 0:
                    print(f"  {scenario}: {scenario_frames}/{expected_frames}", end="\r")
            capture.release()
            if progress:
                print(f"  {scenario}: {scenario_frames}/{expected_frames}")
    finally:
        writer.release()
    return total_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path", type=Path,
        default=Path("/Users/rashid/data/DS/SR/v6/Jul27/extracted_frames"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/cf_dataset_mac.yaml"))
    parser.add_argument("--after", default="8_0", help="Include scenario IDs strictly after this ID.")
    parser.add_argument("--camera", choices=("front_view", "back_view"), default="front_view")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--output-fps", type=float)
    parser.add_argument("--save-every-n", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate existing scenario videos.")
    parser.add_argument(
        "--annotated-masks", action="store_true",
        help="Overlay configured safety-area masks using per-frame annotation labels.",
    )
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=Path("reports/safety_area_annotations/saved_annotation"),
    )
    parser.add_argument(
        "--areas", nargs="+", help="Safety areas to overlay; default: every configured mask.",
    )
    parser.add_argument(
        "--mask-opacity", type=float, default=0.28,
        help="Colored mask fill opacity in [0, 1].",
    )
    parser.add_argument(
        "--missing-annotations", choices=("skip", "error", "unlabeled"), default="skip",
        help="Behavior when a scenario/camera annotation CSV is unavailable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.video_fps <= 0 or (args.output_fps is not None and args.output_fps <= 0):
        parser.error("FPS values must be positive")
    if args.save_every_n < 1:
        parser.error("--save-every-n must be at least 1")
    if not 0 <= args.mask_opacity <= 1:
        parser.error("--mask-opacity must be between 0 and 1")
    scenario_key(args.after)
    return args


def main() -> int:
    args = parse_args()
    args.base_path = args.base_path.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.annotations_dir = args.annotations_dir.expanduser().resolve()
    if not args.base_path.is_dir():
        raise FileNotFoundError(f"Base path not found: {args.base_path}")
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    output = (
        args.output.expanduser().resolve() if args.output else
        args.base_path / "unified_videos"
        / (
            f"all_scenarios_after_{args.after}_{args.camera}_annotated_masks.mp4"
            if args.annotated_masks else
            f"all_scenarios_after_{args.after}_{args.camera}.mp4"
        )
    )
    scenarios = discover_scenarios(args.base_path, args.after)
    if not scenarios:
        print(f"No scenario directories found after {args.after}")
        return 1
    print(f"Selected {len(scenarios)} scenarios: {', '.join(scenarios)}")

    masks = load_masks(args.config, args.areas) if args.annotated_masks else {}
    videos = []
    for scenario in scenarios:
        path = scenario_video_path(args.base_path, scenario, args.camera)
        if not generate_video(args, scenario, path):
            continue
        selected_path = path
        if args.annotated_masks:
            annotations = annotation_path(
                args.annotations_dir, scenario, args.camera
            )
            if not annotations.is_file():
                message = f"No annotation CSV for {scenario}/{args.camera}: {annotations}"
                if args.missing_annotations == "error":
                    raise FileNotFoundError(message)
                if args.missing_annotations == "skip":
                    print(f"[skip] {message}")
                    continue
                print(f"[unlabeled] {message}")
                annotations = None
            selected_path = annotated_video_path(
                args.base_path, scenario, args.camera
            )
            if not args.dry_run and not create_annotated_video(
                path, selected_path, annotations, masks, scenario,
                args.save_every_n, args.mask_opacity, args.force, args.progress,
            ):
                print(f"[warning] no annotated frames produced for {scenario}")
                continue
        videos.append((scenario, selected_path))
    if args.dry_run:
        print(f"[dry-run] Unified output would be: {output}")
        return 0
    if not videos:
        print("No videos are available to combine")
        return 1

    frames = combine_videos(videos, output, args.output_fps, args.progress)
    print(f"[done] Combined {len(videos)} scenarios and {frames} frames")
    print(f"[output] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
