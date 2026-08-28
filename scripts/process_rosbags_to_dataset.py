#!/usr/bin/env python3
"""Convert configured MCAP camera streams into raw frames or safety-area crops."""

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from cv_bridge import CvBridge

from process_rosbags_from_config import (
    IMAGE_TYPES,
    ProgressBar,
    camera_topic,
    decode_frame,
    inspect_bag,
    load_config,
    resolve_bag_path,
    resolve_masks,
    resolve_scenarios,
    topic_message_counts,
)


def crop_safety_area(frame, mask, target_size=128, keep_aspect=True):
    """Apply a mask, crop its bounding box, and resize for model input."""
    height, width = frame.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    masked = cv2.bitwise_and(frame, frame, mask=binary_mask)
    crop = masked[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if not keep_aspect:
        return cv2.resize(
            crop, (target_size, target_size), interpolation=cv2.INTER_AREA
        )

    crop_height, crop_width = crop.shape[:2]
    scale = min(target_size / crop_width, target_size / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    output = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - resized_width) // 2
    y_offset = (target_size - resized_height) // 2
    output[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized
    return output


def output_camera_name(topic):
    if topic.startswith("/camera/"):
        return topic.split("/")[2]
    return topic.strip("/").replace("/", "_")


def process_scenario(config, scenario_id, args, output_base):
    bag_path = resolve_bag_path(config, scenario_id)
    if not bag_path.is_file():
        print(f"[ERROR] [{scenario_id}] Bag not found: {bag_path}", file=sys.stderr)
        return False

    reader, topic_types = inspect_bag(bag_path)
    requested_topics = [camera_topic(camera) for camera in args.camera]
    available_topics = {
        topic: topic_types[topic]
        for topic in requested_topics
        if topic_types.get(topic) in IMAGE_TYPES
    }
    if not available_topics:
        print(f"[ERROR] [{scenario_id}] No requested image topics found", file=sys.stderr)
        return False

    bridge = CvBridge()
    camera_names = {
        topic: output_camera_name(topic) for topic in available_topics
    }
    masks_by_topic = {}
    output_dirs = {}
    for topic, camera_name in camera_names.items():
        camera_root = output_base / scenario_id / camera_name
        if args.process_to == "frames":
            raw_dir = camera_root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[topic] = {"raw": raw_dir}
            print(f"[{scenario_id}] {topic} -> {raw_dir}")
        else:
            masks = resolve_masks(config, args, camera_name)
            if not masks:
                raise ValueError("No safety-area masks selected or configured")
            masks_by_topic[topic] = masks
            output_dirs[topic] = {}
            for area_name in masks:
                area_dir = camera_root / "processed" / area_name
                area_dir.mkdir(parents=True, exist_ok=True)
                output_dirs[topic][area_name] = area_dir
                print(f"[{scenario_id}] {topic}, {area_name} -> {area_dir}")

    message_counts = topic_message_counts(bag_path)
    expected = {
        topic: int(min(
            math.ceil(message_counts.get(topic, 0) / args.save_every_n),
            args.max_frames or math.inf,
        ))
        for topic in available_topics
    }
    progress = ProgressBar(
        sum(expected.values()),
        f"[{scenario_id}] {args.process_to}",
        enabled=args.progress,
    )
    seen = {topic: 0 for topic in available_topics}
    processed = {topic: 0 for topic in available_topics}
    saved = {topic: 0 for topic in available_topics}

    try:
        while reader.has_next():
            topic, serialized_data, _ = reader.read_next()
            if topic not in available_topics:
                continue
            seen[topic] += 1
            if (seen[topic] - 1) % args.save_every_n != 0:
                continue
            if args.max_frames and processed[topic] >= args.max_frames:
                if all(value >= args.max_frames for value in processed.values()):
                    break
                continue

            frame_index = processed[topic]
            processed[topic] += 1
            try:
                frame, _ = decode_frame(
                    serialized_data, available_topics[topic], bridge
                )
                if args.process_to == "frames":
                    output_path = output_dirs[topic]["raw"] / (
                        f"frame_{frame_index:06d}.{args.image_format}"
                    )
                    if not cv2.imwrite(str(output_path), frame):
                        raise RuntimeError(f"Failed to write {output_path}")
                else:
                    for area_name, mask in masks_by_topic[topic].items():
                        crop = crop_safety_area(
                            frame, mask,
                            target_size=args.target_size,
                            keep_aspect=not args.stretch,
                        )
                        if crop is None:
                            print(
                                f"[WARNING] Empty mask for {area_name}, frame {frame_index}",
                                file=sys.stderr,
                            )
                            continue
                        output_path = output_dirs[topic][area_name] / (
                            f"s-{scenario_id}_s-{area_name}_f-{frame_index:06d}.{args.image_format}"
                        )
                        if not cv2.imwrite(str(output_path), crop):
                            raise RuntimeError(f"Failed to write {output_path}")
                saved[topic] += 1
            except Exception as error:
                print(
                    f"[ERROR] {scenario_id}, {topic}, frame {frame_index}: {error}",
                    file=sys.stderr,
                )
            finally:
                progress.update()
    finally:
        progress.close()

    for topic in available_topics:
        print(
            f"[{scenario_id}] done: {topic} -> "
            f"seen={seen[topic]}, processed={processed[topic]}, saved={saved[topic]}"
        )
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save rosbag camera streams as raw frames or safety-area crops."
    )
    parser.add_argument("--config", default="configs/cf_dataset_mac.yaml")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenario", nargs="+", help="Scenario ID(s).")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--camera", nargs="+", help="Camera names or topic paths.")
    parser.add_argument(
        "--process-to",
        required=True,
        choices=["frames", "safety-areas"],
        help="Save full frames or mask-cropped safety-area model inputs.",
    )
    parser.add_argument(
        "--safety-areas",
        nargs="+",
        help="Areas from data.mask_types; default is every configured area.",
    )
    parser.add_argument("--output-dir", help="Default: dataset_base/extracted_frames")
    parser.add_argument("--save-every-n", type=int, default=1)
    parser.add_argument("--max-frames", type=int, help="First N sampled frames per camera.")
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="Stretch crops to target size instead of preserving aspect ratio.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.save_every_n < 1:
        parser.error("--save-every-n must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.target_size < 1:
        parser.error("--target-size must be at least 1")
    if args.process_to == "frames" and args.safety_areas:
        parser.error("--safety-areas is only valid with --process-to safety-areas")
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    scenario_ids = resolve_scenarios(config, args.scenario, args.all)
    args.camera = args.camera or [
        config.get("playback_options", {}).get("camera", "back_view")
    ]
    output_base = Path(
        args.output_dir
        or Path(config["data"]["dataset_base"]) / "extracted_frames"
    ).expanduser()

    print(f"Mode: {args.process_to}")
    print(f"Scenarios: {', '.join(scenario_ids)}")
    print(f"Cameras: {', '.join(args.camera)}")
    print(f"Output base: {output_base}")
    results = [
        process_scenario(config, scenario_id, args, output_base)
        for scenario_id in scenario_ids
    ]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
