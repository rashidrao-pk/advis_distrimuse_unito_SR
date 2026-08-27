#!/usr/bin/env python3
"""Extract raw or compressed camera frames from MCAP rosbags listed in YAML."""

import argparse
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, Info, SequentialReader, StorageOptions
from sensor_msgs.msg import CompressedImage, Image


IMAGE_TYPES = {
    "sensor_msgs/msg/Image": Image,
    "sensor_msgs/msg/CompressedImage": CompressedImage,
}


class ProgressBar:
    """Small dependency-free terminal progress bar."""

    def __init__(self, total, description, enabled=False, width=32):
        self.total = max(0, int(total))
        self.description = description
        self.enabled = enabled
        self.width = width
        self.current = 0
        self.started = time.monotonic()
        self.last_draw = 0.0

    def update(self, amount=1):
        if not self.enabled:
            return
        self.current += amount
        now = time.monotonic()
        if now - self.last_draw >= 0.1 or self.current >= self.total:
            self._draw(now)

    def _draw(self, now):
        fraction = min(1.0, self.current / self.total) if self.total else 0.0
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(now - self.started, 1e-9)
        rate = self.current / elapsed
        percent = 100.0 * fraction if self.total else 0.0
        print(
            f"\r{self.description} [{bar}] {self.current}/{self.total} "
            f"({percent:5.1f}%) {rate:5.1f} frame/s",
            end="",
            flush=True,
        )
        self.last_draw = now

    def close(self):
        if self.enabled:
            self._draw(time.monotonic())
            print()


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if "data" not in config or "scenario_options" not in config:
        raise ValueError("Config must contain 'data' and 'scenario_options' sections")
    return config


def resolve_scenarios(config, requested, process_all):
    options = config["scenario_options"]
    if process_all:
        scenario_ids = list(options)
    elif requested:
        scenario_ids = requested
    else:
        selected = str(config.get("scenario", {}).get("id", ""))
        if not selected:
            raise ValueError("No scenario selected in config; use --scenario or --all")
        scenario_ids = [selected]

    unknown = [scenario_id for scenario_id in scenario_ids if scenario_id not in options]
    if unknown:
        raise ValueError(f"Unknown scenario ID(s): {', '.join(unknown)}")
    return scenario_ids


def resolve_bag_path(config, scenario_id):
    relative_or_absolute = Path(
        config["scenario_options"][scenario_id]["rosbag_path"]
    ).expanduser()
    if relative_or_absolute.is_absolute():
        return relative_or_absolute
    return Path(config["data"]["dataset_base"]).expanduser() / relative_or_absolute


def camera_topic(camera):
    if camera.startswith("/"):
        return camera
    return f"/camera/{camera}/image_raw"


def decode_frame(serialized_data, message_type, bridge):
    message_class = IMAGE_TYPES[message_type]
    message = deserialize_message(serialized_data, message_class)
    if message_type == "sensor_msgs/msg/CompressedImage":
        frame = cv2.imdecode(
            np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
    else:
        frame = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
    if frame is None:
        raise RuntimeError("Image message decoded to an empty frame")
    return frame, message


def inspect_bag(bag_path):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="mcap"),
        ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    return reader, topic_types


def topic_message_counts(bag_path):
    metadata = Info().read_metadata(str(bag_path), "mcap")
    return {
        item.topic_metadata.name: item.message_count
        for item in metadata.topics_with_message_count
    }


def resolve_masks(config, args, camera_name):
    """Load only the requested safety-area masks from data.mask_types."""
    configured = config.get("data", {}).get("mask_types") or {}
    if camera_name in configured and isinstance(configured[camera_name], dict):
        configured = configured[camera_name]
    if not isinstance(configured, dict):
        raise ValueError("data.mask_types must be a mapping of area names to paths")

    requested_areas = args.safety_areas or list(configured)
    unknown = [area for area in requested_areas if area not in configured]
    if unknown:
        available = ", ".join(configured) or "none"
        raise ValueError(
            f"Unknown safety area(s): {', '.join(unknown)}. "
            f"Available in YAML: {available}"
        )
    mask_paths = {area: configured[area] for area in requested_areas}

    masks = {}
    for area_name, mask_path in mask_paths.items():
        path = Path(mask_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Mask not found for {area_name}: {path}")
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask for {area_name}: {path}")
        masks[area_name] = mask
    return masks


def make_masked_frame(
    frame, masks, blur_ksize=31, dim_factor=0.35, outline_thickness=6
):
    """Keep safety areas sharp and blur/dim everything outside them."""
    height, width = frame.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    resized_masks = []

    for mask in masks.values():
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
        resized_masks.append(binary_mask)

    kernel_size = max(3, int(blur_ksize))
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    outside = np.clip(
        blurred.astype(np.float32) * float(dim_factor), 0, 255
    ).astype(np.uint8)
    result = np.where(combined_mask[..., None] > 0, frame, outside)

    for binary_mask in resized_masks:
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            result, contours, -1, (255, 255, 255), int(outline_thickness)
        )
    return result


def format_video_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    return f"{minutes:02d}:{seconds:05.2f}"


def wrap_text(text, max_characters=72):
    words = str(text).split()
    lines = []
    current = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_characters:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or ["No description provided"]


def add_video_overlay(
    frame, scenario_id, description, camera_name, frame_number,
    total_frames, video_fps,
):
    """Add scenario and video-position information to the top-left corner."""
    output = frame.copy()
    current_seconds = (frame_number - 1) / video_fps
    total_seconds = total_frames / video_fps
    lines = [
        f"Scenario: {scenario_id}",
        *[f"Description: {line}" if index == 0 else f"             {line}"
          for index, line in enumerate(wrap_text(description))],
        f"Camera: {camera_name}",
        f"Frame: {frame_number}/{total_frames}",
        (
            f"Video time: {format_video_time(current_seconds)}"
            f" / {format_video_time(total_seconds)}"
        ),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(frame.shape[1] / 1920 * 0.75, 0.75))
    thickness = 2
    line_height = int(32 * font_scale / 0.75)
    text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    box_width = min(frame.shape[1] - 24, text_width + 28)
    box_height = len(lines) * line_height + 28

    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (12 + box_width, 12 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output, line, (26, 42 + index * line_height), font,
            font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return output


def extract_scenario(config, scenario_id, args, output_base):
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

    print(f"\n[{scenario_id}] {bag_path}")
    for topic, message_type in topic_types.items():
        print(f"  available: {topic} -> {message_type}")
    for topic in requested_topics:
        if topic not in available_topics:
            print(f"  [WARNING] Requested image topic is unavailable: {topic}")

    if not available_topics:
        print(f"  [ERROR] No requested image topics found", file=sys.stderr)
        return False
    if args.dry_run:
        return True

    bridge = CvBridge()
    seen = {topic: 0 for topic in available_topics}
    processed = {topic: 0 for topic in available_topics}
    saved = {topic: 0 for topic in available_topics}
    output_dirs = {}
    camera_names = {}
    topic_masks = {}
    video_writers = {}
    video_paths = {}
    for topic in available_topics:
        camera_name = topic.split("/")[2] if topic.startswith("/camera/") else topic.strip("/").replace("/", "_")
        camera_names[topic] = camera_name
        output_dir = output_base / scenario_id / camera_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[topic] = output_dir
        print(f"  output: {topic} -> {output_dir}")
        if args.generate_masked_video:
            topic_masks[topic] = resolve_masks(config, args, camera_name)
            if not topic_masks[topic]:
                raise ValueError(
                    "Masked video requested, but data.mask_types contains no masks."
                )
            video_paths[topic] = output_base / scenario_id / f"{scenario_id}_{camera_name}_masked.mp4"

    message_counts = topic_message_counts(bag_path)
    expected_frames = {
        topic: int(min(
            math.ceil(message_counts.get(topic, 0) / args.save_every_n),
            args.max_frames or math.inf,
        ))
        for topic in available_topics
    }
    progress_total = sum(expected_frames.values())
    progress = ProgressBar(
        progress_total, f"  [{scenario_id}] processing", enabled=args.progress
    )

    try:
        while reader.has_next():
            topic, serialized_data, bag_timestamp_ns = reader.read_next()
            if topic not in available_topics:
                continue

            seen[topic] += 1
            if (seen[topic] - 1) % args.save_every_n != 0:
                continue
            if args.max_frames and processed[topic] >= args.max_frames:
                if all(count >= args.max_frames for count in processed.values()):
                    break
                continue

            processed[topic] += 1

            try:
                frame, message = decode_frame(
                    serialized_data, available_topics[topic], bridge
                )
                stamp = message.header.stamp
                message_timestamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                timestamp_ns = message_timestamp_ns or bag_timestamp_ns

                if not args.no_save_frames:
                    filename = (
                        f"{scenario_id}_{camera_names[topic]}_"
                        f"{saved[topic]:06d}_{timestamp_ns}.{args.image_format}"
                    )
                    output_path = output_dirs[topic] / filename
                    if not cv2.imwrite(str(output_path), frame):
                        raise RuntimeError(f"OpenCV failed to write {output_path}")

                if args.generate_masked_video:
                    masked_frame = make_masked_frame(
                        frame,
                        topic_masks[topic],
                        blur_ksize=args.mask_blur_ksize,
                        dim_factor=args.mask_dim_factor,
                        outline_thickness=args.mask_outline_thickness,
                    )
                    scenario_description = config["scenario_options"][scenario_id].get(
                        "description"
                    )
                    if not scenario_description:
                        selected_scenario = config.get("scenario", {})
                        if str(selected_scenario.get("id")) == scenario_id:
                            scenario_description = selected_scenario.get("description")
                    masked_frame = add_video_overlay(
                        masked_frame,
                        scenario_id=scenario_id,
                        description=scenario_description or "No description provided",
                        camera_name=camera_names[topic],
                        frame_number=processed[topic],
                        total_frames=expected_frames[topic],
                        video_fps=args.video_fps,
                    )
                    if topic not in video_writers:
                        height, width = masked_frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(video_paths[topic]),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.video_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"Could not create video: {video_paths[topic]}"
                            )
                        video_writers[topic] = writer
                        print(f"  masked video: {video_paths[topic]}")
                    video_writers[topic].write(masked_frame)

                saved[topic] += 1
            except Exception as error:
                print(f"  [ERROR] {topic}, message {seen[topic]}: {error}", file=sys.stderr)
            finally:
                progress.update()
    finally:
        for writer in video_writers.values():
            writer.release()
        progress.close()

    for topic in available_topics:
        print(f"  done: {topic} -> seen={seen[topic]}, saved={saved[topic]}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract camera frames from MCAP scenarios defined in a YAML config."
    )
    parser.add_argument(
        "--config", default="configs/cf_dataset_mac.yaml", help="Dataset YAML file."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--scenario", nargs="+", help="Scenario ID(s), for example: 1_0 3_1."
    )
    selection.add_argument("--all", action="store_true", help="Process all scenarios.")
    parser.add_argument(
        "--camera",
        nargs="+",
        help="Camera names or full topic paths. Default: playback_options.camera.",
    )
    parser.add_argument("--output-dir", help="Override the output base directory.")
    parser.add_argument("--save-every-n", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        help=(
            "Process only the first N sampled frames per camera. Omit this option "
            "to process the complete video."
        ),
    )
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Do not save individual images (useful with --generate-masked-video).",
    )
    parser.add_argument(
        "--generate-masked-video",
        action="store_true",
        help="Generate a masked MP4 for each selected scenario/camera.",
    )
    parser.add_argument(
        "--safety-areas",
        nargs="+",
        help=(
            "Safety areas to use from data.mask_types in the YAML, for example "
            "PLeft PRight. Default: all configured areas."
        ),
    )
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--mask-blur-ksize", type=int, default=31)
    parser.add_argument("--mask-dim-factor", type=float, default=0.35)
    parser.add_argument("--mask-outline-thickness", type=int, default=6)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show an optional terminal progress bar while processing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; save nothing.")
    args = parser.parse_args()
    if args.save_every_n < 1:
        parser.error("--save-every-n must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.video_fps <= 0:
        parser.error("--video-fps must be greater than zero")
    if args.no_save_frames and not args.generate_masked_video and not args.dry_run:
        parser.error("--no-save-frames requires --generate-masked-video")
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

    print(f"Scenarios: {', '.join(scenario_ids)}")
    print(f"Camera(s): {', '.join(args.camera)}")
    print(f"Output base: {output_base}")
    print(f"Mode: {'dry run' if args.dry_run else 'extract frames'}")

    results = [
        extract_scenario(config, scenario_id, args, output_base)
        for scenario_id in scenario_ids
    ]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
