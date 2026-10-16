#!/usr/bin/env python3
"""Export original-resolution MCAP camera frames for R2AD grouped scenario splits.

Reuses the ROS2/MCAP readers from process_rosbags_from_config.py.
No masks, ROI cropping, or image resizing are applied.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
from cv_bridge import CvBridge

from process_rosbags_from_config import (
    IMAGE_TYPES, ProgressBar, camera_topic, decode_frame, inspect_bag,
    load_config, resolve_bag_path, resolve_scenarios, topic_message_counts,
)


def resolve_category(scenario_id, config):
    """Use scenario_categories, e.g. {'1-7': 'normal', '8-16': 'anomalous'}."""
    major = int(scenario_id.split("_", 1)[0])
    for key, value in (config.get("scenario_categories") or {}).items():
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", str(key))
        if not match:
            continue
        low = int(match.group(1))
        high = int(match.group(2) or low)
        if low <= major <= high:
            category = str(value).strip().lower()
            if category in ("normal", "expected"):
                return "normal"
            if category in ("anomalous", "anomaly", "unexpected", "abnormal"):
                return "unexpected"
            raise ValueError(f"Unsupported category {value!r} for {key!r}")
    raise ValueError(
        f"No scenario_categories mapping for scenario {scenario_id}. "
        "Add it to the YAML or use --category normal/unexpected."
    )


def get_source_root(config, category, output_dir):
    """Use existing YAML paths when present, without changing the source config."""
    base = Path(config["data"]["dataset_base"]).expanduser()
    if output_dir:
        return Path(output_dir).expanduser() / category
    r2ad = config.get("r2ad") or {}
    configured = r2ad.get("normal_source" if category == "normal" else "unexpected_source")
    if configured and "REPLACE_ME" not in str(configured):
        path = Path(configured).expanduser()
        return path if path.is_absolute() else base / path
    return base / "r2ad" / "sources" / category


def process_one(config, scenario_id, args):
    category = args.category if args.category != "auto" else resolve_category(scenario_id, config)
    bag_path = resolve_bag_path(config, scenario_id)
    if not bag_path.is_file():
        print(f"[ERROR] {scenario_id}: MCAP not found: {bag_path}", file=sys.stderr)
        return False

    reader, types = inspect_bag(bag_path)
    requested = {camera_topic(c): c for c in args.camera}
    available = {topic: types[topic] for topic in requested if types.get(topic) in IMAGE_TYPES}
    if not available:
        print(f"[ERROR] {scenario_id}: requested camera topics not found. Bag has: {list(types)}", file=sys.stderr)
        return False

    source_root = get_source_root(config, category, args.output_dir)
    dirs = {}
    for topic, camera in requested.items():
        if topic not in available:
            continue
        # Keep the grouped folder exactly one level below normal_source/unexpected_source.
        # For multiple cameras, encode the camera in file names to avoid collisions.
        path = source_root / f"scenario_{scenario_id}"
        path.mkdir(parents=True, exist_ok=True)
        dirs[topic] = path
        print(f"[{scenario_id}] {category} / {camera}: {path}")

    counts = topic_message_counts(bag_path)
    total = sum(min(math.ceil(counts.get(t, 0) / args.save_every_n),
                    args.max_frames or math.inf) for t in available)
    progress = ProgressBar(total, f"[{scenario_id}] full-resolution frames", enabled=args.progress)
    bridge = CvBridge()
    seen = {t: 0 for t in available}
    saved = {t: 0 for t in available}
    failed = {t: 0 for t in available}
    first_shape = {}
    try:
        while reader.has_next():
            topic, data, _timestamp = reader.read_next()
            if topic not in available:
                continue
            seen[topic] += 1
            if (seen[topic] - 1) % args.save_every_n != 0:
                continue
            if args.max_frames is not None and saved[topic] >= args.max_frames:
                if all(saved[t] >= args.max_frames for t in available):
                    break
                continue
            sample_idx = (seen[topic] - 1) // args.save_every_n
            try:
                frame, _ = decode_frame(data, available[topic], bridge)
                first_shape.setdefault(topic, list(frame.shape))
                cam_name = requested[topic].strip("/").split("/")[-1]
                # Bare frame_000001.png for a single camera.
                prefix = "frame" if len(available) == 1 else f"{cam_name}_frame"
                path = dirs[topic] / f"{prefix}_{sample_idx:06d}.{args.image_format}"
                if path.exists() and not args.overwrite:
                    raise FileExistsError(f"{path} exists (use --overwrite to replace it)")
                options = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality] if args.image_format == "jpg" else []
                if not cv2.imwrite(str(path), frame, options):
                    raise OSError(f"Cannot write {path}")
                saved[topic] += 1
            except Exception as exc:
                failed[topic] += 1
                print(f"\n[ERROR] {scenario_id} {topic} sample {sample_idx}: {exc}", file=sys.stderr)
            finally:
                progress.update()
            if args.max_frames is not None and all(saved[t] >= args.max_frames for t in available):
                break
    finally:
        progress.close()

    for topic in available:
        print(f"[{scenario_id}] {topic}: seen={seen[topic]} saved={saved[topic]} failed={failed[topic]}, shape={first_shape.get(topic)}")
    if any(failed.values()):
        print("[WARNING] Some frames failed; inspect errors before using this scenario.", file=sys.stderr)
    metadata = {
        "scenario": scenario_id,
        "category": category,
        "bag": str(bag_path),
        "camera_topics": list(available),
        "frame_shapes": first_shape,
        "saved_counts": saved,
        "failed_counts": failed,
        "save_every_n": args.save_every_n,
        "original_resolution": True,
    }
    # Metadata is outside scenario images to avoid interfering with frame readers.
    metadata_dir = source_root / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / f"scenario_{scenario_id}.json").write_text(json.dumps(metadata, indent=2))
    return bool(sum(saved.values())) and not any(failed.values())


def parse_args():
    p = argparse.ArgumentParser(description="Extract full-size MCAP images into grouped R2AD scenarios.")
    p.add_argument("--config", default="configs/cf_dataset_mac.yaml")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--scenario", nargs="+")
    group.add_argument("--all", action="store_true")
    p.add_argument("--camera", nargs="+", default=None)
    p.add_argument("--process-to", choices=["frames"], default="frames",
                   help="Compatibility with original script; only full frames are supported.")
    p.add_argument("--target-size", type=int, default=None,
                   help="Ignored deliberately: R2AD requires unresized full-resolution frames.")
    p.add_argument("--category", choices=["auto", "normal", "unexpected"], default="auto")
    p.add_argument("--output-dir", default=None,
                   help="Optional common parent for normal/ and unexpected/. Otherwise use r2ad.*_source.")
    p.add_argument("--save-every-n", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--image-format", choices=["png", "jpg"], default="png")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()
    if args.save_every_n < 1 or (args.max_frames is not None and args.max_frames < 1):
        p.error("--save-every-n and --max-frames must be positive")
    if args.target_size is not None:
        print("[INFO] --target-size is ignored: R2AD keeps the original camera resolution.")
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    scenarios = resolve_scenarios(config, args.scenario, args.all)
    args.camera = args.camera or [config.get("playback_options", {}).get("camera", "back_view")]
    print(f"[R2AD] scenarios={scenarios} cameras={args.camera} classification={args.category}")
    status = [process_one(config, scenario, args) for scenario in scenarios]
    if not all(status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
