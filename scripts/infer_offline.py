#!/usr/bin/env python3
"""Run trained safety-area models on folders, video files, or ROS bags."""

import argparse
import csv
import json
import re
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from matplotlib import colormaps
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import utils_model as utmc
from utils_model import Decoder, Discriminator, Encoder


ALL_AREAS = ("PLeft", "PRight", "RoboArm", "ConvBelt")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cf_dataset_epito.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline anomaly inference for cropped data, frames, MP4, or rosbag."
    )
    parser.add_argument(
        "--input_type", required=True,
        choices=("cropped", "frames", "video", "rosbag"),
    )
    parser.add_argument(
        "--input", type=Path,
        help="Input path. Cropped mode defaults to data.training from --config.",
    )
    parser.add_argument(
        "--scenario",
        help="Rosbag scenario ID such as 13_1; resolved under data.dataset_base/rosbags.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset_version", default="V6")
    parser.add_argument("--safety_areas", nargs="+", default=list(ALL_AREAS))
    parser.add_argument(
        "--mask", action="append", default=[], metavar="AREA=PATH",
        help="Full-frame mask. Repeat once per selected area.",
    )
    parser.add_argument(
        "--topic", default="/camera/back_view/image_raw",
        help="Image topic used in rosbag mode.",
    )
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--threshold_dir", type=Path)
    parser.add_argument("--latent_dims", type=int)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument(
        "--skip-first", "--skip_first", dest="skip_first", type=int, default=0
    )
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model_variant", choices=("old", "new"), default="old")
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--output_video", type=Path)
    parser.add_argument("--timeline_video", type=Path)
    parser.add_argument("--timeline_png", type=Path)
    parser.add_argument("--output_fps", type=float, default=10.0)
    parser.add_argument("--timeline_history", type=int, default=500)
    parser.add_argument("--timeline_seconds", type=float, default=4.0)
    args = parser.parse_args()

    if args.frame_stride < 1:
        parser.error("--frame_stride must be at least 1")
    if args.skip_first < 0:
        parser.error("--skip-first must be zero or greater")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max_frames must be at least 1")
    if args.output_fps <= 0 or args.timeline_history < 2 or args.timeline_seconds < 0:
        parser.error(
            "--output_fps must be positive, --timeline_history at least 2, "
            "and --timeline_seconds non-negative"
        )
    if len(args.safety_areas) == 1 and args.safety_areas[0].upper() == "ALL":
        args.safety_areas = list(ALL_AREAS)
    unknown = sorted(set(args.safety_areas).difference(ALL_AREAS))
    if unknown:
        parser.error(f"Unknown safety area(s): {', '.join(unknown)}")
    return args


def scenario_id_from_rosbag(path):
    """Return IDs such as 13_1 from Jul27_Scenario_13_1_... names."""
    match = re.search(r"(?:^|_)Scenario_(\d+_\d+)(?:_|$)", path.name, re.IGNORECASE)
    return match.group(1) if match else None


def resolve_rosbag_scenario(scenario_id, data_config):
    scenario_id = re.sub(r"^Scenario_", "", str(scenario_id), flags=re.IGNORECASE)
    if not re.fullmatch(r"\d+_\d+", scenario_id):
        raise ValueError(f"Invalid scenario ID {scenario_id!r}; expected a value like 13_1")
    dataset_base = data_config.get("dataset_base")
    if not dataset_base:
        raise ValueError("Scenario lookup requires data.dataset_base in --config")
    rosbag_root = Path(dataset_base).expanduser() / "rosbags"
    matches = sorted(rosbag_root.glob(f"*_Scenario_{scenario_id}_*"))
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise ValueError(
            f"Expected one rosbag for scenario {scenario_id} under {rosbag_root}; "
            f"found {len(matches)}: {names}"
        )
    return matches[0].resolve(), scenario_id


def resolve_path(value, repository_root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def load_settings(args):
    repository_root = Path(__file__).resolve().parent.parent
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    data_config = config.get("data") or {}
    model_config = config.get("models") or {}
    args.scenario_id = None
    if args.input_type == "rosbag":
        if args.scenario and args.input is not None:
            raise ValueError("Use either --scenario or --input for rosbag mode, not both")
        scenario = args.scenario
        if scenario is None and args.input is not None:
            candidate = args.input.expanduser()
            if len(candidate.parts) == 1 and not candidate.exists():
                scenario = str(args.input)
        if scenario is not None:
            args.input, args.scenario_id = resolve_rosbag_scenario(
                scenario, data_config
            )
    if args.input is None:
        if args.input_type != "cropped" or not data_config.get("training"):
            raise ValueError("Provide --input, or use --scenario ID for rosbag mode")
        args.input = Path(data_config["training"])
    args.input = args.input.expanduser().resolve()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.input_type == "rosbag" and args.scenario_id is None:
        args.scenario_id = scenario_id_from_rosbag(args.input)

    checkpoint_value = args.checkpoints or model_config.get("checkpoints")
    if not checkpoint_value:
        checkpoint_value = f"results/{args.dataset_version}/models_{args.dataset_version}"
    args.checkpoints = resolve_path(checkpoint_value, repository_root)
    args.threshold_dir = (
        args.threshold_dir.expanduser().resolve()
        if args.threshold_dir
        else repository_root / "results" / args.dataset_version / "thresholds"
    )
    args.latent_dims = args.latent_dims or int(model_config.get("latent_dims", 64))
    args.config_masks_dir = None
    if data_config.get("masks"):
        args.config_masks_dir = Path(data_config["masks"]).expanduser().resolve()
    args.output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else repository_root / "results" / args.dataset_version / "offline_inference" /
        f"{args.input_type}_scores.csv"
    )
    output_root = args.output_csv.parent
    scenario_id = (
        scenario_id_from_rosbag(args.input)
        if args.input_type == "rosbag" else None
    )
    default_video_name = (
        f"rosbag_{scenario_id}_detections.mp4" if scenario_id
        else f"{args.input_type}_detections.mp4"
    )
    args.output_video = (
        args.output_video.expanduser().resolve()
        if args.output_video else output_root / default_video_name
    )
    args.timeline_video = (
        args.timeline_video.expanduser().resolve()
        if args.timeline_video else output_root / f"{args.input_type}_timeline.mp4"
    )
    args.timeline_png = (
        args.timeline_png.expanduser().resolve()
        if args.timeline_png else output_root / f"{args.input_type}_timeline.png"
    )
    return args


def parse_masks(values, areas, masks_dir=None):
    masks = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --mask {value!r}; expected AREA=PATH")
        area, raw_path = value.split("=", 1)
        if area in masks:
            raise ValueError(f"Duplicate mask for {area}")
        path = Path(raw_path).expanduser().resolve()
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask for {area}: {path}")
        masks[area] = mask
    missing = [area for area in areas if area not in masks]
    if missing and masks_dir and masks_dir.is_dir():
        candidates = [
            path for path in masks_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        for area in missing:
            matches = [path for path in candidates if area.lower() in path.stem.lower()]
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple masks found for {area} in {masks_dir}; "
                    "use --mask AREA=PATH"
                )
            if matches:
                mask = cv2.imread(str(matches[0]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Cannot read mask for {area}: {matches[0]}")
                masks[area] = mask
                print(f"[mask] {area}: {matches[0]}")
    missing = [area for area in areas if area not in masks]
    if missing:
        raise ValueError(
            "Full-frame input requires masks for: " + ", ".join(missing)
        )
    return masks


def load_threshold(threshold_dir, area):
    path = threshold_dir / area / f"threshold_{area}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Threshold config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    return {
        "threshold": float(config["threshold"]),
        "offset": int(config["offset"]),
        "sigma": float(config["sigma"]),
        "quantile": float(config["quantile"]),
    }


def load_models(args, device):
    models = OrderedDict()
    for area in args.safety_areas:
        encoder = Encoder(z_size=args.latent_dims).to(device)
        decoder = Decoder(z_size=args.latent_dims).to(device)
        discriminator = Discriminator().to(device)
        optimizer_ed, optimizer_d = utmc.get_optimizers(
            encoder, decoder, discriminator, verbose=False
        )
        suffix = f"{area}_{args.latent_dims}"
        history, checkpoint_config = utmc.load_model(
            encoder, decoder, discriminator, optimizer_ed, optimizer_d,
            str(args.checkpoints), suffix, device=device, verbose=False,
            model_variant=args.model_variant,
        )
        if not history and checkpoint_config is None:
            raise RuntimeError(
                f"Checkpoint not found: {args.checkpoints / f'model_{suffix}.pt'}"
            )
        encoder.eval()
        decoder.eval()
        models[area] = {
            "encoder": encoder,
            "decoder": decoder,
            "threshold": load_threshold(args.threshold_dir, area),
        }
        print(f"[loaded] {area}: model_{suffix}.pt")
    return models


def resize_for_model(image, target_size=128):
    """Resize exactly like the stretched safety-area crops used for training."""
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)


def crop_area(frame, mask):
    height, width = frame.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(binary > 0)
    if not len(xs) or not len(ys):
        raise ValueError("Safety-area mask is empty")
    masked = cv2.bitwise_and(frame, frame, mask=binary)
    return masked[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def to_tensor(image, normalize, device):
    resized = resize_for_model(image)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return normalize(tensor).unsqueeze(0).to(device)


def tensor_to_hwc(tensor):
    return (
        tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        * 0.5 + 0.5
    )


def distance_offset(image_a, image_b, offset):
    height, width, _ = image_a.shape
    distance = np.full((height, width), np.inf, dtype=np.float32)
    for row_offset in range(-offset, offset + 1):
        for col_offset in range(-offset, offset + 1):
            a_r0, a_r1 = max(0, row_offset), min(height, height + row_offset)
            b_r0, b_r1 = max(0, -row_offset), min(height, height - row_offset)
            a_c0, a_c1 = max(0, col_offset), min(width, width + col_offset)
            b_c0, b_c1 = max(0, -col_offset), min(width, width - col_offset)
            delta = (
                image_a[a_r0:a_r1, a_c0:a_c1]
                - image_b[b_r0:b_r1, b_c0:b_c1]
            )
            local = np.sqrt((delta ** 2).sum(axis=2)).astype(np.float32)
            distance[a_r0:a_r1, a_c0:a_c1] = np.minimum(
                distance[a_r0:a_r1, a_c0:a_c1], local
            )
    return distance


def infer_crop(image, area, model, normalize, device):
    input_tensor = to_tensor(image, normalize, device)
    with torch.no_grad():
        mean, _ = model["encoder"](input_tensor)
        reconstruction = model["decoder"](mean)
    original = tensor_to_hwc(input_tensor.squeeze(0))
    reconstructed = tensor_to_hwc(reconstruction.squeeze(0))
    threshold_config = model["threshold"]
    distance = distance_offset(original, reconstructed, threshold_config["offset"])
    if threshold_config["sigma"] > 0:
        distance = gaussian_filter(distance, sigma=threshold_config["sigma"])
    score = float(np.quantile(distance, threshold_config["quantile"]))
    threshold = threshold_config["threshold"]
    normalized = score / threshold
    original_bgr = (original[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
    reconstructed_bgr = (
        reconstructed[..., ::-1] * 255
    ).clip(0, 255).astype(np.uint8)
    heat = np.clip(distance / max(2.0 * threshold, 1e-6), 0.0, 1.0)
    anomaly_rgb = (colormaps["Reds"](heat)[..., :3] * 255).astype(np.uint8)
    anomaly_bgr = cv2.cvtColor(anomaly_rgb, cv2.COLOR_RGB2BGR)
    return {
        "safety_area": area,
        "anomaly_score": score,
        "threshold": threshold,
        "normalized_score": normalized,
        "is_anomalous": normalized > 1.0,
        "original_bgr": original_bgr,
        "reconstructed_bgr": reconstructed_bgr,
        "anomaly_bgr": anomaly_bgr,
    }


def image_paths(root):
    if root.is_file():
        if root.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file: {root}")
        return [root]
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sampled_total(frame_count, args):
    remaining = max(0, int(frame_count) - args.skip_first)
    total = (remaining + args.frame_stride - 1) // args.frame_stride
    return min(total, args.max_frames) if args.max_frames else total


def rosbag_topic_total(args):
    bag_dir = args.input if args.input.is_dir() else args.input.parent
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return args.max_frames
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    bag_info = metadata.get("rosbag2_bagfile_information") or {}
    for entry in bag_info.get("topics_with_message_count") or []:
        topic = (entry.get("topic_metadata") or {}).get("name")
        if topic == args.topic:
            return sampled_total(entry.get("message_count", 0), args)
    return args.max_frames


def input_progress_total(args):
    """Return an accurate tqdm total for bounded and complete input runs."""
    if args.input_type == "rosbag":
        return rosbag_topic_total(args)
    if args.input_type == "frames":
        return sampled_total(len(image_paths(args.input)), args)
    if args.input_type == "video":
        capture = cv2.VideoCapture(str(args.input))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        return sampled_total(count, args) if count > 0 else args.max_frames
    if args.input_type == "cropped":
        counts = []
        for area in args.safety_areas:
            area_root = args.input / area if (args.input / area).is_dir() else args.input
            counts.append(len(image_paths(area_root)[args.skip_first::args.frame_stride]))
        total = min(counts) if counts else 0
        return min(total, args.max_frames) if args.max_frames else total
    return args.max_frames


def iter_cropped(args):
    has_area_directories = all((args.input / area).is_dir() for area in args.safety_areas)
    if len(args.safety_areas) > 1 and not has_area_directories:
        raise ValueError(
            "Cropped input with multiple safety areas must contain one directory "
            "per area. Select a single area for a directory of crops."
        )
    for area in args.safety_areas:
        area_root = args.input / area if (args.input / area).is_dir() else args.input
        paths = image_paths(area_root)
        if not paths:
            raise FileNotFoundError(f"No images found for {area}: {area_root}")
        selected_paths = paths[args.skip_first::args.frame_stride]
        if args.max_frames:
            selected_paths = selected_paths[:args.max_frames]
        for path in selected_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[warning] skipped unreadable image: {path}")
                continue
            yield str(path), area, image



def iter_cropped_groups(args):
    """Yield one sorted crop per area for a combined dashboard frame."""
    paths_by_area = OrderedDict()
    for area in args.safety_areas:
        area_root = args.input / area if (args.input / area).is_dir() else args.input
        paths = image_paths(area_root)[args.skip_first::args.frame_stride]
        if not paths:
            raise FileNotFoundError(f"No images found for {area}: {area_root}")
        paths_by_area[area] = paths
    group_count = min(len(paths) for paths in paths_by_area.values())
    if args.max_frames:
        group_count = min(group_count, args.max_frames)
    for index in range(group_count):
        crops = OrderedDict()
        source_paths = OrderedDict()
        for area, paths in paths_by_area.items():
            path = paths[index]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read cropped image: {path}")
            crops[area] = image
            source_paths[area] = str(path)
        yield f"combined_{index:06d}", crops, source_paths


def area_grid(images, results=None):
    """Arrange all selected safety-area images in a labelled 2x2 grid."""
    canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    positions = ((0, 0), (480, 0), (0, 270), (480, 270))
    for (area, image), (x, y) in zip(images.items(), positions):
        tile = fit_image(image, 480, 270, (0, 0, 0))
        result = results.get(area) if results else None
        color = (0, 0, 255) if result and result["is_anomalous"] else (0, 220, 0)
        label = area
        if result:
            status = "UNEXPECTED" if result["is_anomalous"] else "normal"
            label = f"{area}  {status}: {result['normalized_score']:.2f}x"
        cv2.rectangle(tile, (0, 0), (480, 34), (20, 20, 20), -1)
        cv2.putText(
            tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, color, 2, cv2.LINE_AA,
        )
        canvas[y:y + 270, x:x + 480] = tile
    return canvas

def iter_frames(args):
    paths = image_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"No images found: {args.input}")
    for index, path in enumerate(paths):
        if index < args.skip_first or (index - args.skip_first) % args.frame_stride:
            continue
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is not None:
            yield str(path), frame


def iter_video(args):
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= args.skip_first and (index - args.skip_first) % args.frame_stride == 0:
                yield f"{args.input}#frame={index}", frame
            index += 1
    finally:
        capture.release()



def decode_cdr_compressed_image(serialized):
    """Extract JPEG/PNG bytes from a ROS 2 CDR CompressedImage message."""
    import struct

    data = memoryview(serialized)
    if len(data) < 16:
        raise ValueError("CompressedImage CDR payload is too short")
    little_endian = data[1] == 1
    byte_order = "<" if little_endian else ">"
    offset = 4

    def align(alignment):
        nonlocal offset
        relative = offset - 4
        offset = 4 + ((relative + alignment - 1) // alignment) * alignment

    def uint32():
        nonlocal offset
        align(4)
        value = struct.unpack_from(byte_order + "I", data, offset)[0]
        offset += 4
        return value

    # std_msgs/Header.stamp: int32 sec + uint32 nanosec
    align(4)
    offset += 8
    # std_msgs/Header.frame_id and sensor_msgs/CompressedImage.format strings.
    for _ in range(2):
        length = uint32()
        if offset + length > len(data):
            raise ValueError("Invalid CDR string length in CompressedImage")
        offset += length
    encoded_length = uint32()
    if offset + encoded_length > len(data):
        raise ValueError("Invalid image-data length in CompressedImage")
    return np.frombuffer(data[offset:offset + encoded_length], dtype=np.uint8)


def mcap_file_from_input(input_path):
    if input_path.is_file():
        return input_path
    files = sorted(input_path.glob("*.mcap"))
    if len(files) != 1:
        raise ValueError(
            f"Expected one .mcap file in {input_path}, found {len(files)}"
        )
    return files[0]


def iter_mcap_without_ros(args):
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 modules are unavailable. Install the lightweight fallback "
            "with: python -m pip install mcap"
        ) from error

    mcap_path = mcap_file_from_input(args.input)
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream)
        index = 0
        found_topic = False
        for _, channel, message in reader.iter_messages(topics=[args.topic]):
            found_topic = True
            if index >= args.skip_first and (index - args.skip_first) % args.frame_stride == 0:
                encoded = decode_cdr_compressed_image(message.data)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(
                        f"Could not decode compressed frame at {message.log_time}"
                    )
                label = args.scenario_id or args.input.name
                yield f"scenario={label}#timestamp={message.log_time}", frame
            index += 1
        if not found_topic:
            raise ValueError(f"Image topic {args.topic!r} not found in {mcap_path}")

def iter_rosbag(args):
    """Read ROS 2 image messages, decoding compressed images without CvBridge."""
    try:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from sensor_msgs.msg import CompressedImage, Image
    except ImportError:
        yield from iter_mcap_without_ros(args)
        return
    image_types = {
        "sensor_msgs/msg/CompressedImage": CompressedImage,
        "sensor_msgs/msg/Image": Image,
    }
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(args.input), storage_id="mcap"),
        ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    message_type = topic_types.get(args.topic)
    if message_type not in image_types:
        available = sorted(
            topic for topic, kind in topic_types.items() if kind in image_types
        )
        raise ValueError(
            f"Image topic {args.topic!r} not found. Available: {', '.join(available)}"
        )

    bridge = None
    if message_type == "sensor_msgs/msg/Image":
        try:
            from cv_bridge import CvBridge
        except ImportError as error:
            raise RuntimeError(
                "Raw sensor_msgs/Image decoding requires cv_bridge; compressed "
                "image bags do not."
            ) from error
        bridge = CvBridge()

    index = 0
    while reader.has_next():
        topic, serialized, timestamp = reader.read_next()
        if topic != args.topic:
            continue
        if index >= args.skip_first and (index - args.skip_first) % args.frame_stride == 0:
            message = deserialize_message(serialized, image_types[message_type])
            if message_type == "sensor_msgs/msg/CompressedImage":
                frame = cv2.imdecode(
                    np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
            else:
                frame = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            if frame is None:
                raise RuntimeError(
                    f"Decoded an empty image at rosbag timestamp {timestamp}"
                )
            label = args.scenario_id or args.input.name
            yield f"scenario={label}#timestamp={timestamp}", frame
        index += 1



def public_result(result):
    """Drop image arrays before writing a result to CSV."""
    return {
        key: result[key] for key in (
            "safety_area", "anomaly_score", "threshold",
            "normalized_score", "is_anomalous",
        )
    }


def annotate_full_frame(frame, masks, results):
    """Draw each safety-area boundary, status, and normalized score."""
    output = frame.copy()
    height, width = output.shape[:2]
    for area, result in results.items():
        mask = masks[area]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        color = (0, 0, 255) if result["is_anomalous"] else (0, 220, 0)
        cv2.drawContours(output, contours, -1, color, 3)
        if contours:
            x, y, _, _ = cv2.boundingRect(np.vstack(contours))
            status = "UNEXPECTED" if result["is_anomalous"] else "normal"
            label = f"{area} {status}: {result['normalized_score']:.2f}x"
            cv2.putText(
                output, label, (x, max(28, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA,
            )
    return output

def fit_image(image, width, height, background=(0, 0, 0)):
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w = max(1, int(source_w * scale))
    new_h = max(1, int(source_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def paste_patch(canvas, patch, mask):
    """Project a stretched model patch back into its safety-area bounding box."""
    height, width = canvas.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(binary > 0)
    if not len(xs) or not len(ys):
        return
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    crop_width = x2 - x1 + 1
    crop_height = y2 - y1 + 1

    resized = cv2.resize(
        patch, (crop_width, crop_height), interpolation=cv2.INTER_AREA
    )
    region_mask = binary[y1:y2 + 1, x1:x2 + 1, None] > 0
    target = canvas[y1:y2 + 1, x1:x2 + 1]
    canvas[y1:y2 + 1, x1:x2 + 1] = np.where(region_mask, resized, target)


def make_advis_dashboard(frame, masks, results, sample_id, cropped_area=None):
    """Create the four-panel ADVIS dashboard used by the Zenoh viewer."""
    width, height, padding = 1600, 1000, 16
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    panel_w = (width - 3 * padding) // 2
    panel_h = (height - 3 * padding) // 2
    boxes = (
        (padding, padding, padding + panel_w, padding + panel_h),
        (2 * padding + panel_w, padding, width - padding, padding + panel_h),
        (padding, 2 * padding + panel_h, padding + panel_w, height - padding),
        (2 * padding + panel_w, 2 * padding + panel_h, width - padding, height - padding),
    )
    titles = ("Input View", "Unexpected Situations View", "AI View", "Details")
    inner_boxes = []
    for title, (x1, y1, x2, y2) in zip(titles, boxes):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (45, 45, 45), 1)
        cv2.putText(
            canvas, title, (x1 + 12, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (20, 20, 20), 2, cv2.LINE_AA,
        )
        inner_boxes.append((x1 + 12, y1 + 42, x2 - 12, y2 - 12))

    if cropped_area is not None:
        input_view = area_grid(frame, results)
        anomaly_view = area_grid(
            OrderedDict((area, result["anomaly_bgr"]) for area, result in results.items()),
            results,
        )
        ai_view = area_grid(
            OrderedDict(
                (area, result["reconstructed_bgr"])
                for area, result in results.items()
            ),
            results,
        )
    else:
        input_view = annotate_full_frame(frame, masks, results)
        anomaly_view = np.full_like(frame, 255)
        ai_view = np.zeros_like(frame)
        for area, result in results.items():
            paste_patch(anomaly_view, result["anomaly_bgr"], masks[area])
            paste_patch(ai_view, result["reconstructed_bgr"], masks[area])
        anomaly_view = annotate_full_frame(anomaly_view, masks, results)
        ai_view = annotate_full_frame(ai_view, masks, results)

    for image, box, background in (
        (input_view, inner_boxes[0], (0, 0, 0)),
        (anomaly_view, inner_boxes[1], (255, 255, 255)),
        (ai_view, inner_boxes[2], (0, 0, 0)),
    ):
        x1, y1, x2, y2 = box
        canvas[y1:y2, x1:x2] = fit_image(image, x2 - x1, y2 - y1, background)

    x1, y1, x2, y2 = inner_boxes[3]
    details = canvas[y1:y2, x1:x2]
    details[:] = (245, 245, 245)
    cv2.line(details, (20, 24), (details.shape[1] - 20, 24), (40, 40, 40), 2)
    cv2.putText(
        details, f"Sample: {str(sample_id)[-75:]}", (25, 58),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA,
    )
    headers = ("Safety Area", "RawVal", "Threshold", "Score", "Status")
    columns = (25, 230, 345, 475, 590)
    y = 105
    for column, header in zip(columns, headers):
        cv2.putText(
            details, header, (column, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (25, 25, 25), 2, cv2.LINE_AA,
        )
    y += 20
    cv2.line(details, (20, y), (details.shape[1] - 20, y), (80, 80, 80), 1)
    y += 38
    display_names = {
        "PLeft": "Pallet Left", "PRight": "Pallet Right",
        "RoboArm": "Robo Arm", "ConvBelt": "Conveyor Belt",
    }
    for area, result in results.items():
        status = "UNEXPECTED" if result["is_anomalous"] else "normal"
        color = (0, 0, 210) if result["is_anomalous"] else (0, 145, 0)
        values = (
            display_names.get(area, area),
            f"{result['anomaly_score']:.3f}",
            f"{result['threshold']:.3f}",
            f"{result['normalized_score']:.3f}",
            status,
        )
        for index, (column, value) in enumerate(zip(columns, values)):
            cv2.putText(
                details, value, (column, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, color if index >= 3 else (35, 35, 35),
                2 if index >= 3 else 1, cv2.LINE_AA,
            )
        y += 20
        cv2.line(details, (20, y), (details.shape[1] - 20, y), (150, 150, 150), 1)
        y += 38
    return canvas

def render_dashboard(visual, histories, latest, sample_id, final=False):
    """Render a fixed-size frame with detections and per-area timelines."""
    canvas = np.full((720, 1280, 3), 18, dtype=np.uint8)
    if visual is not None:
        view = cv2.resize(visual, (1240, 400), interpolation=cv2.INTER_AREA)
        canvas[42:442, 20:1260] = view
    heading = "Final detection timeline" if final else str(sample_id)
    cv2.putText(
        canvas, heading[:110], (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (245, 245, 245), 2, cv2.LINE_AA,
    )

    areas = list(histories)
    panel_top, panel_bottom = (70, 700) if final else (458, 700)
    row_height = max(52, (panel_bottom - panel_top) // max(1, len(areas)))
    chart_left, chart_right = 235, 1245
    for row, area in enumerate(areas):
        y_top = panel_top + row * row_height
        y_bottom = min(panel_bottom, y_top + row_height - 8)
        values = list(histories[area])
        current = latest.get(area)
        color = (0, 0, 255) if current and current["is_anomalous"] else (0, 220, 0)
        label = area
        if current:
            label += (
                f"  score={current['anomaly_score']:.4f}  "
                f"norm={current['normalized_score']:.2f}x"
            )
        cv2.putText(
            canvas, label, (20, y_top + 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, color, 1, cv2.LINE_AA,
        )
        threshold_y = int(y_bottom - (1.0 / 2.5) * (y_bottom - y_top))
        cv2.line(
            canvas, (chart_left, threshold_y), (chart_right, threshold_y),
            (80, 80, 230), 1, cv2.LINE_AA,
        )
        if len(values) >= 2:
            clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 2.5)
            xs = np.linspace(chart_left, chart_right, len(clipped)).astype(np.int32)
            ys = (
                y_bottom - clipped / 2.5 * max(1, y_bottom - y_top)
            ).astype(np.int32)
            points = np.column_stack((xs, ys)).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], False, (255, 190, 40), 2, cv2.LINE_AA)
        cv2.rectangle(
            canvas, (chart_left, y_top), (chart_right, y_bottom),
            (90, 90, 90), 1,
        )
    return canvas


class VideoOutput:
    def __init__(self, path, fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1600, 1000)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {path}")

    def write(self, frame):
        if frame.shape[:2] != (1000, 1600):
            frame = cv2.resize(frame, (1600, 1000), interpolation=cv2.INTER_AREA)
        self.writer.write(frame)

    def close(self):
        self.writer.release()

def main():
    args = load_settings(parse_args())
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[device] {device}")
    print(f"[input] {args.input_type}: {args.input}")
    print(f"[checkpoints] {args.checkpoints}")
    models = load_models(args, device)
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    masks = None
    if args.input_type != "cropped":
        masks = parse_masks(args.mask, args.safety_areas, args.config_masks_dir)

    histories = OrderedDict(
        (area, deque(maxlen=args.timeline_history)) for area in args.safety_areas
    )
    latest = OrderedDict()
    rows = []
    video = VideoOutput(args.output_video, args.output_fps)
    timeline_video = VideoOutput(args.timeline_video, args.output_fps)
    try:
        if args.input_type == "cropped":
            progress = tqdm(
                iter_cropped_groups(args), total=input_progress_total(args),
                desc="Offline inference [cropped]", unit="combined frame",
                dynamic_ncols=True,
            )
            for sample_id, crops, source_paths in progress:
                frame_results = OrderedDict()
                for area, crop in crops.items():
                    result = infer_crop(
                        crop, area, models[area], normalize, device
                    )
                    rows.append({
                        "sample_id": source_paths[area],
                        **public_result(result),
                    })
                    histories[area].append(float(result["normalized_score"]))
                    latest[area] = result
                    frame_results[area] = result
                status_text = ", ".join(
                    f"{area}={result['normalized_score']:.2f}x"
                    for area, result in frame_results.items()
                )
                progress.set_postfix_str(status_text)
                dashboard = make_advis_dashboard(
                    crops, None, frame_results, sample_id, cropped_area="ALL"
                )
                video.write(dashboard)
                timeline_video.write(
                    render_dashboard(None, histories, latest, sample_id, final=True)
                )
        else:
            factories = {
                "frames": iter_frames,
                "video": iter_video,
                "rosbag": iter_rosbag,
            }
            processed_frames = 0
            progress = tqdm(
                factories[args.input_type](args), total=input_progress_total(args),
                desc=f"Offline inference [{args.input_type}]", unit="frame",
                dynamic_ncols=True,
            )
            for sample_id, frame in progress:
                frame_results = OrderedDict()
                for area in args.safety_areas:
                    crop = crop_area(frame, masks[area])
                    result = infer_crop(
                        crop, area, models[area], normalize, device
                    )
                    rows.append({"sample_id": sample_id, **public_result(result)})
                    histories[area].append(float(result["normalized_score"]))
                    latest[area] = result
                    frame_results[area] = result
                status_text = ", ".join(
                    f"{area}={result['normalized_score']:.2f}x"
                    for area, result in frame_results.items()
                )
                progress.set_postfix_str(status_text)
                dashboard = make_advis_dashboard(
                    frame, masks, frame_results, sample_id
                )
                video.write(dashboard)
                timeline_video.write(
                    render_dashboard(None, histories, latest, sample_id, final=True)
                )
                processed_frames += 1
                if args.max_frames and processed_frames >= args.max_frames:
                    break

        final_timeline = render_dashboard(
            None, histories, latest, "Final detection timeline", final=True
        )
        args.timeline_png.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.timeline_png), final_timeline):
            raise RuntimeError(f"Cannot save timeline image: {args.timeline_png}")
    finally:
        video.close()
        timeline_video.close()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "sample_id", "safety_area", "anomaly_score", "threshold",
        "normalized_score", "is_anomalous",
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {len(rows):,} area results -> {args.output_csv}")
    print(f"[save] detection video -> {args.output_video}")
    print(f"[save] timeline video -> {args.timeline_video}")
    print(f"[save] final timeline -> {args.timeline_png}")


if __name__ == "__main__":
    main()
