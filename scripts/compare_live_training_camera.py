#!/usr/bin/env python3
"""Compare one live ROS camera frame with a full training-source frame."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRAINING_NAME = re.compile(
    r"^s-(?P<scenario>.+?)_s-(?P<area>.+?)_f-(?P<frame>\d+)"
)


def load_config(path: Path) -> dict:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def find_training_reference(
    config: dict, area: str, camera: str, explicit_reference: Path | None = None,
) -> tuple[Path, Path | None]:
    """Return a full raw source frame and the training crop that selected it."""
    if explicit_reference is not None:
        reference = explicit_reference.expanduser().resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"Training reference not found: {reference}")
        return reference, None

    data = config.get("data") or {}
    training = Path(data.get("training", "")).expanduser().resolve()
    normal_dir = training / area / "normal"
    if not normal_dir.is_dir():
        raise FileNotFoundError(f"Training normal directory not found: {normal_dir}")
    crops = sorted(
        path for path in normal_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not crops:
        raise FileNotFoundError(f"No training images found in: {normal_dir}")

    dataset_session_root = training.parent
    checked = []
    # Search evenly through the sorted training set rather than assuming its
    # first item has already been extracted as a full raw frame.
    indices = np.linspace(0, len(crops) - 1, min(len(crops), 200), dtype=int)
    for index in indices:
        crop = crops[int(index)]
        match = TRAINING_NAME.match(crop.stem)
        if not match:
            continue
        scenario = match.group("scenario")
        frame_id = int(match.group("frame"))
        raw_dir = dataset_session_root / "extracted_frames" / scenario / camera / "raw"
        candidates = [
            raw_dir / f"frame_{frame_id:06d}.png",
            raw_dir / f"frame_{frame_id:06d}.jpg",
            raw_dir / f"frame{frame_id:06d}.png",
        ]
        checked.extend(candidates)
        for reference in candidates:
            if reference.is_file():
                return reference.resolve(), crop.resolve()

    # Some installations retain the processed training crops but remove their
    # original normal-scenario raw frames.  Return an actual training crop and
    # let main() apply the identical safety-area crop to the live image.
    return crops[len(crops) // 2].resolve(), crops[len(crops) // 2].resolve()


def crop_live_like_training(
    live_bgr: np.ndarray, config: dict, area: str, target_size: int,
) -> tuple[np.ndarray, float]:
    """Apply the dataset mask/crop and return pixels-per-crop-pixel scale."""
    mask_value = ((config.get("data") or {}).get("mask_types") or {}).get(area)
    if not mask_value:
        raise KeyError(f"No mask configured for training area {area!r}")
    mask_path = Path(mask_value).expanduser()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load configured {area} mask: {mask_path}")
    height, width = live_bgr.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if not len(xs):
        raise ValueError(f"Configured {area} mask is empty")
    masked = cv2.bitwise_and(live_bgr, live_bgr, mask=binary)
    crop = masked[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop_height, crop_width = crop.shape[:2]
    resize_scale = min(target_size / crop_width, target_size / crop_height)
    resized_width = max(1, round(crop_width * resize_scale))
    resized_height = max(1, round(crop_height * resize_scale))
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    output = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - resized_width) // 2
    y_offset = (target_size - resized_height) // 2
    output[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return output, 1.0 / resize_scale


def load_static_background_mask(config: dict, shape: tuple[int, ...]) -> np.ndarray | None:
    """Use pixels outside all safety areas for camera-pose matching."""
    mask_paths = (config.get("data") or {}).get("mask_types") or {}
    union = np.zeros(shape[:2], dtype=np.uint8)
    loaded = 0
    for value in mask_paths.values():
        mask = cv2.imread(str(Path(value).expanduser()), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != union.shape:
            mask = cv2.resize(
                mask, (union.shape[1], union.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        union[mask > 0] = 255
        loaded += 1
    if not loaded:
        return None
    background = cv2.bitwise_not(union)
    # Erode the usable background away from mask edges, where small alignment
    # differences otherwise generate unstable features.
    background = cv2.erode(background, np.ones((9, 9), np.uint8), iterations=1)
    if cv2.countNonZero(background) < 0.05 * background.size:
        return None
    return background


def estimate_alignment(
    training_bgr: np.ndarray, live_bgr: np.ndarray,
    feature_mask: np.ndarray | None = None,
) -> dict:
    """Estimate the affine transform that maps the live view to training."""
    if training_bgr is None or live_bgr is None:
        raise ValueError("Both training and live images are required")

    train_h, train_w = training_bgr.shape[:2]
    original_live_shape = live_bgr.shape[:2]
    if live_bgr.shape[:2] != (train_h, train_w):
        live_for_match = cv2.resize(live_bgr, (train_w, train_h), interpolation=cv2.INTER_AREA)
    else:
        live_for_match = live_bgr

    train_gray = cv2.cvtColor(training_bgr, cv2.COLOR_BGR2GRAY)
    live_gray = cv2.cvtColor(live_for_match, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=10)
    train_keypoints, train_descriptors = orb.detectAndCompute(train_gray, feature_mask)
    live_keypoints, live_descriptors = orb.detectAndCompute(live_gray, feature_mask)
    if train_descriptors is None or live_descriptors is None:
        raise RuntimeError("Not enough visual features in one or both frames")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(live_descriptors, train_descriptors, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 8:
        raise RuntimeError(f"Only {len(good)} reliable feature matches; need at least 8")

    live_points = np.float32(
        [live_keypoints[match.queryIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    training_points = np.float32(
        [train_keypoints[match.trainIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        live_points, training_points,
        method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=5000, confidence=0.995, refineIters=20,
    )
    if matrix is None or inlier_mask is None:
        raise RuntimeError("Could not estimate live-to-training alignment")

    inliers = int(inlier_mask.ravel().sum())
    a, minus_b, shift_x = matrix[0]
    b, _, shift_y = matrix[1]
    scale = float(math.hypot(a, b))
    rotation_deg = float(math.degrees(math.atan2(b, a)))
    inlier_ratio = inliers / len(good)
    same_angle = (
        len(good) >= 20
        and inliers >= 12
        and inlier_ratio >= 0.25
        and abs(rotation_deg) <= 1.0
        and abs(scale - 1.0) <= 0.02
    )

    aligned = cv2.warpAffine(
        live_for_match, matrix, (train_w, train_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    difference = cv2.absdiff(training_bgr, aligned)
    return {
        "matrix": matrix,
        "aligned_live": aligned,
        "difference": difference,
        "training_keypoints": len(train_keypoints),
        "live_keypoints": len(live_keypoints),
        "reliable_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "shift_x_pixels": float(shift_x),
        "shift_y_pixels": float(shift_y),
        "scale": scale,
        "rotation_degrees": rotation_deg,
        "same_camera_angle": bool(same_angle),
        "training_resolution": [train_w, train_h],
        "live_resolution": [original_live_shape[1], original_live_shape[0]],
    }


def capture_live_frame(topic: str, message_type: str, timeout: float) -> np.ndarray:
    """Capture one ROS Image or CompressedImage frame."""
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, Image
    except ImportError as exc:
        raise RuntimeError("ROS capture requires rclpy, sensor_msgs, and cv_bridge") from exc

    class CaptureNode(Node):
        def __init__(self):
            super().__init__("advis_camera_alignment_capture")
            self.frame = None
            self.bridge = CvBridge()
            self.subscription = None
            self.kind = None
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.qos = qos
            if message_type == "auto":
                self.timer = self.create_timer(0.2, self.discover)
            else:
                self.subscribe(message_type)

        def discover(self):
            types = {info.topic_type for info in self.get_publishers_info_by_topic(topic)}
            if "sensor_msgs/msg/CompressedImage" in types:
                self.subscribe("compressed")
            elif "sensor_msgs/msg/Image" in types:
                self.subscribe("raw")

        def subscribe(self, kind):
            if self.subscription is not None:
                return
            self.kind = kind
            cls = CompressedImage if kind == "compressed" else Image
            self.subscription = self.create_subscription(cls, topic, self.on_frame, self.qos)
            if hasattr(self, "timer"):
                self.timer.cancel()

        def on_frame(self, message):
            if self.kind == "compressed":
                self.frame = cv2.imdecode(
                    np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
            else:
                self.frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    rclpy.init(args=None)
    node = CaptureNode()
    deadline = time.monotonic() + timeout
    try:
        while node.frame is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.frame is None:
            raise TimeoutError(f"No frame received from {topic} within {timeout:.1f}s")
        return node.frame.copy()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def panel(image: np.ndarray, title: str, width: int = 800) -> np.ndarray:
    scale = width / image.shape[1]
    resized = cv2.resize(
        image, (width, max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    cv2.rectangle(resized, (0, 0), (width, 44), (20, 20, 20), -1)
    cv2.putText(
        resized, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return resized


def save_report(
    output_dir: Path, training: np.ndarray, live: np.ndarray, result: dict,
    training_path: Path, training_crop: Path | None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    live_at_training_size = cv2.resize(
        live, (training.shape[1], training.shape[0]), interpolation=cv2.INTER_AREA
    )
    overlay = cv2.addWeighted(training, 0.5, result["aligned_live"], 0.5, 0)
    top = np.hstack([
        panel(training, "Training source frame"),
        panel(live_at_training_size, "Current live frame"),
    ])
    bottom = np.hstack([
        panel(overlay, "Overlay after estimated alignment"),
        panel(result["difference"], "Absolute difference after alignment"),
    ])
    report_image = np.vstack([top, bottom])
    image_path = output_dir / "camera_alignment_comparison.png"
    cv2.imwrite(str(image_path), report_image)

    serializable = {
        key: value for key, value in result.items()
        if key not in {"matrix", "aligned_live", "difference"}
    }
    serializable.update({
        "training_reference": str(training_path),
        "training_crop": str(training_crop) if training_crop else None,
        "estimated_live_to_training_matrix": result["matrix"].tolist(),
        "recommended_frame_shift_y": result.get(
            "recommended_frame_shift_y", int(round(result["shift_y_pixels"]))
        ),
        "interpretation": (
            "approximately_same_angle"
            if result["same_camera_angle"] else "camera_pose_mismatch_or_low_confidence"
        ),
    })
    json_path = output_dir / "camera_alignment_report.json"
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return image_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare a current ROS camera frame with one training-source frame."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera-topic", default="/camera/back_view/image_raw")
    parser.add_argument(
        "--message-type", choices=("auto", "compressed", "raw"), default="auto"
    )
    parser.add_argument("--camera", default="back_view")
    parser.add_argument("--training-area", default="PLeft")
    parser.add_argument("--training-reference", type=Path)
    parser.add_argument("--live-frame", type=Path)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/camera_alignment"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    reference_path, training_crop = find_training_reference(
        config, args.training_area, args.camera, args.training_reference
    )
    training = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if training is None:
        raise RuntimeError(f"Could not decode training reference: {reference_path}")

    if args.live_frame:
        live = cv2.imread(str(args.live_frame.expanduser()), cv2.IMREAD_COLOR)
        if live is None:
            raise RuntimeError(f"Could not decode live frame: {args.live_frame}")
    else:
        live = capture_live_frame(args.camera_topic, args.message_type, args.timeout)

    comparison_mode = "full_source_frame"
    source_pixels_per_comparison_pixel = 1.0
    if training_crop is not None and reference_path == training_crop:
        comparison_mode = "training_safety_area_crop"
        live, source_pixels_per_comparison_pixel = crop_live_like_training(
            live, config, args.training_area, training.shape[0]
        )
        feature_mask = None
    else:
        feature_mask = load_static_background_mask(config, training.shape)
    result = estimate_alignment(training, live, feature_mask)
    result["comparison_mode"] = comparison_mode
    result["source_pixels_per_comparison_pixel"] = source_pixels_per_comparison_pixel
    result["recommended_frame_shift_y"] = int(round(
        result["shift_y_pixels"] * source_pixels_per_comparison_pixel
    ))
    image_path, json_path = save_report(
        args.output_dir.expanduser().resolve(), training, live, result,
        reference_path, training_crop,
    )

    print(f"Training reference : {reference_path}")
    if training_crop:
        print(f"Selected by crop   : {training_crop}")
    print(f"Live resolution    : {result['live_resolution']}")
    print(f"Training resolution: {result['training_resolution']}")
    print(f"Reliable matches   : {result['reliable_matches']}")
    print(f"Inliers            : {result['inliers']} ({result['inlier_ratio']:.1%})")
    print(f"Live → training X  : {result['shift_x_pixels']:+.2f} px")
    print(f"Live → training Y  : {result['shift_y_pixels']:+.2f} px")
    print(f"Rotation           : {result['rotation_degrees']:+.3f} degrees")
    print(f"Scale              : {result['scale']:.5f}")
    print(f"Comparison mode    : {comparison_mode}")
    print(
        "Camera angle       : "
        + ("approximately the same" if result["same_camera_angle"] else "different or uncertain")
    )
    print(f"Suggested option   : --frame_shift_y {result['recommended_frame_shift_y']}")
    print(f"Comparison image   : {image_path}")
    print(f"JSON report        : {json_path}")


if __name__ == "__main__":
    main()
