#!/usr/bin/env python3
"""Run the offline ADVIS inference pipeline on a live ROS 2 image topic.

The node intentionally reuses :mod:`infer_offline` for model loading,
preprocessing, TAAS scoring, calibrated thresholds, and temporal rolling.  A
rosbag can therefore be replayed with ``ros2 bag play`` and evaluated with the
same settings used by offline inference.

ROS dependencies are imported defensively so that helper functions in this
module can still be unit-tested on machines without ROS 2.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import infer_offline as offline

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import String

    ROS_IMPORT_ERROR = None
except ImportError as exc:  # Allow non-ROS unit tests to import this module.
    rclpy = None
    Node = object
    HistoryPolicy = QoSProfile = ReliabilityPolicy = None
    CompressedImage = Image = String = None
    ROS_IMPORT_ERROR = exc


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cf_dataset_epito.yaml"
ROS_IMAGE_TYPES = {
    "sensor_msgs/msg/CompressedImage": "compressed",
    "sensor_msgs/msg/Image": "raw",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Live anomaly inference from a ROS 2 Image/CompressedImage topic. "
            "It uses the same checkpoints, thresholds, TAAS, and rolling policy "
            "as scripts/infer_offline.py."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset_version", "--dataset-version", default="V6")
    parser.add_argument(
        "--camera_topic", "--camera-topic", "--topic",
        dest="camera_topic", default="/camera/back_view/image_raw",
    )
    parser.add_argument(
        "--message_type", "--message-type",
        choices=("auto", "compressed", "raw"), default="auto",
        help=(
            "ROS image type. 'auto' waits for a publisher and reads its topic "
            "type (default: auto)."
        ),
    )
    parser.add_argument(
        "--safety_areas", "--safety-areas", "--safety_area", "--safety-area",
        dest="safety_areas", nargs="+", default=list(offline.ALL_AREAS),
    )
    parser.add_argument(
        "--mask", action="append", default=[], metavar="AREA=PATH",
        help=(
            "Override one full-frame mask. Repeat for multiple areas. Without "
            "this option, masks are loaded from data.mask_types in the YAML."
        ),
    )

    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--threshold_dir", "--threshold-dir", type=Path)
    parser.add_argument(
        "--threshold_strategy", "--threshold-strategy",
        choices=("max", "percentile", "mean_std"), default="max",
    )
    parser.add_argument(
        "--threshold_percentile", "--threshold-percentile",
        type=float, default=99.0,
    )
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument(
        "--taas_backend", "--taas-backend",
        choices=("auto", "cython", "numpy"), default="auto",
    )
    parser.add_argument(
        "--taas_variant", "--taas-variant",
        choices=("canonical", "minimization"), default="canonical",
    )
    parser.add_argument(
        "--rolling", choices=("none", "mean", "min", "max"), default="none",
    )
    parser.add_argument(
        "--rolling_window", "--rolling-window", type=int, default=5,
    )
    parser.add_argument("--latent_dims", "--latent-dims", type=int)
    parser.add_argument("--model_variant", "--model-variant", choices=("old", "new"), default="old")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--frame_stride", "--frame-stride", type=int, default=1)
    parser.add_argument("--max_frames", "--max-frames", type=int)
    parser.add_argument(
        "--process_period", "--process-period", type=float, default=0.01,
        help="Seconds between checks for a new frame (default: 0.01).",
    )
    parser.add_argument("--timeline_history", "--timeline-history", type=int, default=500)
    parser.add_argument("--log_every_n", "--log-every-n", type=int, default=1)
    parser.add_argument("--verbose_level", "--verbose-level", type=int, default=1)
    parser.add_argument("--profile_timing", "--profile-timing", action="store_true")
    parser.add_argument(
        "--reliable", action="store_true",
        help="Use RELIABLE input QoS instead of sensor-style BEST_EFFORT.",
    )

    parser.add_argument(
        "--detections_topic", "--detections-topic",
        default="/advis/detections",
        help="std_msgs/String JSON detection topic; use an empty value to disable.",
    )
    parser.add_argument(
        "--dashboard_topic", "--dashboard-topic", default="",
        help=(
            "Optional sensor_msgs/CompressedImage dashboard topic. Rendering is "
            "disabled when this is empty, which improves live FPS."
        ),
    )
    parser.add_argument("--dashboard_jpeg_quality", "--dashboard-jpeg-quality", type=int, default=85)
    parser.add_argument("--add_score_name", "--add-score-name", action="store_true")
    parser.add_argument("--add_fps_details", "--add-fps-details", action="store_true")

    parser.add_argument("--publish_rulex", "--publish-rulex", action="store_true")
    parser.add_argument("--rulex_topic", "--rulex-topic", default="/rulex/data")
    parser.add_argument("--attach_image_on_anomaly", "--attach-image-on-anomaly", action="store_true")

    parser.add_argument(
        "--publish_zenoh", "--publish-zenoh", action="store_true",
        help="Publish state compatible with the existing ADVIS Zenoh viewers.",
    )
    parser.add_argument("--zenoh_endpoint", "--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh_dashboard_key", "--zenoh-dashboard-key", default="advis/vis/dashboard/state")
    parser.add_argument("--zenoh_timeline_key", "--zenoh-timeline-key", default="advis/vis/timeline/state")
    parser.add_argument(
        "--publish_debug_media", "--publish-debug-media",
        "--debug_mode", "--debug-mode", action="store_true",
        help=(
            "Publish a separate technical diagnostics state for the Zenoh "
            "debug viewer. Requires --publish_zenoh."
        ),
    )
    parser.add_argument(
        "--zenoh_debug_key", "--zenoh-debug-key",
        default="advis/vis/debug/state",
    )
    parser.add_argument("--zenoh_jpeg_quality", "--zenoh-jpeg-quality", type=int, default=85)
    parser.add_argument("--zenoh_log_level", "--zenoh-log-level", default="error")

    args = parser.parse_args(argv)
    if len(args.safety_areas) == 1 and args.safety_areas[0].upper() == "ALL":
        args.safety_areas = list(offline.ALL_AREAS)
    unknown = sorted(set(args.safety_areas).difference(offline.ALL_AREAS))
    if unknown:
        parser.error(f"Unknown safety area(s): {', '.join(unknown)}")
    if args.frame_stride < 1 or args.rolling_window < 1 or args.log_every_n < 1:
        parser.error("--frame_stride, --rolling_window, and --log_every_n must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max_frames must be at least 1")
    if args.process_period <= 0 or args.timeline_history < 2:
        parser.error("--process_period must be positive and --timeline_history at least 2")
    if args.offset < 0 or args.sigma < 0 or not 0.0 <= args.quantile <= 1.0:
        parser.error("--offset/--sigma must be non-negative and --quantile in [0, 1]")
    if not 1 <= args.dashboard_jpeg_quality <= 100 or not 1 <= args.zenoh_jpeg_quality <= 100:
        parser.error("JPEG quality values must be in [1, 100]")
    if args.publish_debug_media and not args.publish_zenoh:
        parser.error("--publish_debug_media requires --publish_zenoh")
    return args


def load_live_settings(args):
    """Resolve YAML model/mask settings without requiring an offline input path."""
    repository_root = Path(__file__).resolve().parent.parent
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    model_config = config.get("models") or {}
    data_config = config.get("data") or {}
    checkpoint_value = args.checkpoints or model_config.get("checkpoints")
    if checkpoint_value is None:
        checkpoint_value = f"results/{args.dataset_version}/train/models"
    args.checkpoints = offline.resolve_path(checkpoint_value, repository_root)
    args.threshold_dir = (
        args.threshold_dir.expanduser().resolve()
        if args.threshold_dir
        else repository_root / "results" / args.dataset_version / "thresholds"
    )
    args.latent_dims = args.latent_dims or int(model_config.get("latent_dims", 64))
    args.config_mask_paths = {
        area: offline.resolve_path(path, repository_root)
        for area, path in (data_config.get("mask_types") or {}).items()
    }
    args.config_masks_dir = (
        offline.resolve_path(data_config["masks"], repository_root)
        if data_config.get("masks") else None
    )
    return args


def _parse_mask_overrides(values):
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --mask {value!r}; expected AREA=PATH")
        area, raw_path = value.split("=", 1)
        if area not in offline.ALL_AREAS:
            raise ValueError(f"Unknown mask area {area!r}")
        if area in overrides:
            raise ValueError(f"Duplicate mask override for {area}")
        overrides[area] = raw_path
    return overrides


def load_live_masks(args):
    """Load selected masks from CLI overrides, YAML mask_types, or masks dir."""
    overrides = _parse_mask_overrides(args.mask)
    paths = {}
    for area in args.safety_areas:
        raw_path = overrides.get(area, args.config_mask_paths.get(area))
        if raw_path:
            paths[area] = Path(raw_path).expanduser().resolve()

    missing = [area for area in args.safety_areas if area not in paths]
    if missing and args.config_masks_dir and args.config_masks_dir.is_dir():
        candidates = [
            path for path in args.config_masks_dir.iterdir()
            if path.is_file() and path.suffix.lower() in offline.IMAGE_SUFFIXES
        ]
        for area in missing:
            matches = [path for path in candidates if area.lower() in path.stem.lower()]
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple masks found for {area}; use --mask {area}=PATH"
                )
            if matches:
                paths[area] = matches[0]

    missing = [area for area in args.safety_areas if area not in paths]
    if missing:
        raise ValueError(
            "No mask configured for: " + ", ".join(missing) +
            ". Add data.mask_types entries or use --mask AREA=PATH."
        )

    masks = OrderedDict()
    for area in args.safety_areas:
        path = paths[area]
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask for {area}: {path}")
        masks[area] = mask
        print(f"[mask] {area}: {path}")
    return masks


def select_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def detect_topic_message_type(topic_types):
    """Return compressed/raw from ROS type names, or None when unavailable."""
    supported = {ROS_IMAGE_TYPES[name] for name in topic_types if name in ROS_IMAGE_TYPES}
    if len(supported) > 1:
        raise RuntimeError(
            "The camera topic reports both Image and CompressedImage. Select "
            "one explicitly with --message_type raw|compressed."
        )
    return next(iter(supported), None)


def frame_meta(msg_id, frame_id, stamp):
    return {
        "msg_id": int(msg_id),
        "corr_frame_id": str(frame_id),
        "stamp": {
            "sec": int(getattr(stamp, "sec", 0) or 0),
            "nanosec": int(getattr(stamp, "nanosec", 0) or 0),
        },
    }


def serializable_results(results):
    """Expose current result names and legacy viewer aliases."""
    cleaned = OrderedDict()
    for area in offline.ALL_AREAS:
        if area not in results:
            continue
        result = results[area]
        public = offline.public_result(result)
        public.update({
            "score": float(public["anomaly_score"]),
            "norm_score": float(public["normalized_score"]),
            "status": "UNEXPECTED" if public["is_anomalous"] else "normal",
            # Explicit aliases make calibration and inference independently
            # understandable in remote dashboard payloads.  They are the same
            # by design because load_threshold validates the requested variant.
            "calibration_taas_variant": public.get("taas_variant", "canonical"),
            "inference_taas_variant": public.get("taas_variant", "canonical"),
        })
        cleaned[area] = public
    return cleaned


def detection_payload(
    *, msg_id, frame_id, stamp, results, processing_fps, source_fps,
    received_frames, processed_frames, dropped_frames,
):
    return {
        "schema_version": 1,
        "frame_meta": frame_meta(msg_id, frame_id, stamp),
        "processing_fps": float(processing_fps),
        "source_fps": float(source_fps),
        "received_frames": int(received_frames),
        "processed_frames": int(processed_frames),
        "dropped_frames": int(dropped_frames),
        "any_anomalous": any(result["is_anomalous"] for result in results.values()),
        "areas": serializable_results(results),
    }


def encode_image(image, extension=".jpg", params=None):
    ok, encoded = cv2.imencode(extension, image, params or [])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {extension}")
    return encoded.tobytes()


def make_zenoh_config(zenoh, endpoint):
    return zenoh.Config.from_json5(
        '{mode:"client",connect:{endpoints:["' + endpoint + '"]}}'
    )


def pack_dashboard_state(
    msgpack, *, msg_id, frame_id, stamp, frame_bgr, geometries, results,
    jpeg_quality, runtime_meta=None,
):
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    areas = OrderedDict()
    for area, result in results.items():
        geometry = geometries[area]
        x1, y1, x2, y2 = geometry["bbox"]
        areas[area] = {
            # The legacy viewer expects an inclusive bbox.
            "bbox": [int(x1), int(y1), int(x2 - 1), int(y2 - 1)],
            "resize_meta": None,
            "mask_png": encode_image(geometry["binary"], ".png"),
            "orig_patch_jpg": encode_image(result["original_bgr"], ".jpg", params),
            "recon_patch_jpg": encode_image(result["reconstructed_bgr"], ".jpg", params),
            "anom_patch_jpg": encode_image(result["anomaly_bgr"], ".jpg", params),
        }
    payload = {
        "frame_meta": frame_meta(msg_id, frame_id, stamp),
        "frame_bgr_jpg": encode_image(frame_bgr, ".jpg", params),
        "latest_results": serializable_results(results),
        "area_inputs": areas,
        "runtime_meta": dict(runtime_meta or {}),
    }
    return msgpack.packb(payload, use_bin_type=True)


def pack_timeline_state(msgpack, *, msg_id, frame_id, stamp, histories, results):
    payload = {
        "frame_meta": frame_meta(msg_id, frame_id, stamp),
        "score_history": OrderedDict((area, list(values)) for area, values in histories.items()),
        "latest_results": serializable_results(results),
    }
    return msgpack.packb(payload, use_bin_type=True)


def pack_debug_state(
    msgpack, *, msg_id, frame_id, stamp, results, runtime_meta=None,
):
    """Pack full technical metadata without duplicating dashboard images."""
    payload = {
        "schema_version": 1,
        "frame_meta": frame_meta(msg_id, frame_id, stamp),
        "runtime_meta": dict(runtime_meta or {}),
        "latest_results": serializable_results(results),
    }
    return msgpack.packb(payload, use_bin_type=True)


class LiveInferenceNode(Node):
    """Latest-frame ROS inference node using the current offline implementation."""

    def __init__(self, args):
        super().__init__("advis_live_inference")
        self.args = args
        self.device = select_device(args.cpu)
        self.backend = offline.resolve_taas_backend(args.taas_backend)
        self.profiler = offline.TimingProfiler(args.profile_timing)

        self.log(1, f"[device] {self.device}")
        self.log(
            1,
            f"[TAAS] variant={args.taas_variant}, backend={self.backend}, "
            f"offset={args.offset}, sigma={args.sigma}, quantile={args.quantile}",
        )
        self.masks = load_live_masks(args)
        self.models = offline.load_models(args, self.device)
        self.normalize = offline.normalize_model_input
        self.geometries = None
        self.score_windows = OrderedDict(
            (area, deque(maxlen=args.rolling_window)) for area in args.safety_areas
        )
        self.histories = OrderedDict(
            (area, deque(maxlen=args.timeline_history)) for area in args.safety_areas
        )

        self.received_frames = 0
        self.accepted_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.latest_msg = None
        self.latest_msg_id = 0
        self.source_started = None
        self.processing_started = None
        self.last_results = OrderedDict()
        self._shutdown_requested = False

        sensor_reliability = (
            ReliabilityPolicy.RELIABLE if args.reliable
            else ReliabilityPolicy.BEST_EFFORT
        )
        self.sensor_qos = QoSProfile(
            reliability=sensor_reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription = None
        self.discovery_timer = None
        if args.message_type == "auto":
            self.discovery_timer = self.create_timer(0.5, self.discover_topic_type)
            self.log(1, f"[subscriber] waiting to discover type of {args.camera_topic}")
        else:
            self.create_image_subscription(args.message_type)

        self.detections_pub = None
        if args.detections_topic:
            self.detections_pub = self.create_publisher(
                String, args.detections_topic, self.publisher_qos
            )
            self.log(1, f"[publisher] JSON detections: {args.detections_topic}")

        self.dashboard_pub = None
        if args.dashboard_topic:
            self.dashboard_pub = self.create_publisher(
                CompressedImage, args.dashboard_topic, self.publisher_qos
            )
            self.log(1, f"[publisher] dashboard JPEG: {args.dashboard_topic}")

        self.rulex_pub = None
        self.rulex_area_cls = None
        self.rulex_result_cls = None
        self.bridge = None
        if args.publish_rulex:
            self.setup_rulex_publisher()

        self.zenoh_session = None
        self.zenoh_dashboard_pub = None
        self.zenoh_timeline_pub = None
        self.zenoh_debug_pub = None
        self.msgpack = None
        if args.publish_zenoh:
            self.setup_zenoh_publishers()

        self.process_timer = self.create_timer(args.process_period, self.process_latest)
        self.wait_started = time.monotonic()
        self.wait_timer = self.create_timer(5.0, self.report_waiting)

    def log(self, level, message):
        if self.args.verbose_level >= level:
            self.get_logger().info(message)

    def discover_topic_type(self):
        if self.subscription is not None:
            return
        infos = self.get_publishers_info_by_topic(self.args.camera_topic)
        topic_types = {info.topic_type for info in infos}
        try:
            message_type = detect_topic_message_type(topic_types)
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            self.discovery_timer.cancel()
            return
        if message_type:
            self.create_image_subscription(message_type)
            self.discovery_timer.cancel()

    def create_image_subscription(self, message_type):
        ros_type = CompressedImage if message_type == "compressed" else Image
        self.subscription = self.create_subscription(
            ros_type, self.args.camera_topic, self.store_latest, self.sensor_qos
        )
        self.message_type = message_type
        self.log(
            1,
            f"[subscriber] {self.args.camera_topic} ({message_type}, "
            f"stride={self.args.frame_stride}, depth=1)",
        )

    def setup_rulex_publisher(self):
        try:
            from cv_bridge import CvBridge
            from distrimuse_ros2_api.msg import RulexAreaScore, RulexDetectionResult
        except ImportError as exc:
            raise RuntimeError(
                "--publish_rulex requires cv_bridge and distrimuse_ros2_api"
            ) from exc
        self.bridge = CvBridge()
        self.rulex_area_cls = RulexAreaScore
        self.rulex_result_cls = RulexDetectionResult
        self.rulex_pub = self.create_publisher(
            RulexDetectionResult, self.args.rulex_topic, self.publisher_qos
        )
        self.log(1, f"[publisher] RulexDetectionResult: {self.args.rulex_topic}")

    def setup_zenoh_publishers(self):
        try:
            import msgpack
            import zenoh
        except ImportError as exc:
            raise RuntimeError(
                "--publish_zenoh requires the zenoh and msgpack Python packages"
            ) from exc
        zenoh.init_log_from_env_or(self.args.zenoh_log_level)
        self.zenoh_session = zenoh.open(
            make_zenoh_config(zenoh, self.args.zenoh_endpoint)
        )
        self.zenoh_dashboard_pub = self.zenoh_session.declare_publisher(
            self.args.zenoh_dashboard_key,
            encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        )
        self.zenoh_timeline_pub = self.zenoh_session.declare_publisher(
            self.args.zenoh_timeline_key,
            encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        )
        if self.args.publish_debug_media:
            self.zenoh_debug_pub = self.zenoh_session.declare_publisher(
                self.args.zenoh_debug_key,
                encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
            )
        self.msgpack = msgpack
        self.log(1, f"[publisher] Zenoh dashboard: {self.args.zenoh_dashboard_key}")
        self.log(1, f"[publisher] Zenoh timeline: {self.args.zenoh_timeline_key}")
        if self.zenoh_debug_pub is not None:
            self.log(1, f"[publisher] Zenoh debug: {self.args.zenoh_debug_key}")

    def report_waiting(self):
        if self.received_frames == 0:
            elapsed = time.monotonic() - self.wait_started
            self.log(
                1,
                f"[wait] no frames after {elapsed:.1f}s on {self.args.camera_topic}; "
                "verify ros2 topic type/hz and QoS",
            )
        else:
            self.wait_timer.cancel()

    def store_latest(self, msg):
        self.received_frames += 1
        if self.source_started is None:
            self.source_started = time.monotonic()
            self.log(1, "[subscriber] first frame received")
        if (self.received_frames - 1) % self.args.frame_stride:
            return
        self.accepted_frames += 1
        if self.latest_msg is not None:
            self.dropped_frames += 1
        self.latest_msg = msg
        self.latest_msg_id = self.received_frames

    def decode_frame(self, msg):
        with self.profiler.measure("input_decode"):
            if self.message_type == "compressed":
                frame = cv2.imdecode(
                    np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
            else:
                if self.bridge is None:
                    try:
                        from cv_bridge import CvBridge
                    except ImportError as exc:
                        raise RuntimeError("Raw Image input requires cv_bridge") from exc
                    self.bridge = CvBridge()
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if frame is None or not frame.size:
            raise RuntimeError("ROS image decoded to an empty frame")
        return frame

    def process_latest(self):
        if self.latest_msg is None or self._shutdown_requested:
            return
        msg = self.latest_msg
        msg_id = self.latest_msg_id
        self.latest_msg = None
        started = time.monotonic()

        try:
            frame = self.decode_frame(msg)
            if self.geometries is None:
                with self.profiler.measure("mask_prepare_once"):
                    self.geometries = OrderedDict(
                        (area, offline.prepare_mask_geometry(mask, frame.shape))
                        for area, mask in self.masks.items()
                    )
            if self.processing_started is None:
                self.processing_started = time.monotonic()

            results = OrderedDict()
            for area in self.args.safety_areas:
                with self.profiler.measure("mask_crop"):
                    crop = offline.crop_area(
                        frame, self.masks[area], self.geometries[area]
                    )
                result = offline.infer_crop(
                    crop, area, self.models[area], self.normalize, self.device,
                    profiler=self.profiler, taas_backend=self.backend,
                )
                offline.apply_rolling_policy(
                    result, self.score_windows[area], self.args.rolling,
                    self.args.rolling_window,
                )
                results[area] = result
                self.histories[area].append(float(result["normalized_score"]))

            self.processed_frames += 1
            self.profiler.source_frames = self.processed_frames
            self.last_results = results
            elapsed = time.monotonic() - self.processing_started
            processing_fps = self.processed_frames / max(elapsed, 1e-9)
            source_elapsed = time.monotonic() - self.source_started
            source_fps = self.received_frames / max(source_elapsed, 1e-9)
            payload = detection_payload(
                msg_id=msg_id,
                frame_id=msg.header.frame_id,
                stamp=msg.header.stamp,
                results=results,
                processing_fps=processing_fps,
                source_fps=source_fps,
                received_frames=self.received_frames,
                processed_frames=self.processed_frames,
                dropped_frames=self.dropped_frames,
            )
            self.publish_outputs(msg, frame, results, payload, processing_fps)

            if self.processed_frames == 1 or self.processed_frames % self.args.log_every_n == 0:
                statuses = " | ".join(
                    f"{area}={'ANOMALY' if result['is_anomalous'] else 'normal'} "
                    f"{result['normalized_score']:.3f}x"
                    for area, result in results.items()
                )
                instant_fps = 1.0 / max(time.monotonic() - started, 1e-9)
                self.log(
                    1,
                    f"[frame {self.processed_frames}] avg={processing_fps:.2f} fps, "
                    f"instant={instant_fps:.2f} fps, dropped={self.dropped_frames} | "
                    f"{statuses}",
                )

            if self.args.max_frames and self.processed_frames >= self.args.max_frames:
                self.log(1, f"[done] reached --max_frames {self.args.max_frames}")
                self._shutdown_requested = True
                rclpy.shutdown()
        except Exception as exc:
            self.get_logger().error(f"Failed to process ROS frame {msg_id}: {exc}")
            raise

    def publish_outputs(self, source_msg, frame, results, payload, processing_fps):
        if self.detections_pub is not None:
            message = String()
            message.data = json.dumps(payload, separators=(",", ":"))
            self.detections_pub.publish(message)

        if self.dashboard_pub is not None:
            with self.profiler.measure("dashboard_render"):
                dashboard = offline.make_advis_dashboard(
                    frame, self.masks, results,
                    sample_id=f"ROS frame {payload['frame_meta']['msg_id']}",
                    add_score_name=self.args.add_score_name,
                    add_fps_details=self.args.add_fps_details,
                    processing_fps=processing_fps,
                    output_fps=processing_fps,
                    mask_geometries=self.geometries,
                )
                encoded = encode_image(
                    dashboard, ".jpg",
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.args.dashboard_jpeg_quality],
                )
            message = CompressedImage()
            message.header = source_msg.header
            message.format = "jpeg"
            message.data = encoded
            self.dashboard_pub.publish(message)

        if self.rulex_pub is not None:
            self.publish_rulex(source_msg, frame, results)

        if self.zenoh_session is not None:
            runtime_meta = {
                "processing_fps": payload["processing_fps"],
                "source_fps": payload["source_fps"],
                "received_frames": payload["received_frames"],
                "processed_frames": payload["processed_frames"],
                "dropped_frames": payload["dropped_frames"],
            }
            dashboard_payload = pack_dashboard_state(
                self.msgpack,
                msg_id=payload["frame_meta"]["msg_id"],
                frame_id=source_msg.header.frame_id,
                stamp=source_msg.header.stamp,
                frame_bgr=frame,
                geometries=self.geometries,
                results=results,
                jpeg_quality=self.args.zenoh_jpeg_quality,
                runtime_meta=runtime_meta,
            )
            timeline_payload = pack_timeline_state(
                self.msgpack,
                msg_id=payload["frame_meta"]["msg_id"],
                frame_id=source_msg.header.frame_id,
                stamp=source_msg.header.stamp,
                histories=self.histories,
                results=results,
            )
            self.zenoh_dashboard_pub.put(dashboard_payload)
            self.zenoh_timeline_pub.put(timeline_payload)
            if self.zenoh_debug_pub is not None:
                debug_payload = pack_debug_state(
                    self.msgpack,
                    msg_id=payload["frame_meta"]["msg_id"],
                    frame_id=source_msg.header.frame_id,
                    stamp=source_msg.header.stamp,
                    results=results,
                    runtime_meta=runtime_meta,
                )
                self.zenoh_debug_pub.put(debug_payload)

    def publish_rulex(self, source_msg, frame, results):
        area_class = self.rulex_area_cls
        enum_names = {
            "RoboArm": "AREA_A",
            "ConvBelt": "AREA_B",
            "PLeft": "AREA_C",
            "PRight": "AREA_D",
        }
        message = self.rulex_result_cls()
        area_scores = []
        for area, result in results.items():
            area_message = area_class()
            area_message.area = getattr(area_class, enum_names[area])
            area_message.anomaly = bool(result["is_anomalous"])
            area_scores.append(area_message)
        message.area_scores = area_scores
        if self.args.attach_image_on_anomaly and any(
            result["is_anomalous"] for result in results.values()
        ):
            message.image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            message.image.header = source_msg.header
        self.rulex_pub.publish(message)

    def close_external_publishers(self):
        for publisher in (
            self.zenoh_dashboard_pub,
            self.zenoh_timeline_pub,
            self.zenoh_debug_pub,
        ):
            if publisher is not None:
                try:
                    publisher.undeclare()
                except Exception:
                    pass
        if self.zenoh_session is not None:
            try:
                self.zenoh_session.close()
            except Exception:
                pass


def main(argv=None):
    args = parse_args(argv)
    if rclpy is None:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable. Run this script in the Pixi/ROS "
            f"environment. Original import error: {ROS_IMPORT_ERROR}"
        )
    args = load_live_settings(args)
    rclpy.init(args=None)
    node = None
    try:
        node = LiveInferenceNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Stopped by user.")
    finally:
        if node is not None:
            node.profiler.report()
            node.close_external_publishers()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
