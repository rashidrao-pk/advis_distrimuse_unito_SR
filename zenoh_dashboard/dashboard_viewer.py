from __future__ import annotations

import argparse
import os
import threading
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import msgpack
import numpy as np
import zenoh

ALL_SAFETY_AREAS = ["PRight", "PLeft", "RoboArm", "ConvBelt"]
AREA_DISPLAY_NAMES = {
    "PRight": "Pallet Right (A)",
    "PLeft": "Pallet Left (B)",
    "RoboArm": "Robo Arm (C)",
    "ConvBelt": "Conveyor Belt (D)",
}

STATUS_COLORS = {
    "running": (35, 150, 35),
    "waiting": (0, 150, 220),
    "inference_stopped": (0, 140, 255),
    "camera_offline": (0, 0, 210),
    "camera_unknown": (100, 100, 100),
}


def connection_status(
    now: float,
    inference_last_received: Optional[float],
    camera_last_received: Optional[float],
    started_at: float,
    inference_timeout: float,
    camera_timeout: float,
    camera_monitor_available: bool = True,
) -> Tuple[str, str, str]:
    """Return status key, title, and explanation for the dashboard banner."""
    inference_fresh = (
        inference_last_received is not None
        and now - inference_last_received <= inference_timeout
    )
    camera_fresh = (
        camera_last_received is not None
        and now - camera_last_received <= camera_timeout
    )

    if inference_fresh:
        return "running", "INFERENCE RUNNING", "Camera stream and detections are live"
    if camera_fresh:
        return (
            "inference_stopped",
            "CAMERA LIVE - INFERENCE STOPPED",
            "Camera frames continue, but no new ADVIS detection was received",
        )
    if now - started_at <= max(inference_timeout, camera_timeout):
        return "waiting", "WAITING FOR DATA", "Waiting for camera and inference messages"
    if camera_monitor_available:
        return (
            "camera_offline",
            "CAMERA OFFLINE",
            "No camera frames are being published; inference results are stale",
        )
    return (
        "camera_unknown",
        "INFERENCE STOPPED - CAMERA STATUS UNKNOWN",
        "No detections received and the ROS camera monitor is unavailable",
    )


class ROSCameraMonitor:
    """Monitor camera-message freshness independently from ADVIS inference."""

    TYPE_NAMES = {
        "raw": "sensor_msgs/msg/Image",
        "compressed": "sensor_msgs/msg/CompressedImage",
    }

    def __init__(self, topic: str, message_type: str = "auto"):
        self.topic = topic
        self.requested_type = message_type
        self.last_received = None
        self.available = False
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            # rclpy's default ~/.ros/log location may live on a slow or
            # unavailable mounted home directory.  A logging failure prevents
            # the camera monitor from starting even though DDS itself is fine.
            ros_log_dir = Path(tempfile.gettempdir()) / "advis_dashboard_ros_logs"
            ros_log_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))

            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import CompressedImage, Image

            context = Context()
            rclpy.init(args=None, context=context)
            node = Node("advis_dashboard_camera_monitor", context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            classes = {"raw": Image, "compressed": CompressedImage}
            subscription = None
            subscribed_type = None
            next_discovery = 0.0
            first_frame_reported = False
            self.available = True

            def on_frame(_message) -> None:
                nonlocal first_frame_reported
                self.last_received = time.monotonic()
                if not first_frame_reported:
                    print(f"[camera monitor] first frame received on {self.topic}")
                    first_frame_reported = True

            while not self._stop.is_set() and context.ok():
                now = time.monotonic()
                if now >= next_discovery:
                    discovered = set()
                    for info in node.get_publishers_info_by_topic(self.topic):
                        for kind, type_name in self.TYPE_NAMES.items():
                            if info.topic_type == type_name:
                                discovered.add(kind)
                    desired = self.requested_type
                    if desired == "auto":
                        desired = (
                            "compressed" if "compressed" in discovered
                            else "raw" if "raw" in discovered else None
                        )
                    if desired is not None and desired != subscribed_type:
                        if subscription is not None:
                            node.destroy_subscription(subscription)
                        subscription = node.create_subscription(
                            classes[desired], self.topic, on_frame, qos
                        )
                        subscribed_type = desired
                        print(
                            f"[camera monitor] subscribed to {self.topic} "
                            f"({desired})"
                        )
                    next_discovery = now + 0.5
                executor.spin_once(timeout_sec=0.1)

            if subscription is not None:
                node.destroy_subscription(subscription)
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()
            context.shutdown()
        except Exception as exc:
            self.available = False
            self.error = str(exc)
            print(f"[camera monitor] unavailable: {exc}")


def ordered_area_list(areas: Iterable[str]) -> List[str]:
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


def decode_image(payload: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, flags)
    if image is None:
        raise ValueError("cv2.imdecode failed")
    return image


def ensure_gray(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def prepare_binary_mask(mask: np.ndarray, frame_shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = frame_shape_hw
    mask = ensure_gray(mask)
    if mask is None:
        raise ValueError("mask cannot be None")
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def extract_mask_contours(mask_gray: np.ndarray, frame_shape_hw: Tuple[int, int]) -> Tuple[List[np.ndarray], np.ndarray]:
    mask_bin = prepare_binary_mask(mask_gray, frame_shape_hw)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_bin


def create_union_mask(area_inputs: Dict[str, dict], frame_shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = frame_shape_hw
    union_mask = np.zeros((h, w), dtype=np.uint8)

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        mask_bin = info.get("mask_bin")
        if mask_bin is None:
            continue
        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        union_mask = np.maximum(union_mask, mask_bin)

    return union_mask


def overlay_outside_safety_blur(
    frame_bgr: np.ndarray,
    area_inputs: Dict[str, dict],
    blur_ksize: int = 1,
    darken_factor: float = 0.85,
) -> np.ndarray:
    if len(area_inputs) == 0:
        return frame_bgr.copy()

    union_mask = create_union_mask(area_inputs, frame_bgr.shape[:2])
    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)
    darkened = (blurred.astype(np.float32) * darken_factor).clip(0, 255).astype(np.uint8)
    union_mask_3 = cv2.cvtColor(union_mask, cv2.COLOR_GRAY2BGR)
    return np.where(union_mask_3 > 0, frame_bgr, darkened)


def resize_and_center(image: Optional[np.ndarray], target_w: int, target_h: int, bg_color=(0, 0, 0)) -> np.ndarray:
    if image is None:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def scale_contours(contours: List[np.ndarray], scale: float, x_off: int, y_off: int) -> List[np.ndarray]:
    scaled = []
    for cnt in contours:
        cnt_scaled = cnt.astype(np.float32).copy()
        cnt_scaled[:, 0, 0] = x_off + cnt_scaled[:, 0, 0] * scale
        cnt_scaled[:, 0, 1] = y_off + cnt_scaled[:, 0, 1] * scale
        scaled.append(cnt_scaled.astype(np.int32))
    return scaled


def unletterbox_patch(patch_bgr: Optional[np.ndarray], resize_meta: Optional[dict]) -> Optional[np.ndarray]:
    if patch_bgr is None or resize_meta is None:
        return patch_bgr

    x_off = int(resize_meta.get("x_off", 0))
    y_off = int(resize_meta.get("y_off", 0))
    new_w = int(resize_meta.get("new_w", patch_bgr.shape[1]))
    new_h = int(resize_meta.get("new_h", patch_bgr.shape[0]))

    if new_w <= 0 or new_h <= 0:
        return patch_bgr

    h, w = patch_bgr.shape[:2]
    x1 = max(0, x_off)
    y1 = max(0, y_off)
    x2 = min(w, x_off + new_w)
    y2 = min(h, y_off + new_h)

    if x2 <= x1 or y2 <= y1:
        return patch_bgr

    cropped = patch_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return patch_bgr
    return cropped


def paste_area_result_in_full_frame(
    target_canvas: np.ndarray,
    patch_bgr: Optional[np.ndarray],
    bbox: Optional[Tuple[int, int, int, int]],
    mask_bin: Optional[np.ndarray],
    resize_meta: Optional[dict] = None,
    keep_background: bool = False,
    background_canvas: Optional[np.ndarray] = None,
) -> np.ndarray:
    if patch_bgr is None or bbox is None or mask_bin is None:
        return target_canvas

    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1
    if crop_w <= 0 or crop_h <= 0:
        return target_canvas

    if resize_meta is not None:
        patch_bgr = unletterbox_patch(patch_bgr, resize_meta)

    if patch_bgr is None or patch_bgr.size == 0:
        return target_canvas

    patch_resized = cv2.resize(patch_bgr, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
    mask_crop = mask_bin[y1:y2 + 1, x1:x2 + 1]
    mask_crop_3 = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
    roi = target_canvas[y1:y2 + 1, x1:x2 + 1]

    if keep_background and background_canvas is not None:
        bg_roi = background_canvas[y1:y2 + 1, x1:x2 + 1]
        blended = np.where(mask_crop_3 > 0, patch_resized, bg_roi)
    else:
        blended = np.where(mask_crop_3 > 0, patch_resized, roi)

    target_canvas[y1:y2 + 1, x1:x2 + 1] = blended
    return target_canvas


def unpack_dashboard_state(payload: bytes) -> dict:
    obj = msgpack.unpackb(payload, raw=False)
    frame_bgr = decode_image(obj["frame_bgr_jpg"], cv2.IMREAD_COLOR)

    area_inputs = OrderedDict()
    for area in ordered_area_list(obj["area_inputs"].keys()):
        info = obj["area_inputs"][area]
        mask_bin = decode_image(info["mask_png"], cv2.IMREAD_GRAYSCALE)
        contours, mask_bin = extract_mask_contours(mask_bin, frame_bgr.shape[:2])
        area_inputs[area] = {
            "bbox": tuple(info["bbox"]) if info.get("bbox") is not None else None,
            "resize_meta": info.get("resize_meta"),
            "mask_bin": mask_bin,
            "contours": contours,
            "orig_patch_bgr": decode_image(info["orig_patch_jpg"], cv2.IMREAD_COLOR),
            "recon_patch_bgr": decode_image(info["recon_patch_jpg"], cv2.IMREAD_COLOR),
            "anom_patch_bgr": decode_image(info["anom_patch_jpg"], cv2.IMREAD_COLOR),
        }

    return {
        "frame_meta": obj["frame_meta"],
        "frame_bgr": frame_bgr,
        "latest_results": obj["latest_results"],
        "area_inputs": area_inputs,
        "runtime_meta": obj.get("runtime_meta", {}),
    }


def _summary_value(results, key, fallback_key=None, default="unknown"):
    """Return one common metadata value, or a compact mixed-value summary."""
    values = []
    for area_name in ordered_area_list(results.keys()):
        result = results[area_name]
        value = result.get(key)
        if value is None and fallback_key is not None:
            value = result.get(fallback_key)
        if value is not None and str(value) not in values:
            values.append(str(value))
    if not values:
        return str(default)
    if len(values) == 1:
        return values[0]
    return "mixed[" + ", ".join(values[:2]) + (", ...]" if len(values) > 2 else "]")


def dashboard_detail_lines(results, runtime_meta=None):
    """Build calibration/inference descriptions for the dashboard panel."""
    if not results:
        return []
    calibration_variant = _summary_value(
        results, "calibration_taas_variant", "taas_variant", "legacy-unspecified"
    )
    inference_variant = _summary_value(
        results, "inference_taas_variant", "taas_variant", "legacy-unspecified"
    )
    calibration_score = _summary_value(
        results, "calibration_score_func", "score_func"
    )
    inference_score = _summary_value(
        results, "inference_score_func", "score_func"
    )
    strategy = _summary_value(results, "threshold_strategy")
    offset = _summary_value(results, "offset", default="-")
    sigma = _summary_value(results, "sigma", default="-")
    quantile = _summary_value(results, "quantile", default="-")
    rolling = _summary_value(results, "rolling_policy", default="none")
    rolling_window = _summary_value(results, "rolling_window", default="1")

    lines = [
        ("Calibration", f"{calibration_variant} | {calibration_score}"),
        ("Tau policy", f"{strategy} | offset={offset}, sigma={sigma}, q={quantile}"),
        ("Inference", f"{inference_variant} | {inference_score}"),
        ("Temporal", f"{rolling} | window={rolling_window}"),
    ]
    runtime_meta = runtime_meta or {}
    if runtime_meta:
        processing_fps = runtime_meta.get("processing_fps")
        source_fps = runtime_meta.get("source_fps")
        dropped = runtime_meta.get("dropped_frames", 0)
        processing_text = "-" if processing_fps is None else f"{float(processing_fps):.2f}"
        source_text = "-" if source_fps is None else f"{float(source_fps):.2f}"
        lines.append((
            "Runtime",
            f"inference={processing_text} fps | source={source_text} fps | dropped={dropped}",
        ))
    return lines


def _fit_text(text, max_width, font_scale, thickness):
    """Truncate an OpenCV label to the available pixel width."""
    text = str(text)
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    while text and cv2.getTextSize(
        text + suffix, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )[0][0] > max_width:
        text = text[:-1]
    return text + suffix


def draw_text_table(
    panel, results, frame_id=None, corr_frame_id=None, corr_stamp=None,
    runtime_meta=None,
):
    h, w = panel.shape[:2]
    panel[:] = (245, 245, 245)

    y = 18
    if frame_id is not None and corr_stamp is not None:
        frame_text = (
            f"Frame {frame_id} | {corr_frame_id} | "
            f"{corr_stamp['sec']}.{int(corr_stamp['nanosec']):09d}"
        )
        cv2.putText(
            panel, _fit_text(frame_text, w - 40, 0.55, 1), (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA,
        )
        y += 13
    cv2.line(panel, (15, y), (w - 15, y), (70, 70, 70), 1)
    y += 24

    headers = ["Safety Area", "RawVal", "Threshold", "Score", "Status"]
    header_bold = [2, 1, 1, 2, 2]
    col_x = [20, 190, 285, 410, 515]

    for i, hdr in enumerate(headers):
        cv2.putText(panel, hdr, (col_x[i], y), cv2.FONT_HERSHEY_SIMPLEX, 0.59,
                     (30, 30, 30), header_bold[i], cv2.LINE_AA)

    y += 9
    cv2.line(panel, (15, y), (w - 15, y), (70, 70, 70), 1)
    y += 23

    for area_name in ordered_area_list(results.keys()):
        r = results[area_name]
        raw_score = r.get("score", None)
        thr = r.get("threshold", None)
        norm = r.get("norm_score", None)
        status = r.get("status", "unknown")
        is_anom = bool(r.get("is_anomalous", False))
        color = (0, 0, 180) if is_anom else (0, 140, 0)

        vals = [
            AREA_DISPLAY_NAMES.get(area_name, area_name),
            "-" if raw_score is None else f"{raw_score:.3f}",
            "-" if thr is None else f"{thr:.3f}",
            "-" if norm is None else f"{norm:.3f}",
            status,
        ]

        for i, val in enumerate(vals):
            cv2.putText(
                panel,
                str(val),
                (col_x[i], y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                color if i >= 3 else (30, 30, 30),
                2 if i >= 3 else 1,
                cv2.LINE_AA,
            )

        y += 8
        cv2.line(panel, (15, y), (w - 15, y), (165, 165, 165), 1)
        y += 23

    y += 1
    cv2.line(panel, (15, y), (w - 15, y), (70, 70, 70), 1)
    # Give the calibration/inference block more visual separation from the
    # per-area score table.
    y += 33
    detail_colors = {
        "Calibration": (0, 115, 190),
        "Tau policy": (0, 115, 190),
        "Inference": (175, 75, 0),
        "Temporal": (120, 60, 120),
        "Runtime": (50, 120, 50),
    }
    for label, value in dashboard_detail_lines(results, runtime_meta):
        if y > h - 7:
            break
        cv2.putText(
            panel, f"{label}:", (20, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.47, detail_colors.get(label, (40, 40, 40)), 2, cv2.LINE_AA,
        )
        label_width = cv2.getTextSize(
            f"{label}:", cv2.FONT_HERSHEY_SIMPLEX, 0.47, 2
        )[0][0]
        value_x = 28 + label_width
        cv2.putText(
            panel, _fit_text(value, w - value_x - 12, 0.47, 1),
            (value_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
            (35, 35, 35), 1, cv2.LINE_AA,
        )
        y += 22

    return panel


def draw_dashboard_panel(
    frame_bgr, area_inputs, latest_results, frame_id=None, width=1600,
    height=1000, corr_frame_id=None, corr_stamp=None, runtime_meta=None,
):
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)

    pad = 16
    panel_w = (width - 3 * pad) // 2
    panel_h = (height - 3 * pad) // 2

    tl = (pad, pad, pad + panel_w, pad + panel_h)
    tr = (2 * pad + panel_w, pad, width - pad, pad + panel_h)
    bl = (pad, 2 * pad + panel_h, pad + panel_w, height - pad)
    br = (2 * pad + panel_w, 2 * pad + panel_h, width - pad, height - pad)

    def draw_panel_title(title, box):
        x1, y1, x2, y2 = box
        cv2.putText(canvas, title, (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)

    draw_panel_title("Input View", tl)
    draw_panel_title("Unexpected Situations View", tr)
    draw_panel_title("AI View", bl)
    draw_panel_title("Details", br)

    inner_margin = 12
    title_h = 40

    def inner_box(box):
        x1, y1, x2, y2 = box
        return (x1 + inner_margin, y1 + title_h, x2 - inner_margin, y2 - inner_margin)

    tl_in = inner_box(tl)
    tr_in = inner_box(tr)
    bl_in = inner_box(bl)
    br_in = inner_box(br)

    h, w = frame_bgr.shape[:2]

    input_vis = overlay_outside_safety_blur(frame_bgr, area_inputs)
    input_full = input_vis.copy()

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        input_full = paste_area_result_in_full_frame(
            input_full,
            info.get("orig_patch_bgr"),
            info.get("bbox"),
            info.get("mask_bin"),
            resize_meta=info.get("resize_meta"),
            keep_background=True,
            background_canvas=input_vis,
        )

    tl_w = tl_in[2] - tl_in[0]
    tl_h = tl_in[3] - tl_in[1]
    scale = min(tl_w / w, tl_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    tl_img = cv2.resize(input_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_off = tl_in[0] + (tl_w - new_w) // 2
    y_off = tl_in[1] + (tl_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = tl_img

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        contours = info.get("contours", [])
        rr = latest_results.get(area_name, {})
        is_anom = bool(rr.get("is_anomalous", False))
        color = (0, 0, 255) if is_anom else (255, 255, 255)
        scaled = scale_contours(contours, scale, x_off, y_off)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, color, 2)
            pt = scaled[0][0][0]
            # label = f"{AREA_DISPLAY_NAMES.get(area_name, area_name)}: {rr.get('norm_score', 0):.2f}" if "norm_score" in rr else area_name
            label = f"{rr.get('status', '')}: {rr.get('norm_score', 0):.2f}"
            # cv2.putText(canvas, label, (int(pt[0]), max(20, int(pt[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            pt = scaled[0][0][0]
            label = f"{rr.get('status', '')}: {rr.get('norm_score', 0):.2f}"

            x_text = int(pt[0])
            y_text = max(20, int(pt[1]) - 8)
            if area_name == "PLeft":
                x_text -= 30   # try -30, -40, or -50
            elif area_name == "ConvBelt":
                text_width = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )[0][0]
                contour_points = np.concatenate(scaled, axis=0).reshape(-1, 2)
                x_text = max(tl_in[0] + 4, int(contour_points[:, 0].max()) - text_width - 6)
                y_text = max(
                    tl_in[1] + 18,
                    min(tl_in[3] - 6, int(contour_points[:, 1].max()) - 6),
                )

            cv2.putText(
                canvas,
                label,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    recon_full = np.zeros_like(frame_bgr)
    anom_full = np.full_like(frame_bgr, 255)

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        recon_full = paste_area_result_in_full_frame(
            recon_full,
            info.get("recon_patch_bgr"),
            info.get("bbox"),
            info.get("mask_bin"),
            resize_meta=info.get("resize_meta"),
        )
        anom_full = paste_area_result_in_full_frame(
            anom_full,
            info.get("anom_patch_bgr"),
            info.get("bbox"),
            info.get("mask_bin"),
            resize_meta=info.get("resize_meta"),
        )

    tr_w = tr_in[2] - tr_in[0]
    tr_h = tr_in[3] - tr_in[1]
    anom_disp = resize_and_center(anom_full, tr_w, tr_h, bg_color=(255, 255, 255))
    canvas[tr_in[1]:tr_in[1] + tr_h, tr_in[0]:tr_in[0] + tr_w] = anom_disp

    scale_tr = min(tr_w / w, tr_h / h)
    new_w_tr = max(1, int(w * scale_tr))
    new_h_tr = max(1, int(h * scale_tr))
    x_off_tr = tr_in[0] + (tr_w - new_w_tr) // 2
    y_off_tr = tr_in[1] + (tr_h - new_h_tr) // 2

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        contours = info.get("contours", [])
        rr = latest_results.get(area_name, {})
        is_anom = bool(rr.get("is_anomalous", False))
        color = (0, 0, 255) if is_anom else (0, 128, 0)
        scaled = scale_contours(contours, scale_tr, x_off_tr, y_off_tr)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, color, 2)
            pt = scaled[0][0][0]
            label = f"{rr.get('status', '')}: {rr.get('norm_score', 0):.2f}" if "norm_score" in rr else area_name
            # cv2.putText(canvas, label, (int(pt[0]), int(pt[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            pt = scaled[0][0][0]
            label = f"{rr.get('status', '')}: {rr.get('norm_score', 0):.2f}" if "norm_score" in rr else area_name

            x_text = int(pt[0])
            y_text = int(pt[1]) - 5
            if area_name == "PLeft":
                x_text -= 40
            elif area_name == "ConvBelt":
                text_width = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )[0][0]
                contour_points = np.concatenate(scaled, axis=0).reshape(-1, 2)
                x_text = max(tr_in[0] + 4, int(contour_points[:, 0].max()) - text_width - 6)
                y_text = max(
                    tr_in[1] + 18,
                    min(tr_in[3] - 6, int(contour_points[:, 1].max()) - 6),
                )

            cv2.putText(
                canvas,
                label,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    bl_w = bl_in[2] - bl_in[0]
    bl_h = bl_in[3] - bl_in[1]
    recon_disp = resize_and_center(recon_full, bl_w, bl_h, bg_color=(0, 0, 0))
    canvas[bl_in[1]:bl_in[1] + bl_h, bl_in[0]:bl_in[0] + bl_w] = recon_disp

    scale_bl = min(bl_w / w, bl_h / h)
    x_off_bl = bl_in[0] + (bl_w - max(1, int(w * scale_bl))) // 2
    y_off_bl = bl_in[1] + (bl_h - max(1, int(h * scale_bl))) // 2

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        contours = info.get("contours", [])
        rr = latest_results.get(area_name, {})
        is_anom = bool(rr.get("is_anomalous", False))
        color = (0, 0, 255) if is_anom else (0, 180, 0)
        scaled = scale_contours(contours, scale_bl, x_off_bl, y_off_bl)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, color, 2)

    details_panel = np.full((br_in[3] - br_in[1], br_in[2] - br_in[0], 3), 245, dtype=np.uint8)
    details_panel = draw_text_table(
        details_panel, latest_results, frame_id=frame_id,
        corr_frame_id=corr_frame_id, corr_stamp=corr_stamp,
        runtime_meta=runtime_meta,
    )
    canvas[br_in[1]:br_in[3], br_in[0]:br_in[2]] = details_panel
    return canvas


def make_config(endpoint: str) -> zenoh.Config:
    return zenoh.Config.from_json5(
        f'''
    {{
      mode: "client",
      connect: {{
        endpoints: ["{endpoint}"]
      }}
    }}
    '''
    )


def render_from_payload(raw: bytes, width: int, height: int) -> np.ndarray:
    state = unpack_dashboard_state(raw)
    meta = state["frame_meta"]
    return draw_dashboard_panel(
        state["frame_bgr"],
        state["area_inputs"],
        state["latest_results"],
        frame_id=meta["msg_id"],
        width=width,
        height=height,
        corr_frame_id=meta["corr_frame_id"],
        corr_stamp=meta["stamp"],
        runtime_meta=state.get("runtime_meta", {}),
    )


def draw_connection_banner(
    dashboard: np.ndarray,
    status_key: str,
    title: str,
    explanation: str,
    stale: bool,
) -> np.ndarray:
    """Draw a compact live indicator or a prominent stale-state banner."""
    output = dashboard.copy()
    h, w = output.shape[:2]
    color = STATUS_COLORS[status_key]

    if status_key == "running":
        border = max(8, min(h, w) // 120)
        cv2.rectangle(
            output, (border // 2, border // 2),
            (w - 1 - border // 2, h - 1 - border // 2),
            color, border,
        )

        # Put the healthy-state text in the title row of the Details subplot.
        pad = 16
        panel_w = (w - 3 * pad) // 2
        panel_h = (h - 3 * pad) // 2
        details_x1 = 2 * pad + panel_w
        details_y1 = 2 * pad + panel_h
        status_x = details_x1 + 145
        status_y = details_y1 + 28
        status_text = "CAMERA ALIVE | INFERENCE RUNNING"
        available_width = max(1, w - pad - status_x - 12)
        cv2.putText(
            output, _fit_text(status_text, available_width, 0.58, 2),
            (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            color, 2, cv2.LINE_AA,
        )
        return output

    if stale:
        output = cv2.addWeighted(output, 0.42, np.zeros_like(output), 0.58, 0)

    banner_h = max(76, int(h * 0.09))
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    output = cv2.addWeighted(overlay, 0.90, output, 0.10, 0)

    title_scale = max(0.75, min(1.25, w / 1500.0))
    cv2.putText(
        output, title, (24, int(banner_h * 0.48)),
        cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output, _fit_text(explanation, w - 48, 0.58, 1),
        (24, banner_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return output


def empty_dashboard(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    text = "ADVIS Dashboard - no inference frame received"
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    cv2.putText(
        image, text, ((width - size[0]) // 2, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 210, 210), 2, cv2.LINE_AA,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser("Remote ADVIS dashboard viewer")
    parser.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh-key", default="advis/vis/dashboard/state")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--camera-topic", default="/camera/back_view/image_raw")
    parser.add_argument(
        "--camera-message-type", choices=("auto", "raw", "compressed"),
        default="auto",
    )
    parser.add_argument(
        "--camera-timeout", type=float, default=2.0,
        help="Seconds without a ROS camera frame before showing CAMERA OFFLINE.",
    )
    parser.add_argument(
        "--inference-timeout", type=float, default=3.0,
        help="Seconds without a Zenoh detection before showing inference stopped.",
    )
    parser.add_argument(
        "--no-camera-monitor", action="store_true",
        help="Disable direct ROS camera monitoring (camera state becomes unknown).",
    )
    args = parser.parse_args()
    if args.camera_timeout <= 0 or args.inference_timeout <= 0:
        parser.error("--camera-timeout and --inference-timeout must be positive")

    zenoh.init_log_from_env_or("error")
    config = make_config(args.zenoh_endpoint)
    started_at = time.monotonic()
    camera_monitor = None
    if not args.no_camera_monitor:
        camera_monitor = ROSCameraMonitor(
            args.camera_topic, args.camera_message_type
        )
        camera_monitor.start()

    shared = {
        "raw": None,
        "generation": 0,
        "inference_last_received": None,
    }

    def on_dashboard_sample(sample) -> None:
        try:
            shared["raw"] = sample.payload.to_bytes()
            shared["generation"] += 1
            shared["inference_last_received"] = time.monotonic()
        except Exception as exc:
            print(f"Dashboard receive error: {exc}")

    try:
        with zenoh.open(config) as session:
            got_any = False
            for reply in session.get(args.zenoh_key):
                if getattr(reply, "ok", None) is None:
                    continue
                try:
                    # Stored state supplies the last image, but is deliberately
                    # not considered a fresh inference heartbeat.
                    shared["raw"] = reply.ok.payload.to_bytes()
                    shared["generation"] += 1
                    got_any = True
                except Exception as exc:
                    print(f"Skipping invalid stored dashboard payload: {exc}")

            if not got_any:
                print("No stored dashboard state yet.")

            rendered = empty_dashboard(args.width, args.height)
            rendered_generation = -1
            with session.declare_subscriber(args.zenoh_key, on_dashboard_sample):
                while True:
                    if shared["generation"] != rendered_generation:
                        try:
                            rendered = render_from_payload(
                                shared["raw"], args.width, args.height
                            )
                            rendered_generation = shared["generation"]
                        except Exception as exc:
                            print(f"Dashboard render error: {exc}")
                            rendered_generation = shared["generation"]

                    now = time.monotonic()
                    camera_last = (
                        camera_monitor.last_received
                        if camera_monitor is not None else None
                    )
                    monitor_available = (
                        camera_monitor is not None and camera_monitor.available
                    )
                    status_key, title, explanation = connection_status(
                        now,
                        shared["inference_last_received"],
                        camera_last,
                        started_at,
                        args.inference_timeout,
                        args.camera_timeout,
                        monitor_available,
                    )
                    display = draw_connection_banner(
                        rendered, status_key, title, explanation,
                        stale=status_key != "running",
                    )
                    cv2.imshow("ADVIS Dashboard", display)
                    if (cv2.waitKey(50) & 0xFF) == 27:
                        break
    finally:
        if camera_monitor is not None:
            camera_monitor.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
