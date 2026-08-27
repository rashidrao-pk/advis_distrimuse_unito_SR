import os
import time
import argparse
from collections import OrderedDict, deque

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage, CompressedImage
from cv_bridge import CvBridge


ALL_SAFETY_AREAS = ["PLeft", "PRight", "RoboArm", "ConvBelt"]

AREA_DISPLAY_NAMES = {
    "PLeft": "Pallet Left",
    "PRight": "Pallet Right",
    "RoboArm": "Robo Arm",
    "ConvBelt": "Conveyor Belt",
}


def ordered_area_list(areas):
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


def _ensure_gray(mask):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _prepare_binary_mask(mask, frame_shape_hw):
    h, w = frame_shape_hw
    mask = _ensure_gray(mask)
    if mask is None:
        return None
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _extract_mask_contours(mask_gray, frame_shape_hw):
    mask_bin = _prepare_binary_mask(mask_gray, frame_shape_hw)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_bin


def _crop_with_mask(frame, mask_gray):
    mask_bin = _prepare_binary_mask(mask_gray, frame.shape[:2])
    masked_full = cv2.bitwise_and(frame, frame, mask=mask_bin)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, masked_full, mask_bin

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = masked_full[y_min:y_max + 1, x_min:x_max + 1]
    bbox = (x_min, y_min, x_max, y_max)
    return cropped, bbox, masked_full, mask_bin


def _resize_128(image, keep_aspect=True, target=(128, 128), return_meta=False):
    target_w, target_h = target

    if image is None:
        return (None, None) if return_meta else None

    if not keep_aspect:
        out = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        meta = {
            "new_w": target_w,
            "new_h": target_h,
            "x_off": 0,
            "y_off": 0,
            "target_w": target_w,
            "target_h": target_h,
            "orig_h": image.shape[0],
            "orig_w": image.shape[1],
        }
        return (out, meta) if return_meta else out

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return (None, None) if return_meta else None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    meta = {
        "new_w": new_w,
        "new_h": new_h,
        "x_off": x_off,
        "y_off": y_off,
        "target_w": target_w,
        "target_h": target_h,
        "orig_h": h,
        "orig_w": w,
        "scale": scale,
    }

    return (canvas, meta) if return_meta else canvas


def create_union_mask(area_inputs, frame_shape_hw):
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


def overlay_outside_safety_blur(frame_bgr, area_inputs, blur_ksize=31, darken_factor=0.35):
    if len(area_inputs) == 0:
        return frame_bgr.copy()

    union_mask = create_union_mask(area_inputs, frame_bgr.shape[:2])

    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)
    darkened = (blurred.astype(np.float32) * darken_factor).clip(0, 255).astype(np.uint8)

    union_mask_3 = cv2.cvtColor(union_mask, cv2.COLOR_GRAY2BGR)
    out = np.where(union_mask_3 > 0, frame_bgr, darkened)
    return out


def resize_and_center(image, target_w, target_h, bg_color=(0, 0, 0)):
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


def scale_contours(contours, scale, x_off, y_off):
    scaled = []
    for cnt in contours:
        cnt_scaled = cnt.astype(np.float32).copy()
        cnt_scaled[:, 0, 0] = x_off + cnt_scaled[:, 0, 0] * scale
        cnt_scaled[:, 0, 1] = y_off + cnt_scaled[:, 0, 1] * scale
        scaled.append(cnt_scaled.astype(np.int32))
    return scaled


def draw_preprocessing_dashboard(
    frame_bgr, area_inputs, width=1600, height=1000, rows=2,
    stream_details=None,
):
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)

    pad = 16
    details_h = 170 if stream_details else 0
    content_bottom = height - details_h - (pad if stream_details else 0)
    panel_w = (width - 3 * pad) // 2
    panel_h = (
        content_bottom - 2 * pad
        if rows == 1
        else (content_bottom - 3 * pad) // 2
    )

    tl = (pad, pad, pad + panel_w, pad + panel_h)
    tr = (2 * pad + panel_w, pad, width - pad, pad + panel_h)

    def draw_panel_title(title, box):
        x1, y1, x2, y2 = box
        cv2.putText(canvas, title, (x1 + 12, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)

    draw_panel_title("Input Frame + Safety Areas", tl)
    draw_panel_title("Masked Full Frame", tr)

    if rows == 2:
        bl = (pad, 2 * pad + panel_h, pad + panel_w, content_bottom - pad)
        br = (2 * pad + panel_w, 2 * pad + panel_h, width - pad, content_bottom - pad)
        draw_panel_title("Per-Area Crops", bl)
        draw_panel_title("Final 128x128 Model Inputs", br)

    inner_margin = 12
    title_h = 40

    def inner_box(box):
        x1, y1, x2, y2 = box
        return (x1 + inner_margin, y1 + title_h, x2 - inner_margin, y2 - inner_margin)

    tl_in = inner_box(tl)
    tr_in = inner_box(tr)

    h, w = frame_bgr.shape[:2]

    # -----------------------------
    # TOP-LEFT: input + contours
    # -----------------------------
    tl_w = tl_in[2] - tl_in[0]
    tl_h = tl_in[3] - tl_in[1]
    scale = min(tl_w / w, tl_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    inp = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_off = tl_in[0] + (tl_w - new_w) // 2
    y_off = tl_in[1] + (tl_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = inp

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        contours = info.get("contours", [])
        bbox = info.get("bbox")

        scaled = scale_contours(contours, scale, x_off, y_off)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, (0, 255, 255), 2)
            pt = scaled[0][0][0]
            cv2.putText(canvas, AREA_DISPLAY_NAMES.get(area_name, area_name),
                        (int(pt[0]), max(20, int(pt[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            rx1 = int(x_off + x1 * scale)
            ry1 = int(y_off + y1 * scale)
            rx2 = int(x_off + x2 * scale)
            ry2 = int(y_off + y2 * scale)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

    # -----------------------------
    # TOP-RIGHT: masked frame
    # -----------------------------
    masked_bg = overlay_outside_safety_blur(frame_bgr, area_inputs)
    tr_w = tr_in[2] - tr_in[0]
    tr_h = tr_in[3] - tr_in[1]
    masked_disp = resize_with_blurred_background(masked_bg, tr_w, tr_h)
    canvas[tr_in[1]:tr_in[1] + tr_h, tr_in[0]:tr_in[0] + tr_w] = masked_disp

    if rows == 2:
        bl_in = inner_box(bl)
        br_in = inner_box(br)

        # BOTTOM-LEFT: crops grid
        crop_panel = np.full(
            (bl_in[3] - bl_in[1], bl_in[2] - bl_in[0], 3), 245, dtype=np.uint8
        )
        draw_area_grid(crop_panel, area_inputs, key_name="crop")
        canvas[bl_in[1]:bl_in[3], bl_in[0]:bl_in[2]] = crop_panel

        # BOTTOM-RIGHT: resized 128x128 inputs
        model_panel = np.full(
            (br_in[3] - br_in[1], br_in[2] - br_in[0], 3), 245, dtype=np.uint8
        )
        draw_area_grid(model_panel, area_inputs, key_name="resized", show_meta=True)
        canvas[br_in[1]:br_in[3], br_in[0]:br_in[2]] = model_panel

    if stream_details:
        details_box = (pad, height - details_h, width - pad, height - pad)
        draw_panel_title("ROS Stream Details", details_box)
        x_left = details_box[0] + 16
        x_right = details_box[0] + (details_box[2] - details_box[0]) // 2
        y_start = details_box[1] + 58
        line_gap = 28

        left_lines = [
            f"Topic: {stream_details.get('topic', '-')}",
            f"Advertised type(s): {', '.join(stream_details.get('types', [])) or 'discovering...'}",
            (
                f"Available: raw={'YES' if stream_details.get('raw_available') else 'NO'}"
                f" | compressed={'YES' if stream_details.get('compressed_available') else 'NO'}"
            ),
        ]
        right_lines = [
            (
                f"FPS: {stream_details.get('fps', 0.0):.2f}"
                f" | Resolution: {stream_details.get('resolution', '-')}"
            ),
            (
                f"Encoding/format: {stream_details.get('encoding', '-')}"
                f" | Publishers: {stream_details.get('publishers', 0)}"
            ),
            (
                f"Frame ID: {stream_details.get('frame_id', '-')}"
                f" | Received frames: {stream_details.get('frame_count', 0)}"
            ),
        ]
        for idx, text in enumerate(left_lines):
            cv2.putText(
                canvas, text, (x_left, y_start + idx * line_gap),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (25, 25, 25), 1, cv2.LINE_AA,
            )
        for idx, text in enumerate(right_lines):
            cv2.putText(
                canvas, text, (x_right, y_start + idx * line_gap),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (25, 25, 25), 1, cv2.LINE_AA,
            )

    return canvas


def draw_area_grid(panel, area_inputs, key_name="crop", show_meta=False):
    areas = ordered_area_list(area_inputs.keys())
    h, w = panel.shape[:2]

    if len(areas) == 0:
        cv2.putText(panel, "No active areas / no masks matched", (30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        return panel

    cols = 2
    rows = 2
    pad = 12
    cell_w = (w - (cols + 1) * pad) // cols
    cell_h = (h - (rows + 1) * pad) // rows

    for idx, area_name in enumerate(areas[:4]):
        r = idx // cols
        c = idx % cols
        x1 = pad + c * (cell_w + pad)
        y1 = pad + r * (cell_h + pad)
        x2 = x1 + cell_w
        y2 = y1 + cell_h

        cv2.rectangle(panel, (x1, y1), (x2, y2), (80, 80, 80), 1)
        cv2.putText(panel, AREA_DISPLAY_NAMES.get(area_name, area_name),
                    (x1 + 8, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

        img = area_inputs[area_name].get(key_name)
        if img is not None:
            disp_h = cell_h - 40
            disp_w = cell_w - 16
            disp = resize_and_center(img, disp_w, disp_h, bg_color=(0, 0, 0))
            panel[y1 + 32:y1 + 32 + disp_h, x1 + 8:x1 + 8 + disp_w] = disp

        if show_meta:
            meta = area_inputs[area_name].get("resize_meta")
            bbox = area_inputs[area_name].get("bbox")
            if meta is not None:
                text1 = f"new:{meta['new_w']}x{meta['new_h']} off:({meta['x_off']},{meta['y_off']})"
                cv2.putText(panel, text1, (x1 + 8, y2 - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                text2 = f"bbox: ({x_min},{y_min})-({x_max},{y_max})"
                cv2.putText(panel, text2, (x1 + 8, y2 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

    return panel


def resize_with_blurred_background(image, target_w, target_h):
    """Fit an image without black letterbox bars."""
    background = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    background = cv2.GaussianBlur(background, (31, 31), 0)
    foreground = resize_and_center(
        image, target_w, target_h, bg_color=(235, 235, 235)
    )

    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    fitted_w = max(1, int(round(w * scale)))
    fitted_h = max(1, int(round(h * scale)))
    x_off = (target_w - fitted_w) // 2
    y_off = (target_h - fitted_h) // 2
    background[
        y_off:y_off + fitted_h, x_off:x_off + fitted_w
    ] = foreground[
        y_off:y_off + fitted_h, x_off:x_off + fitted_w
    ]
    return background


def draw_dual_camera_dashboard(camera_states, width=1800, height=1000):
    """Draw back/front camera rows with stream details in the last column."""
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    pad = 14
    row_h = (height - 3 * pad) // 2
    details_w = max(360, int(width * 0.25))
    image_w = (width - details_w - 4 * pad) // 2

    def box_for(row, column):
        y1 = pad + row * (row_h + pad)
        if column == 0:
            x1, x2 = pad, pad + image_w
        elif column == 1:
            x1, x2 = 2 * pad + image_w, 2 * pad + 2 * image_w
        else:
            x1, x2 = 3 * pad + 2 * image_w, width - pad
        return x1, y1, x2, y1 + row_h

    def title(text, box):
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 30), 1)
        cv2.putText(
            canvas, text, (x1 + 10, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA,
        )

    for row, camera_key in enumerate(("back", "front")):
        state = camera_states[camera_key]
        label = camera_key.upper()
        input_box = box_for(row, 0)
        masked_box = box_for(row, 1)
        details_box = box_for(row, 2)
        title(f"{label}: Input + Safety Areas", input_box)
        title(f"{label}: Masked Frame", masked_box)
        title(f"{label}: ROS Details", details_box)

        frame = state.get("frame")
        is_online = (
            frame is not None
            and time.monotonic() - state.get("last_received", 0.0) <= 2.0
        )
        if not is_online:
            offline_text = f"{label} CAMERA OFFLINE"
            for box in (input_box, masked_box):
                x1, y1, x2, y2 = box
                cv2.putText(
                    canvas, offline_text,
                    (x1 + 25, y1 + (y2 - y1) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (40, 40, 210), 2,
                    cv2.LINE_AA,
                )
        else:
            area_inputs = state["area_inputs"]
            annotated = frame.copy()
            for area_name, info in area_inputs.items():
                contours = info.get("contours", [])
                bbox = info.get("bbox")
                cv2.drawContours(annotated, contours, -1, (0, 255, 255), 2)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated, AREA_DISPLAY_NAMES.get(area_name, area_name),
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA,
                    )

            masked = overlay_outside_safety_blur(frame, area_inputs)
            for image, box in ((annotated, input_box), (masked, masked_box)):
                x1, y1, x2, y2 = box
                inner_x, inner_y = x1 + 8, y1 + 40
                inner_w, inner_h = x2 - x1 - 16, y2 - y1 - 48
                display = resize_with_blurred_background(image, inner_w, inner_h)
                canvas[
                    inner_y:inner_y + inner_h, inner_x:inner_x + inner_w
                ] = display

        details = state["details"]
        status = "ONLINE" if is_online else "OFFLINE"
        status_color = (20, 135, 20) if is_online else (40, 40, 210)
        dx, dy = details_box[0] + 12, details_box[1] + 58
        lines = [
            (f"Status: {status}", status_color),
            (f"Topic: {details.get('topic', '-')}", (25, 25, 25)),
            (f"Type: {', '.join(details.get('types', [])) or 'not detected'}", (25, 25, 25)),
            (
                f"Raw: {'YES' if details.get('raw_available') else 'NO'}"
                f"   Compressed: {'YES' if details.get('compressed_available') else 'NO'}",
                (25, 25, 25),
            ),
            (
                f"FPS: {details.get('fps', 0.0):.2f}"
                f"   Resolution: {details.get('resolution', '-')}",
                (25, 25, 25),
            ),
            (f"Encoding: {details.get('encoding', '-')}", (25, 25, 25)),
            (f"Frame ID: {details.get('frame_id', '-')}", (25, 25, 25)),
            (
                f"Publishers: {details.get('publishers', 0)}"
                f"   Frames: {details.get('frame_count', 0)}",
                (25, 25, 25),
            ),
        ]
        for idx, (text, color) in enumerate(lines):
            cv2.putText(
                canvas, text, (dx, dy + idx * 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

    return canvas


class CheckInputNode(Node):
    def __init__(self, args):
        super().__init__("check_input_node")

        self.args = args
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_log_t = time.time()
        self.camera_states = {}
        for camera_key, topic in (
            ("back", args.camera_topic),
            ("front", args.front_camera_topic),
        ):
            self.camera_states[camera_key] = {
                "frame": None,
                "area_inputs": OrderedDict(),
                "last_received": 0.0,
                "frame_times": deque(maxlen=120),
                "last_topic_info_t": 0.0,
                "details": {
                    "topic": topic,
                    "types": [],
                    "raw_available": False,
                    "compressed_available": False,
                    "fps": 0.0,
                    "resolution": "-",
                    "encoding": "-",
                    "publishers": 0,
                    "frame_id": "-",
                    "frame_count": 0,
                },
            }
        self.stream_details = self.camera_states["back"]["details"]

        self.areas = (
            ALL_SAFETY_AREAS
            if "ALL" in args.safety_area
            else ordered_area_list(args.safety_area)
        )

        if len(args.area_names) != len(args.static_mask_paths):
            raise ValueError("area_names and static_mask_paths must have the same length")

        missing_masks = [area for area in self.areas if area not in args.area_names]
        if missing_masks:
            raise ValueError(
                "No mask supplied for selected area(s): " + ", ".join(missing_masks)
            )

        self.area_masks = OrderedDict()
        pairs = list(zip(args.area_names, args.static_mask_paths))
        pairs = sorted(
            pairs,
            key=lambda x: ALL_SAFETY_AREAS.index(x[0]) if x[0] in ALL_SAFETY_AREAS else 999
        )

        for area_name, mask_path in pairs:
            if area_name not in self.areas:
                continue
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {area_name}: {mask_path}")

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not load mask for {area_name}: {mask_path}")

            self.area_masks[area_name] = mask
            self.get_logger().info(f"[mask] loaded {area_name}: {mask_path} shape={mask.shape}")

        self.front_area_masks = self.area_masks
        if args.dual_camera_dashboard and args.front_static_mask_paths:
            if len(args.area_names) != len(args.front_static_mask_paths):
                raise ValueError(
                    "area_names and front_static_mask_paths must have the same length"
                )
            self.front_area_masks = OrderedDict()
            for area_name, mask_path in zip(
                args.area_names, args.front_static_mask_paths
            ):
                if area_name not in self.areas:
                    continue
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(
                        f"Front mask not found for {area_name}: {mask_path}"
                    )
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(
                        f"Could not load front mask for {area_name}: {mask_path}"
                    )
                self.front_area_masks[area_name] = mask
                self.get_logger().info(
                    f"[front mask] loaded {area_name}: {mask_path} shape={mask.shape}"
                )
        elif args.dual_camera_dashboard:
            self.get_logger().warning(
                "No front masks supplied; temporarily reusing the back-camera masks."
            )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if args.use_compressed:
            self.subscription = self.create_subscription(
                CompressedImage,
                args.camera_topic,
                lambda msg: self.callback_compressed(msg, "back"),
                sensor_qos,
            )
            self.get_logger().info(f"Subscribed to CompressedImage topic: {args.camera_topic}")
            if args.dual_camera_dashboard:
                self.front_subscription = self.create_subscription(
                    CompressedImage,
                    args.front_camera_topic,
                    lambda msg: self.callback_compressed(msg, "front"),
                    sensor_qos,
                )
        else:
            self.subscription = self.create_subscription(
                RosImage,
                args.camera_topic,
                lambda msg: self.callback_raw(msg, "back"),
                sensor_qos,
            )
            self.get_logger().info(f"Subscribed to Image topic: {args.camera_topic}")
            if args.dual_camera_dashboard:
                self.front_subscription = self.create_subscription(
                    RosImage,
                    args.front_camera_topic,
                    lambda msg: self.callback_raw(msg, "front"),
                    sensor_qos,
                )

        if args.dual_camera_dashboard:
            message_type = "CompressedImage" if args.use_compressed else "Image"
            self.get_logger().info(
                f"Subscribed to {message_type} front topic: {args.front_camera_topic}"
            )

        self.get_logger().info(f"Active areas: {self.areas}")

    def update_stream_details(
        self, msg, frame_bgr, received_type, encoding, camera_key
    ):
        now = time.monotonic()
        state = self.camera_states[camera_key]
        frame_times = state["frame_times"]
        frame_times.append(now)
        fps = 0.0
        if len(frame_times) > 1:
            elapsed = frame_times[-1] - frame_times[0]
            if elapsed > 0:
                fps = (len(frame_times) - 1) / elapsed

        h, w = frame_bgr.shape[:2]
        details = state["details"]
        details.update({
            "fps": fps,
            "resolution": f"{w}x{h}",
            "encoding": encoding or "-",
            "frame_id": msg.header.frame_id or "-",
            "frame_count": details["frame_count"] + 1,
        })

        if now - state["last_topic_info_t"] >= 1.0:
            topic_types = dict(self.get_topic_names_and_types())
            topic = details["topic"]
            base_topic = (
                topic[:-len("/compressed")]
                if topic.endswith("/compressed")
                else topic
            )
            related_topics = {topic, base_topic, f"{base_topic}/compressed"}
            related_types = {
                msg_type
                for topic_name in related_topics
                for msg_type in topic_types.get(topic_name, [])
            }
            advertised_types = topic_types.get(topic, []) or [received_type]
            details.update({
                "types": advertised_types,
                "raw_available": "sensor_msgs/msg/Image" in related_types,
                "compressed_available": "sensor_msgs/msg/CompressedImage" in related_types,
                "publishers": len(self.get_publishers_info_by_topic(topic)),
            })
            state["last_topic_info_t"] = now

    def preprocess_area(self, frame_bgr, area_name, area_masks=None):
        masks = area_masks if area_masks is not None else self.area_masks
        mask = masks[area_name]

        contours, mask_bin = _extract_mask_contours(mask, frame_bgr.shape[:2])
        cropped, bbox, masked_full, mask_bin = _crop_with_mask(frame_bgr, mask)

        if cropped is None:
            return {
                "status": "mask_failed",
                "crop": None,
                "resized": None,
                "bbox": None,
                "contours": contours,
                "mask_bin": mask_bin,
                "masked_full": masked_full,
                "resize_meta": None,
            }

        resized, resize_meta = _resize_128(
            cropped,
            keep_aspect=self.args.keep_aspect,
            target=(self.args.target_size, self.args.target_size),
            return_meta=True,
        )

        return {
            "status": "ok",
            "crop": cropped.copy(),
            "resized": resized.copy(),
            "bbox": bbox,
            "contours": contours,
            "mask_bin": mask_bin,
            "masked_full": masked_full,
            "resize_meta": resize_meta,
        }

    def process_frame(self, frame_bgr, camera_key="back"):
        self.frame_count += 1

        area_masks = (
            self.front_area_masks if camera_key == "front" else self.area_masks
        )
        area_inputs = OrderedDict()
        for area_name in self.areas:
            area_inputs[area_name] = self.preprocess_area(
                frame_bgr, area_name, area_masks
            )

        state = self.camera_states[camera_key]
        state["frame"] = frame_bgr
        state["area_inputs"] = area_inputs
        state["last_received"] = time.monotonic()

        if self.args.show_dashboard:
            if self.args.dual_camera_dashboard:
                dashboard = draw_dual_camera_dashboard(
                    self.camera_states,
                    width=self.args.dashboard_width,
                    height=self.args.dashboard_height,
                )
            else:
                dashboard = draw_preprocessing_dashboard(
                    frame_bgr,
                    area_inputs,
                    width=self.args.dashboard_width,
                    height=self.args.dashboard_height,
                    rows=self.args.dashboard_rows,
                    stream_details=(
                        state["details"]
                        if self.args.show_stream_details
                        else None
                    ),
                )
            cv2.imshow("ADVIS Preprocessing Check", dashboard)

        else:
            # basic separate windows
            frame_vis = frame_bgr.copy()
            for area_name in self.areas:
                info = area_inputs[area_name]
                contours = info.get("contours", [])
                bbox = info.get("bbox")

                cv2.drawContours(frame_vis, contours, -1, (0, 255, 255), 2)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_vis, AREA_DISPLAY_NAMES.get(area_name, area_name),
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                crop = info.get("crop")
                resized = info.get("resized")
                if crop is not None:
                    cv2.imshow(f"{area_name} - crop", crop)
                if resized is not None:
                    cv2.imshow(f"{area_name} - model_input_128", resized)

            cv2.imshow("Input Frame", frame_vis)

        if self.frame_count % self.args.log_every_n == 0:
            msg_parts = [f"frame={self.frame_count}"]
            for area_name in self.areas:
                info = area_inputs[area_name]
                bbox = info.get("bbox")
                meta = info.get("resize_meta")
                if bbox is None:
                    msg_parts.append(f"{area_name}: mask_failed")
                else:
                    msg_parts.append(
                        f"{area_name}: bbox={bbox}, resized={self.args.target_size}x{self.args.target_size}, meta={meta}"
                    )
            self.get_logger().info(" | ".join(msg_parts))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.get_logger().info("ESC pressed. Shutting down.")
            rclpy.shutdown()

    def callback_compressed(self, msg, camera_key="back"):
        try:
            frame_bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                self.get_logger().error("Failed to decode compressed frame.")
                return
            self.update_stream_details(
                msg, frame_bgr, "sensor_msgs/msg/CompressedImage", msg.format,
                camera_key,
            )
            self.process_frame(frame_bgr, camera_key)
        except Exception as e:
            self.get_logger().error(f"Compressed callback error: {e}")

    def callback_raw(self, msg, camera_key="back"):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.update_stream_details(
                msg, frame_bgr, "sensor_msgs/msg/Image", msg.encoding,
                camera_key,
            )
            self.process_frame(frame_bgr, camera_key)
        except Exception as e:
            self.get_logger().error(f"Raw callback error: {e}")


def parse_args():
    p = argparse.ArgumentParser("Check preprocessing / model inputs from ROS camera topic")

    p.add_argument("--camera_topic", default="/camera/back_view/image_raw")
    p.add_argument(
        "--front_camera_topic",
        default="/camera/front_view/image_raw",
        help="Front-camera topic used by --dual_camera_dashboard.",
    )
    p.add_argument("--use_compressed", action="store_true",
                   help="Use sensor_msgs/CompressedImage instead of sensor_msgs/Image")

    p.add_argument(
        "--safety_area",
        nargs="+",
        default=["ALL"],
        choices=["ALL", *ALL_SAFETY_AREAS],
        help="One or more safety areas, or ALL.",
    )
    p.add_argument("--area_names", nargs="+", default=["PLeft", "PRight", "RoboArm", "ConvBelt"])
    p.add_argument("--static_mask_paths", nargs="+", required=True)
    p.add_argument(
        "--front_static_mask_paths",
        nargs="+",
        help="Front-camera masks. Back-camera masks are reused when omitted.",
    )

    p.add_argument("--target_size", type=int, default=128)
    p.add_argument("--keep_aspect", action="store_true", default=True)

    p.add_argument("--show_dashboard", action="store_true")
    p.add_argument(
        "--dual_camera_dashboard",
        action="store_true",
        help="Show back and front camera rows with details in the last column.",
    )
    p.add_argument(
        "--show_stream_details",
        action="store_true",
        help="Add a bottom row with live ROS topic, type, FPS, and frame details.",
    )
    p.add_argument(
        "--dashboard_rows",
        type=int,
        choices=[1, 2],
        default=2,
        help="1 shows only the top two panels; 2 shows all four panels.",
    )
    p.add_argument("--dashboard_width", type=int, default=1600)
    p.add_argument("--dashboard_height", type=int, default=1000)

    p.add_argument("--log_every_n", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CheckInputNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
