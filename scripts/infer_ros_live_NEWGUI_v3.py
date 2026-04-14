import os
import json
import time
import argparse
from collections import deque, OrderedDict

import cv2
import numpy as np

import torch
import torchvision.transforms as transforms

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

import utils as ut
import utils_model as utmc
from utils_model import Encoder, Decoder, Discriminator


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


############################################################################################################
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


def colorize_anomaly_map(dist_map, clip_max=None):
    if dist_map is None:
        return None

    dm = dist_map.astype(np.float32)

    if clip_max is None:
        clip_max = np.percentile(dm, 99.5)
        clip_max = max(clip_max, 1e-6)

    dm = np.clip(dm / clip_max, 0.0, 1.0)
    dm_u8 = (dm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(dm_u8, cv2.COLORMAP_JET)
    return heat


def paste_area_result_in_full_frame(
    target_canvas,
    patch_bgr,
    bbox,
    mask_bin,
    keep_background=False,
    background_canvas=None
):
    if patch_bgr is None or bbox is None or mask_bin is None:
        return target_canvas

    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1

    if crop_w <= 0 or crop_h <= 0:
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


def draw_text_table(panel, results, frame_id=None):
    h, w = panel.shape[:2]
    panel[:] = (245, 245, 245)

    title_y = 35
    cv2.putText(panel, "Details", (w // 2 - 50, title_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    y = 70
    cv2.line(panel, (20, y), (w - 20, y), (40, 40, 40), 2)
    y += 35

    if frame_id is not None:
        cv2.putText(panel, f"Frame: {frame_id}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        y += 20
        cv2.line(panel, (20, y), (w - 20, y), (40, 40, 40), 1)
        y += 35

    headers = ["Safety Area", "Raw Score", "Threshold", "Norm Score", "Status"]
    col_x = [30, 240, 370, 510, 670]

    for i, hdr in enumerate(headers):
        cv2.putText(panel, hdr, (col_x[i], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)

    y += 20
    cv2.line(panel, (20, y), (w - 20, y), (40, 40, 40), 1)
    y += 35

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
            cv2.putText(panel, str(val), (col_x[i], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color if i >= 3 else (30, 30, 30),
                        2 if i == 4 else 1,
                        cv2.LINE_AA)

        y += 20
        cv2.line(panel, (20, y), (w - 20, y), (120, 120, 120), 1)
        y += 35

    return panel
############################################################################################################


def draw_dashboard_panel(frame_bgr, area_inputs, latest_results, frame_id=None, width=1600, height=1000):
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
        cv2.putText(canvas, title, (x1 + 12, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)

    draw_panel_title("Input Image with detections", tl)
    draw_panel_title("Anomaly Map", tr)
    draw_panel_title("Reconstructed Image", bl)
    draw_panel_title("Details", br)

    inner_margin = 12
    title_h = 40

    def inner_box(box):
        x1, y1, x2, y2 = box
        return (
            x1 + inner_margin,
            y1 + title_h,
            x2 - inner_margin,
            y2 - inner_margin,
        )

    tl_in = inner_box(tl)
    tr_in = inner_box(tr)
    bl_in = inner_box(bl)
    br_in = inner_box(br)

    input_vis = overlay_outside_safety_blur(frame_bgr, area_inputs)

    h, w = frame_bgr.shape[:2]
    tl_w = tl_in[2] - tl_in[0]
    tl_h = tl_in[3] - tl_in[1]
    scale = min(tl_w / w, tl_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    tl_img = cv2.resize(input_vis, (new_w, new_h), interpolation=cv2.INTER_AREA)
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
            label = f"{AREA_DISPLAY_NAMES.get(area_name, area_name)}: {rr.get('norm_score', 0):.2f}" if "norm_score" in rr else area_name
            cv2.putText(canvas, label, (int(pt[0]), max(20, int(pt[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    base_black = np.zeros_like(frame_bgr)
    recon_full = base_black.copy()
    anom_full = np.full_like(frame_bgr, 255)

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        bbox = info.get("bbox")
        mask_bin = info.get("mask_bin")
        recon_patch = info.get("recon_patch_bgr")
        anom_patch = info.get("anom_patch_bgr")

        recon_full = paste_area_result_in_full_frame(
            recon_full, recon_patch, bbox, mask_bin
        )

        anom_full = paste_area_result_in_full_frame(
            anom_full, anom_patch, bbox, mask_bin
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
            cv2.putText(canvas, label, (int(pt[0]), int(pt[1]) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    bl_w = bl_in[2] - bl_in[0]
    bl_h = bl_in[3] - bl_in[1]
    recon_disp = resize_and_center(recon_full, bl_w, bl_h, bg_color=(0, 0, 0))
    canvas[bl_in[1]:bl_in[1] + bl_h, bl_in[0]:bl_in[0] + bl_w] = recon_disp

    scale_bl = min(bl_w / w, bl_h / h)
    new_w_bl = max(1, int(w * scale_bl))
    new_h_bl = max(1, int(h * scale_bl))
    x_off_bl = bl_in[0] + (bl_w - new_w_bl) // 2
    y_off_bl = bl_in[1] + (bl_h - new_h_bl) // 2

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        contours = info.get("contours", [])
        scaled = scale_contours(contours, scale_bl, x_off_bl, y_off_bl)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, (0, 180, 0), 2)

    details_panel = np.full((br_in[3] - br_in[1], br_in[2] - br_in[0], 3), 245, dtype=np.uint8)
    details_panel = draw_text_table(details_panel, latest_results, frame_id=frame_id)
    canvas[br_in[1]:br_in[3], br_in[0]:br_in[2]] = details_panel

    return canvas


####################################################################################################################
def _ensure_gray(mask):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _prepare_binary_mask(mask, frame_shape_hw):
    h, w = frame_shape_hw
    mask = _ensure_gray(mask)
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
        return None, None, masked_full

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = masked_full[y_min:y_max + 1, x_min:x_max + 1]
    bbox = (x_min, y_min, x_max, y_max)
    return cropped, bbox, masked_full


def _resize_128(image, keep_aspect=True, target=(128, 128)):
    target_w, target_h = target

    if image is None:
        return None

    if not keep_aspect:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def tensor_to_hwc_float32(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32) * 0.5 + 0.5)


def _compute_distance_offset_np(imgA: np.ndarray, imgB: np.ndarray, offset: int) -> np.ndarray:
    H, W, _ = imgA.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    for di in range(-offset, offset + 1):
        for dj in range(-offset, offset + 1):
            i0a = max(0, di)
            i1a = min(H, H + di)
            i0b = max(0, -di)
            i1b = min(H, H - di)

            j0a = max(0, dj)
            j1a = min(W, W + dj)
            j0b = max(0, -dj)
            j1b = min(W, W - dj)

            diff = imgA[i0a:i1a, j0a:j1a] - imgB[i0b:i1b, j0b:j1b]
            d = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)
            dist[i0a:i1a, j0a:j1a] = np.minimum(dist[i0a:i1a, j0a:j1a], d)
    return dist


def compute_anomaly_score_pair(imgA, imgB, offset=2, quantile=0.995):
    dist = _compute_distance_offset_np(imgA, imgB, offset)
    return float(np.quantile(dist, quantile)), dist


def load_threshold(threshold_dir: str, safety_area: str) -> float:
    json_path = os.path.join(threshold_dir, safety_area, f"threshold_{safety_area}.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        return float(d["threshold"])
    raise FileNotFoundError(f"Threshold file not found: {json_path}")


def build_suffix_for_area(area, args):
    class P:
        pass

    params = P()
    params.subgroup = area
    params.latent_dims = args.latent_dims
    params.z_dim = args.latent_dims
    params.dataset_type = args.dataset_source_name
    params.subgroup_mask = args.subgroup_mask
    params.target_size = (128, 128)

    if not hasattr(args, "save_figures"):
        args.save_figures = False
    if not hasattr(args, "train"):
        args.train = False
    if not hasattr(args, "test"):
        args.test = False
    if not hasattr(args, "inference"):
        args.inference = True

    paths = P()
    cwd = os.getcwd()
    paths.path_codes = cwd
    paths.path_codes_local = cwd
    paths.path_results_local = cwd
    paths.path_results_cloud = cwd
    paths.path_models = os.path.join(cwd, args.checkpoints)

    suffix, _ = ut.get_create_results_path(
        area,
        params,
        args,
        paths,
        save_path_type=args.save_path_type,
        dir="scripts/results",
        verbose=False,
    )
    return suffix


def load_models_and_thresholds(areas, args, device, log_fn=None):
    models = OrderedDict()
    thresholds = OrderedDict()

    checkpoint_root = os.path.join(os.getcwd(), args.checkpoints)

    for area in ordered_area_list(areas):
        if log_fn:
            log_fn(1, f"[model-load] preparing area={area}")

        enc = Encoder(z_size=args.latent_dims).to(device)
        dec = Decoder(z_size=args.latent_dims).to(device)
        dis = Discriminator().to(device)

        optED, optD = utmc.get_optimizers(enc, dec, dis, verbose=False)
        suffix = build_suffix_for_area(area, args)

        if log_fn:
            log_fn(2, f"[model-load] area={area} suffix={suffix}")
            log_fn(2, f"[model-load] checkpoint_root={checkpoint_root}")

        history = utmc.load_model(
            enc, dec, dis, optED, optD,
            checkpoint_root, suffix, device=device, verbose=False
        )

        if len(history) == 0:
            raise RuntimeError(f"No checkpoint found for area={area}, suffix={suffix}")

        enc.eval()
        dec.eval()

        tau = load_threshold(args.threshold_dir, area)

        models[area] = {"encoder": enc, "decoder": dec, "suffix": suffix}
        thresholds[area] = tau

        if log_fn:
            log_fn(1, f"[loaded] {area}: suffix={suffix}, tau={tau:.6f}")

    return models, thresholds


def draw_timeline_panel(score_history, latest_results, width=1000, height=500, max_points=200):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    left_pad = 80
    right_pad = 20
    top_pad = 40
    bottom_pad = 40

    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    cv2.rectangle(canvas, (left_pad, top_pad), (left_pad + plot_w, top_pad + plot_h), (80, 80, 80), 1)

    y_thr = top_pad + int(plot_h * 0.5)
    cv2.line(canvas, (left_pad, y_thr), (left_pad + plot_w, y_thr), (0, 0, 255), 1)
    cv2.putText(canvas, "thr=1.0", (left_pad + 8, y_thr - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    for val in [0.0, 0.5, 1.0, 1.5, 2.0]:
        yy = top_pad + int(plot_h * (1.0 - min(val, 2.0) / 2.0))
        cv2.line(canvas, (left_pad - 5, yy), (left_pad, yy), (180, 180, 180), 1)
        cv2.putText(canvas, f"{val:.1f}", (10, yy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(canvas, "ADVIS Live Anomaly Timeline (normalized scores)",
                (left_pad, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    keys = ordered_area_list(score_history.keys())
    for idx, area_name in enumerate(keys):
        vals = list(score_history[area_name])

        if len(vals) >= 2:
            pts = []
            recent_vals = vals[-max_points:]
            for i, v in enumerate(recent_vals):
                x = left_pad + int(i * (plot_w / max(1, max_points - 1)))
                v_clip = max(0.0, min(2.0, float(v)))
                y = top_pad + int(plot_h * (1.0 - v_clip / 2.0))
                pts.append((x, y, float(v)))

            for i in range(1, len(pts)):
                p0 = pts[i - 1]
                p1 = pts[i]
                seg_color = (0, 0, 255) if (p0[2] > 1.0 or p1[2] > 1.0) else (255, 255, 255)
                cv2.line(canvas, (p0[0], p0[1]), (p1[0], p1[1]), seg_color, 2)

        latest = latest_results.get(area_name, {})
        latest_norm = float(latest.get("norm_score", 0.0)) if "norm_score" in latest else 0.0
        legend_color = (0, 0, 255) if latest_norm > 1.0 else (255, 255, 255)

        label = AREA_DISPLAY_NAMES.get(area_name, area_name)
        if "norm_score" in latest:
            label += f"  {latest['norm_score']:.3f}"
        if "status" in latest:
            label += f"  [{latest['status']}]"

        legend_y = top_pad + 20 + 28 * idx
        cv2.line(canvas, (width - 360, legend_y - 5), (width - 320, legend_y - 5), legend_color, 3)
        cv2.putText(canvas, label, (width - 310, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, legend_color, 1, cv2.LINE_AA)

    return canvas


class LiveRosAnomalyInfer(Node):
    def __init__(self, args):
        super().__init__("live_ros_anomaly_infer")

        self.args = args
        self.verbose_level = args.verbose_level
        self.log_every_n = args.log_every_n

        self.vlog(1, "[startup] initializing node")

        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.bridge = CvBridge()

        self.vlog(1, f"[startup] torch.cuda.is_available()={torch.cuda.is_available()}")
        self.vlog(1, f"[startup] requested cpu flag={args.cpu}")
        self.vlog(1, f"[startup] selected device={self.device}")

        if self.device.type == "cuda":
            self.vlog(1, f"[startup] cuda device count={torch.cuda.device_count()}")
            self.vlog(1, f"[startup] cuda current device={torch.cuda.current_device()}")
            self.vlog(1, f"[startup] cuda device name={torch.cuda.get_device_name(torch.cuda.current_device())}")

        self.areas = ALL_SAFETY_AREAS if args.safety_area.upper() == "ALL" else ordered_area_list([args.safety_area])
        self.vlog(1, f"[startup] active areas={self.areas}")

        self.vlog(1, "[startup] loading models and thresholds...")
        t0 = time.time()
        self.models, self.thresholds = load_models_and_thresholds(
            self.areas, args, self.device, log_fn=self.vlog
        )
        self.vlog(1, f"[startup] models loaded in {time.time() - t0:.3f}s")

        self.area_masks = OrderedDict()
        if len(args.area_names) != len(args.static_mask_paths):
            raise ValueError("area_names and static_mask_paths must have the same length")

        self.vlog(1, "[startup] loading masks...")
        pairs = list(zip(args.area_names, args.static_mask_paths))
        pairs = sorted(pairs, key=lambda x: ALL_SAFETY_AREAS.index(x[0]) if x[0] in ALL_SAFETY_AREAS else 999)

        for area_name, mask_path in pairs:
            if area_name not in self.areas:
                continue
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {area_name}: {mask_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not load mask for {area_name}: {mask_path}")
            self.area_masks[area_name] = mask
            self.vlog(1, f"[mask] loaded area={area_name} path={mask_path} shape={mask.shape}")

        self.normalize = transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subscription = self.create_subscription(
            RosImage,
            args.camera_topic,
            self.ros_callback_store_latest,
            sensor_qos,
        )

        self.frame_count = 0
        self.processed_count = 0
        self.max_frames = args.max_frames
        self.start_time = time.time()
        self.first_callback_seen = False

        self.latest_msg = None
        self.latest_msg_id = 0
        self.last_processed_msg_id = 0
        self.is_processing = False
        self.process_start_time = None

        self.show_timeline = args.show_timeline
        self.timeline_history_len = args.timeline_history
        self.timeline_width = args.timeline_width
        self.timeline_height = args.timeline_height

        self.score_history = OrderedDict(
            (area, deque(maxlen=self.timeline_history_len)) for area in self.areas
        )
        self.latest_results = OrderedDict(
            (area, {"status": "waiting"}) for area in self.areas
        )

        self.wait_start = time.time()
        self.wait_timer = self.create_timer(5.0, self.wait_for_first_frame_log)
        self.process_timer = self.create_timer(args.process_period, self.process_latest_frame)

        self.vlog(1, f"[subscriber] subscribed to {args.camera_topic}")
        self.vlog(1, f"[subscriber] frame_stride={self.args.frame_stride}")
        self.vlog(1, f"[subscriber] max_frames={self.max_frames}")
        self.vlog(1, f"[subscriber] process_period={self.args.process_period}s")

    def vlog(self, level, msg):
        if self.verbose_level >= level:
            self.get_logger().info(msg)

    def wait_for_first_frame_log(self):
        if not self.first_callback_seen:
            waited = time.time() - self.wait_start
            self.vlog(1, f"[wait] still waiting for first frame after {waited:.1f}s on {self.args.camera_topic}")
        else:
            try:
                self.wait_timer.cancel()
            except Exception:
                pass

    def ros_callback_store_latest(self, msg):
        self.frame_count += 1
        self.latest_msg = msg
        self.latest_msg_id = self.frame_count

        if not self.first_callback_seen:
            self.first_callback_seen = True
            self.vlog(1, "[callback] first ROS frame received")

        self.vlog(4, f"[callback-store] stored latest raw frame #{self.latest_msg_id}")

    def preprocess_area(self, frame_bgr, area_name):
        t0 = time.time()

        mask = self.area_masks[area_name]

        contours, mask_bin = _extract_mask_contours(mask, frame_bgr.shape[:2])

        cropped, bbox, _ = _crop_with_mask(frame_bgr, mask)
        if cropped is None:
            self.vlog(3, f"[preprocess] {area_name}: crop failed")
            return None, None, None

        resized = _resize_128(cropped, keep_aspect=True, target=(128, 128))
        if resized is None:
            self.vlog(3, f"[preprocess] {area_name}: resize failed")
            return None, None, None

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        input_tensor = self.normalize(input_tensor)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        self.vlog(
            3,
            f"[preprocess] {area_name}: bbox={bbox}, tensor_shape={tuple(input_tensor.shape)}, time={time.time() - t0:.4f}s"
        )

        vis_data = {
            "crop": cropped.copy(),
            "resized": resized.copy(),
            "bbox": bbox,
            "contours": contours,
            "mask_bin": mask_bin,
        }

        return input_tensor, bbox, vis_data

    def infer_area(self, input_tensor, area_name):
        t0 = time.time()

        enc = self.models[area_name]["encoder"]
        dec = self.models[area_name]["decoder"]
        tau = self.thresholds[area_name]

        with torch.no_grad():
            mu, logvar = enc(input_tensor)
            z = mu
            recon = dec(z)

        orig_hwc = tensor_to_hwc_float32(input_tensor.squeeze(0))
        recon_hwc = tensor_to_hwc_float32(recon.squeeze(0))

        score, dist_map = compute_anomaly_score_pair(
            orig_hwc, recon_hwc,
            offset=self.args.offset,
            quantile=self.args.quantile,
        )

        norm_score = score / tau
        is_anom = norm_score > 1.0

        orig_bgr = (orig_hwc[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        recon_bgr = (recon_hwc[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        anom_bgr = colorize_anomaly_map(dist_map)

        self.vlog(
            3,
            f"[infer] {area_name}: score={score:.6f}, tau={tau:.6f}, norm={norm_score:.6f}, time={time.time() - t0:.4f}s"
        )

        return {
            "score": score,
            "threshold": tau,
            "norm_score": norm_score,
            "is_anomalous": is_anom,
            "status": "UNEXPECTED" if is_anom else "normal",
            "orig_patch_bgr": orig_bgr,
            "recon_patch_bgr": recon_bgr,
            "anom_patch_bgr": anom_bgr,
            "dist_map": dist_map,
        }

    def process_latest_frame(self):
        if self.is_processing:
            return
        if self.latest_msg is None:
            return
        if self.latest_msg_id == self.last_processed_msg_id:
            return

        msg = self.latest_msg
        msg_id = self.latest_msg_id

        self.is_processing = True
        process_t0 = time.time()

        try:
            if self.args.frame_stride > 1 and (msg_id % self.args.frame_stride != 0):
                self.last_processed_msg_id = msg_id
                self.vlog(4, f"[process] skipped by frame_stride: raw frame={msg_id}")
                return

            self.vlog(2, f"[process] using latest raw frame #{msg_id}")

            t_convert = time.time()
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if self.processed_count == 0:
                self.process_start_time = time.time()

            self.processed_count += 1

            self.vlog(
                2,
                f"[process] processed_count={self.processed_count}, shape={frame_bgr.shape}, "
                f"convert_time={time.time() - t_convert:.4f}s"
            )

            results = OrderedDict()
            area_inputs = OrderedDict()

            for area in self.areas:
                area_t0 = time.time()
                self.vlog(3, f"[area-start] {area}")

                input_tensor, bbox, vis_data = self.preprocess_area(frame_bgr, area)
                if input_tensor is None:
                    results[area] = {"status": "mask_failed"}
                    self.latest_results[area] = {"status": "mask_failed"}
                    self.vlog(2, f"[area-end] {area}: mask_failed in {time.time() - area_t0:.4f}s")
                    continue

                out = self.infer_area(input_tensor, area)

                if vis_data is not None:
                    vis_data["recon_patch_bgr"] = out.get("recon_patch_bgr")
                    vis_data["anom_patch_bgr"] = out.get("anom_patch_bgr")
                    vis_data["orig_patch_bgr"] = out.get("orig_patch_bgr")
                    area_inputs[area] = vis_data

                out["bbox"] = bbox
                results[area] = out

                self.score_history[area].append(float(out["norm_score"]))
                self.latest_results[area] = out

                self.vlog(
                    2,
                    f"[area-end] {area}: status={out['status']} norm={out['norm_score']:.4f} "
                    f"in {time.time() - area_t0:.4f}s"
                )

            self.last_processed_msg_id = msg_id

            proc_elapsed = time.time() - self.process_start_time if self.process_start_time else 0.0
            avg_fps = self.processed_count / max(proc_elapsed, 1e-6)
            inst_fps = 1.0 / max(time.time() - process_t0, 1e-6)

            if self.processed_count % self.log_every_n == 0 or self.processed_count == 1:
                msg_parts = [
                    f"raw_frame={msg_id}",
                    f"processed={self.processed_count}",
                    f"avg_fps={avg_fps:.2f}",
                    f"inst_fps={inst_fps:.2f}",
                ]
                for area in self.areas:
                    r = results[area]
                    if "score" in r:
                        msg_parts.append(
                            f"{area}: score={r['score']:.5f}, norm={r['norm_score']:.3f}, status={r['status']}"
                        )
                    else:
                        msg_parts.append(f"{area}: status={r['status']}")
                self.vlog(1, " | ".join(msg_parts))

            if self.args.show_model_input:
                dashboard = draw_dashboard_panel(
                    frame_bgr,
                    area_inputs,
                    self.latest_results,
                    frame_id=msg_id,
                    width=self.args.model_input_width,
                    height=self.args.model_input_height,
                )
                cv2.imshow("ADVIS Dashboard", dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.vlog(1, "[gui] ESC pressed, shutting down")
                    rclpy.shutdown()
                    return

            if self.show_timeline:
                panel = draw_timeline_panel(
                    self.score_history,
                    self.latest_results,
                    width=self.timeline_width,
                    height=self.timeline_height,
                    max_points=self.timeline_history_len,
                )
                cv2.imshow("ADVIS Timeline", panel)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.vlog(1, "[gui] ESC pressed, shutting down")
                    rclpy.shutdown()
                    return

            self.vlog(3, f"[process] total time={time.time() - process_t0:.4f}s")

            if self.max_frames is not None and self.processed_count >= self.max_frames:
                self.vlog(1, "[process] reached max_frames, shutting down")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"[process-error] {e}")
        finally:
            self.is_processing = False


def parse_args():
    p = argparse.ArgumentParser("Live ROS anomaly inference")

    p.add_argument("--camera_topic", default="/camera/back_view/image_raw")
    p.add_argument("--safety_area", default="ALL")
    p.add_argument("--area_names", nargs="+", default=["PLeft", "PRight", "RoboArm", "ConvBelt"])
    p.add_argument("--static_mask_paths", nargs="+", required=True)

    p.add_argument("--threshold_dir", required=True)
    p.add_argument("--checkpoints", default="scripts/results/models")
    p.add_argument("--latent_dims", type=int, default=64)

    p.add_argument("--offset", type=int, default=2)
    p.add_argument("--quantile", type=float, default=0.995)
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--dataset_source_name", default="SR")
    p.add_argument("--subgroup_mask", default="MASK")
    p.add_argument("--save_path_type", default="local")
    p.add_argument("--save_figures", action="store_true", default=False)

    p.add_argument("--verbose_level", type=int, default=2)
    p.add_argument("--log_every_n", type=int, default=1)
    p.add_argument("--process_period", type=float, default=0.05)

    p.add_argument("--show_timeline", action="store_true")
    p.add_argument("--timeline_history", type=int, default=200)
    p.add_argument("--timeline_width", type=int, default=1000)
    p.add_argument("--timeline_height", type=int, default=500)

    p.add_argument("--show_model_input", action="store_true")
    p.add_argument("--model_input_width", type=int, default=1600)
    p.add_argument("--model_input_height", type=int, default=1000)

    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = LiveRosAnomalyInfer(args)
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