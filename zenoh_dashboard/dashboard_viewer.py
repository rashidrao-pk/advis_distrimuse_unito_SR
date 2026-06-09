from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import msgpack
import numpy as np
import zenoh

ALL_SAFETY_AREAS = ["PLeft", "PRight", "RoboArm", "ConvBelt"]
AREA_DISPLAY_NAMES = {
    "PLeft": "Pallet Left",
    "PRight": "Pallet Right",
    "RoboArm": "Robo Arm",
    "ConvBelt": "Conveyor Belt",
}


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
    }


def draw_text_table(panel, results, frame_id=None, corr_frame_id=None, corr_stamp=None):
    h, w = panel.shape[:2]
    panel[:] = (245, 245, 245)

    title_y = 10
    # cv2.putText(panel, "Details", (w // 2 - 50, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    y = 30
    cv2.line(panel, (20, y), (w - 20, y), (40, 40, 40), 2)
    y += 35

    # if frame_id is not None and corr_stamp is not None:
    #     cv2.putText(
    #         panel,
    #         f"Frame: {frame_id} CFID: {corr_frame_id} @ {corr_stamp['sec']}.{corr_stamp['nanosec']}",
    #         (30, y),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.8,
    #         (20, 20, 20),
    #         2,
    #         cv2.LINE_AA,
    #     )
    #     y += 20
    #     cv2.line(panel, (20, y), (w - 20, y), (40, 40, 40), 1)
    #     y += 35
    if frame_id is not None and corr_stamp is not None:
        # Line 1 → Frame ID
        cv2.putText(
            panel,
            f"Frame: {frame_id}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        y += 25

        # Line 2 → camera + timestamp
        cv2.putText(
            panel,
            f"{corr_frame_id} @ {corr_stamp['sec']}.{corr_stamp['nanosec']}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        y += 40

    
    headers = ["Safety Area", "RawVal", "Threshold", "Score", "Status"]
    header_bold = [2,1,1,2,2]
    
    col_x = [30, 200, 300, 440, 540]

    for i, hdr in enumerate(headers):
        cv2.putText(panel, hdr, (col_x[i], y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                     (30, 30, 30), header_bold[i], cv2.LINE_AA)

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
            cv2.putText(
                panel,
                str(val),
                (col_x[i], y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color if i >= 3 else (30, 30, 30),
                header_bold[i],
                cv2.LINE_AA,
            )

        y += 20
        cv2.line(panel, (20, y), (w - 20, y), (120, 120, 120), 1)
        y += 35

    return panel


def draw_dashboard_panel(frame_bgr, area_inputs, latest_results, frame_id=None, width=1600, height=1000, corr_frame_id=None, corr_stamp=None):
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
            if area_name == "PLeft":
                x_text -= 30   # try -30, -40, or -50

            cv2.putText(
                canvas,
                label,
                (x_text, max(20, int(pt[1]) - 8)),
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
            if area_name == "PLeft":
                x_text -= 40

            cv2.putText(
                canvas,
                label,
                (x_text, int(pt[1]) - 5),
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
    details_panel = draw_text_table(details_panel, latest_results, frame_id=frame_id, corr_frame_id=corr_frame_id, corr_stamp=corr_stamp)
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


def render_from_payload(raw: bytes, width: int, height: int) -> None:
    state = unpack_dashboard_state(raw)
    meta = state["frame_meta"]
    image = draw_dashboard_panel(
        state["frame_bgr"],
        state["area_inputs"],
        state["latest_results"],
        frame_id=meta["msg_id"],
        width=width,
        height=height,
        corr_frame_id=meta["corr_frame_id"],
        corr_stamp=meta["stamp"],
    )
    cv2.imshow("ADVIS Dashboard", image)
    cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser("Remote ADVIS dashboard viewer")
    parser.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh-key", default="advis/vis/dashboard/state")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    zenoh.init_log_from_env_or("error")
    config = make_config(args.zenoh_endpoint)

    with zenoh.open(config) as session:
        got_any = False
        for reply in session.get(args.zenoh_key):
            if getattr(reply, "ok", None) is None:
                continue
            try:
                render_from_payload(reply.ok.payload.to_bytes(), args.width, args.height)
                got_any = True
            except Exception as exc:
                print(f"Skipping invalid stored dashboard payload: {exc}")

        if not got_any:
            print("No stored dashboard state yet.")

        with session.declare_subscriber(args.zenoh_key) as subscriber:
            while True:
                try:
                    sample = subscriber.recv()
                    render_from_payload(sample.payload.to_bytes(), args.width, args.height)
                except Exception as exc:
                    print(f"Dashboard render error: {exc}")
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
