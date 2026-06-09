from __future__ import annotations

import argparse
import random
import time
from collections import OrderedDict
from typing import Iterable, List, Tuple

import cv2
import msgpack
import numpy as np
import zenoh

ALL_SAFETY_AREAS = ["PLeft", "PRight", "RoboArm", "ConvBelt"]


def ordered_area_list(areas: Iterable[str]) -> List[str]:
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


def encode_image(image: np.ndarray, ext: str = ".jpg", params=None) -> bytes:
    if image is None:
        raise ValueError("image cannot be None")
    ok, encoded = cv2.imencode(ext, image, params or [])
    if not ok:
        raise ValueError(f"cv2.imencode failed for {ext}")
    return encoded.tobytes()


def serializable_results(results) -> OrderedDict:
    cleaned = OrderedDict()
    for area in ordered_area_list(results.keys()):
        src = results[area]
        cleaned[area] = {
            "score": float(src.get("score", 0.0)),
            "threshold": float(src.get("threshold", 1.0)),
            "norm_score": float(src.get("norm_score", 0.0)),
            "is_anomalous": bool(src.get("is_anomalous", False)),
            "status": str(src.get("status", "unknown")),
        }
    return cleaned


def frame_meta(msg_id: int) -> dict:
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1_000_000_000)
    return {
        "msg_id": int(msg_id),
        "corr_frame_id": "fake_dashboard",
        "stamp": {"sec": sec, "nanosec": nanosec},
    }


def pack_dashboard_state(*, msg_id: int, frame_bgr: np.ndarray, area_inputs, latest_results, jpeg_quality: int = 85) -> bytes:
    payload = {
        "frame_meta": frame_meta(msg_id),
        "frame_bgr_jpg": encode_image(frame_bgr, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]),
        "latest_results": serializable_results(latest_results),
        "area_inputs": OrderedDict(),
    }

    for area in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area]
        payload["area_inputs"][area] = {
            "bbox": list(info["bbox"]) if info.get("bbox") is not None else None,
            "resize_meta": info.get("resize_meta"),
            "mask_png": encode_image(info["mask_bin"], ".png"),
            "orig_patch_jpg": encode_image(info["orig_patch_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]),
            "recon_patch_jpg": encode_image(info["recon_patch_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]),
            "anom_patch_jpg": encode_image(info["anom_patch_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]),
        }

    return msgpack.packb(payload, use_bin_type=True)


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


def random_color(rng: random.Random) -> Tuple[int, int, int]:
    return (
        rng.randint(0, 255),
        rng.randint(0, 255),
        rng.randint(0, 255),
    )


def draw_random_rectangles(canvas: np.ndarray, rng: random.Random, count: int, fill: bool = True) -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape[:2]
    for _ in range(count):
        x1 = rng.randint(0, max(0, w - 20))
        y1 = rng.randint(0, max(0, h - 20))
        x2 = rng.randint(x1 + 10, min(w - 1, x1 + max(20, w // 3)))
        y2 = rng.randint(y1 + 10, min(h - 1, y1 + max(20, h // 3)))
        thickness = -1 if fill and rng.random() < 0.8 else rng.randint(1, 4)
        cv2.rectangle(out, (x1, y1), (x2, y2), random_color(rng), thickness)
    return out


def make_frame(width: int, height: int, rng: random.Random) -> np.ndarray:
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    frame = draw_random_rectangles(frame, rng, count=12, fill=True)
    return frame


def choose_bbox(frame_w: int, frame_h: int, area_index: int, rng: random.Random) -> Tuple[int, int, int, int]:
    # Keep the areas spread around the frame so contours and labels are easy to see.
    centers = [
        (frame_w * 0.25, frame_h * 0.30),
        (frame_w * 0.75, frame_h * 0.30),
        (frame_w * 0.35, frame_h * 0.72),
        (frame_w * 0.72, frame_h * 0.72),
    ]
    cx_base, cy_base = centers[area_index % len(centers)]
    bw = rng.randint(max(80, frame_w // 8), max(120, frame_w // 4))
    bh = rng.randint(max(80, frame_h // 8), max(120, frame_h // 4))
    cx = int(cx_base + rng.randint(-70, 70))
    cy = int(cy_base + rng.randint(-50, 50))
    x1 = max(0, min(frame_w - bw - 1, cx - bw // 2))
    y1 = max(0, min(frame_h - bh - 1, cy - bh // 2))
    x2 = min(frame_w - 1, x1 + bw)
    y2 = min(frame_h - 1, y1 + bh)
    return x1, y1, x2, y2


def make_mask(frame_w: int, frame_h: int, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def make_recon_patch(orig_patch: np.ndarray, rng: random.Random) -> np.ndarray:
    recon = np.full_like(orig_patch, 255)
    recon = draw_random_rectangles(recon, rng, count=max(2, (orig_patch.shape[0] * orig_patch.shape[1]) // 15000), fill=True)
    return recon


def make_anom_patch(orig_patch: np.ndarray, rng: random.Random, anomalous: bool) -> np.ndarray:
    anom = np.full_like(orig_patch, 255)
    rect_count = 4 if anomalous else 2
    anom = draw_random_rectangles(anom, rng, count=rect_count, fill=True)
    if anomalous:
        h, w = anom.shape[:2]
        cv2.rectangle(anom, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    return anom


def next_norm_value(rng: random.Random, current: float) -> float:
    value = current + rng.uniform(-0.12, 0.12)
    if rng.random() < 0.12:
        value += rng.uniform(0.35, 0.9)
    if value > 1.4:
        value -= rng.uniform(0.05, 0.25)
    return max(0.0, min(2.0, value))


def build_state(frame_w: int, frame_h: int, rng: random.Random, current_values) -> Tuple[np.ndarray, OrderedDict, OrderedDict]:
    frame = make_frame(frame_w, frame_h, rng)
    area_inputs = OrderedDict()
    latest_results = OrderedDict()

    for area_index, area in enumerate(ordered_area_list(ALL_SAFETY_AREAS)):
        norm = next_norm_value(rng, current_values[area])
        current_values[area] = norm
        anomalous = norm > 1.0
        score = norm * rng.uniform(0.85, 1.15)
        threshold = max(0.4, score / max(norm, 0.01))

        bbox = choose_bbox(frame_w, frame_h, area_index, rng)
        mask = make_mask(frame_w, frame_h, bbox)
        x1, y1, x2, y2 = bbox

        # Paint a visible area rectangle on the full frame.
        cv2.rectangle(frame, (x1, y1), (x2, y2), random_color(rng), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        orig_patch = frame[y1:y2 + 1, x1:x2 + 1].copy()
        recon_patch = make_recon_patch(orig_patch, rng)
        anom_patch = make_anom_patch(orig_patch, rng, anomalous)

        area_inputs[area] = {
            "bbox": bbox,
            "resize_meta": None,
            "mask_bin": mask,
            "orig_patch_bgr": orig_patch,
            "recon_patch_bgr": recon_patch,
            "anom_patch_bgr": anom_patch,
        }
        latest_results[area] = {
            "score": float(score),
            "threshold": float(threshold),
            "norm_score": float(norm),
            "is_anomalous": anomalous,
            "status": "UNEXPECTED" if anomalous else "normal",
        }

    return frame, area_inputs, latest_results


def main() -> None:
    parser = argparse.ArgumentParser("Fake ADVIS dashboard publisher")
    parser.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh-key", default="advis/vis/dashboard/state")
    parser.add_argument("--rate", type=float, default=2.0, help="Publish rate in Hz")
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--zenoh-log-level", default="error")
    args = parser.parse_args()

    period = 1.0 / max(args.rate, 0.1)
    rng = random.Random(args.seed)
    current_values = OrderedDict((area, rng.uniform(0.2, 0.9)) for area in ordered_area_list(ALL_SAFETY_AREAS))

    zenoh.init_log_from_env_or(args.zenoh_log_level)
    config = make_config(args.zenoh_endpoint)

    with zenoh.open(config) as session:
        pub = session.declare_publisher(
            args.zenoh_key,
            encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        )

        msg_id = 0
        print(f"Publishing fake dashboard data to {args.zenoh_key} via {args.zenoh_endpoint}")
        try:
            while True:
                msg_id += 1
                frame_bgr, area_inputs, latest_results = build_state(
                    args.frame_width,
                    args.frame_height,
                    rng,
                    current_values,
                )
                payload = pack_dashboard_state(
                    msg_id=msg_id,
                    frame_bgr=frame_bgr,
                    area_inputs=area_inputs,
                    latest_results=latest_results,
                    jpeg_quality=args.jpeg_quality,
                )
                pub.put(payload)

                summary = " | ".join(
                    f"{area}={latest_results[area]['norm_score']:.2f}"
                    for area in ordered_area_list(latest_results.keys())
                )
                print(f"published #{msg_id}: {summary}")
                time.sleep(period)
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            try:
                pub.undeclare()
            except Exception:
                pass


if __name__ == "__main__":
    main()
