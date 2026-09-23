#!/usr/bin/env python3
"""Render detailed ADVIS inference diagnostics from a separate Zenoh stream."""

from __future__ import annotations

import argparse
import time

import cv2
import msgpack
import numpy as np
import zenoh


ALL_SAFETY_AREAS = ("PLeft", "PRight", "RoboArm", "ConvBelt")
AREA_DISPLAY_NAMES = {
    "PLeft": "Pallet Left",
    "PRight": "Pallet Right",
    "RoboArm": "Robo Arm",
    "ConvBelt": "Conveyor Belt",
}


def ordered_area_list(areas):
    order = {name: index for index, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(areas, key=lambda name: order.get(name, 999))


def unpack_debug_state(payload):
    obj = msgpack.unpackb(payload, raw=False)
    return {
        "schema_version": obj.get("schema_version", 1),
        "frame_meta": obj["frame_meta"],
        "runtime_meta": obj.get("runtime_meta", {}),
        "latest_results": obj.get("latest_results", {}),
    }


def fit_text(text, max_width, font_scale=0.52, thickness=1):
    text = str(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, font_scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    while text and cv2.getTextSize(
        text + suffix, font, font_scale, thickness
    )[0][0] > max_width:
        text = text[:-1]
    return text + suffix


def format_number(value, digits=4):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def draw_labeled_line(
    canvas, label, value, x, y, max_width, label_color=(80, 80, 80),
    value_color=(35, 35, 35),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    cv2.putText(
        canvas, f"{label}:", (x, y), font, scale, label_color, 2, cv2.LINE_AA
    )
    label_width = cv2.getTextSize(f"{label}:", font, scale, 2)[0][0]
    value_x = x + label_width + 10
    cv2.putText(
        canvas, fit_text(value, max_width - label_width - 10, scale, 1),
        (value_x, y), font, scale, value_color, 1, cv2.LINE_AA,
    )


def draw_area_card(canvas, area, result, box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    anomalous = bool(result.get("is_anomalous", False))
    accent = (0, 0, 210) if anomalous else (0, 145, 0)
    background = (249, 249, 249)
    canvas[y1:y2, x1:x2] = background
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (145, 145, 145), 1)
    cv2.rectangle(canvas, (x1, y1), (x1 + 7, y2), accent, -1)

    status = "ANOMALY" if anomalous else "NORMAL"
    cv2.putText(
        canvas, AREA_DISPLAY_NAMES.get(area, area), (x1 + 22, y1 + 31),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, status, (x2 - 145, y1 + 31), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, accent, 2, cv2.LINE_AA,
    )
    cv2.line(canvas, (x1 + 16, y1 + 44), (x2 - 16, y1 + 44), (190, 190, 190), 1)

    score = result.get("anomaly_score", result.get("score"))
    threshold = result.get("threshold")
    normalized = result.get("normalized_score", result.get("norm_score"))
    y = y1 + 69
    draw_labeled_line(
        canvas, "Detection",
        f"score={format_number(score)} | tau={format_number(threshold)} | "
        f"normalized={format_number(normalized, 3)}x",
        x1 + 20, y, width - 40, value_color=accent,
    )
    y += 27

    calibration_variant = result.get(
        "calibration_taas_variant", result.get("taas_variant", "legacy-unspecified")
    )
    calibration_score = result.get(
        "calibration_score_func", result.get("score_func", "unknown")
    )
    draw_labeled_line(
        canvas, "Calibration",
        f"variant={calibration_variant} | function={calibration_score}",
        x1 + 20, y, width - 40, label_color=(0, 115, 190),
    )
    y += 27

    inference_variant = result.get(
        "inference_taas_variant", result.get("taas_variant", "legacy-unspecified")
    )
    inference_score = result.get(
        "inference_score_func", result.get("score_func", "unknown")
    )
    draw_labeled_line(
        canvas, "Inference",
        f"variant={inference_variant} | function={inference_score}",
        x1 + 20, y, width - 40, label_color=(175, 75, 0),
    )
    y += 27

    draw_labeled_line(
        canvas, "Backend",
        f"{result.get('inference_score_backend', 'legacy-unspecified')} | "
        f"reconstruction={result.get('reconstruction_mode', 'unknown')}",
        x1 + 20, y, width - 40, label_color=(175, 75, 0),
    )
    y += 27

    draw_labeled_line(
        canvas, "Threshold",
        f"strategy={result.get('threshold_strategy', 'unknown')} | "
        f"offset={result.get('offset', '-')} | sigma={result.get('sigma', '-')} | "
        f"q={result.get('quantile', '-')}",
        x1 + 20, y, width - 40, label_color=(0, 115, 190),
    )
    y += 27

    rolling_policy = result.get("rolling_policy", "none")
    rolling_window = result.get("rolling_window", 1)
    rolling_count = result.get("rolling_count", 1)
    instantaneous = result.get("instantaneous_normalized_score", normalized)
    draw_labeled_line(
        canvas, "Temporal",
        f"policy={rolling_policy} | window={rolling_window} | available={rolling_count} | "
        f"instantaneous={format_number(instantaneous, 3)}x",
        x1 + 20, y, width - 40, label_color=(120, 60, 120),
    )


def draw_debug_panel(state, width=1500, height=900):
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    frame_meta = state.get("frame_meta", {})
    stamp = frame_meta.get("stamp", {})
    runtime = state.get("runtime_meta", {})
    results = state.get("latest_results", {})

    cv2.putText(
        canvas, "ADVIS Live Inference Debug", (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (25, 25, 25), 2, cv2.LINE_AA,
    )
    frame_text = (
        f"Frame {frame_meta.get('msg_id', '-')} | "
        f"{frame_meta.get('corr_frame_id', '-')} | "
        f"{stamp.get('sec', 0)}.{int(stamp.get('nanosec', 0)):09d}"
    )
    cv2.putText(
        canvas, frame_text, (28, 72), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (65, 65, 65), 1, cv2.LINE_AA,
    )
    runtime_text = (
        f"Runtime {format_number(runtime.get('processing_fps'), 2)} fps | "
        f"Source {format_number(runtime.get('source_fps'), 2)} fps | "
        f"Rx {runtime.get('received_frames', '-')} | "
        f"Done {runtime.get('processed_frames', '-')} | "
        f"Dropped {runtime.get('dropped_frames', '-')}"
    )
    cv2.putText(
        canvas, fit_text(runtime_text, width // 2 - 35, 0.58, 1),
        (width // 2, 42), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (40, 110, 40), 1, cv2.LINE_AA,
    )
    cv2.line(canvas, (24, 88), (width - 24, 88), (100, 100, 100), 1)

    areas = ordered_area_list(results.keys())
    if not areas:
        cv2.putText(
            canvas, "Waiting for inference results...", (40, 145),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1, cv2.LINE_AA,
        )
        return canvas

    pad = 24
    gap = 18
    top = 108
    columns = 2 if len(areas) > 1 else 1
    rows = (len(areas) + columns - 1) // columns
    card_width = (width - 2 * pad - gap * (columns - 1)) // columns
    card_height = (height - top - pad - gap * (rows - 1)) // rows
    for index, area in enumerate(areas):
        row, column = divmod(index, columns)
        x1 = pad + column * (card_width + gap)
        y1 = top + row * (card_height + gap)
        draw_area_card(
            canvas, area, results[area],
            (x1, y1, x1 + card_width, y1 + card_height),
        )
    return canvas


def make_config(endpoint):
    return zenoh.Config.from_json5(
        '{mode:"client",connect:{endpoints:["' + endpoint + '"]}}'
    )


def render_from_payload(raw, width, height):
    image = draw_debug_panel(unpack_debug_state(raw), width, height)
    cv2.imshow("ADVIS Debug", image)
    cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser("Remote ADVIS technical debug viewer")
    parser.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh-key", default="advis/vis/debug/state")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    zenoh.init_log_from_env_or("error")
    with zenoh.open(make_config(args.zenoh_endpoint)) as session:
        got_any = False
        for reply in session.get(args.zenoh_key):
            if getattr(reply, "ok", None) is None:
                continue
            try:
                render_from_payload(reply.ok.payload.to_bytes(), args.width, args.height)
                got_any = True
            except Exception as exc:
                print(f"Skipping invalid stored debug payload: {exc}")
        if not got_any:
            print("No stored debug state yet.")

        with session.declare_subscriber(args.zenoh_key) as subscriber:
            while True:
                try:
                    sample = subscriber.recv()
                    render_from_payload(sample.payload.to_bytes(), args.width, args.height)
                except Exception as exc:
                    print(f"Debug render error: {exc}")
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                time.sleep(0.001)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
