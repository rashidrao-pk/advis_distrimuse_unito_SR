from __future__ import annotations

import argparse
import time
from typing import Iterable, List

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
AREA_COLORS = {
    "PLeft": (255, 190, 40),
    "PRight": (60, 210, 60),
    "RoboArm": (220, 90, 220),
    "ConvBelt": (40, 200, 255),
}


def ordered_area_list(areas: Iterable[str]) -> List[str]:
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


def unpack_timeline_state(payload: bytes) -> dict:
    obj = msgpack.unpackb(payload, raw=False)
    return {
        "frame_meta": obj["frame_meta"],
        "score_history": obj["score_history"],
        "latest_results": obj["latest_results"],
        # These fields are absent in legacy stored messages.
        "history_meta": obj.get("history_meta", []),
        "source_meta": obj.get("source_meta", {}),
        "runtime_meta": obj.get("runtime_meta", {}),
    }


def _axis_values(history_meta, count, requested_mode):
    """Return aligned x values, effective mode, and a human-readable label."""
    samples = list(history_meta or [])[-count:]
    keys = {
        "elapsed": ("ros_elapsed_seconds", "ROS/rosbag elapsed time [s]"),
        "frame": ("source_message_index", "Received source message / frame number"),
        "ros-time": ("ros_timestamp", "ROS timestamp"),
    }
    key, label = keys[requested_mode]
    if len(samples) == count and all(sample.get(key) is not None for sample in samples):
        return np.asarray([float(sample[key]) for sample in samples]), requested_mode, label
    return np.arange(1, count + 1, dtype=np.float64), "sample", "Recent processed sample"


def _x_tick_label(value, mode):
    if mode in {"frame", "sample"}:
        return str(int(round(value)))
    if mode == "ros-time":
        whole = int(value)
        tenths = int(round((value - whole) * 10.0)) % 10
        return time.strftime("%H:%M:%S", time.localtime(whole)) + f".{tenths}"
    return f"{value:.1f}"


def draw_timeline_panel(
    score_history, latest_results, width=1200, height=600, max_points=200,
    history_meta=None, source_meta=None, runtime_meta=None, frame_meta=None,
    x_axis="elapsed",
):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    source_meta = source_meta or {}
    runtime_meta = runtime_meta or {}
    frame_meta = frame_meta or {}
    stamp = frame_meta.get("stamp", {})
    source_name = source_meta.get("source_name") or "legacy/unspecified"
    scenario_id = source_meta.get("scenario_id") or "-"
    camera = frame_meta.get("corr_frame_id") or source_meta.get("camera_topic", "-")
    current_sample = history_meta[-1] if history_meta else {}

    cv2.putText(
        canvas, "ADVIS Live Anomaly Timeline", (25, 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA,
    )
    context = (
        f"Source: {source_name} | Scenario: {scenario_id} | Camera: {camera} | "
        f"Processed frame: {current_sample.get('processed_frame', '-')} | "
        f"Source frame: {current_sample.get('source_message_index', frame_meta.get('msg_id', '-'))}"
    )
    cv2.putText(
        canvas, context[:170], (25, 55), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (195, 195, 195), 1, cv2.LINE_AA,
    )
    runtime_text = (
        f"ROS stamp: {stamp.get('sec', 0)}.{int(stamp.get('nanosec', 0)):09d} | "
        f"Inference: {float(runtime_meta.get('processing_fps', 0.0)):.2f} fps | "
        f"Source: {float(runtime_meta.get('source_fps', 0.0)):.2f} fps | "
        f"Received: {runtime_meta.get('received_frames', '-')} | "
        f"Processed: {runtime_meta.get('processed_frames', '-')} | "
        f"Dropped: {runtime_meta.get('dropped_frames', '-')}"
    )
    cv2.putText(
        canvas, runtime_text[:180], (25, 78), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (130, 200, 130), 1, cv2.LINE_AA,
    )

    left_pad = 78
    right_pad = 275
    top_pad = 100
    bottom_pad = 65

    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    cv2.rectangle(canvas, (left_pad, top_pad), (left_pad + plot_w, top_pad + plot_h), (80, 80, 80), 1)

    all_recent = [
        float(value)
        for values in score_history.values()
        for value in list(values)[-max_points:]
    ]
    y_max = max(2.0, max(all_recent, default=0.0) * 1.10)
    y_ticks = np.linspace(0.0, y_max, 5)
    for val in y_ticks:
        yy = top_pad + int(plot_h * (1.0 - float(val) / y_max))
        cv2.line(canvas, (left_pad, yy), (left_pad + plot_w, yy), (48, 48, 48), 1)
        cv2.line(canvas, (left_pad - 5, yy), (left_pad, yy), (180, 180, 180), 1)
        cv2.putText(canvas, f"{val:.2f}", (12, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    y_thr = top_pad + int(plot_h * (1.0 - min(1.0, y_max) / y_max))
    for x in range(left_pad, left_pad + plot_w, 14):
        cv2.line(canvas, (x, y_thr), (min(x + 7, left_pad + plot_w), y_thr), (0, 0, 235), 1)
    cv2.putText(canvas, "threshold = 1.0", (left_pad + 8, y_thr - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 235), 1, cv2.LINE_AA)

    keys = ordered_area_list(score_history.keys())
    count = min(max_points, max((len(score_history[key]) for key in keys), default=0))
    x_values, effective_axis, x_label = _axis_values(history_meta, count, x_axis)
    if count:
        x_min, x_max = float(x_values[0]), float(x_values[-1])
    else:
        x_min, x_max = 0.0, 1.0
    if np.isclose(x_min, x_max):
        x_max = x_min + 1.0

    for tick in np.linspace(x_min, x_max, 6):
        xx = left_pad + int((tick - x_min) / (x_max - x_min) * plot_w)
        cv2.line(canvas, (xx, top_pad + plot_h), (xx, top_pad + plot_h + 5), (180, 180, 180), 1)
        label = _x_tick_label(tick, effective_axis)
        label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        cv2.putText(canvas, label, (xx - label_w // 2, top_pad + plot_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    axis_width = cv2.getTextSize(x_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(canvas, x_label, (left_pad + (plot_w - axis_width) // 2, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 205, 205), 1, cv2.LINE_AA)

    for idx, area_name in enumerate(keys):
        vals = list(score_history[area_name])[-count:]
        area_color = AREA_COLORS.get(area_name, (235, 235, 235))

        if len(vals) >= 2:
            pts = []
            area_x_values = x_values[-len(vals):]
            for x_value, v in zip(area_x_values, vals):
                x = left_pad + int((float(x_value) - x_min) / (x_max - x_min) * plot_w)
                v_clip = max(0.0, min(y_max, float(v)))
                y = top_pad + int(plot_h * (1.0 - v_clip / y_max))
                pts.append((x, y, float(v)))

            for i in range(1, len(pts)):
                p0 = pts[i - 1]
                p1 = pts[i]
                seg_color = (
                    (0, 0, 255)
                    if (p0[2] > 1.0 or p1[2] > 1.0)
                    else area_color
                )
                cv2.line(canvas, (p0[0], p0[1]), (p1[0], p1[1]), seg_color, 2)

        latest = latest_results.get(area_name, {})
        latest_norm = float(latest.get("norm_score", 0.0)) if "norm_score" in latest else 0.0
        legend_color = (0, 0, 255) if latest_norm > 1.0 else area_color

        label = AREA_DISPLAY_NAMES.get(area_name, area_name)
        if "norm_score" in latest:
            label += f"  {latest['norm_score']:.3f}"
        status = str(latest.get("status", "unknown"))

        legend_x = left_pad + plot_w + 20
        legend_y = top_pad + 24 + 52 * idx
        cv2.line(canvas, (legend_x, legend_y - 5), (legend_x + 35, legend_y - 5), legend_color, 3)
        cv2.putText(canvas, label, (legend_x + 45, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, legend_color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas, f"[{status}]", (legend_x + 45, legend_y + 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.41, legend_color, 1, cv2.LINE_AA,
        )

    if count:
        current_x = left_pad + plot_w
        cv2.line(canvas, (current_x, top_pad), (current_x, top_pad + plot_h), (0, 215, 255), 2)
        triangle = np.asarray([
            [current_x - 7, top_pad + 2],
            [current_x + 7, top_pad + 2],
            [current_x, top_pad + 13],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [triangle], (0, 215, 255))
        cv2.putText(
            canvas, "current", (current_x - 54, top_pad + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 255), 1, cv2.LINE_AA,
        )

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


def render_from_payload(raw: bytes, width: int, height: int, history: int, x_axis: str) -> None:
    state = unpack_timeline_state(raw)
    image = draw_timeline_panel(
        state["score_history"],
        state["latest_results"],
        width=width,
        height=height,
        max_points=history,
        history_meta=state["history_meta"],
        source_meta=state["source_meta"],
        runtime_meta=state["runtime_meta"],
        frame_meta=state["frame_meta"],
        x_axis=x_axis,
    )
    cv2.imshow("ADVIS Timeline", image)
    cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser("Remote ADVIS timeline viewer")
    parser.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh-key", default="advis/vis/timeline/state")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument(
        "--x-axis", choices=("elapsed", "frame", "ros-time"), default="elapsed",
        help="Timeline x-axis (default: elapsed ROS/rosbag time).",
    )
    args = parser.parse_args()

    zenoh.init_log_from_env_or("error")
    config = make_config(args.zenoh_endpoint)

    with zenoh.open(config) as session:
        got_any = False
        for reply in session.get(args.zenoh_key):
            if getattr(reply, "ok", None) is None:
                continue
            try:
                render_from_payload(
                    reply.ok.payload.to_bytes(), args.width, args.height,
                    args.history, args.x_axis,
                )
                got_any = True
            except Exception as exc:
                print(f"Skipping invalid stored timeline payload: {exc}")

        if not got_any:
            print("No stored timeline state yet.")

        with session.declare_subscriber(args.zenoh_key) as subscriber:
            while True:
                try:
                    sample = subscriber.recv()
                    render_from_payload(
                        sample.payload.to_bytes(), args.width, args.height,
                        args.history, args.x_axis,
                    )
                except Exception as exc:
                    print(f"Timeline render error: {exc}")
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
