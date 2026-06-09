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


def ordered_area_list(areas: Iterable[str]) -> List[str]:
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


def unpack_timeline_state(payload: bytes) -> dict:
    obj = msgpack.unpackb(payload, raw=False)
    return {
        "frame_meta": obj["frame_meta"],
        "score_history": obj["score_history"],
        "latest_results": obj["latest_results"],
    }


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
    cv2.putText(canvas, "thr=1.0", (left_pad + 8, y_thr - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    for val in [0.0, 0.5, 1.0, 1.5, 2.0]:
        yy = top_pad + int(plot_h * (1.0 - min(val, 2.0) / 2.0))
        cv2.line(canvas, (left_pad - 5, yy), (left_pad, yy), (180, 180, 180), 1)
        cv2.putText(canvas, f"{val:.1f}", (10, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(canvas, "ADVIS Live Anomaly Timeline (normalized scores)", (left_pad, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

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
        cv2.putText(canvas, label, (width - 310, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, legend_color, 1, cv2.LINE_AA)

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


def render_from_payload(raw: bytes, width: int, height: int, history: int) -> None:
    state = unpack_timeline_state(raw)
    image = draw_timeline_panel(
        state["score_history"],
        state["latest_results"],
        width=width,
        height=height,
        max_points=history,
    )
    cv2.imshow("ADVIS Timeline", image)
    cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser("Remote ADVIS timeline viewer")
    parser.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh-key", default="advis/vis/timeline/state")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--history", type=int, default=500)
    args = parser.parse_args()

    zenoh.init_log_from_env_or("error")
    config = make_config(args.zenoh_endpoint)

    with zenoh.open(config) as session:
        got_any = False
        for reply in session.get(args.zenoh_key):
            if getattr(reply, "ok", None) is None:
                continue
            try:
                render_from_payload(reply.ok.payload.to_bytes(), args.width, args.height, args.history)
                got_any = True
            except Exception as exc:
                print(f"Skipping invalid stored timeline payload: {exc}")

        if not got_any:
            print("No stored timeline state yet.")

        with session.declare_subscriber(args.zenoh_key) as subscriber:
            while True:
                try:
                    sample = subscriber.recv()
                    render_from_payload(sample.payload.to_bytes(), args.width, args.height, args.history)
                except Exception as exc:
                    print(f"Timeline render error: {exc}")
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
