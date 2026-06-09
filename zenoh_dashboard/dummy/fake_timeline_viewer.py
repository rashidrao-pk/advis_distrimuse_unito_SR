from __future__ import annotations

import argparse
import math
import time
from collections import deque
from typing import Dict, Iterable, List

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


def draw_timeline_panel(score_history, latest_results, areas, width=1000, height=500, max_points=200):
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
    cv2.putText(
        canvas,
        "thr=1.0",
        (left_pad + 8, y_thr - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    for val in [0.0, 0.5, 1.0, 1.5, 2.0]:
        yy = top_pad + int(plot_h * (1.0 - min(val, 2.0) / 2.0))
        cv2.line(canvas, (left_pad - 5, yy), (left_pad, yy), (180, 180, 180), 1)
        cv2.putText(
            canvas,
            f"{val:.1f}",
            (10, yy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "ADVIS Live Anomaly Timeline (normalized scores)",
        (left_pad, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    keys = ordered_area_list([a for a in areas if a in score_history])

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
        f"""
    {{
      mode: "client",
      connect: {{
        endpoints: ["{endpoint}"]
      }}
    }}
    """
    )


def render_from_payload(raw: bytes, width: int, height: int, history: int, areas: List[str]) -> None:
    state = unpack_timeline_state(raw)
    image = draw_timeline_panel(
        state["score_history"],
        state["latest_results"],
        areas=areas,
        width=width,
        height=height,
        max_points=history,
    )
    cv2.imshow("ADVIS Timeline", image)
    cv2.waitKey(1)


class FakeTimelineGenerator:
    def __init__(
        self,
        areas: List[str],
        history: int = 500,
        fps: float = 20.0,
        right_delay_sec: float = 4.0,
        right_unexpected_sec: float = 6.0,
        cycle_sec: float = 14.0,
    ):
        self.areas = ordered_area_list(areas)
        self.history = history
        self.fps = fps
        self.t = 0

        self.right_delay_frames = int(right_delay_sec * fps)
        self.right_unexpected_frames = int(right_unexpected_sec * fps)
        self.cycle_frames = int(cycle_sec * fps)

        self.score_history: Dict[str, deque] = {
            area: deque(maxlen=history) for area in self.areas
        }

        self.latest_results: Dict[str, dict] = {
            area: {"norm_score": 0.0, "status": "NORMAL"} for area in self.areas
        }

        for _ in range(history):
            self.step(initial_fill=True)

    def _small_normal_wave(self, t: int, base: float, amp1: float = 0.04, amp2: float = 0.02, phase: float = 0.0):
        return base + amp1 * math.sin(0.08 * t + phase) + amp2 * math.sin(0.19 * t + 0.7 + phase)

    def _pright_scripted_score(self, t: int) -> float:
        pos = t % max(1, self.cycle_frames)

        base_normal = self._small_normal_wave(t, base=0.20, amp1=0.035, amp2=0.015, phase=0.8)

        if pos < self.right_delay_frames:
            return base_normal

        anomaly_end = self.right_delay_frames + self.right_unexpected_frames
        if pos < anomaly_end:
            local_t = pos - self.right_delay_frames

            ramp_frames = max(8, int(0.6 * self.fps))
            fall_frames = max(8, int(0.8 * self.fps))

            if local_t < ramp_frames:
                ratio = local_t / ramp_frames
                return 0.35 + ratio * 1.05

            if local_t > self.right_unexpected_frames - fall_frames:
                tail_t = local_t - (self.right_unexpected_frames - fall_frames)
                ratio = tail_t / fall_frames
                return 1.35 - ratio * 0.25 + 0.05 * math.sin(0.25 * t)

            return 1.28 + 0.08 * math.sin(0.18 * t) + 0.04 * math.sin(0.47 * t)

        return self._small_normal_wave(t, base=0.18, amp1=0.03, amp2=0.015, phase=1.1)

    def _area_score(self, area: str, t: int) -> float:
        if area == "PLeft":
            return self._small_normal_wave(t, base=0.22, amp1=0.03, amp2=0.015, phase=0.0)

        if area == "PRight":
            return self._pright_scripted_score(t)

        if area == "RoboArm":
            return self._small_normal_wave(t, base=0.30, amp1=0.06, amp2=0.03, phase=2.0) + 0.03 * abs(math.sin(0.22 * t))

        if area == "ConvBelt":
            return self._small_normal_wave(t, base=0.14, amp1=0.02, amp2=0.01, phase=0.5)

        return 0.2

    def step(self, initial_fill: bool = False):
        self.t += 1

        frame_meta = {
            "frame_id": self.t,
            "ts": time.time(),
        }

        for area in self.areas:
            v = self._area_score(area, self.t)
            v = max(0.0, min(2.0, float(v)))

            self.score_history[area].append(v)
            self.latest_results[area] = {
                "norm_score": v,
                "status": "ANOMALY" if v > 1.0 else "NORMAL",
            }

        return {
            "frame_meta": frame_meta,
            "score_history": {k: list(v) for k, v in self.score_history.items()},
            "latest_results": self.latest_results,
        }

    def pack(self) -> bytes:
        state = self.step()
        return msgpack.packb(state, use_bin_type=True)


def run_fake_viewer(
    width: int,
    height: int,
    history: int,
    fps: float,
    areas: List[str],
    right_delay_sec: float,
    right_unexpected_sec: float,
    cycle_sec: float,
):
    gen = FakeTimelineGenerator(
        areas=areas,
        history=history,
        fps=fps,
        right_delay_sec=right_delay_sec,
        right_unexpected_sec=right_unexpected_sec,
        cycle_sec=cycle_sec,
    )

    while True:
        raw = gen.pack()
        render_from_payload(raw, width, height, history, areas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        time.sleep(max(0.001, 1.0 / fps))

    cv2.destroyAllWindows()


def run_zenoh_viewer(endpoint: str, key: str, width: int, height: int, history: int, areas: List[str]):
    zenoh.init_log_from_env_or("error")
    config = make_config(endpoint)

    with zenoh.open(config) as session:
        got_any = False
        for reply in session.get(key):
            if getattr(reply, "ok", None) is None:
                continue
            try:
                render_from_payload(reply.ok.payload.to_bytes(), width, height, history, areas)
                got_any = True
            except Exception as exc:
                print(f"Skipping invalid stored timeline payload: {exc}")

        if not got_any:
            print("No stored timeline state yet.")

        with session.declare_subscriber(key) as subscriber:
            while True:
                try:
                    sample = subscriber.recv()
                    render_from_payload(sample.payload.to_bytes(), width, height, history, areas)
                except Exception as exc:
                    print(f"Timeline render error: {exc}")

                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

                time.sleep(0.001)

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser("Remote ADVIS timeline viewer")
    parser.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh-key", default="advis/vis/timeline/state")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--fake", action="store_true", help="Run with synthetic timeline data")
    parser.add_argument("--fake-fps", type=float, default=20.0, help="Refresh rate for fake mode")
    parser.add_argument(
        "--areas",
        nargs="+",
        default=ALL_SAFETY_AREAS,
        help="List of safety areas to visualize. Example: --areas PLeft PRight",
    )

    parser.add_argument("--right-delay-sec", type=float, default=4.0, help="How long PRight stays normal first")
    parser.add_argument("--right-unexpected-sec", type=float, default=6.0, help="How long PRight stays unexpected")
    parser.add_argument("--cycle-sec", type=float, default=14.0, help="Total cycle length before repeating")

    args = parser.parse_args()

    invalid = [a for a in args.areas if a not in ALL_SAFETY_AREAS]
    if invalid:
        raise ValueError(f"Invalid area names: {invalid}. Valid choices: {ALL_SAFETY_AREAS}")

    selected_areas = ordered_area_list(args.areas)

    if args.fake:
        run_fake_viewer(
            width=args.width,
            height=args.height,
            history=args.history,
            fps=args.fake_fps,
            areas=selected_areas,
            right_delay_sec=args.right_delay_sec,
            right_unexpected_sec=args.right_unexpected_sec,
            cycle_sec=args.cycle_sec,
        )
    else:
        run_zenoh_viewer(
            endpoint=args.zenoh_endpoint,
            key=args.zenoh_key,
            width=args.width,
            height=args.height,
            history=args.history,
            areas=selected_areas,
        )


if __name__ == "__main__":
    main()