#!/usr/bin/env python3
"""Live ADVIS inference with optional vertical camera-frame alignment.

This entry point intentionally reuses :mod:`inference_live` and changes only
the decoded frame.  Masks remain fixed in their training-time coordinates.
A positive ``--frame_shift_y`` moves image content down; a negative value
moves it up.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import inference_live as base


def shift_frame_vertical(
    frame: np.ndarray, shift_y: int, fill: str = "replicate",
) -> np.ndarray:
    """Translate a BGR frame vertically without resizing or wrapping it."""
    if frame is None or frame.ndim not in (2, 3) or frame.size == 0:
        raise ValueError("frame must be a non-empty image array")

    shift_y = int(shift_y)
    if shift_y == 0:
        return frame

    height = frame.shape[0]
    if abs(shift_y) >= height:
        raise ValueError(
            f"frame shift {shift_y} must be smaller than frame height {height}"
        )
    if fill not in {"replicate", "black"}:
        raise ValueError("fill must be 'replicate' or 'black'")

    shifted = np.empty_like(frame)
    if shift_y > 0:
        shifted[shift_y:] = frame[:-shift_y]
        if fill == "replicate":
            shifted[:shift_y] = frame[0]
        else:
            shifted[:shift_y] = 0
    else:
        amount = -shift_y
        shifted[:-amount] = frame[amount:]
        if fill == "replicate":
            shifted[-amount:] = frame[-1]
        else:
            shifted[-amount:] = 0
    return shifted


def parse_args(argv=None):
    """Parse alignment flags here and all standard flags in inference_live."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    alignment_parser = argparse.ArgumentParser(add_help=False)
    alignment_parser.add_argument(
        "--frame_shift_y", "--frame-shift-y", type=int, default=0,
        help=(
            "Vertical camera alignment in pixels. Positive moves image content "
            "down; negative moves it up (default: 0)."
        ),
    )
    alignment_parser.add_argument(
        "--frame_shift_fill", "--frame-shift-fill",
        choices=("replicate", "black"), default="replicate",
        help="Fill exposed rows using the edge pixel or black (default: replicate).",
    )
    alignment, remaining = alignment_parser.parse_known_args(raw_args)
    args = base.parse_args(remaining)
    args.frame_shift_y = alignment.frame_shift_y
    args.frame_shift_fill = alignment.frame_shift_fill
    return args


class ShiftedLiveInferenceNode(base.LiveInferenceNode):
    """Apply camera alignment before masks, crops, inference, and publishing."""

    def __init__(self, args):
        super().__init__(args)
        self.log(
            1,
            f"[frame alignment] shift_y={args.frame_shift_y}px "
            f"(positive=down), fill={args.frame_shift_fill}",
        )

    def decode_frame(self, msg):
        frame = super().decode_frame(msg)
        if self.args.frame_shift_y:
            with self.profiler.measure("frame_alignment_shift"):
                frame = shift_frame_vertical(
                    frame,
                    self.args.frame_shift_y,
                    self.args.frame_shift_fill,
                )
        return frame

    def publish_outputs(self, source_msg, frame, results, payload, processing_fps):
        payload["frame_alignment"] = {
            "vertical_shift_pixels": int(self.args.frame_shift_y),
            "direction": (
                "down" if self.args.frame_shift_y > 0
                else "up" if self.args.frame_shift_y < 0
                else "none"
            ),
            "fill": self.args.frame_shift_fill,
        }
        return super().publish_outputs(
            source_msg, frame, results, payload, processing_fps
        )


def main(argv=None):
    args = parse_args(argv)
    if base.rclpy is None:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable. Run this script in the "
            f"Pixi/ROS environment. Original import error: {base.ROS_IMPORT_ERROR}"
        )
    args = base.load_live_settings(args)
    base.rclpy.init(args=None)
    node = None
    try:
        node = ShiftedLiveInferenceNode(args)
        base.rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Stopped by user.")
    finally:
        if node is not None:
            node.profiler.report()
            node.close_external_publishers()
            node.destroy_node()
        if base.rclpy.ok():
            base.rclpy.shutdown()


if __name__ == "__main__":
    main()
