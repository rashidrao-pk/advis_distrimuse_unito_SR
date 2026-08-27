#!/usr/bin/env python3
"""Publish frames from a local camera or video file as ROS 2 Image messages."""

import argparse
from pathlib import Path
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def parse_source(value: str):
    """Interpret an integer as a camera index and anything else as a file path."""
    try:
        return int(value)
    except ValueError:
        return value


class LocalFrameBroadcaster(Node):
    def __init__(self, args):
        super().__init__("local_frame_broadcaster")
        self.args = args
        self.bridge = CvBridge()
        self.frame_count = 0

        source = parse_source(args.source)
        if isinstance(source, str) and not Path(source).is_file():
            raise FileNotFoundError(f"Video source does not exist: {source}")

        # AVFoundation is the native camera backend on macOS. Video files work
        # better with OpenCV's automatic backend selection.
        backend = cv2.CAP_AVFOUNDATION if isinstance(source, int) else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(source, backend)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open source {args.source!r}. "
                "For a camera, allow terminal/Pixi camera access in macOS settings."
            )

        if args.width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(Image, args.topic, qos)
        self.timer = self.create_timer(1.0 / args.fps, self.publish_frame)
        self.get_logger().info(
            f"Publishing {args.source!r} at {args.fps:g} FPS on {args.topic}"
        )

    def publish_frame(self):
        ok, frame = self.capture.read()
        if not ok and self.args.loop and not isinstance(parse_source(self.args.source), int):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()

        if not ok:
            self.get_logger().info("Source ended; shutting down.")
            rclpy.shutdown()
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        self.publisher.publish(msg)
        self.frame_count += 1

        if self.args.max_frames and self.frame_count >= self.args.max_frames:
            self.get_logger().info(f"Published {self.frame_count} frames; shutting down.")
            rclpy.shutdown()

    def close(self):
        self.capture.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Broadcast a local Mac camera or video file on a ROS 2 image topic."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (default: 0) or path to a video file.",
    )
    parser.add_argument("--topic", default="/camera/back_view/image_raw")
    parser.add_argument("--frame-id", default="local_camera")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--loop", action="store_true", help="Loop a video file.")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = LocalFrameBroadcaster(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Give AVFoundation a moment to release the camera before process exit.
        time.sleep(0.05)


if __name__ == "__main__":
    main()
