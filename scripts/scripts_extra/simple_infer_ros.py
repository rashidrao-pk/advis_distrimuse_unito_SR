#!/usr/bin/env python3
"""
Minimal ROS2 "inference" node for quick end-to-end testing.

What it does:
- subscribes to a camera topic
- receives frames
- performs a trivial placeholder "inference"
- publishes RulexDetectionResult on a ROS2 topic

This is intended only to verify that:
1) camera subscription works
2) custom ROS2 messages are available
3) publishing back to ROS works
"""

import argparse
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage

from distrimuse_ros2_api.msg import RulexAreaScore, RulexDetectionResult


AREA_NAME_TO_ENUM = {
    "RoboArm": RulexAreaScore.AREA_A,
    "ConvBelt": RulexAreaScore.AREA_B,
    "PLeft": RulexAreaScore.AREA_C,
    "PRight": RulexAreaScore.AREA_D,
}

class SimpleRosInfer(Node):
    def __init__(self, args):
        super().__init__("simple_ros_infer")

        self.args = args
        self.bridge = CvBridge()
        self.frame_count = 0
        self.start_time = time.time()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            RosImage,
            args.camera_topic,
            self.image_callback,
            sensor_qos,
        )

        self.pub = self.create_publisher(
            RulexDetectionResult,
            args.rulex_topic,
            10,
        )

        self.get_logger().info(f"Subscribed to: {args.camera_topic}")
        self.get_logger().info(f"Publishing to: {args.rulex_topic}")
        self.get_logger().info(f"Areas: {args.area_names}")

    def fake_inference(self, frame_bgr: np.ndarray):
        """
        Very simple placeholder inference:
        - converts to grayscale
        - computes mean brightness
        - marks anomaly if mean brightness is below threshold
          or if toggle mode is enabled, alternates every N frames
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))

        results = {}

        for area in self.args.area_names:
            if self.args.mode == "threshold":
                is_anom = mean_val < self.args.brightness_threshold
            elif self.args.mode == "toggle":
                is_anom = ((self.frame_count // max(1, self.args.toggle_every)) % 2) == 1
            else:
                is_anom = False

            results[area] = {
                "anomaly": bool(is_anom),
                "score": mean_val,
            }

        return results

    def publish_result(self, results, frame_bgr: np.ndarray):
        msg = RulexDetectionResult()
        area_scores = []
        any_anomaly = False

        for area_name in self.args.area_names:
            area_msg = RulexAreaScore()
            area_msg.area = AREA_NAME_TO_ENUM.get(area_name, RulexAreaScore.AREA_A)
            area_msg.anomaly = bool(results[area_name]["anomaly"])

            if area_msg.anomaly:
                any_anomaly = True

            area_scores.append(area_msg)

        msg.area_scores = area_scores

        if self.args.attach_image_on_anomaly and any_anomaly:
            msg.image = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")

        self.pub.publish(msg)

    def image_callback(self, msg: RosImage):
        self.frame_count += 1

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        if self.args.frame_stride > 1 and (self.frame_count % self.args.frame_stride != 0):
            return

        results = self.fake_inference(frame_bgr)
        self.publish_result(results, frame_bgr)

        elapsed = max(time.time() - self.start_time, 1e-6)
        fps = self.frame_count / elapsed

        summary = []
        for area_name in self.args.area_names:
            summary.append(f"{area_name}={results[area_name]['anomaly']}")

        self.get_logger().info(
            f"frame={self.frame_count} fps={fps:.2f} mean_brightness={results[self.args.area_names[0]]['score']:.2f} "
            + " ".join(summary)
        )

        if self.args.show:
            vis = frame_bgr.copy()
            y = 30
            for area_name in self.args.area_names:
                text = f"{area_name}: {'UNEXPECTED' if results[area_name]['anomaly'] else 'normal'}"
                cv2.putText(
                    vis,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if not results[area_name]['anomaly'] else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 30

            cv2.imshow("simple_ros_infer", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.get_logger().info("ESC pressed, shutting down.")
                rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple ROS2 inference and Rulex publisher")
    parser.add_argument("--camera_topic", default="/camera/back_view/image_raw")
    parser.add_argument("--rulex_topic", default="/rulex/detection_result")
    parser.add_argument("--area_names", nargs="+", default=["RoboArm", "ConvBelt", "PLeft", "PRight"])
    parser.add_argument("--frame_stride", type=int, default=1)

    parser.add_argument(
        "--mode",
        choices=["threshold", "toggle", "always_normal"],
        default="toggle",
        help="Dummy inference mode",
    )
    parser.add_argument(
        "--brightness_threshold",
        type=float,
        default=90.0,
        help="Used only in threshold mode",
    )
    parser.add_argument(
        "--toggle_every",
        type=int,
        default=10,
        help="Toggle anomaly state every N processed frames in toggle mode",
    )
    parser.add_argument("--attach_image_on_anomaly", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SimpleRosInfer(args)

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
