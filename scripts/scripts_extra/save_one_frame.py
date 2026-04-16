import os
import argparse

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage, CompressedImage
from cv_bridge import CvBridge


class SaveOneFrameNode(Node):
    def __init__(self, args):
        super().__init__("save_one_frame_node")
        self.args = args
        self.bridge = CvBridge()
        self.saved = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

        if args.use_compressed:
            self.sub = self.create_subscription(
                CompressedImage,
                args.camera_topic,
                self.cb_compressed,
                qos,
            )
            self.get_logger().info(f"Subscribed to compressed topic: {args.camera_topic}")
        else:
            self.sub = self.create_subscription(
                RosImage,
                args.camera_topic,
                self.cb_raw,
                qos,
            )
            self.get_logger().info(f"Subscribed to raw topic: {args.camera_topic}")

    def save_and_quit(self, frame_bgr):
        ok = cv2.imwrite(self.args.output_path, frame_bgr)
        if not ok:
            self.get_logger().error(f"Failed to save frame to: {self.args.output_path}")
        else:
            self.get_logger().info(f"Saved frame to: {self.args.output_path}")
        self.saved = True
        rclpy.shutdown()

    def cb_compressed(self, msg):
        if self.saved:
            return
        try:
            frame_bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                self.get_logger().error("Failed to decode compressed frame")
                return
            self.save_and_quit(frame_bgr)
        except Exception as e:
            self.get_logger().error(f"Compressed callback error: {e}")

    def cb_raw(self, msg):
        if self.saved:
            return
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.save_and_quit(frame_bgr)
        except Exception as e:
            self.get_logger().error(f"Raw callback error: {e}")


def parse_args():
    p = argparse.ArgumentParser("Save one frame from a ROS camera topic")
    p.add_argument("--camera_topic", default="/camera/back_view/image_raw")
    p.add_argument("--output_path", default="back_frame.jpg")
    p.add_argument("--use_compressed", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SaveOneFrameNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()