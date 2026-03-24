import os
from datetime import datetime

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class FrameSaver(Node):
    def __init__(self):
        super().__init__('frame_saver')

        self.declare_parameter('save_dir', '/home/unito/data/saved_frames')
        self.declare_parameter('topics', ['/camera/front_view/image_raw'])
        self.declare_parameter('save_every_n', 1)
        self.declare_parameter('image_format', 'png')

        self.save_dir = self.get_parameter('save_dir').value
        self.topics = list(self.get_parameter('topics').value)
        self.save_every_n = int(self.get_parameter('save_every_n').value)
        self.image_format = str(self.get_parameter('image_format').value).lower()

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.frame_counts = {topic: 0 for topic in self.topics}
        self.saved_counts = {topic: 0 for topic in self.topics}

        self._subs = []
        for topic in self.topics:
            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, t=topic: self.listener_callback(msg, t),
                10
            )
            self._subs.append(sub)
            self.get_logger().info(f"Subscribed to {topic}")

        self.get_logger().info(f"Saving frames to: {self.save_dir}")
        self.get_logger().info(f"Save every N frames: {self.save_every_n}")
        self.get_logger().info(f"Image format: {self.image_format}")

    def listener_callback(self, msg, topic_name):
        try:
            self.frame_counts[topic_name] += 1

            if self.frame_counts[topic_name] % self.save_every_n != 0:
                return

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            topic_clean = topic_name.strip('/').replace('/', '_')
            filename = os.path.join(
                self.save_dir,
                f"{topic_clean}_{timestamp}.{self.image_format}"
            )

            ##TODO--
            # PreProcess Input 
                # FRAME -> safety areas
                # Split 80/20
            # Training
            # Calibration using Val Set
            # score = model(frame)
            # RulexDetectionResult

            ok = cv2.imwrite(filename, frame)
            if not ok:
                self.get_logger().error(f"Failed to save frame to {filename}")
                return

            self.saved_counts[topic_name] += 1

            if self.saved_counts[topic_name] % 20 == 0:
                self.get_logger().info(
                    f"{topic_name}: saved {self.saved_counts[topic_name]} frames"
                )

        except Exception as e:
            self.get_logger().error(f"Error on {topic_name}: {e}")


def main():
    rclpy.init()
    node = FrameSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping frame saver.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()