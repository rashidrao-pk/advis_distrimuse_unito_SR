import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

SAVE_DIR = "/home/unito/saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

class FrameSaver(Node):
    def __init__(self):
        super().__init__('frame_saver')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/front_view/image_raw',   # 🔁 change if needed
            self.listener_callback,
            10
        )

        self.counter = 0
        self.get_logger().info("Frame saver started...")

    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{SAVE_DIR}/frame_{timestamp}.png"

            cv2.imwrite(filename, frame)

            self.counter += 1

            if self.counter % 20 == 0:
                self.get_logger().info(f"Saved {self.counter} frames")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main():
    rclpy.init()
    node = FrameSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
