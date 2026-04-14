import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

class Probe(Node):
    def __init__(self):
        super().__init__('probe_sub')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.count = 0
        self.create_subscription(Image, '/camera/back_view/image_raw', self.cb, qos)
        self.get_logger().info('Subscribed to /camera/back_view/image_raw')

    def cb(self, msg):
        self.count += 1
        self.get_logger().info(
            f"received #{self.count}: "
            f"{msg.width}x{msg.height}, encoding={msg.encoding}, step={msg.step}"
        )
        if self.count >= 3:
            rclpy.shutdown()

rclpy.init()
node = Probe()
rclpy.spin(node)
node.destroy_node()
if rclpy.ok():
    rclpy.shutdown()