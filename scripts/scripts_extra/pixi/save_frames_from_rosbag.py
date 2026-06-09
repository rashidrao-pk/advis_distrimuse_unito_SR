import os
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def topic_to_name(topic):
    return topic.strip("/").replace("/", "_")


def main():
    rclpy.init()

    # =========================
    # Configuration
    # =========================
    bag_path = "/home/unito/advis/bags/recording_20260313_133316"
    save_dir = "/home/unito/data/saved_frames"
    topics = ["/camera/front_view/image_raw"]   # add more topics if needed
    save_every_n = 1
    image_format = "png"   # png / jpg
    use_msg_timestamp = True   # use ros message timestamp in filename

    ensure_dir(save_dir)
    bridge = CvBridge()

    frame_counts = {topic: 0 for topic in topics}
    saved_counts = {topic: 0 for topic in topics}

    # =========================
    # Open bag
    # =========================
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Optional: inspect available topics in bag
    all_topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {x.name: x.type for x in all_topics_and_types}

    print("Topics available in bag:")
    for name, msg_type in topic_type_map.items():
        print(f"  {name} -> {msg_type}")

    for topic in topics:
        if topic not in topic_type_map:
            print(f"[WARNING] Topic not found in bag: {topic}")

    print(f"\nSaving frames to: {save_dir}")
    print(f"Save every N frames: {save_every_n}")
    print(f"Image format: {image_format}")

    # =========================
    # Read messages
    # =========================
    while reader.has_next():
        topic_name, data, t = reader.read_next()

        if topic_name not in topics:
            continue

        if topic_type_map.get(topic_name) != "sensor_msgs/msg/Image":
            print(f"[WARNING] Skipping non-Image topic: {topic_name}")
            continue

        try:
            msg = deserialize_message(data, Image)
            frame_counts[topic_name] += 1

            if frame_counts[topic_name] % save_every_n != 0:
                continue

            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            topic_clean = topic_to_name(topic_name)

            if use_msg_timestamp:
                stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                filename = os.path.join(
                    save_dir,
                    f"{topic_clean}_{stamp_ns}.{image_format}"
                )
            else:
                filename = os.path.join(
                    save_dir,
                    f"{topic_clean}_{saved_counts[topic_name]:06d}.{image_format}"
                )

            ok = cv2.imwrite(filename, frame)
            if not ok:
                print(f"[ERROR] Failed to save frame to {filename}")
                continue

            saved_counts[topic_name] += 1

            if saved_counts[topic_name] % 20 == 0:
                print(f"{topic_name}: saved {saved_counts[topic_name]} frames")

        except Exception as e:
            print(f"[ERROR] Error on {topic_name}: {e}")

    print("\nDone.")
    for topic in topics:
        print(
            f"{topic} -> seen: {frame_counts[topic]}, saved: {saved_counts[topic]}"
        )

    rclpy.shutdown()


if __name__ == "__main__":
    main()