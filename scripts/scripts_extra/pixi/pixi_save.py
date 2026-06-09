import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image


CAMERA_NAME = "back_view"
TOPIC_PREFIX = f"/camera/{CAMERA_NAME}/"

MSG_TYPE_MAP = {
    "sensor_msgs/msg/CompressedImage": CompressedImage,
    "sensor_msgs/msg/Image": Image,
}


def decode_frame(msg):
    if isinstance(msg, CompressedImage):
        return cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
    if isinstance(msg, Image):
        from cv_bridge import CvBridge

        return CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    return None


def extract_frames(bag_path, save_path):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="",
        output_serialization_format="",
    )
    reader.open(storage_options, converter_options)

    topic_type_map = {
        t.name: t.type for t in reader.get_all_topics_and_types()
    }

    matching_topics = {
        name: MSG_TYPE_MAP[typ]
        for name, typ in topic_type_map.items()
        if name.startswith(TOPIC_PREFIX) and typ in MSG_TYPE_MAP
    }

    if not matching_topics:
        print(f"No image topics found for '{CAMERA_NAME}' camera.")
        print("Available topics:")
        for name, typ in sorted(topic_type_map.items()):
            print(f"  {name} [{typ}]")
        return

    save_dir = Path(save_path) / CAMERA_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic not in matching_topics:
            continue

        msg = deserialize_message(data, matching_topics[topic])
        frame = decode_frame(msg)
        if frame is None:
            continue

        filename = save_dir / f"{CAMERA_NAME}_{count:06d}.jpg"
        cv2.imwrite(str(filename), frame)
        count += 1

    print(f"Extracted {count} frames to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=f"Extract {CAMERA_NAME} camera frames from a rosbag"
    )
    parser.add_argument("bag_path", help="Path to the bag file (.mcap)")
    parser.add_argument(
        "--save-path",
        default=".",
        help="Root directory for extracted frames (default: current directory)",
    )
    args = parser.parse_args()
    extract_frames(args.bag_path, args.save_path)


if __name__ == "__main__":
    main()