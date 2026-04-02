import os
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def _ensure_gray(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _crop_with_mask(frame: np.ndarray, mask_gray: np.ndarray):
    if frame is None or mask_gray is None:
        return None, None

    mask_gray = _ensure_gray(mask_gray)

    if mask_gray.shape[:2] != frame.shape[:2]:
        mask_gray = cv2.resize(
            mask_gray,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    masked_full = cv2.bitwise_and(frame, frame, mask=mask_bin)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, masked_full

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = masked_full[y_min:y_max + 1, x_min:x_max + 1]
    return cropped, masked_full


def _resize_128(image: np.ndarray, keep_aspect: bool = True, target=(128, 128)) -> np.ndarray:
    target_w, target_h = target

    if image is None:
        return None

    if not keep_aspect:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _prepare_binary_mask(mask: np.ndarray, frame_shape_hw):
    h, w = frame_shape_hw
    mask = _ensure_gray(mask)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _make_masked_input_visual(
    frame: np.ndarray,
    masks: list,
    blur_ksize: int = 31,
    dim_factor: float = 0.35,
    outline_color=(255, 255, 255),
    outline_thickness: int = 6
):
    """
    Create a visualization where masked safety areas remain sharp,
    while the outside region is blurred + dimmed.
    """
    h, w = frame.shape[:2]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    resized_masks = []

    for m in masks:
        if m is None:
            continue
        mask_bin = _prepare_binary_mask(m, (h, w))
        resized_masks.append(mask_bin)
        combined_mask = cv2.bitwise_or(combined_mask, mask_bin)

    # blurred background
    k = max(3, int(blur_ksize))
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(frame, (k, k), 0)

    # dim blurred background
    dimmed_blurred = np.clip(blurred.astype(np.float32) * dim_factor, 0, 255).astype(np.uint8)

    # keep original content inside safety areas
    combined_mask_3 = cv2.merge([combined_mask, combined_mask, combined_mask])
    result = np.where(combined_mask_3 > 0, frame, dimmed_blurred)

    # draw white outlines around each area
    for mask_bin in resized_masks:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, outline_color, outline_thickness)

    return result


class FrameSaver(Node):
    def __init__(self):
        super().__init__('frame_saver')

        self.declare_parameter('save_dir', '/home/unito/data/saved_frames')
        self.declare_parameter('camera_topic', '/camera/back_view/image_raw')
        self.declare_parameter('area_names', ['ConvBelt', 'PLeft', 'PRight', 'RoboArm'])
        self.declare_parameter('static_mask_paths', [''])
        self.declare_parameter('save_every_n', 1)
        self.declare_parameter('image_format', 'png')
        self.declare_parameter('keep_aspect', True)
        self.declare_parameter('save_masked_full', False)

        # new params
        self.declare_parameter('save_masked_input', True)
        self.declare_parameter('masked_input_subdir', 'masked_input')
        self.declare_parameter('masked_input_blur_ksize', 31)
        self.declare_parameter('masked_input_dim_factor', 0.35)
        self.declare_parameter('masked_input_outline_thickness', 6)

        self.save_dir = self.get_parameter('save_dir').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.area_names = list(self.get_parameter('area_names').value)
        self.static_mask_paths = list(self.get_parameter('static_mask_paths').value)
        self.save_every_n = int(self.get_parameter('save_every_n').value)
        self.image_format = str(self.get_parameter('image_format').value).lower()
        self.keep_aspect = bool(self.get_parameter('keep_aspect').value)
        self.save_masked_full = bool(self.get_parameter('save_masked_full').value)

        self.save_masked_input = bool(self.get_parameter('save_masked_input').value)
        self.masked_input_subdir = str(self.get_parameter('masked_input_subdir').value)
        self.masked_input_blur_ksize = int(self.get_parameter('masked_input_blur_ksize').value)
        self.masked_input_dim_factor = float(self.get_parameter('masked_input_dim_factor').value)
        self.masked_input_outline_thickness = int(self.get_parameter('masked_input_outline_thickness').value)

        os.makedirs(self.save_dir, exist_ok=True)

        if len(self.area_names) != len(self.static_mask_paths):
            raise ValueError(
                f"Length mismatch: area_names={len(self.area_names)} vs static_mask_paths={len(self.static_mask_paths)}"
            )

        self.bridge = CvBridge()
        self.frame_count = 0
        self.saved_count = 0

        self.area_masks = {}
        for area_name, mask_path in zip(self.area_names, self.static_mask_paths):
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {area_name}: {mask_path}")

            mask = _ensure_gray(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            if mask is None:
                raise RuntimeError(f"Could not read mask for {area_name}: {mask_path}")

            self.area_masks[area_name] = mask
            os.makedirs(os.path.join(self.save_dir, area_name), exist_ok=True)

            if self.save_masked_full:
                os.makedirs(os.path.join(self.save_dir, area_name, "masked_full"), exist_ok=True)

            self.get_logger().info(f"Loaded mask for {area_name}: {mask_path}")

        if self.save_masked_input:
            os.makedirs(os.path.join(self.save_dir, self.masked_input_subdir), exist_ok=True)

        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.listener_callback,
            10
        )

        self.get_logger().info(f"Subscribed to camera topic: {self.camera_topic}")
        self.get_logger().info(f"Saving areas to: {self.save_dir}")

    def listener_callback(self, msg):
        try:
            self.frame_count += 1

            if self.frame_count % self.save_every_n != 0:
                return

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # save combined masked_input visualization
            if self.save_masked_input:
                combined_visual = _make_masked_input_visual(
                    frame=frame,
                    masks=[self.area_masks[a] for a in self.area_names],
                    blur_ksize=self.masked_input_blur_ksize,
                    dim_factor=self.masked_input_dim_factor,
                    outline_thickness=self.masked_input_outline_thickness
                )

                masked_input_dir = os.path.join(self.save_dir, self.masked_input_subdir)
                masked_input_path = os.path.join(
                    masked_input_dir,
                    f"masked_input_{timestamp}.{self.image_format}"
                )
                ok = cv2.imwrite(masked_input_path, combined_visual)
                if not ok:
                    self.get_logger().error(f"Failed saving masked input to {masked_input_path}")

            # save per-area cropped outputs
            for area_name in self.area_names:
                mask = self.area_masks[area_name]
                cropped, masked_full = _crop_with_mask(frame, mask)

                if cropped is None:
                    self.get_logger().warning(f"No valid crop for area {area_name}")
                    continue

                resized = _resize_128(
                    cropped,
                    keep_aspect=self.keep_aspect,
                    target=(128, 128)
                )

                out_dir = os.path.join(self.save_dir, area_name)
                out_path = os.path.join(out_dir, f"{area_name}_{timestamp}.{self.image_format}")

                ok = cv2.imwrite(out_path, resized)



                ## LOAD MODELS 
                ## 
                if not ok:
                    self.get_logger().error(f"Failed saving {out_path}")
                    continue

                if self.save_masked_full and masked_full is not None:
                    masked_dir = os.path.join(self.save_dir, area_name, "masked_full")
                    masked_path = os.path.join(masked_dir, f"masked_{timestamp}.{self.image_format}")
                    cv2.imwrite(masked_path, masked_full)

            self.saved_count += 1
            if self.saved_count % 20 == 0:
                self.get_logger().info(f"Processed and saved {self.saved_count} sampled frames")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")


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