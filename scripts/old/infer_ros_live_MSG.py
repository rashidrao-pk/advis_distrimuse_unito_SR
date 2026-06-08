import os
import json
import time
import argparse
from collections import deque

import cv2
import numpy as np

import torch
import torchvision.transforms as transforms

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

from distrimuse_ros2_api.msg import RulexAreaScore, RulexDetectionResult

import utils as ut
import utils_model as utmc
from utils_model import Encoder, Decoder, Discriminator


ALL_SAFETY_AREAS = ["RoboArm", "ConvBelt", "PLeft", "PRight"]


def _ensure_gray(mask):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _prepare_binary_mask(mask, frame_shape_hw):
    h, w = frame_shape_hw
    mask = _ensure_gray(mask)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _crop_with_mask(frame, mask_gray):
    mask_bin = _prepare_binary_mask(mask_gray, frame.shape[:2])
    masked_full = cv2.bitwise_and(frame, frame, mask=mask_bin)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, masked_full

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = masked_full[y_min:y_max + 1, x_min:x_max + 1]
    bbox = (x_min, y_min, x_max, y_max)
    return cropped, bbox, masked_full


def _resize_128(image, keep_aspect=True, target=(128, 128)):
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


def tensor_to_hwc_float32(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32) * 0.5 + 0.5)


def _compute_distance_offset_np(imgA: np.ndarray, imgB: np.ndarray, offset: int) -> np.ndarray:
    H, W, _ = imgA.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    for di in range(-offset, offset + 1):
        for dj in range(-offset, offset + 1):
            i0a = max(0, di)
            i1a = min(H, H + di)
            i0b = max(0, -di)
            i1b = min(H, H - di)

            j0a = max(0, dj)
            j1a = min(W, W + dj)
            j0b = max(0, -dj)
            j1b = min(W, W - dj)

            diff = imgA[i0a:i1a, j0a:j1a] - imgB[i0b:i1b, j0b:j1b]
            d = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)
            dist[i0a:i1a, j0a:j1a] = np.minimum(dist[i0a:i1a, j0a:j1a], d)
    return dist


def compute_anomaly_score_pair(imgA, imgB, offset=2, quantile=0.995):
    dist = _compute_distance_offset_np(imgA, imgB, offset)
    return float(np.quantile(dist, quantile)), dist


def load_threshold(threshold_dir: str, safety_area: str) -> float:
    json_path = os.path.join(threshold_dir, safety_area, f"threshold_{safety_area}.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        return float(d["threshold"])
    raise FileNotFoundError(f"Threshold file not found: {json_path}")


def build_suffix_for_area(area, args):
    class P:
        pass

    params = P()
    params.subgroup = area
    params.latent_dims = args.latent_dims
    params.z_dim = args.latent_dims
    params.dataset_type = args.dataset_source_name
    params.subgroup_mask = args.subgroup_mask
    params.target_size = (128, 128)

    if not hasattr(args, "save_figures"):
        args.save_figures = False
    if not hasattr(args, "train"):
        args.train = False
    if not hasattr(args, "test"):
        args.test = False
    if not hasattr(args, "inference"):
        args.inference = True

    paths = P()
    cwd = os.getcwd()
    paths.path_codes = cwd
    paths.path_codes_local = cwd
    paths.path_results_local = cwd
    paths.path_results_cloud = cwd
    paths.path_models = os.path.join(cwd, args.checkpoints)

    suffix, _ = ut.get_create_results_path(
        area,
        params,
        args,
        paths,
        save_path_type=args.save_path_type,
        dir="scripts/results",
        verbose=False,
    )
    return suffix


def load_models_and_thresholds(areas, args, device, log_fn=None):
    models = {}
    thresholds = {}

    checkpoint_root = os.path.join(os.getcwd(), args.checkpoints)

    for area in areas:
        if log_fn:
            log_fn(1, f"[model-load] preparing area={area}")

        enc = Encoder(z_size=args.latent_dims).to(device)
        dec = Decoder(z_size=args.latent_dims).to(device)
        dis = Discriminator().to(device)

        optED, optD = utmc.get_optimizers(enc, dec, dis, verbose=False)
        suffix = build_suffix_for_area(area, args)

        if log_fn:
            log_fn(2, f"[model-load] area={area} suffix={suffix}")
            log_fn(2, f"[model-load] checkpoint_root={checkpoint_root}")

        history = utmc.load_model(
            enc, dec, dis, optED, optD,
            checkpoint_root, suffix, device=device, verbose=False
        )

        if len(history) == 0:
            raise RuntimeError(f"No checkpoint found for area={area}, suffix={suffix}")

        enc.eval()
        dec.eval()

        tau = load_threshold(args.threshold_dir, area)

        models[area] = {"encoder": enc, "decoder": dec, "suffix": suffix}
        thresholds[area] = tau

        if log_fn:
            log_fn(1, f"[loaded] {area}: suffix={suffix}, tau={tau:.6f}")

    return models, thresholds


def _safe_color(name):
    colors = {
        "RoboArm": (255, 100, 100),
        "ConvBelt": (100, 255, 100),
        "PLeft": (100, 180, 255),
        "PRight": (255, 220, 100),
    }
    return colors.get(name, (200, 200, 200))


def draw_timeline_panel(score_history, latest_results, width=1000, height=500, max_points=200):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    left_pad = 80
    right_pad = 20
    top_pad = 40
    bottom_pad = 40

    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    cv2.rectangle(canvas, (left_pad, top_pad), (left_pad + plot_w, top_pad + plot_h), (80, 80, 80), 1)

    y_thr = top_pad + int(plot_h * 0.5)
    cv2.line(canvas, (left_pad, y_thr), (left_pad + plot_w, y_thr), (0, 0, 180), 1)
    cv2.putText(canvas, "thr=1.0", (left_pad + 8, y_thr - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)

    for val in [0.0, 0.5, 1.0, 1.5, 2.0]:
        yy = top_pad + int(plot_h * (1.0 - min(val, 2.0) / 2.0))
        cv2.line(canvas, (left_pad - 5, yy), (left_pad, yy), (180, 180, 180), 1)
        cv2.putText(canvas, f"{val:.1f}", (10, yy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(canvas, "ADVIS Live Anomaly Timeline (normalized scores)",
                (left_pad, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    keys = list(score_history.keys())
    for idx, area_name in enumerate(keys):
        color = _safe_color(area_name)
        vals = list(score_history[area_name])

        if len(vals) >= 2:
            pts = []
            recent_vals = vals[-max_points:]
            for i, v in enumerate(recent_vals):
                x = left_pad + int(i * (plot_w / max(1, max_points - 1)))
                v_clip = max(0.0, min(2.0, float(v)))
                y = top_pad + int(plot_h * (1.0 - v_clip / 2.0))
                pts.append((x, y))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 2)

        latest = latest_results.get(area_name, {})
        label = area_name
        if "norm_score" in latest:
            label += f"  {latest['norm_score']:.3f}"
        if "status" in latest:
            label += f"  [{latest['status']}]"

        legend_y = top_pad + 20 + 28 * idx
        cv2.line(canvas, (width - 300, legend_y - 5), (width - 260, legend_y - 5), color, 3)
        cv2.putText(canvas, label, (width - 250, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return canvas


class LiveRosAnomalyInfer(Node):
    def __init__(self, args):
        super().__init__("live_ros_anomaly_infer")

        self.args = args
        self.verbose_level = args.verbose_level
        self.log_every_n = args.log_every_n

        self.vlog(1, "[startup] initializing node")

        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.bridge = CvBridge()

        self.vlog(1, f"[startup] torch.cuda.is_available()={torch.cuda.is_available()}")
        self.vlog(1, f"[startup] requested cpu flag={args.cpu}")
        self.vlog(1, f"[startup] selected device={self.device}")

        if self.device.type == "cuda":
            self.vlog(1, f"[startup] cuda device count={torch.cuda.device_count()}")
            self.vlog(1, f"[startup] cuda current device={torch.cuda.current_device()}")
            self.vlog(1, f"[startup] cuda device name={torch.cuda.get_device_name(torch.cuda.current_device())}")

        self.areas = ALL_SAFETY_AREAS if args.safety_area.upper() == "ALL" else [args.safety_area]
        self.vlog(1, f"[startup] active areas={self.areas}")

        self.vlog(1, "[startup] loading models and thresholds...")
        t0 = time.time()
        self.models, self.thresholds = load_models_and_thresholds(
            self.areas, args, self.device, log_fn=self.vlog
        )
        self.vlog(1, f"[startup] models loaded in {time.time() - t0:.3f}s")

        self.area_masks = {}
        if len(args.area_names) != len(args.static_mask_paths):
            raise ValueError("area_names and static_mask_paths must have the same length")

        self.vlog(1, "[startup] loading masks...")
        for area_name, mask_path in zip(args.area_names, args.static_mask_paths):
            if area_name not in self.areas:
                continue
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {area_name}: {mask_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not load mask for {area_name}: {mask_path}")
            self.area_masks[area_name] = mask
            self.vlog(1, f"[mask] loaded area={area_name} path={mask_path} shape={mask.shape}")

        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subscription = self.create_subscription(
            RosImage,
            args.camera_topic,
            self.ros_callback_store_latest,
            sensor_qos,
        )

        self.frame_count = 0
        self.processed_count = 0
        self.max_frames = args.max_frames
        self.start_time = time.time()
        self.first_callback_seen = False

        self.latest_msg = None
        self.latest_msg_id = 0
        self.last_processed_msg_id = 0
        self.is_processing = False
        self.process_start_time = None

        self.show_timeline = args.show_timeline
        self.timeline_history_len = args.timeline_history
        self.timeline_width = args.timeline_width
        self.timeline_height = args.timeline_height

        self.score_history = {
            area: deque(maxlen=self.timeline_history_len) for area in self.areas
        }
        self.latest_results = {
            area: {"status": "waiting"} for area in self.areas
        }

        self.publish_rulex = args.publish_rulex
        self.rulex_topic = args.rulex_topic
        self.attach_image_on_anomaly = args.attach_image_on_anomaly

        self.area_to_rulex = {
            "RoboArm": RulexAreaScore.AREA_A,
            "ConvBelt": RulexAreaScore.AREA_B,
            "PLeft": RulexAreaScore.AREA_C,
            "PRight": RulexAreaScore.AREA_D,
        }

        if self.publish_rulex:
            self.rulex_pub = self.create_publisher(
                RulexDetectionResult,
                self.rulex_topic,
                10
            )
            self.vlog(1, f"[publisher] Rulex publisher created on {self.rulex_topic}")

        self.wait_start = time.time()
        self.wait_timer = self.create_timer(5.0, self.wait_for_first_frame_log)

        self.process_timer = self.create_timer(args.process_period, self.process_latest_frame)

        self.vlog(1, f"[subscriber] subscribed to {args.camera_topic}")
        self.vlog(1, f"[subscriber] frame_stride={self.args.frame_stride}")
        self.vlog(1, f"[subscriber] max_frames={self.max_frames}")
        self.vlog(1, f"[subscriber] process_period={self.args.process_period}s")

    def vlog(self, level, msg):
        if self.verbose_level >= level:
            self.get_logger().info(msg)

    def wait_for_first_frame_log(self):
        if not self.first_callback_seen:
            waited = time.time() - self.wait_start
            self.vlog(1, f"[wait] still waiting for first frame after {waited:.1f}s on {self.args.camera_topic}")
        else:
            try:
                self.wait_timer.cancel()
            except Exception:
                pass

    def ros_callback_store_latest(self, msg):
        self.frame_count += 1
        self.latest_msg = msg
        self.latest_msg_id = self.frame_count

        if not self.first_callback_seen:
            self.first_callback_seen = True
            self.vlog(1, "[callback] first ROS frame received")

        self.vlog(4, f"[callback-store] stored latest raw frame #{self.latest_msg_id}")

    def preprocess_area(self, frame_bgr, area_name):
        t0 = time.time()

        mask = self.area_masks[area_name]
        cropped, bbox, _ = _crop_with_mask(frame_bgr, mask)
        if cropped is None:
            self.vlog(3, f"[preprocess] {area_name}: crop failed")
            return None, None

        resized = _resize_128(cropped, keep_aspect=True, target=(128, 128))
        if resized is None:
            self.vlog(3, f"[preprocess] {area_name}: resize failed")
            return None, None

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        input_tensor = self.normalize(input_tensor)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        self.vlog(3, f"[preprocess] {area_name}: bbox={bbox}, tensor_shape={tuple(input_tensor.shape)}, time={time.time() - t0:.4f}s")
        return input_tensor, bbox

    def infer_area(self, input_tensor, area_name):
        t0 = time.time()

        enc = self.models[area_name]["encoder"]
        dec = self.models[area_name]["decoder"]
        tau = self.thresholds[area_name]

        with torch.no_grad():
            mu, logvar = enc(input_tensor)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            recon = dec(z)

        orig_hwc = tensor_to_hwc_float32(input_tensor.squeeze(0))
        recon_hwc = tensor_to_hwc_float32(recon.squeeze(0))
        score, _ = compute_anomaly_score_pair(
            orig_hwc, recon_hwc,
            offset=self.args.offset,
            quantile=self.args.quantile,
        )
        norm_score = score / tau
        is_anom = norm_score > 1.0

        self.vlog(3, f"[infer] {area_name}: score={score:.6f}, tau={tau:.6f}, norm={norm_score:.6f}, time={time.time() - t0:.4f}s")

        return {
            "score": score,
            "threshold": tau,
            "norm_score": norm_score,
            "is_anomalous": is_anom,
            "status": "UNEXPECTED" if is_anom else "normal",
        }

    def publish_rulex_result(self, results, frame_bgr):
        if not self.publish_rulex:
            return

        msg = RulexDetectionResult()
        area_scores = []
        any_anomaly = False

        for area in self.areas:
            r = results.get(area, {})
            score_msg = RulexAreaScore()
            score_msg.area = self.area_to_rulex.get(area, area)
            score_msg.anomaly = bool(r.get("is_anomalous", False))

            if score_msg.anomaly:
                any_anomaly = True

            area_scores.append(score_msg)

        msg.area_scores = area_scores

        if self.attach_image_on_anomaly and any_anomaly:
            msg.image = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")

        self.rulex_pub.publish(msg)
        self.vlog(2, f"[publisher] published RulexDetectionResult with {len(area_scores)} area scores")

    def process_latest_frame(self):
        if self.is_processing:
            return
        if self.latest_msg is None:
            return
        if self.latest_msg_id == self.last_processed_msg_id:
            return

        msg = self.latest_msg
        msg_id = self.latest_msg_id

        self.is_processing = True
        process_t0 = time.time()

        try:
            if self.args.frame_stride > 1 and (msg_id % self.args.frame_stride != 0):
                self.last_processed_msg_id = msg_id
                self.vlog(4, f"[process] skipped by frame_stride: raw frame={msg_id}")
                return

            self.vlog(2, f"[process] using latest raw frame #{msg_id}")

            t_convert = time.time()
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if self.processed_count == 0:
                self.process_start_time = time.time()

            self.processed_count += 1

            self.vlog(
                2,
                f"[process] processed_count={self.processed_count}, shape={frame_bgr.shape}, "
                f"convert_time={time.time() - t_convert:.4f}s"
            )

            results = {}
            for area in self.areas:
                area_t0 = time.time()
                self.vlog(3, f"[area-start] {area}")

                input_tensor, bbox = self.preprocess_area(frame_bgr, area)
                if input_tensor is None:
                    results[area] = {"status": "mask_failed"}
                    self.latest_results[area] = {"status": "mask_failed"}
                    self.vlog(2, f"[area-end] {area}: mask_failed in {time.time() - area_t0:.4f}s")
                    continue

                out = self.infer_area(input_tensor, area)
                out["bbox"] = bbox
                results[area] = out

                self.score_history[area].append(float(out["norm_score"]))
                self.latest_results[area] = out

                self.vlog(
                    2,
                    f"[area-end] {area}: status={out['status']} norm={out['norm_score']:.4f} "
                    f"in {time.time() - area_t0:.4f}s"
                )

            self.last_processed_msg_id = msg_id

            self.publish_rulex_result(results, frame_bgr)

            proc_elapsed = time.time() - self.process_start_time if self.process_start_time else 0.0
            avg_fps = self.processed_count / max(proc_elapsed, 1e-6)
            inst_fps = 1.0 / max(time.time() - process_t0, 1e-6)

            if self.processed_count % self.log_every_n == 0 or self.processed_count == 1:
                msg_parts = [
                    f"raw_frame={msg_id}",
                    f"processed={self.processed_count}",
                    f"avg_fps={avg_fps:.2f}",
                    f"inst_fps={inst_fps:.2f}",
                ]
                for area in self.areas:
                    r = results[area]
                    if "score" in r:
                        msg_parts.append(
                            f"{area}: score={r['score']:.5f}, norm={r['norm_score']:.3f}, status={r['status']}"
                        )
                    else:
                        msg_parts.append(f"{area}: status={r['status']}")
                self.vlog(1, " | ".join(msg_parts))

            if self.show_timeline:
                panel = draw_timeline_panel(
                    self.score_history,
                    self.latest_results,
                    width=self.timeline_width,
                    height=self.timeline_height,
                    max_points=self.timeline_history_len,
                )
                cv2.imshow("ADVIS Timeline", panel)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.vlog(1, "[gui] ESC pressed, shutting down")
                    rclpy.shutdown()
                    return

            self.vlog(3, f"[process] total time={time.time() - process_t0:.4f}s")

            if self.max_frames is not None and self.processed_count >= self.max_frames:
                self.vlog(1, "[process] reached max_frames, shutting down")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"[process-error] {e}")
        finally:
            self.is_processing = False


def parse_args():
    p = argparse.ArgumentParser("Live ROS anomaly inference")

    p.add_argument("--camera_topic", default="/camera/back_view/image_raw")
    p.add_argument("--safety_area", default="ALL")
    p.add_argument("--area_names", nargs="+", default=["ConvBelt", "PLeft", "PRight", "RoboArm"])
    p.add_argument("--static_mask_paths", nargs="+", required=True)

    p.add_argument("--threshold_dir", required=True)
    p.add_argument("--checkpoints", default="scripts/results/models")
    p.add_argument("--latent_dims", type=int, default=64)

    p.add_argument("--offset", type=int, default=2)
    p.add_argument("--quantile", type=float, default=0.995)
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--dataset_source_name", default="SR")
    p.add_argument("--subgroup_mask", default="MASK")
    p.add_argument("--save_path_type", default="local")
    p.add_argument("--save_figures", action="store_true", default=False)

    p.add_argument("--verbose_level", type=int, default=2)
    p.add_argument("--log_every_n", type=int, default=1)
    p.add_argument("--process_period", type=float, default=0.05)

    p.add_argument("--show_timeline", action="store_true")
    p.add_argument("--timeline_history", type=int, default=200)
    p.add_argument("--timeline_width", type=int, default=1000)
    p.add_argument("--timeline_height", type=int, default=500)

    p.add_argument("--publish_rulex", action="store_true",
                   help="Publish RulexDetectionResult over ROS2")
    p.add_argument("--rulex_topic", default="/rulex/detection_result",
                   help="ROS2 topic for RulexDetectionResult")
    p.add_argument("--attach_image_on_anomaly", action="store_true",
                   help="Attach current frame only when at least one area is anomalous")

    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = LiveRosAnomalyInfer(args)
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