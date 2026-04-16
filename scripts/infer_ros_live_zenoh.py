import os
import json
import time
import argparse
from collections import deque, OrderedDict

import cv2
import msgpack
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torch
import torchvision.transforms as transforms
import zenoh

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage,CompressedImage

from cv_bridge import CvBridge

from distrimuse_ros2_api.msg import RulexAreaScore, RulexDetectionResult

import utils as ut
import utils_model as utmc
from utils_model import Encoder, Decoder, Discriminator


ALL_SAFETY_AREAS = ["PLeft", "PRight", "RoboArm", "ConvBelt"]

THRESHOLD_CMAP_UNEXPECTED = LinearSegmentedColormap.from_list(
    "custom_threshold_cmap",
    [
        (0.0, "white"),
        (0.25, "lightblue"),
        (0.35, "coral"),
        (0.50, "red"),
        (1.0, "purple"),
    ],
)

AREA_DISPLAY_NAMES = {
    "PLeft": "Pallet Left",
    "PRight": "Pallet Right",
    "RoboArm": "Robo Arm",
    "ConvBelt": "Conveyor Belt",
}

AREA_NAME_TO_ENUM = {
    "RoboArm": RulexAreaScore.AREA_A,
    "ConvBelt": RulexAreaScore.AREA_B,
    "PLeft": RulexAreaScore.AREA_C,
    "PRight": RulexAreaScore.AREA_D,
}


def ordered_area_list(areas):
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


############################################################################################################
def colorize_anomaly_map(dist_map, vmin=0.0, vmax=2.0):
    if dist_map is None:
        return None

    dm = dist_map.astype(np.float32)

    denom = max(float(vmax) - float(vmin), 1e-6)
    dm = np.clip((dm - float(vmin)) / denom, 0.0, 1.0)

    rgb = (THRESHOLD_CMAP_UNEXPECTED(dm)[..., :3] * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def unletterbox_patch(patch_bgr, resize_meta):
    """
    Remove padding introduced by _resize_128(... keep_aspect=True).
    """
    if patch_bgr is None or resize_meta is None:
        return patch_bgr

    x_off = int(resize_meta.get("x_off", 0))
    y_off = int(resize_meta.get("y_off", 0))
    new_w = int(resize_meta.get("new_w", patch_bgr.shape[1]))
    new_h = int(resize_meta.get("new_h", patch_bgr.shape[0]))

    if new_w <= 0 or new_h <= 0:
        return patch_bgr

    h, w = patch_bgr.shape[:2]
    x1 = max(0, x_off)
    y1 = max(0, y_off)
    x2 = min(w, x_off + new_w)
    y2 = min(h, y_off + new_h)

    if x2 <= x1 or y2 <= y1:
        return patch_bgr

    cropped = patch_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return patch_bgr
    return cropped


def paste_area_result_in_full_frame(
    target_canvas,
    patch_bgr,
    bbox,
    mask_bin,
    resize_meta=None,
    keep_background=False,
    background_canvas=None
):
    """
    Paste patch back into full-frame canvas. If the patch came from a letterboxed
    128x128 input, first remove the padding using resize_meta so shapes are restored.
    """
    if patch_bgr is None or bbox is None or mask_bin is None:
        return target_canvas

    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1

    if crop_w <= 0 or crop_h <= 0:
        return target_canvas

    if resize_meta is not None:
        patch_bgr = unletterbox_patch(patch_bgr, resize_meta)

    if patch_bgr is None or patch_bgr.size == 0:
        return target_canvas

    patch_resized = cv2.resize(patch_bgr, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
    mask_crop = mask_bin[y1:y2 + 1, x1:x2 + 1]
    mask_crop_3 = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)

    roi = target_canvas[y1:y2 + 1, x1:x2 + 1]

    if keep_background and background_canvas is not None:
        bg_roi = background_canvas[y1:y2 + 1, x1:x2 + 1]
        blended = np.where(mask_crop_3 > 0, patch_resized, bg_roi)
    else:
        blended = np.where(mask_crop_3 > 0, patch_resized, roi)

    target_canvas[y1:y2 + 1, x1:x2 + 1] = blended
    return target_canvas


############################################################################################################
#------------------------------------------------------------------------------------------------------------
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


def _extract_mask_contours(mask_gray, frame_shape_hw):
    mask_bin = _prepare_binary_mask(mask_gray, frame_shape_hw)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_bin


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


def _resize_128(image, keep_aspect=True, target=(128, 128), return_meta=False):
    target_w, target_h = target

    if image is None:
        return (None, None) if return_meta else None

    if not keep_aspect:
        out = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        meta = {
            "new_w": target_w,
            "new_h": target_h,
            "x_off": 0,
            "y_off": 0,
            "target_w": target_w,
            "target_h": target_h,
            "orig_h": image.shape[0],
            "orig_w": image.shape[1],
        }
        return (out, meta) if return_meta else out

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return (None, None) if return_meta else None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    meta = {
        "new_w": new_w,
        "new_h": new_h,
        "x_off": x_off,
        "y_off": y_off,
        "target_w": target_w,
        "target_h": target_h,
        "orig_h": h,
        "orig_w": w,
    }

    return (canvas, meta) if return_meta else canvas


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
    models = OrderedDict()
    thresholds = OrderedDict()

    checkpoint_root = os.path.join(os.getcwd(), args.checkpoints)

    for area in ordered_area_list(areas):
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
            checkpoint_root, suffix, device=device, verbose=False,
            model_variant=args.model_variant,
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


def encode_image(image: np.ndarray, ext: str = ".jpg", params=None) -> bytes:
    if image is None:
        raise ValueError("image cannot be None")
    ok, encoded = cv2.imencode(ext, image, params or [])
    if not ok:
        raise ValueError(f"cv2.imencode failed for {ext}")
    return encoded.tobytes()


def _frame_meta(msg_id: int, corr_frame_id: str, corr_stamp) -> dict:
    return {
        "msg_id": int(msg_id),
        "corr_frame_id": str(corr_frame_id),
        "stamp": {
            "sec": int(getattr(corr_stamp, "sec", 0) or 0),
            "nanosec": int(getattr(corr_stamp, "nanosec", 0) or 0),
        },
    }


def _serializable_results(results):
    cleaned = OrderedDict()
    for area in ordered_area_list(results.keys()):
        src_result = results[area]
        cleaned[area] = {
            "score": float(src_result["score"]) if "score" in src_result else None,
            "threshold": float(src_result["threshold"]) if "threshold" in src_result else None,
            "norm_score": float(src_result["norm_score"]) if "norm_score" in src_result else None,
            "is_anomalous": bool(src_result.get("is_anomalous", False)),
            "status": str(src_result.get("status", "unknown")),
        }
    return cleaned


def ordered_dict_of_lists(mapping):
    return OrderedDict((k, list(mapping[k])) for k in ordered_area_list(mapping.keys()))


def pack_dashboard_state(*, msg_id, corr_frame_id, corr_stamp, frame_bgr, area_inputs, latest_results, jpeg_quality=85):
    payload = {
        "frame_meta": _frame_meta(msg_id, corr_frame_id, corr_stamp),
        "frame_bgr_jpg": encode_image(frame_bgr, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]),
        "latest_results": _serializable_results(latest_results),
        "area_inputs": OrderedDict(),
    }

    for area in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area]
        payload["area_inputs"][area] = {
            "bbox": list(info["bbox"]) if info.get("bbox") is not None else None,
            "resize_meta": info.get("resize_meta"),
            "mask_png": encode_image(info["mask_bin"], ".png"),
            "orig_patch_jpg": encode_image(info["orig_patch_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]),
            "recon_patch_jpg": encode_image(info["recon_patch_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]),
            "anom_patch_jpg": encode_image(info["anom_patch_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]),
        }

    return msgpack.packb(payload, use_bin_type=True)


def pack_timeline_state(*, msg_id, corr_frame_id, corr_stamp, score_history, latest_results):
    payload = {
        "frame_meta": _frame_meta(msg_id, corr_frame_id, corr_stamp),
        "score_history": ordered_dict_of_lists(score_history),
        "latest_results": _serializable_results(latest_results),
    }
    return msgpack.packb(payload, use_bin_type=True)


def make_zenoh_config(endpoint: str) -> zenoh.Config:
    return zenoh.Config.from_json5(
        f"""
    {{
      mode: "client",
      connect: {{ endpoints: ["{endpoint}"] }}
    }}
    """
    )


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

        self.areas = ALL_SAFETY_AREAS if args.safety_area.upper() == "ALL" else ordered_area_list([args.safety_area])
        self.vlog(1, f"[startup] active areas={self.areas}")

        self.vlog(1, "[startup] loading models and thresholds...")
        t0 = time.time()
        self.models, self.thresholds = load_models_and_thresholds(
            self.areas, args, self.device, log_fn=self.vlog
        )
        self.vlog(1, f"[startup] models loaded in {time.time() - t0:.3f}s")

        self.area_masks = OrderedDict()
        if len(args.area_names) != len(args.static_mask_paths):
            raise ValueError("area_names and static_mask_paths must have the same length")

        self.vlog(1, "[startup] loading masks...")
        pairs = list(zip(args.area_names, args.static_mask_paths))
        pairs = sorted(pairs, key=lambda x: ALL_SAFETY_AREAS.index(x[0]) if x[0] in ALL_SAFETY_AREAS else 999)

        for area_name, mask_path in pairs:
            if area_name not in self.areas:
                continue
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {area_name}: {mask_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not load mask for {area_name}: {mask_path}")
            self.area_masks[area_name] = mask
            self.vlog(1, f"[mask] loaded area={area_name} path={mask_path} shape={mask.shape}")

        self.normalize = transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        )

        sensor_qos = QoSProfile(
            # reliability=ReliabilityPolicy.RELIABLE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            args.camera_topic,
            self.ros_callback_store_latest,
            sensor_qos,
        )

        print('self.subscription --> ', self.subscription)

        self.rulex_pub = None
        if args.publish_rulex:
            pub_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                # reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.rulex_pub = self.create_publisher(
                RulexDetectionResult,
                args.rulex_topic,
                pub_qos,
            )

            self.vlog(1, f"[publisher] publishing RulexDetectionResult to {args.rulex_topic}")
            print('='*100)

        zenoh.init_log_from_env_or(args.zenoh_log_level)
        self.zenoh_session = zenoh.open(make_zenoh_config(args.zenoh_endpoint))
        self.zenoh_dashboard_pub = self.zenoh_session.declare_publisher(
            args.zenoh_dashboard_key,
            encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        )
        self.zenoh_timeline_pub = self.zenoh_session.declare_publisher(
            args.zenoh_timeline_key,
            encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        )
        self.vlog(1, f"[publisher] publishing dashboard state to {args.zenoh_dashboard_key}")
        self.vlog(1, f"[publisher] publishing timeline state to {args.zenoh_timeline_key}")

        self.frame_count = 0
        self.processed_count = 0
        self.max_frames = args.max_frames
        self.start_time = time.time()
        self.first_callback_seen = False

        self.first_rulex_publish_done = False

        self.latest_msg = None
        self.latest_msg_id = 0
        self.last_processed_msg_id = 0
        self.is_processing = False
        self.process_start_time = None

        self.timeline_history_len = args.timeline_history

        self.score_history = OrderedDict(
            (area, deque(maxlen=self.timeline_history_len)) for area in self.areas
        )
        self.latest_results = OrderedDict(
            (area, {"status": "waiting"}) for area in self.areas
        )

        self.wait_start = time.time()
        self.wait_timer = self.create_timer(5.0, self.wait_for_first_frame_log)
        self.process_timer = self.create_timer(args.process_period, self.process_latest_frame)

        self.vlog(1, f"[subscriber] subscribed to {args.camera_topic}")
        print('-'*100)
        self.vlog(1, f"[subscriber] frame_stride={self.args.frame_stride}")
        print('-'*100)
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

        _, mask_bin = _extract_mask_contours(mask, frame_bgr.shape[:2])

        cropped, bbox, _ = _crop_with_mask(frame_bgr, mask)
        if cropped is None:
            self.vlog(3, f"[preprocess] {area_name}: crop failed")
            return None, None, None

        resized, resize_meta = _resize_128(cropped, keep_aspect=True, target=(128, 128), return_meta=True)
        if resized is None:
            self.vlog(3, f"[preprocess] {area_name}: resize failed")
            return None, None, None

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        input_tensor = self.normalize(input_tensor)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        self.vlog(
            3,
            f"[preprocess] {area_name}: bbox={bbox}, tensor_shape={tuple(input_tensor.shape)}, "
            f"resize_meta={resize_meta}, time={time.time() - t0:.4f}s"
        )

        vis_data = {
            "bbox": bbox,
            "mask_bin": mask_bin,
            "resize_meta": resize_meta,
        }

        return input_tensor, bbox, vis_data

    def infer_area(self, input_tensor, area_name):
        t0 = time.time()

        enc = self.models[area_name]["encoder"]
        dec = self.models[area_name]["decoder"]
        tau = self.thresholds[area_name]

        with torch.no_grad():
            mu, logvar = enc(input_tensor)
            z = mu
            recon = dec(z)

        orig_hwc = tensor_to_hwc_float32(input_tensor.squeeze(0))
        recon_hwc = tensor_to_hwc_float32(recon.squeeze(0))

        score, dist_map = compute_anomaly_score_pair(
            orig_hwc, recon_hwc,
            offset=self.args.offset,
            quantile=self.args.quantile,
        )

        norm_score = score / tau
        is_anom = norm_score > 1.0

        orig_bgr = (orig_hwc[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        recon_bgr = (recon_hwc[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        anom_bgr = colorize_anomaly_map(dist_map)

        self.vlog(
            3,
            f"[infer] {area_name}: score={score:.6f}, tau={tau:.6f}, norm={norm_score:.6f}, time={time.time() - t0:.4f}s"
        )

        return {
            "score": score,
            "threshold": tau,
            "norm_score": norm_score,
            "is_anomalous": is_anom,
            "status": "UNEXPECTED" if is_anom else "normal",
            "orig_patch_bgr": orig_bgr,
            "recon_patch_bgr": recon_bgr,
            "anom_patch_bgr": anom_bgr,
            "dist_map": dist_map,
        }

    def publish_rulex_result(self, results, frame_bgr, corr_frame_id, corr_stamp):
        if self.rulex_pub is None:
            return

        msg = RulexDetectionResult()
        area_scores = []
        any_anomaly = False

        for area_name in self.areas:
            r = results.get(area_name, {})

            area_msg = RulexAreaScore()
            area_msg.area = AREA_NAME_TO_ENUM.get(area_name, RulexAreaScore.AREA_A)
            area_msg.anomaly = bool(r.get("is_anomalous", False))

            # area_msg.score = float(r.get("norm_score", 0.0))
            # area_msg.frame_id = corr_frame_id
            # area_msg.stamp = corr_stamp

            if area_msg.anomaly:
                any_anomaly = True

            area_scores.append(area_msg)

        msg.area_scores = area_scores

        if self.args.attach_image_on_anomaly and any_anomaly:
            msg.image = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")

        if not self.first_rulex_publish_done:
            print('-'*100)
            print(f"[publish] FIRST RulexDetectionResult publish started for frame_id={corr_frame_id} at stamp={corr_stamp.sec}.{corr_stamp.nanosec}")
            print('-'*100)
            self.first_rulex_publish_done = True

        self.rulex_pub.publish(msg)

        summary = []
        for area_name in self.areas:
            rr = results.get(area_name, {})
            summary.append(
                f"{area_name}=anom:{bool(rr.get('is_anomalous', False))},norm:{float(rr.get('norm_score', 0.0)):.3f}"
            )

        self.vlog(2, f"[publish] sent RulexDetectionResult | " + " | ".join(summary))

    def process_latest_frame(self):
        if self.is_processing:
            return
        if self.latest_msg is None:
            ## TO DO: Sleep for 100 ms
            print("[process] no frame received yet, waiting...")
            time.sleep(0.1)
            return
        if self.latest_msg_id == self.last_processed_msg_id:
            return

        msg = self.latest_msg
        msg_id = self.latest_msg_id
        self.latest_msg = None
        # self.latest_msg_id = NotImplementedError
        self.latest_msg_id = 0

        corr_frame_id = msg.header.frame_id
        corr_stamp = msg.header.stamp

        print(f"[process] new frame to process: id={corr_frame_id}, stamp={corr_stamp}")

        self.is_processing = True
        process_t0 = time.time()

        try:
            # if self.args.frame_stride > 1 and (msg_id % self.args.frame_stride != 0):
            #     self.last_processed_msg_id = msg_id
            #     self.vlog(4, f"[process] skipped by frame_stride: raw frame={msg_id}")
            #     return

            self.vlog(2, f"[process] using latest raw frame #{msg_id}")

            t_convert = time.time()
            # frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame_bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.processed_count == 0:
                self.process_start_time = time.time()

            self.processed_count += 1

            self.vlog(
                2,
                f"[process] processed_count={self.processed_count}, shape={frame_bgr.shape}, "
                f"convert_time={time.time() - t_convert:.4f}s"
            )

            results = OrderedDict()
            area_inputs = OrderedDict()

            for area in self.areas:
                area_t0 = time.time()
                self.vlog(3, f"[area-start] {area}")

                input_tensor, bbox, vis_data = self.preprocess_area(frame_bgr, area)
                if input_tensor is None:
                    results[area] = {"status": "mask_failed"}
                    self.latest_results[area] = {"status": "mask_failed"}
                    self.vlog(2, f"[area-end] {area}: mask_failed in {time.time() - area_t0:.4f}s")
                    continue

                out = self.infer_area(input_tensor, area)

                if vis_data is not None:
                    vis_data["recon_patch_bgr"] = out.get("recon_patch_bgr")
                    vis_data["anom_patch_bgr"] = out.get("anom_patch_bgr")
                    vis_data["orig_patch_bgr"] = out.get("orig_patch_bgr")
                    area_inputs[area] = vis_data

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

            proc_elapsed = time.time() - self.process_start_time if self.process_start_time else 0.0
            avg_fps = self.processed_count / max(proc_elapsed, 1e-6)
            inst_fps = 1.0 / max(time.time() - process_t0, 1e-6)

            # if self.processed_count % self.log_every_n == 0 or self.processed_count == 1:
            msg_parts = [
                f"raw_frame={msg.header.frame_id}",
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

            if self.rulex_pub is not None:
                self.publish_rulex_result(results, frame_bgr, corr_frame_id, corr_stamp)

            dashboard_payload = pack_dashboard_state(
                msg_id=msg_id,
                corr_frame_id=corr_frame_id,
                corr_stamp=corr_stamp,
                frame_bgr=frame_bgr,
                area_inputs=area_inputs,
                latest_results=self.latest_results,
                jpeg_quality=self.args.zenoh_jpeg_quality,
            )
            self.zenoh_dashboard_pub.put(dashboard_payload)

            timeline_payload = pack_timeline_state(
                msg_id=msg_id,
                corr_frame_id=corr_frame_id,
                corr_stamp=corr_stamp,
                score_history=self.score_history,
                latest_results=self.latest_results,
            )
            self.zenoh_timeline_pub.put(timeline_payload)

            self.vlog(3, f"[process] total time={time.time() - process_t0:.4f}s")

            if self.max_frames is not None and self.processed_count >= self.max_frames:
                self.vlog(1, "[process] reached max_frames, shutting down")
                rclpy.shutdown()

        except Exception as e:
            print(f"[process] error processing frame #{msg_id}: {e}")
            self.get_logger().error(f"[process-error] {e}")
        finally:
            self.is_processing = False


def parse_args():
    p = argparse.ArgumentParser("Live ROS anomaly inference")

    p.add_argument("--camera_topic", default="/camera/back_view/image_raw")
    p.add_argument("--rulex_topic", default="/rulex/data")
    p.add_argument("--publish_rulex", action="store_true", default=True,
                   help="Publish RulexDetectionResult on ROS2")
    p.add_argument("--attach_image_on_anomaly", action="store_true",
                   help="Attach current frame to RulexDetectionResult if any area is anomalous")

    p.add_argument("--safety_area", default="ALL")
    p.add_argument("--area_names", nargs="+", default=["PLeft", "PRight", "RoboArm", "ConvBelt"])
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

    p.add_argument("--timeline_history", type=int, default=200)
    p.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    p.add_argument("--zenoh-dashboard-key", default="advis/vis/dashboard/state")
    p.add_argument("--zenoh-timeline-key", default="advis/vis/timeline/state")
    p.add_argument("--zenoh-jpeg-quality", type=int, default=85)
    p.add_argument("--zenoh-log-level", default="error")
    p.add_argument("--model_variant", default="old")
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
            if getattr(node, "zenoh_dashboard_pub", None) is not None:
                node.zenoh_dashboard_pub.undeclare()
            if getattr(node, "zenoh_timeline_pub", None) is not None:
                node.zenoh_timeline_pub.undeclare()
            if getattr(node, "zenoh_session", None) is not None:
                node.zenoh_session.close()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

